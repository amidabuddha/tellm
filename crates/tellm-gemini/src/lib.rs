//! Google Gemini Interactions API client (`POST /v1beta/interactions`).
//!
//! Mapping notes (checked 2026-07-26 against ai.google.dev Interactions API,
//! Gemini 3.6 Flash migration, API key, thinking, Google Search, and image
//! generation docs — re-check live provider docs before changing these
//! mappings):
//! - State: `store: false`; stateless history is carried as raw Interactions
//!   `Step` objects in `ChatRequest::history` and replayed via `input`.
//! - Turn items: return the new `user_input` step plus EVERY response step
//!   verbatim, including `thought`, `google_search_call`, and
//!   `google_search_result` signatures. Stateless mode requires those
//!   signatures back unchanged on future turns.
//! - Thinking: `generation_config.thinking_level` for low/medium/high.
//!   `Off` sends the model's `off_effort` and omits the field when that is
//!   unset; `Max` sends the model's `max_effort`, or `high` when unset,
//!   because `high` is the top level of Gemini 4 Carbon and Argon. Gemini 4
//!   Neon and Hydrogen list `xhigh` above it, which a model config opts into
//!   with `max_effort = "xhigh"` (checked 2026-11-27 against the model pages
//!   on aistudio.google.com/docs, where ai.google.dev now redirects).
//!   Gemini 4 Carbon, Neon, Argon, 3.8 Flash and 3.6 Flash support all
//!   three mapped levels; Carbon, Neon, Argon and 3.8 Flash reject
//!   `minimal`, so `low` is the lowest `off_effort` they take (checked
//!   2026-11-16; Carbon 2026-11-17; Neon 2026-11-27 against Google Cloud's
//!   Gemini 4 Neon developer's guide). Requests do not send the deprecated
//!   sampling parameters and always end in a `user_input` step.
//! - Web search: `tools: [{type: "google_search", search_types:
//!   ["web_search"]}]`.
//! - Images/documents: native `image` and `document` content blocks with
//!   inline base64 `data`.
//! - Image generation: only Gemini image model ids are eligible, recognized
//!   by name (`*-image*` or `*nano-banana*`, see [`is_image_generation_model`]);
//!   request `response_format: {"type": "image"}`.
//! - Usage: the Interaction `usage` object reports `total_input_tokens`
//!   (the prompt), `total_cached_tokens` (the cached part of that prompt),
//!   `total_output_tokens`, `total_thought_tokens`, `total_tool_use_tokens`,
//!   and `grounding_tool_count` entries with a `count` (checked 2026-11-18
//!   against Google's Interactions API reference and the Interactions `Usage`
//!   type of googleapis js-genai). Output is billed including thinking, and
//!   Google Search grounding is billed per search query (ai.google.dev
//!   pricing). Tool-use prompt tokens are not priced: for Google Search they
//!   are the grounding results, and Google's pricing says input tokens from
//!   Grounding with Google Search are not charged (checked 2026-11-18
//!   against cloud.google.com Vertex AI pricing; the ai.google.dev pricing
//!   page bills grounding per query and lists no token charge for it).

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ProviderReply,
    ThinkingLevel, Usage, read_provider_response, usage_count,
};

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// `thinking_level` sent for `ThinkingLevel::Max` unless the model's
/// `max_effort` config names another level. "high" is the top level of
/// Gemini 4 Carbon and Argon; Gemini 4 Neon and Hydrogen also take "xhigh"
/// (checked 2026-11-27), which their config sets as `max_effort`.
pub const DEFAULT_MAX_EFFORT: &str = "high";

#[derive(Clone)]
pub struct Gemini {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    max_effort: Option<String>,
    off_effort: Option<String>,
}

impl Gemini {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        Self {
            // Production turns share a keep-alive/TLS pool; custom-timeout
            // clients remain independent for deterministic tests.
            http: default_http_client(),
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string())),
            api_key: api_key.into(),
            max_effort: None,
            off_effort: None,
        }
    }

    /// Sets the `thinking_level` sent for `ThinkingLevel::Max`, from the
    /// model's `max_effort` config. `None` keeps [`DEFAULT_MAX_EFFORT`].
    pub fn with_max_effort(mut self, max_effort: Option<String>) -> Self {
        self.max_effort = max_effort;
        self
    }

    /// Sets the `thinking_level` sent for `ThinkingLevel::Off`, from the
    /// model's `off_effort` config. `None` omits the field, so the model's
    /// own default applies.
    pub fn with_off_effort(mut self, off_effort: Option<String>) -> Self {
        self.off_effort = off_effort;
        self
    }

    pub fn with_base_url_and_timeout(
        api_key: impl Into<String>,
        base_url: Option<String>,
        request_timeout: Duration,
    ) -> Self {
        Self {
            http: reqwest::Client::builder()
                .connect_timeout(PROVIDER_CONNECT_TIMEOUT.min(request_timeout))
                .timeout(request_timeout)
                .build()
                .expect("valid reqwest client configuration"),
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string())),
            api_key: api_key.into(),
            max_effort: None,
            off_effort: None,
        }
    }
}

fn default_http_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(PROVIDER_CONNECT_TIMEOUT)
                .timeout(PROVIDER_REQUEST_TIMEOUT)
                .build()
                .expect("valid reqwest client configuration")
        })
        .clone()
}

/// Name fragments that mark a Gemini image generation model. `-image` covers
/// gemini-3.1-flash-image and gemini-3-pro-image; `nano-banana` covers
/// gemini-nano-banana-2.1, the current image model, whose id no longer
/// carries `-image` (checked 2026-11-16 against ai.google.dev models and
/// image generation docs).
const IMAGE_MODEL_MARKERS: [&str; 2] = ["-image", "nano-banana"];

/// Whether a Gemini model id names an image generation model. This is a
/// fail-closed naming rule, not a capability lookup: an id that matches no
/// marker is refused image generation before any request is sent.
pub fn is_image_generation_model(model_name: &str) -> bool {
    let model_name = model_name.to_ascii_lowercase();
    IMAGE_MODEL_MARKERS
        .iter()
        .any(|marker| model_name.contains(marker))
}

impl Provider for Gemini {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let user_step = user_input_step(&request.input)?;
        let body = request_body(
            request,
            user_step.clone(),
            self.off_effort.as_deref(),
            self.max_effort.as_deref(),
        )?;
        let response_body = self.send_request(body).await?;
        let steps = response_body
            .get("steps")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let status = response_body.get("status").and_then(Value::as_str);
        if status.is_some_and(|status| status != "completed") {
            return Err(ProviderError::Unsupported(format!(
                "Gemini interaction status {status:?} requires unsupported follow-up handling"
            )));
        }

        let text = extract_text(&steps);
        let images = extract_images(&steps);
        let usage = usage_from_response(&response_body);
        let mut turn_items = Vec::with_capacity(1 + steps.len());
        turn_items.push(user_step);
        turn_items.extend(steps);

        Ok(ChatResponse {
            text,
            images,
            turn_items,
            usage,
        })
    }
}

/// Reads an Interaction `usage` object into disjoint [`Usage`] buckets. The
/// cached tokens are part of the prompt total, so ordinary input is what
/// remains; thinking is billed as output, so it is added to it. Grounding
/// tool counts become web searches.
pub fn usage_from_response(body: &Value) -> Option<Usage> {
    let usage = body.get("usage")?;
    let cached_input_tokens = usage_count(usage, "/total_cached_tokens");
    let thought_tokens = usage_count(usage, "/total_thought_tokens");
    let web_searches = usage
        .get("grounding_tool_count")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|entry| usage_count(entry, "/count"))
        .fold(0_u64, u64::saturating_add);

    Some(Usage {
        input_tokens: usage_count(usage, "/total_input_tokens").saturating_sub(cached_input_tokens),
        cached_input_tokens,
        cache_write_tokens: 0,
        output_tokens: usage_count(usage, "/total_output_tokens").saturating_add(thought_tokens),
        reasoning_tokens: thought_tokens,
        web_searches,
        reported_cost_usd: None,
        context_tokens: None,
    })
}

impl Gemini {
    async fn send_request(&self, body: Value) -> Result<Value, ProviderError> {
        let response = self
            .http
            .post(format!("{}/v1beta/interactions", self.base_url))
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await?;

        match read_provider_response(response).await? {
            ProviderReply::Success(body) => Ok(body),
            ProviderReply::Error { status, body } => Err(ProviderError::Api {
                status,
                message: api_error_message(&body),
            }),
        }
    }
}

fn request_body(
    request: &ChatRequest,
    user_step: Value,
    off_effort: Option<&str>,
    max_effort: Option<&str>,
) -> Result<Value, ProviderError> {
    let mut input = request.history.clone();
    input.push(user_step);

    let mut body = json!({
        "model": request.model,
        "input": input,
        "store": false,
        "stream": false,
    });

    if let Some(system) = &request.system
        && !system.is_empty()
    {
        body["system_instruction"] = json!(system);
    }

    let mut generation_config = serde_json::Map::new();
    if let Some(thinking) = thinking_level(request.thinking, off_effort, max_effort) {
        generation_config.insert("thinking_level".to_string(), json!(thinking));
    }
    if !generation_config.is_empty() {
        body["generation_config"] = Value::Object(generation_config);
    }

    if request.web_search {
        body["tools"] = json!([
            {
                "type": "google_search",
                "search_types": ["web_search"]
            }
        ]);
    }

    if request.image_generation {
        if !is_image_generation_model(&request.model) {
            return Err(ProviderError::Unsupported(format!(
                "Gemini image generation requires an image-capable model such as gemini-nano-banana-2.1, not {}",
                request.model
            )));
        }
        body["response_format"] = json!({ "type": "image" });
    }

    Ok(body)
}

fn user_input_step(input: &[ContentPart]) -> Result<Value, ProviderError> {
    let content = input
        .iter()
        .map(content_part)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "type": "user_input",
        "content": content,
    }))
}

fn content_part(part: &ContentPart) -> Result<Value, ProviderError> {
    match part {
        ContentPart::Text { text } => Ok(json!({
            "type": "text",
            "text": text,
        })),
        ContentPart::Image { media_type, base64 } => {
            if !is_supported_image(media_type) {
                return Err(ProviderError::Unsupported(format!(
                    "Gemini Interactions does not support image media type {media_type}"
                )));
            }
            Ok(json!({
                "type": "image",
                "data": base64,
                "mime_type": media_type,
            }))
        }
        ContentPart::Document {
            media_type, base64, ..
        } => {
            if !is_supported_document(media_type) {
                return Err(ProviderError::Unsupported(format!(
                    "Gemini Interactions does not support document media type {media_type}"
                )));
            }
            Ok(json!({
                "type": "document",
                "data": base64,
                "mime_type": media_type,
            }))
        }
    }
}

/// Off omits the level unless the model's `off_effort` config names one,
/// and Max takes `max_effort` before [`DEFAULT_MAX_EFFORT`].
fn thinking_level<'a>(
    thinking: ThinkingLevel,
    off_effort: Option<&'a str>,
    max_effort: Option<&'a str>,
) -> Option<&'a str> {
    match thinking {
        ThinkingLevel::Off => off_effort,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High => Some("high"),
        ThinkingLevel::Max => Some(max_effort.unwrap_or(DEFAULT_MAX_EFFORT)),
    }
}

fn extract_text(steps: &[Value]) -> String {
    let mut text = String::new();
    for step in steps {
        if step.get("type").and_then(Value::as_str) != Some("model_output") {
            continue;
        }
        for content in step
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if content.get("type").and_then(Value::as_str) == Some("text")
                && let Some(part) = content.get("text").and_then(Value::as_str)
            {
                text.push_str(part);
            }
        }
    }
    text
}

fn extract_images(steps: &[Value]) -> Vec<GeneratedImage> {
    steps
        .iter()
        .filter(|step| step.get("type").and_then(Value::as_str) == Some("model_output"))
        .flat_map(|step| {
            step.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .filter(|content| content.get("type").and_then(Value::as_str) == Some("image"))
        .filter_map(|content| {
            Some(GeneratedImage {
                media_type: content
                    .get("mime_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/png")
                    .to_string(),
                base64: content.get("data").and_then(Value::as_str)?.to_string(),
            })
        })
        .collect()
}

/// Error messages pass through as [`ProviderError::Api`]. Google's API
/// error codes include none for input over the model's context window
/// (checked 2026-11-19 against ai.google.dev, API errors), so tellm does not
/// guess one from the message.
fn api_error_message(body: &Value) -> String {
    body.get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
        .or_else(|| body.get("message").and_then(Value::as_str))
        .unwrap_or("unknown Gemini API error")
        .to_string()
}

fn is_supported_image(media_type: &str) -> bool {
    matches!(
        media_type,
        "image/png"
            | "image/jpeg"
            | "image/webp"
            | "image/heic"
            | "image/heif"
            | "image/gif"
            | "image/bmp"
            | "image/tiff"
    )
}

fn is_supported_document(media_type: &str) -> bool {
    matches!(media_type, "application/pdf" | "text/csv")
}

fn trim_trailing_slash(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}
