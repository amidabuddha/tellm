//! Google Gemini Interactions API client (`POST /v1beta/interactions`).
//!
//! Mapping notes (checked 2026-07-05 against ai.google.dev Interactions API,
//! API key, thinking, Google Search, and image generation docs — re-check
//! live provider docs before changing these mappings):
//! - State: `store: false`; stateless history is carried as raw Interactions
//!   `Step` objects in `ChatRequest::history` and replayed via `input`.
//! - Turn items: return the new `user_input` step plus EVERY response step
//!   verbatim, including `thought`, `google_search_call`, and
//!   `google_search_result` signatures. Stateless mode requires those
//!   signatures back unchanged on future turns.
//! - Thinking: `generation_config.thinking_level` for low/medium/high; omit
//!   for `Off`; map `Max` to `high` because Interactions has no `max`.
//! - Web search: `tools: [{type: "google_search", search_types:
//!   ["web_search"]}]`.
//! - Images/documents: native `image` and `document` content blocks with
//!   inline base64 `data`.
//! - Image generation: only Gemini image model ids (`*-image*`) are eligible;
//!   request `response_format: {"type": "image"}`.

use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ThinkingLevel,
};

pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub struct Gemini {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl Gemini {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        Self::with_base_url_and_timeout(api_key, base_url, PROVIDER_REQUEST_TIMEOUT)
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
        }
    }
}

pub fn is_image_generation_model(model_name: &str) -> bool {
    model_name.to_ascii_lowercase().contains("-image")
}

impl Provider for Gemini {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let user_step = user_input_step(&request.input)?;
        let body = request_body(request, user_step.clone())?;
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
        let mut turn_items = Vec::with_capacity(1 + steps.len());
        turn_items.push(user_step);
        turn_items.extend(steps);

        Ok(ChatResponse {
            text,
            images,
            turn_items,
        })
    }
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

        let status = response.status();
        let response_body: Value = response.json().await?;
        if !status.is_success() {
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: api_error_message(&response_body),
            });
        }

        Ok(response_body)
    }
}

fn request_body(request: &ChatRequest, user_step: Value) -> Result<Value, ProviderError> {
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
    if let Some(thinking) = thinking_level(request.thinking) {
        generation_config.insert("thinking_level".to_string(), json!(thinking));
    }
    if let Some(max_tokens) = request.max_tokens {
        generation_config.insert("max_output_tokens".to_string(), json!(max_tokens));
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
                "Gemini image generation requires an image-capable model such as gemini-3.1-flash-image, not {}",
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

fn thinking_level(thinking: ThinkingLevel) -> Option<&'static str> {
    match thinking {
        ThinkingLevel::Off => None,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High | ThinkingLevel::Max => Some("high"),
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
