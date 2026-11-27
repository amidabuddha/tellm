//! OpenAI Responses API client (`POST /v1/responses`) - used for OpenAI,
//! Meta Model API, and xAI (Grok) via `base_url` switch.
//!
//! Mapping notes (checked 2026-07-04 against platform.openai.com, refreshed
//! 2026-07-09 against docs.x.ai for Grok 4.5, and checked 2026-07-09 against
//! dev.meta.ai - re-check live provider docs before changing these mappings):
//! - Reasoning: `reasoning: {"effort": ...}`. The two ends come from the
//!   model's config: `Off` sends its `off_effort` and omits the field when
//!   that is unset, and `Max` sends its `max_effort`.
//!   Include `reasoning.encrypted_content` when reasoning is requested so
//!   stateless history can replay reasoning items later. xAI and Meta get
//!   the flag at every level, `Off` included: their models reason anyway and
//!   return encrypted reasoning only on request (checked 2026-11-16 against
//!   docs.x.ai and dev.meta.ai).
//! - State: `store: false`; history is carried by raw input/output items in
//!   `ChatRequest::history` / `ChatResponse::turn_items`.
//! - System prompts: the first item of `input` on every provider, never
//!   top-level `instructions`, so the prompt is part of the cached prefix.
//!   The role is `developer` on OpenAI and Meta and `system` on xAI (checked
//!   2026-11-19, see `system_message`).
//! - Web search: OpenAI and Meta Model API use `{"type": "web_search"}`;
//!   xAI uses `web_search` plus `x_search` for tellm's search toggle.
//! - Image generation: `{"type": "image_generation"}` tool (OpenAI only);
//!   Meta Model API supports image understanding, not image generation.
//!   Results come back as base64 in `image_generation_call` output items.
//! - Files/images: `input_file` / `input_image` content parts. OpenAI takes
//!   PNG, JPEG, WEBP and GIF images (checked 2026-11-27, see
//!   `is_supported_openai_image`); xAI and Meta check their own.
//! - `status: "incomplete"`: a partial answer gets a closing note, one
//!   without text is an error, and `content_filter` is a refusal (checked
//!   2026-11-27, see `incomplete_note`).
//! - Context window: OpenAI's `error.code` `context_length_exceeded` becomes
//!   `ProviderError::ContextWindowExceeded`; xAI and Meta document no code
//!   for it (checked 2026-11-19, see [`CONTEXT_LENGTH_EXCEEDED`]).
//! - Usage: `usage.input_tokens` includes cache reads and writes;
//!   `input_tokens_details.cached_tokens` counts billed reads and
//!   `input_tokens_details.cache_write_tokens` billed writes, so ordinary input
//!   is the remainder; `output_tokens` includes
//!   `output_tokens_details.reasoning_tokens` (checked 2026-11-18 against
//!   developers.openai.com, prompt caching). Web search is billed per call,
//!   counted here as `web_search_call` output items (developers.openai.com
//!   pricing). xAI adds `usage.cost_in_usd_ticks`, the billed cost after
//!   caching discounts and server-side tool fees at 10^10 ticks per US dollar
//!   (checked 2026-11-18 against docs.x.ai, cost tracking). Meta Model API
//!   reports the same OpenAI-shaped fields and no cost.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    CUT_OFF_NOTE, ChatRequest, ChatResponse, ContentPart, GeneratedImage, OUTPUT_LIMIT_BEFORE_TEXT,
    Provider, ProviderError, ProviderReply, ThinkingLevel, Usage, check_image_types,
    read_provider_response, trim_trailing_slash, usage_count,
};

pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const META_MODEL_API_BASE_URL: &str = "https://api.meta.ai/v1";
pub const XAI_BASE_URL: &str = "https://api.x.ai/v1";
pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// `reasoning.effort` sent for `ThinkingLevel::Max` unless the model's
/// `max_effort` config names another tier. Which tier tops a model's list
/// differs per model ("max" on some, nothing above "xhigh" or "high" on
/// others), so tellm does not guess from the model name. "xhigh" is accepted
/// by GPT-5.2 and later, by every Muse model, and by every current Grok
/// (4.5 treats it as "high"); an older OpenAI model that stops at "high"
/// needs `max_effort = "high"`.
pub const DEFAULT_MAX_EFFORT: &str = "xhigh";

#[derive(Clone)]
pub struct Responses {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    max_effort: Option<String>,
    off_effort: Option<String>,
}

impl Responses {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        Self {
            // Reuse the production connection pool across rooms and model
            // entries; authentication is still attached per request.
            http: default_http_client(),
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| OPENAI_BASE_URL.to_string())),
            api_key: api_key.into(),
            max_effort: None,
            off_effort: None,
        }
    }

    /// Sets the `reasoning.effort` sent for `ThinkingLevel::Max`, from the
    /// model's `max_effort` config. `None` keeps [`DEFAULT_MAX_EFFORT`].
    pub fn with_max_effort(mut self, max_effort: Option<String>) -> Self {
        self.max_effort = max_effort;
        self
    }

    /// Sets the `reasoning.effort` sent for `ThinkingLevel::Off`, from the
    /// model's `off_effort` config. `None` omits `reasoning`, so the model
    /// reasons at its own default.
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
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| OPENAI_BASE_URL.to_string())),
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

impl Provider for Responses {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let is_xai = self.is_xai_request(request);
        let is_meta = self.is_meta_model_api_request(request);
        if is_xai && request.image_generation {
            return Err(ProviderError::Unsupported(
                "xAI Responses does not support OpenAI image_generation".to_string(),
            ));
        }
        if is_meta && request.image_generation {
            return Err(ProviderError::Unsupported(
                "Meta Model API Responses supports image understanding, not image_generation"
                    .to_string(),
            ));
        }

        // xAI and Meta publish their own image limits; they answer for them.
        if !is_xai && !is_meta {
            check_image_types(
                &request.input,
                "OpenAI Responses",
                is_supported_openai_image,
            )?;
        }

        let user_message = user_message(&request.input);
        let body = request_body(
            request,
            user_message.clone(),
            is_xai,
            is_meta,
            self.off_effort.as_deref(),
            self.max_effort.as_deref(),
        );
        let response_body = self.send_request(body, !is_xai && !is_meta).await?;
        let output = response_body
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        if let Some(refusal) = extract_refusal(&output) {
            return Err(ProviderError::Refusal(refusal));
        }

        let mut text = extract_text(&output);
        let images = extract_images(&output);
        if let Some(note) = incomplete_note(&response_body, &text, &images)? {
            text.push_str(note);
        }
        let usage = usage_from_response(&response_body, &output);
        let mut turn_items = Vec::with_capacity(1 + output.len());
        turn_items.push(user_message);
        turn_items.extend(output);

        Ok(ChatResponse {
            text,
            images,
            turn_items,
            usage,
        })
    }
}

/// Appended to a reply that stopped early for a reason other than the
/// output limit. Like [`CUT_OFF_NOTE`], only the user sees it.
pub const INCOMPLETE_NOTE: &str = "\n\n(Reply cut off before it finished.)";
/// Refusal text for a reply the provider's content filter stopped.
pub const CONTENT_FILTER_REFUSAL: &str = "the provider's content filter stopped the reply";

/// The closing note for a reply with `status: "incomplete"`, or the error
/// that replaces it.
///
/// Checked 2026-11-27 against developers.openai.com (reasoning guide;
/// Responses reference): a reply that runs out of tokens has `status:
/// "incomplete"` and `incomplete_details.reason: "max_output_tokens"`, which
/// also covers reaching the context window, and its output can be partial or
/// hold nothing but reasoning. The SDKs also name the reason
/// `content_filter`. Without text or an image the turn is an error rather
/// than an empty answer; a filtered reply is a refusal.
fn incomplete_note(
    body: &Value,
    text: &str,
    images: &[GeneratedImage],
) -> Result<Option<&'static str>, ProviderError> {
    if body.get("status").and_then(Value::as_str) != Some("incomplete") {
        return Ok(None);
    }
    let reason = body
        .pointer("/incomplete_details/reason")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    if reason == "content_filter" {
        return Err(ProviderError::Refusal(CONTENT_FILTER_REFUSAL.to_string()));
    }
    let output_limit = reason == "max_output_tokens";
    if text.trim().is_empty() && images.is_empty() {
        return Err(ProviderError::Api {
            status: 200,
            message: if output_limit {
                OUTPUT_LIMIT_BEFORE_TEXT.to_string()
            } else {
                format!("the reply stopped before any answer text (incomplete: {reason})")
            },
        });
    }
    Ok(Some(if output_limit {
        CUT_OFF_NOTE
    } else {
        INCOMPLETE_NOTE
    }))
}

/// Checked 2026-11-27 against developers.openai.com (images and vision,
/// input requirements): PNG, JPEG, WEBP and non-animated GIF.
fn is_supported_openai_image(media_type: &str) -> bool {
    matches!(
        media_type,
        "image/png" | "image/jpeg" | "image/webp" | "image/gif"
    )
}

/// xAI's `cost_in_usd_ticks` unit: 10^10 ticks per US dollar (checked
/// 2026-11-18 against docs.x.ai, cost tracking).
pub const USD_TICKS_PER_DOLLAR: f64 = 10_000_000_000.0;

/// Reads a Responses `usage` block into disjoint [`Usage`] buckets. Cache
/// reads and writes are counted inside `input_tokens`, so ordinary input is
/// what remains after both. Web searches are the `web_search_call` items in
/// `output`, since search is billed per call.
pub fn usage_from_response(body: &Value, output: &[Value]) -> Option<Usage> {
    let usage = body.get("usage")?;
    let cached_input_tokens = usage_count(usage, "/input_tokens_details/cached_tokens");
    let cache_write_tokens = usage_count(usage, "/input_tokens_details/cache_write_tokens");
    let input_tokens = usage_count(usage, "/input_tokens")
        .saturating_sub(cached_input_tokens)
        .saturating_sub(cache_write_tokens);
    let web_searches = output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("web_search_call"))
        .count() as u64;
    let reported_cost_usd = usage
        .get("cost_in_usd_ticks")
        .and_then(Value::as_u64)
        .map(|ticks| ticks as f64 / USD_TICKS_PER_DOLLAR);

    Some(Usage {
        input_tokens,
        cached_input_tokens,
        cache_write_tokens,
        output_tokens: usage_count(usage, "/output_tokens"),
        reasoning_tokens: usage_count(usage, "/output_tokens_details/reasoning_tokens"),
        web_searches,
        reported_cost_usd,
        context_tokens: None,
    })
}

impl Responses {
    /// `is_openai` says whether OpenAI's error codes apply to the reply.
    async fn send_request(&self, body: Value, is_openai: bool) -> Result<Value, ProviderError> {
        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        match read_provider_response(response).await? {
            ProviderReply::Success(body) => Ok(body),
            ProviderReply::Error { status, body } => Err(api_error(status, &body, is_openai)),
        }
    }

    fn is_xai_request(&self, request: &ChatRequest) -> bool {
        is_xai_endpoint(Some(&self.base_url), &request.model)
    }

    fn is_meta_model_api_request(&self, request: &ChatRequest) -> bool {
        is_meta_model_api_endpoint(Some(&self.base_url), &request.model)
    }
}

/// Whether a Responses request targets xAI rather than OpenAI. Shared with
/// the runtime so capability gating can't drift from request routing.
pub fn is_xai_endpoint(base_url: Option<&str>, model_name: &str) -> bool {
    base_url.is_some_and(|url| url.contains("api.x.ai"))
        || model_name
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("grok-"))
}

/// Whether a Responses request targets Meta Model API. Shared with the
/// runtime so `/imagegen` gating matches the provider backstop.
pub fn is_meta_model_api_endpoint(base_url: Option<&str>, model_name: &str) -> bool {
    // Checked 2026-07-09 against dev.meta.ai Model API docs: the Responses
    // base URL is https://api.meta.ai/v1. Rechecked 2026-11-17 (choose a
    // model): every Model API id starts with "muse-", as in muse-wonder-1.0,
    // muse-spark-1.3 and muse-ember-1.0, so the name test covers all of them
    // when a proxy hides the base URL.
    base_url.is_some_and(|url| url.contains("api.meta.ai"))
        || model_name
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("muse-"))
}

fn request_body(
    request: &ChatRequest,
    user_message: Value,
    is_xai: bool,
    is_meta: bool,
    off_effort: Option<&str>,
    max_effort: Option<&str>,
) -> Value {
    let mut input = Vec::new();
    if let Some(system) = &request.system {
        input.push(system_message(system, is_xai));
    }
    input.extend(request.history.clone());
    input.push(user_message);

    let mut body = json!({
        "model": request.model,
        "input": input,
        "store": false,
    });

    let effort = responses_effort(request.thinking, off_effort, max_effort);
    if let Some(effort) = effort {
        // Checked 2026-07-09 against OpenAI/xAI/Meta Responses reasoning docs.
        body["reasoning"] = json!({ "effort": effort });
    }
    if effort.is_some() || is_xai || is_meta {
        // Checked 2026-11-16 against docs.x.ai (reasoning, best practices)
        // and dev.meta.ai (Responses API). Grok and Muse models reason even
        // when no effort is sent, and both APIs return `encrypted_content`
        // only when asked (grok-4.7 always does). Under `store: false` Meta
        // answers a replayed reasoning item that lacks it with HTTP 400, so
        // `Off` must still ask. OpenAI returns it by default and keeps
        // accepting the flag.
        body["include"] = json!(["reasoning.encrypted_content"]);
    }

    let tools = tools(request, is_xai);
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }

    body
}

/// The system prompt as the first item of `input`, on every Responses
/// provider. Top-level `instructions` is never sent, so the prompt is part of
/// the prefix the provider caches. Checked 2026-11-19:
/// - OpenAI (developers.openai.com, latest model and keep the prefix
///   unchanged): models starting with GPT-6.2 append top-level
///   `instructions` after `input` and bill them at the uncached input rate;
///   reusable instructions belong in a `developer` message as the first item
///   in `input`. Input messages take the roles user, assistant, system and
///   developer, with `input_text` content parts (Responses reference).
/// - Meta (dev.meta.ai, Responses API): input messages take the roles
///   developer, system, user and assistant; `developer` is recommended for
///   behavior instructions, and `system` content applies at the same level.
/// - xAI (docs.x.ai, Responses reference and prompt caching): its examples
///   put the system prompt first with the `system` role and never mention
///   `developer`, so xAI keeps `system`.
fn system_message(system: &str, is_xai: bool) -> Value {
    let role = if is_xai { "system" } else { "developer" };
    json!({
        "role": role,
        "content": [{ "type": "input_text", "text": system }],
    })
}

fn tools(request: &ChatRequest, is_xai: bool) -> Vec<Value> {
    let mut tools = Vec::new();
    if request.web_search {
        if is_xai {
            // Checked 2026-07-09 against docs.x.ai Web Search and X Search
            // docs: both tools are available through the Responses surface.
            tools.push(json!({ "type": "web_search" }));
            tools.push(json!({ "type": "x_search" }));
        } else {
            // Checked 2026-07-04 against OpenAI web search docs and
            // 2026-07-09 against Meta Model API search-grounding docs.
            tools.push(json!({ "type": "web_search" }));
        }
    }
    if request.image_generation {
        // Checked 2026-07-04 against OpenAI image generation tool docs.
        tools.push(json!({ "type": "image_generation" }));
    }
    tools
}

fn user_message(parts: &[ContentPart]) -> Value {
    json!({
        "role": "user",
        "content": content_parts_to_responses(parts),
    })
}

fn content_parts_to_responses(parts: &[ContentPart]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => json!({
                "type": "input_text",
                "text": text,
            }),
            ContentPart::Image { media_type, base64 } => json!({
                "type": "input_image",
                "image_url": data_url(media_type, base64),
            }),
            ContentPart::Document {
                media_type,
                base64,
                name,
            } => json!({
                "type": "input_file",
                "filename": name.as_deref().unwrap_or("document"),
                "file_data": data_url(media_type, base64),
            }),
        })
        .collect()
}

/// Maps our five thinking levels onto `reasoning.effort`. Low, medium and
/// high mean the same everywhere; the two ends are the model's config.
///
/// `Off` sends the model's `off_effort` and omits the field when that is
/// unset, so every family then reasons at its own default. What the lowest
/// tier is differs per model: some OpenAI models accept "none", while Grok
/// and Muse models cannot disable reasoning, and Muse Wonder's default is
/// "max" with "minimal" as its shortest pass (checked 2026-11-17 against
/// dev.meta.ai reasoning and Muse Wonder docs).
///
/// The tier above `high` is the part that differs per model, so it is the
/// model's `max_effort` config rather than a list of model names here, and
/// [`DEFAULT_MAX_EFFORT`] when that is unset. Checked 2026-11-16:
/// - OpenAI (developers.openai.com model pages): GPT-5.6, GPT-6 and GPT-6.1
///   list "max" above "xhigh"; older families stop at "xhigh".
/// - Meta (dev.meta.ai choose-a-model, Responses docs): "max" only on
///   muse-wonder-1.0, muse-spark-1.3 and muse-ember-1.0; "xhigh" elsewhere.
/// - xAI (docs.x.ai reasoning docs): Grok 4.6/4.7 stop at "xhigh", and Grok
///   4.5 treats "xhigh" as "high". No Grok model lists "max".
fn responses_effort<'a>(
    thinking: ThinkingLevel,
    off_effort: Option<&'a str>,
    max_effort: Option<&'a str>,
) -> Option<&'a str> {
    thinking.effort(off_effort, max_effort, DEFAULT_MAX_EFFORT)
}

fn extract_text(output: &[Value]) -> String {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flat_map(|content| content.iter())
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

fn extract_refusal(output: &[Value]) -> Option<String> {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flat_map(|content| content.iter())
        .find(|block| block.get("type").and_then(Value::as_str) == Some("refusal"))
        .and_then(|block| block.get("refusal").and_then(Value::as_str))
        .map(str::to_string)
}

fn extract_images(output: &[Value]) -> Vec<GeneratedImage> {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image_generation_call"))
        .filter_map(|item| item.get("result").and_then(Value::as_str))
        .map(|base64| GeneratedImage {
            media_type: "image/png".to_string(),
            base64: base64.to_string(),
        })
        .collect()
}

/// OpenAI's error code for input that exceeds the model's context window,
/// read from `error.code`. Checked 2026-11-19 against developers.openai.com:
/// the Responses reference says such a request "will fail with a 400 error"
/// without naming a code, and OpenAI's error tables define this one as
/// "Input exceeds the model's context window" (Agents API errors and
/// recovery). xAI and Meta document no code for it (docs.x.ai debugging
/// errors and Responses reference; dev.meta.ai Responses API, which says
/// only that an over-context request fails with HTTP 400), so their errors
/// are not read this way.
pub const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

/// Turns an error response into a [`ProviderError`], telling a request over
/// the context window apart from other API errors where the provider
/// documents a code for it.
fn api_error(status: u16, body: &Value, is_openai: bool) -> ProviderError {
    let message = api_error_message(body);
    if is_openai
        && body.pointer("/error/code").and_then(Value::as_str) == Some(CONTEXT_LENGTH_EXCEEDED)
    {
        ProviderError::ContextWindowExceeded { status, message }
    } else {
        ProviderError::Api { status, message }
    }
}

fn api_error_message(value: &Value) -> String {
    value
        .pointer("/error/message")
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("unknown OpenAI Responses API error")
        .to_string()
}

fn data_url(media_type: &str, base64: &str) -> String {
    format!("data:{media_type};base64,{base64}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_sends_the_configured_tier_or_omits_the_field() {
        assert_eq!(responses_effort(ThinkingLevel::Off, None, None), None);
        assert_eq!(
            responses_effort(ThinkingLevel::Off, None, Some("max")),
            None,
            "the configured max tier must not leak into off"
        );
        assert_eq!(
            responses_effort(ThinkingLevel::Off, Some("minimal"), Some("max")),
            Some("minimal")
        );
    }

    #[test]
    fn effort_low_medium_high_ignore_the_configured_ends() {
        for (off_effort, max_effort) in [(None, None), (Some("none"), Some("max"))] {
            assert_eq!(
                responses_effort(ThinkingLevel::Low, off_effort, max_effort),
                Some("low")
            );
            assert_eq!(
                responses_effort(ThinkingLevel::Medium, off_effort, max_effort),
                Some("medium")
            );
            assert_eq!(
                responses_effort(ThinkingLevel::High, off_effort, max_effort),
                Some("high")
            );
        }
    }

    #[test]
    fn usage_splits_cache_reads_and_writes_out_of_input() {
        let body = json!({
            "usage": {
                "input_tokens": 12000,
                "input_tokens_details": { "cached_tokens": 4000, "cache_write_tokens": 7000 },
                "output_tokens": 900,
                "output_tokens_details": { "reasoning_tokens": 600 },
                "total_tokens": 12900
            }
        });
        let output = [
            json!({ "type": "web_search_call", "id": "ws_1" }),
            json!({ "type": "web_search_call", "id": "ws_2" }),
            json!({ "type": "message", "content": [] }),
        ];
        let usage = usage_from_response(&body, &output).unwrap();
        assert_eq!(usage.input_tokens, 1000);
        assert_eq!(usage.cached_input_tokens, 4000);
        assert_eq!(usage.cache_write_tokens, 7000);
        assert_eq!(usage.output_tokens, 900);
        assert_eq!(usage.reasoning_tokens, 600);
        assert_eq!(usage.web_searches, 2);
        assert_eq!(usage.reported_cost_usd, None);
    }

    #[test]
    fn usage_reads_the_xai_billed_cost_and_tolerates_missing_details() {
        let body = json!({
            "usage": { "input_tokens": 199, "output_tokens": 1, "cost_in_usd_ticks": 158500 }
        });
        let usage = usage_from_response(&body, &[]).unwrap();
        assert_eq!(usage.input_tokens, 199);
        assert_eq!(usage.cached_input_tokens, 0);
        assert_eq!(usage.output_tokens, 1);
        assert_eq!(usage.reported_cost_usd, Some(0.000_015_85));

        assert_eq!(usage_from_response(&json!({}), &[]), None);
    }

    #[test]
    fn max_sends_the_configured_tier_or_the_safe_default() {
        // No model names are consulted: an unset value sends the default tier.
        assert_eq!(
            responses_effort(ThinkingLevel::Max, None, None),
            Some("xhigh")
        );
        assert_eq!(DEFAULT_MAX_EFFORT, "xhigh");
        assert_eq!(
            responses_effort(ThinkingLevel::Max, Some("low"), Some("max")),
            Some("max")
        );
        assert_eq!(
            responses_effort(ThinkingLevel::Max, None, Some("high")),
            Some("high")
        );
    }
}
