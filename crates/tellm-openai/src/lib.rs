//! OpenAI Responses API client (`POST /v1/responses`) - used for OpenAI,
//! Meta Model API, and xAI (Grok) via `base_url` switch.
//!
//! Mapping notes (checked 2026-07-04 against platform.openai.com, refreshed
//! 2026-07-09 against docs.x.ai for Grok 4.5, and checked 2026-07-09 against
//! dev.meta.ai - re-check live provider docs before changing these mappings):
//! - Reasoning: `reasoning: {"effort": ...}`; omit for ThinkingLevel::Off.
//!   Include `reasoning.encrypted_content` when reasoning is requested so
//!   stateless history can replay reasoning items later. xAI and Meta get
//!   the flag at every level, `Off` included: their models reason anyway and
//!   return encrypted reasoning only on request (checked 2026-11-16 against
//!   docs.x.ai and dev.meta.ai).
//! - State: `store: false`; history is carried by raw input/output items in
//!   `ChatRequest::history` / `ChatResponse::turn_items`.
//! - System prompts: OpenAI uses top-level `instructions`; xAI currently
//!   rejects `instructions`, so system prompts are sent as input message items.
//! - Web search: OpenAI and Meta Model API use `{"type": "web_search"}`;
//!   xAI uses `web_search` plus `x_search` for tellm's search toggle.
//! - Image generation: `{"type": "image_generation"}` tool (OpenAI only);
//!   Meta Model API supports image understanding, not image generation.
//!   Results come back as base64 in `image_generation_call` output items.
//! - Files/images: `input_file` / `input_image` content parts.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ThinkingLevel,
};

pub const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const META_MODEL_API_BASE_URL: &str = "https://api.meta.ai/v1";
pub const XAI_BASE_URL: &str = "https://api.x.ai/v1";
pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub struct Responses {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl Responses {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        Self {
            // Reuse the production connection pool across rooms and model
            // entries; authentication is still attached per request.
            http: default_http_client(),
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| OPENAI_BASE_URL.to_string())),
            api_key: api_key.into(),
        }
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

        let user_message = user_message(&request.input);
        let body = request_body(request, user_message.clone(), is_xai, is_meta);
        let response_body = self.send_request(body).await?;
        let output = response_body
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        if let Some(refusal) = extract_refusal(&output) {
            return Err(ProviderError::Refusal(refusal));
        }

        let text = extract_text(&output);
        let images = extract_images(&output);
        let mut turn_items = Vec::with_capacity(1 + output.len());
        turn_items.push(user_message);
        turn_items.extend(output);

        Ok(ChatResponse {
            text,
            images,
            turn_items,
        })
    }
}

impl Responses {
    async fn send_request(&self, body: Value) -> Result<Value, ProviderError> {
        let response = self
            .http
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
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
    // base URL is https://api.meta.ai/v1 and the launch model is muse-spark-1.1.
    base_url.is_some_and(|url| url.contains("api.meta.ai"))
        || model_name
            .get(..10)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("muse-spark"))
}

fn request_body(request: &ChatRequest, user_message: Value, is_xai: bool, is_meta: bool) -> Value {
    let mut input = Vec::new();
    if is_xai && let Some(system) = &request.system {
        // Checked 2026-07-04 against xAI Responses docs: `instructions` is
        // rejected, so system prompts must ride as compatible input messages.
        input.push(json!({
            "role": "system",
            "content": [{ "type": "input_text", "text": system }],
        }));
    }
    input.extend(request.history.clone());
    input.push(user_message);

    let mut body = json!({
        "model": request.model,
        "input": input,
        "store": false,
    });

    if !is_xai && let Some(system) = &request.system {
        // Checked 2026-07-04 against OpenAI Responses docs: top-level
        // `instructions` is the native system/developer guidance field.
        body["instructions"] = json!(system);
    }

    let effort = responses_effort(request.thinking, &request.model, is_xai);
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

/// Maps our five thinking levels onto each Responses model family's accepted
/// `reasoning.effort` vocabulary. `Off` always omits the field (every family
/// then reasons at its own default — Grok and Muse Spark cannot disable
/// reasoning at all, so omission is the closest honest mapping). The families
/// differ only at the top tier:
/// - GPT-5.6 (sol/terra/luna): none/low/medium/high/xhigh/**max** — Max→"max"
///   (checked 2026-07-09 against developers.openai.com latest-model guide).
/// - Older OpenAI (gpt-5.5 …): none/minimal/low/medium/high/xhigh — Max→"xhigh".
/// - Meta: minimal/low/medium/high/xhigh everywhere, no "none"; **max** only on
///   muse-spark-1.3, muse-wonder-1.0 and muse-ember-1.0 — Max→"max" there,
///   Max→"xhigh" on muse-spark-1.1/1.2 and the `-contributor` variants
///   (checked 2026-11-16 against dev.meta.ai choose-a-model and Responses docs).
/// - xAI Grok 4.6/4.7: low/medium/high/xhigh, default high, cannot disable;
///   Grok 4.5 stops at high and treats xhigh as high — Max→"xhigh" for all,
///   never "max" (checked 2026-11-16 against docs.x.ai reasoning docs).
fn responses_effort(
    thinking: ThinkingLevel,
    model_name: &str,
    is_xai: bool,
) -> Option<&'static str> {
    match thinking {
        ThinkingLevel::Off => None,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High => Some("high"),
        ThinkingLevel::Max if is_xai => Some("xhigh"),
        ThinkingLevel::Max if supports_max_effort(model_name) => Some("max"),
        ThinkingLevel::Max => Some("xhigh"),
    }
}

/// Whether the model accepts the `max` reasoning effort tier above `xhigh`.
/// The OpenAI GPT-5.6 family lists it, and so do three Meta models, matched by
/// exact id because their `-contributor` variants and older versions do not.
/// Everything else stays on the safe xhigh mapping until its docs are checked
/// and it is added here.
fn supports_max_effort(model_name: &str) -> bool {
    model_name.starts_with("gpt-5.6")
        || matches!(
            model_name,
            "muse-spark-1.3" | "muse-wonder-1.0" | "muse-ember-1.0"
        )
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

fn trim_trailing_slash(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effort_off_omits_field_for_every_family() {
        for (model, is_xai) in [
            ("gpt-5.6-sol", false),
            ("gpt-5.5", false),
            ("muse-spark-1.1", false),
            ("grok-4.5", true),
        ] {
            assert_eq!(
                responses_effort(ThinkingLevel::Off, model, is_xai),
                None,
                "{model}"
            );
        }
    }

    #[test]
    fn effort_low_medium_high_are_family_independent() {
        for (model, is_xai) in [
            ("gpt-5.6-sol", false),
            ("gpt-5.5", false),
            ("grok-4.5", true),
        ] {
            assert_eq!(
                responses_effort(ThinkingLevel::Low, model, is_xai),
                Some("low")
            );
            assert_eq!(
                responses_effort(ThinkingLevel::Medium, model, is_xai),
                Some("medium")
            );
            assert_eq!(
                responses_effort(ThinkingLevel::High, model, is_xai),
                Some("high")
            );
        }
    }

    #[test]
    fn max_maps_to_the_top_tier_each_family_accepts() {
        // GPT-5.6 family gains the new "max" tier.
        assert_eq!(
            responses_effort(ThinkingLevel::Max, "gpt-5.6-sol", false),
            Some("max")
        );
        assert_eq!(
            responses_effort(ThinkingLevel::Max, "gpt-5.6", false),
            Some("max")
        );
        // Older OpenAI tops out at xhigh.
        assert_eq!(
            responses_effort(ThinkingLevel::Max, "gpt-5.5", false),
            Some("xhigh")
        );
        // Meta lists "max" on three exact ids; older Muse Spark versions and
        // the contributor variants stop at xhigh.
        for model in ["muse-spark-1.3", "muse-wonder-1.0", "muse-ember-1.0"] {
            assert_eq!(
                responses_effort(ThinkingLevel::Max, model, false),
                Some("max"),
                "{model}"
            );
        }
        for model in [
            "muse-spark-1.1",
            "muse-spark-1.2",
            "muse-spark-1.3-contributor",
        ] {
            assert_eq!(
                responses_effort(ThinkingLevel::Max, model, false),
                Some("xhigh"),
                "{model}"
            );
        }
        // Grok 4.6+ accept xhigh and 4.5 treats it as high. No Grok accepts
        // "max", even under a name that would otherwise earn it — is_xai wins.
        assert_eq!(
            responses_effort(ThinkingLevel::Max, "grok-4.5", true),
            Some("xhigh")
        );
        assert_eq!(
            responses_effort(ThinkingLevel::Max, "grok-4.7", true),
            Some("xhigh")
        );
        assert_eq!(
            responses_effort(ThinkingLevel::Max, "gpt-5.6-sol", true),
            Some("xhigh")
        );
    }

    #[test]
    fn supports_max_effort_recognizes_only_verified_models() {
        assert!(supports_max_effort("gpt-5.6"));
        assert!(supports_max_effort("gpt-5.6-sol"));
        assert!(supports_max_effort("gpt-5.6-terra"));
        assert!(supports_max_effort("muse-spark-1.3"));
        assert!(supports_max_effort("muse-wonder-1.0"));
        assert!(supports_max_effort("muse-ember-1.0"));
        assert!(!supports_max_effort("gpt-5.5"));
        assert!(!supports_max_effort("grok-4.5"));
        assert!(!supports_max_effort("muse-spark-1.1"));
        assert!(!supports_max_effort("muse-spark-1.3-contributor"));
    }
}
