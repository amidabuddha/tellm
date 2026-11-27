//! OpenAI chat-completions-compatible client (`POST /chat/completions`).
//!
//! This is the catch-all wire format: Ollama (`http://localhost:11434/v1`),
//! DeepSeek, OpenRouter, Moonshot, and anything else that speaks the
//! completions dialect. `base_url` is mandatory — there is no default,
//! because "compat" always means "someone else's endpoint".
//!
//! Mapping notes (checked 2026-07-04 against platform.openai.com and
//! docs.ollama.com - re-check live provider docs before changing these mappings):
//! - Reasoning: `reasoning_effort` where the provider supports it; silently
//!   ignored by those that don't (that's the dialect's convention). `Off`
//!   omits it unless the model's `off_effort` names a value.
//! - Web search / image generation: not part of this dialect —
//!   `ProviderError::Unsupported` so the user gets told instead of guessing.
//! - Images: `image_url` content parts (data URLs); documents: unsupported.
//! - Context window: only OpenRouter documents an error for it, so only its
//!   `error.metadata.error_type` becomes `ProviderError::ContextWindowExceeded`
//!   (checked 2026-11-19, see [`OPENROUTER_CONTEXT_LENGTH_EXCEEDED`]).
//! - Usage: `usage.prompt_tokens` includes cached input;
//!   `prompt_tokens_details.cached_tokens` counts cache reads and
//!   `prompt_tokens_details.cache_write_tokens` cache writes where the endpoint
//!   bills them, and `completion_tokens` includes
//!   `completion_tokens_details.reasoning_tokens` (checked 2026-11-18 against
//!   openrouter.ai usage accounting). DeepSeek reports cache reads as
//!   `prompt_cache_hit_tokens` instead (checked 2026-11-18 against
//!   api-docs.deepseek.com, context caching). OpenRouter adds `usage.cost`,
//!   the amount charged in credits, which are US dollars; it is trusted only
//!   from openrouter.ai, because the field has no shared meaning in the
//!   dialect.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ThinkingLevel,
    Usage, usage_count,
};

pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// `reasoning_effort` sent for `ThinkingLevel::Max` unless the model's
/// `max_effort` config names another tier. "max" is not in the dialect and
/// is an invalid enum value on endpoints that do not define it, so the
/// default stays at "high".
pub const DEFAULT_MAX_EFFORT: &str = "high";

#[derive(Clone)]
pub struct Compat {
    http: reqwest::Client,
    chat_completions_url: String,
    api_key: Option<String>,
    max_effort: Option<String>,
    off_effort: Option<String>,
}

impl Compat {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::with_client(api_key, base_url, default_http_client())
    }

    pub fn with_base_url_and_timeout(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        request_timeout: Duration,
    ) -> Self {
        let http = build_http_client(request_timeout);
        Self::with_client(api_key, base_url, http)
    }

    /// Sets the `reasoning_effort` sent for `ThinkingLevel::Max`, from the
    /// model's `max_effort` config. `None` keeps [`DEFAULT_MAX_EFFORT`].
    pub fn with_max_effort(mut self, max_effort: Option<String>) -> Self {
        self.max_effort = max_effort;
        self
    }

    /// Sets the `reasoning_effort` sent for `ThinkingLevel::Off`, from the
    /// model's `off_effort` config. `None` omits the field, so the endpoint's
    /// default applies.
    pub fn with_off_effort(mut self, off_effort: Option<String>) -> Self {
        self.off_effort = off_effort;
        self
    }

    fn with_client(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        http: reqwest::Client,
    ) -> Self {
        let api_key = api_key.into();
        Self {
            http,
            chat_completions_url: chat_completions_url(base_url.into()),
            api_key: (!api_key.trim().is_empty()).then_some(api_key),
            max_effort: None,
            off_effort: None,
        }
    }
}

impl Provider for Compat {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        validate_supported(request)?;

        let user_message = user_message(&request.input);
        let body = request_body(
            request,
            user_message.clone(),
            self.off_effort.as_deref(),
            self.max_effort.as_deref(),
        );
        let response_body = self.send_request(body).await?;
        let choice = response_body
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .cloned()
            .unwrap_or_default();
        let message = choice.get("message").cloned().unwrap_or_else(|| {
            json!({
                "role": "assistant",
                "content": null,
            })
        });

        if let Some(refusal) = message.get("refusal").and_then(Value::as_str) {
            return Err(ProviderError::Refusal(refusal.to_string()));
        }
        if choice.get("finish_reason").and_then(Value::as_str) == Some("content_filter") {
            return Err(ProviderError::Refusal(
                "response blocked by content filter".to_string(),
            ));
        }

        let text = extract_text(&message);
        let usage = usage_from_response(
            &response_body,
            is_openrouter_url(&self.chat_completions_url),
        );
        Ok(ChatResponse {
            text,
            images: Vec::<GeneratedImage>::new(),
            turn_items: vec![user_message, message],
            usage,
        })
    }
}

/// Whether a chat-completions URL points at OpenRouter, whose `usage.cost` is
/// the amount actually charged.
pub fn is_openrouter_url(url: &str) -> bool {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "openrouter.ai" || host.ends_with(".openrouter.ai"))
}

/// Reads a chat-completions `usage` block into disjoint [`Usage`] buckets.
/// `prompt_tokens` includes cached input, so ordinary input is what remains
/// after cache reads and writes. `trust_reported_cost` accepts `usage.cost`
/// as US dollars; only OpenRouter documents that meaning.
pub fn usage_from_response(body: &Value, trust_reported_cost: bool) -> Option<Usage> {
    let usage = body.get("usage")?;
    let cached_input_tokens = match usage.get("prompt_cache_hit_tokens") {
        // DeepSeek's own field for cache reads.
        Some(_) => usage_count(usage, "/prompt_cache_hit_tokens"),
        None => usage_count(usage, "/prompt_tokens_details/cached_tokens"),
    };
    let cache_write_tokens = usage_count(usage, "/prompt_tokens_details/cache_write_tokens");
    let input_tokens = usage_count(usage, "/prompt_tokens")
        .saturating_sub(cached_input_tokens)
        .saturating_sub(cache_write_tokens);
    let reported_cost_usd = trust_reported_cost
        .then(|| usage.get("cost").and_then(Value::as_f64))
        .flatten()
        .filter(|cost| cost.is_finite() && *cost >= 0.0);

    Some(Usage {
        input_tokens,
        cached_input_tokens,
        cache_write_tokens,
        output_tokens: usage_count(usage, "/completion_tokens"),
        reasoning_tokens: usage_count(usage, "/completion_tokens_details/reasoning_tokens"),
        web_searches: 0,
        reported_cost_usd,
        context_tokens: None,
    })
}

impl Compat {
    async fn send_request(&self, body: Value) -> Result<Value, ProviderError> {
        let mut request = self.http.post(&self.chat_completions_url);
        if let Some(api_key) = &self.api_key {
            request = request.bearer_auth(api_key);
        }
        let response = request.json(&body).send().await?;

        let status = response.status();
        let response_body: Value = response.json().await?;
        if !status.is_success() {
            return Err(api_error(
                status.as_u16(),
                &response_body,
                is_openrouter_url(&self.chat_completions_url),
            ));
        }

        Ok(response_body)
    }
}

fn validate_supported(request: &ChatRequest) -> Result<(), ProviderError> {
    if request.web_search {
        return Err(ProviderError::Unsupported(
            "chat completions compat does not support provider-native web search".to_string(),
        ));
    }
    if request.image_generation {
        return Err(ProviderError::Unsupported(
            "chat completions compat does not support image generation".to_string(),
        ));
    }
    if request
        .input
        .iter()
        .any(|part| matches!(part, ContentPart::Document { .. }))
    {
        return Err(ProviderError::Unsupported(
            "chat completions compat does not support documents".to_string(),
        ));
    }
    Ok(())
}

fn request_body(
    request: &ChatRequest,
    user_message: Value,
    off_effort: Option<&str>,
    max_effort: Option<&str>,
) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        // Checked 2026-07-04 against OpenAI chat-completions docs.
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.extend(request.history.clone());
    messages.push(user_message);

    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "stream": false,
    });

    if let Some(effort) = reasoning_effort(request.thinking, off_effort, max_effort) {
        // Checked 2026-07-04 against Ollama OpenAI-compat docs; other
        // compatible providers either support or ignore this pass-through.
        body["reasoning_effort"] = json!(effort);
    }

    body
}

fn user_message(parts: &[ContentPart]) -> Value {
    json!({
        "role": "user",
        "content": content_parts_to_chat(parts),
    })
}

fn content_parts_to_chat(parts: &[ContentPart]) -> Value {
    if parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
    {
        return Value::String(
            parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        );
    }

    Value::Array(
        parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(json!({
                    "type": "text",
                    "text": text,
                })),
                ContentPart::Image { media_type, base64 } => Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": data_url(media_type, base64),
                    },
                })),
                ContentPart::Document { .. } => None,
            })
            .collect(),
    )
}

/// Checked 2026-07-05: the chat-completions dialect defines reasoning_effort
/// values none/minimal/low/medium/high/xhigh — "max" is not in the dialect
/// and is an invalid enum value (not an ignorable unknown parameter), so Max
/// sends "high" unless the model's `max_effort` config names a tier its
/// endpoint documents (DeepSeek accepts "max", checked 2026-11-16 against
/// api-docs.deepseek.com). Off omits the field unless the model's
/// `off_effort` config names a value, such as "none" where the endpoint can
/// disable reasoning.
fn reasoning_effort<'a>(
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

fn extract_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// OpenRouter's typed error for a request over the model's context window:
/// HTTP 400 with `error.metadata.error_type` set to this value on chat
/// completions (checked 2026-11-19 against openrouter.ai, errors and
/// debugging). The chat-completions dialect has no shared code for it, and
/// DeepSeek's error codes name none (checked 2026-11-19 against
/// api-docs.deepseek.com), so only an openrouter.ai URL is read this way.
pub const OPENROUTER_CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

/// Turns an error response into a [`ProviderError`], telling a request over
/// the context window apart from other API errors where the endpoint
/// documents a code for it.
fn api_error(status: u16, body: &Value, is_openrouter: bool) -> ProviderError {
    let message = api_error_message(body);
    let error_type = body
        .pointer("/error/metadata/error_type")
        .and_then(Value::as_str);
    if is_openrouter && error_type == Some(OPENROUTER_CONTEXT_LENGTH_EXCEEDED) {
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
        .unwrap_or("unknown chat completions API error")
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

fn chat_completions_url(base_url: String) -> String {
    let mut url = trim_trailing_slash(base_url);
    url.push_str("/chat/completions");
    url
}

fn default_http_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    // Client clones retain the same connection pool, so models using the
    // default timeout do not rebuild TLS and pooling state per message.
    CLIENT
        .get_or_init(|| build_http_client(PROVIDER_REQUEST_TIMEOUT))
        .clone()
}

fn build_http_client(request_timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(PROVIDER_CONNECT_TIMEOUT.min(request_timeout))
        .timeout(request_timeout)
        .build()
        .expect("valid reqwest client configuration")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_openrouter_context_errors_are_read_as_the_context_window() {
        let body = json!({
            "error": {
                "code": 400,
                "message": "This endpoint's maximum context length is 131072 tokens.",
                "metadata": { "error_type": "context_length_exceeded" }
            }
        });

        assert!(matches!(
            api_error(400, &body, true),
            ProviderError::ContextWindowExceeded { status: 400, .. }
        ));
        assert!(matches!(
            api_error(400, &body, false),
            ProviderError::Api { status: 400, .. }
        ));
        assert!(matches!(
            api_error(400, &json!({ "error": { "message": "bad" } }), true),
            ProviderError::Api { status: 400, .. }
        ));
    }
}
