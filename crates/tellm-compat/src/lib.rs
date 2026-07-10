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
//!   ignored by those that don't (that's the dialect's convention).
//! - Web search / image generation: not part of this dialect —
//!   `ProviderError::Unsupported` so the user gets told instead of guessing.
//! - Images: `image_url` content parts (data URLs); documents: unsupported.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ThinkingLevel,
};

pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub struct Compat {
    http: reqwest::Client,
    chat_completions_url: String,
    api_key: Option<String>,
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
        }
    }
}

impl Provider for Compat {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        validate_supported(request)?;

        let user_message = user_message(&request.input);
        let body = request_body(request, user_message.clone());
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
        Ok(ChatResponse {
            text,
            images: Vec::<GeneratedImage>::new(),
            turn_items: vec![user_message, message],
        })
    }
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
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: api_error_message(&response_body),
            });
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

fn request_body(request: &ChatRequest, user_message: Value) -> Value {
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

    if let Some(max_tokens) = request.max_tokens {
        // Checked 2026-07-04 against Ollama OpenAI-compat docs.
        body["max_tokens"] = json!(max_tokens);
    }

    if let Some(effort) = reasoning_effort(request.thinking) {
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
/// clamps to "high".
fn reasoning_effort(thinking: ThinkingLevel) -> Option<&'static str> {
    match thinking {
        ThinkingLevel::Off => None,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High | ThinkingLevel::Max => Some("high"),
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
