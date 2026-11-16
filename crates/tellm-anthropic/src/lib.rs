//! Anthropic Messages API client (`POST /v1/messages`).
//!
//! Mapping notes (checked 2026-07-04 against platform.claude.com — re-check
//! live provider docs before changing these mappings):
//! - Thinking: `thinking: {"type": "adaptive"}` + `output_config: {"effort": ...}`.
//!   `budget_tokens` is rejected on current models — never send it.
//! - Prompt caching: `cache_control: {"type": "ephemeral"}` on the system
//!   block; keep the system prompt byte-stable (no timestamps).
//! - Web search: `{"type": "web_search_20260318", "name": "web_search",
//!   "allowed_callers": ["direct"], "max_uses": N}`. The `["direct"]` pin is
//!   REQUIRED for tellm: from `_20260209` onward the default routes search
//!   through server-side code execution (dynamic filtering), which
//!   complicates the response shape and isn't wanted for plain chat.
//! - Multi-turn: assistant content blocks must be echoed back verbatim,
//!   including `web_search_tool_result` blocks with `encrypted_content`
//!   (missing/modified => 400). That is what `ChatResponse::turn_items`
//!   exists for — return the raw user + assistant message objects.
//! - Thinking blocks are bound to everything sent before them: `system`,
//!   `tools`, and every earlier message (checked 2026-11-16 against
//!   platform.claude.com, preserved thinking / prefix checks). Replaying one
//!   after any of that changed is a 400 on Claude Fable 5.2 ("The block is
//!   bound to a different conversation") and can be one on the other models
//!   with preserved thinking. Verbatim, append-only history keeps them valid;
//!   once the runtime prunes earlier turns or the tool list changes, every
//!   kept turn must go through [`without_thinking_blocks`] before it is sent
//!   again.
//! - Documents/images: native `document` / `image` content blocks (base64) —
//!   PDFs pass through without client-side extraction.
//! - Handle `stop_reason: "refusal"` before reading content
//!   (`ProviderError::Refusal`).

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ThinkingLevel,
};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// checked 2026-07-04 against platform.claude.com/docs/en/api/messages/create
pub const API_VERSION: &str = "2023-06-01";
/// checked 2026-07-04 against platform.claude.com (web-search-tool docs)
pub const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20260318";
pub const WEB_SEARCH_MAX_USES: u32 = 5;
/// Thinking tokens count toward max_tokens; Anthropic's adaptive-thinking
/// examples use 16000 to leave room for thinking + text (checked 2026-07-12).
pub const DEFAULT_MAX_TOKENS: u32 = 16000;
/// Effort sent for `ThinkingLevel::Max` unless the model's `max_effort`
/// config names another tier. Valid on every adaptive-thinking model.
pub const DEFAULT_MAX_EFFORT: &str = "max";
pub const MAX_PAUSE_TURN_CONTINUATIONS: usize = 5;
pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub struct Anthropic {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    max_effort: Option<String>,
}

impl Anthropic {
    pub fn new(api_key: impl Into<String>, base_url: Option<String>) -> Self {
        Self {
            // Production calls share one pool so repeated Telegram turns reuse
            // DNS, TLS sessions, and keep-alive connections. Timeout-specific
            // test clients remain isolated below.
            http: default_http_client(),
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string())),
            api_key: api_key.into(),
            max_effort: None,
        }
    }

    /// Sets the effort value sent for `ThinkingLevel::Max`, from the model's
    /// `max_effort` config. `None` keeps [`DEFAULT_MAX_EFFORT`].
    pub fn with_max_effort(mut self, max_effort: Option<String>) -> Self {
        self.max_effort = max_effort;
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

impl Provider for Anthropic {
    async fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        if request.image_generation {
            return Err(ProviderError::Unsupported(
                "Anthropic Messages does not support image generation".to_string(),
            ));
        }

        let user_message = json!({
            "role": "user",
            "content": content_parts_to_anthropic(&request.input),
        });

        let mut messages = request.history.clone();
        messages.push(user_message.clone());
        let mut assistant_messages = Vec::new();
        let mut text_parts = Vec::new();

        for continuation in 0..=MAX_PAUSE_TURN_CONTINUATIONS {
            let body = request_body(request, messages.clone(), self.max_effort.as_deref());
            let response_body = self.send_request(body).await?;

            let stop_reason = response_body
                .get("stop_reason")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let content = response_body
                .get("content")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let text = extract_text(&content);

            if stop_reason == "refusal" {
                return Err(ProviderError::Refusal(if text.is_empty() {
                    refusal_message(&response_body)
                } else {
                    text
                }));
            }

            let assistant_message = json!({
                "role": "assistant",
                "content": content,
            });

            if !text.is_empty() {
                text_parts.push(text);
            }

            if stop_reason == "pause_turn" {
                if continuation == MAX_PAUSE_TURN_CONTINUATIONS {
                    return Err(ProviderError::Api {
                        status: 200,
                        message: "Anthropic pause_turn continuation limit exceeded".to_string(),
                    });
                }
                messages.push(assistant_message.clone());
                assistant_messages.push(assistant_message);
                continue;
            }

            assistant_messages.push(assistant_message);
            let mut turn_items = Vec::with_capacity(1 + assistant_messages.len());
            turn_items.push(user_message);
            turn_items.extend(assistant_messages);

            return Ok(ChatResponse {
                text: text_parts.join(""),
                images: Vec::<GeneratedImage>::new(),
                turn_items,
            });
        }

        unreachable!("pause_turn loop always returns")
    }
}

impl Anthropic {
    async fn send_request(&self, body: Value) -> Result<Value, ProviderError> {
        let response = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
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

/// Rewrites one stored turn (the `turn_items` of an earlier
/// [`ChatResponse`]) so it can still be replayed after the runtime changed
/// what precedes it: earlier turns were pruned, or the tool list changed.
///
/// Checked 2026-11-16 against platform.claude.com (preserved thinking:
/// prefix checks, change tools or instructions). Such a change invalidates
/// the thinking in every kept turn, and thinking may only be left out as a
/// leading run, a trailing run, or completely — never from the middle — so
/// all of it goes: `thinking` and `redacted_thinking`, empty-text blocks
/// included. Text and tool blocks stay valid without it, and
/// `encrypted_content` is left untouched.
///
/// An assistant message that held nothing but thinking has nothing left to
/// replay and is removed. If that leaves the turn without any assistant
/// message, the result is empty so the caller drops the turn whole instead
/// of keeping a question without its answer.
pub fn without_thinking_blocks(turn_items: Vec<Value>) -> Vec<Value> {
    let mut had_assistant = false;
    let mut kept_assistant = false;
    let mut kept = Vec::with_capacity(turn_items.len());

    for mut item in turn_items {
        if item.get("role").and_then(Value::as_str) == Some("assistant") {
            had_assistant = true;
            if let Some(content) = item.get_mut("content").and_then(Value::as_array_mut) {
                let blocks_before = content.len();
                content.retain(|block| !is_thinking_block(block));
                if content.is_empty() && blocks_before > 0 {
                    continue;
                }
            }
            kept_assistant = true;
        }
        kept.push(item);
    }

    if had_assistant && !kept_assistant {
        return Vec::new();
    }
    kept
}

fn is_thinking_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("thinking" | "redacted_thinking")
    )
}

fn request_body(request: &ChatRequest, messages: Vec<Value>, max_effort: Option<&str>) -> Value {
    let mut body = json!({
        "model": request.model,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": messages,
    });

    if let Some(system) = &request.system {
        // Checked 2026-07-04 against prompt-caching docs: explicit cache
        // breakpoints can be placed on system content blocks.
        body["system"] = json!([
            {
                "type": "text",
                "text": system,
                "cache_control": { "type": "ephemeral" },
            }
        ]);
    }

    if request.web_search {
        // Checked 2026-07-04 against web-search-tool docs. `allowed_callers:
        // ["direct"]` prevents dynamic filtering through code execution.
        body["tools"] = json!([
            {
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "allowed_callers": ["direct"],
                "max_uses": WEB_SEARCH_MAX_USES,
            }
        ]);
    }

    if let Some(effort) = anthropic_effort(request.thinking, max_effort) {
        // Checked 2026-07-04 against adaptive-thinking docs: adaptive mode uses
        // `thinking: {type: "adaptive"}` plus `output_config.effort`.
        body["thinking"] = json!({ "type": "adaptive" });
        body["output_config"] = json!({ "effort": effort });
    }

    body
}

fn content_parts_to_anthropic(parts: &[ContentPart]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => json!({
                "type": "text",
                "text": text,
            }),
            ContentPart::Image { media_type, base64 } => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64,
                },
            }),
            ContentPart::Document {
                media_type, base64, ..
            } => json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64,
                },
            }),
        })
        .collect()
}

fn anthropic_effort(thinking: ThinkingLevel, max_effort: Option<&str>) -> Option<&str> {
    match thinking {
        ThinkingLevel::Off => None,
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High => Some("high"),
        ThinkingLevel::Max => Some(max_effort.unwrap_or(DEFAULT_MAX_EFFORT)),
    }
}

fn extract_text(content: &[Value]) -> String {
    content
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

fn api_error_message(value: &Value) -> String {
    value
        .pointer("/error/message")
        .or_else(|| value.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("unknown Anthropic API error")
        .to_string()
}

fn refusal_message(value: &Value) -> String {
    value
        .pointer("/stop_details/explanation")
        .or_else(|| value.pointer("/stop_details/category"))
        .and_then(Value::as_str)
        .unwrap_or("request declined")
        .to_string()
}

fn trim_trailing_slash(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}
