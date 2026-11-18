//! Anthropic Messages API client (`POST /v1/messages`).
//!
//! Mapping notes (checked 2026-07-04 against platform.claude.com — re-check
//! live provider docs before changing these mappings):
//! - Thinking: `thinking: {"type": "adaptive"}` + `output_config: {"effort": ...}`.
//!   `budget_tokens` is rejected on current models — never send it. `Off`
//!   sends neither, unless the model's `off_effort` names an effort, which
//!   then goes out without a `thinking` field.
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
//! - Handle `stop_reason: "max_tokens"`: thinking counts toward the limit, so
//!   a reply can stop mid-answer or before any text. A cut-off answer gets
//!   [`CUT_OFF_NOTE`]; one without text is an error instead of an empty turn.
//! - Usage: `usage.input_tokens` counts only input that was neither read from
//!   nor written to the cache; `cache_read_input_tokens` and
//!   `cache_creation_input_tokens` are reported beside it, and
//!   `server_tool_use.web_search_requests` counts billed searches (checked
//!   2026-11-18 against platform.claude.com, prompt caching and pricing).
//!   A reply that took `pause_turn` continuations sums every request.

use std::sync::OnceLock;
use std::time::Duration;

use serde_json::{Value, json};
use tellm_core::{
    ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError, ThinkingLevel,
    Usage, usage_count,
};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// checked 2026-07-04 against platform.claude.com/docs/en/api/messages/create
pub const API_VERSION: &str = "2023-06-01";
/// checked 2026-07-04 against platform.claude.com (web-search-tool docs)
pub const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20260318";
pub const WEB_SEARCH_MAX_USES: u32 = 5;
/// `max_tokens` sent unless the model's `max_output_tokens` config names
/// another limit. Thinking tokens count toward max_tokens; Anthropic's
/// adaptive-thinking examples use 16000 to leave room for thinking + text
/// (checked 2026-07-12).
///
/// Requests are not streamed, so a larger limit means a longer wait for one
/// HTTP response; see [`request_timeout_for`].
pub const DEFAULT_MAX_TOKENS: u32 = 16000;
/// The `max_tokens` a non-streamed request is expected to fit inside one
/// [`PROVIDER_REQUEST_TIMEOUT`]. Anthropic's SDKs refuse a non-streamed
/// request above 21,333 because it could outlast their 10-minute timeout
/// (checked 2026-11-16 against platform.claude.com, thinking and API errors).
pub const MAX_TOKENS_PER_REQUEST_TIMEOUT: u32 = 21_333;
/// Effort sent for `ThinkingLevel::Max` unless the model's `max_effort`
/// config names another tier. Valid on every adaptive-thinking model.
pub const DEFAULT_MAX_EFFORT: &str = "max";
pub const MAX_PAUSE_TURN_CONTINUATIONS: usize = 5;
/// Appended to the text of a reply that stopped at `max_tokens`, so a cut-off
/// answer is not mistaken for a finished one. Only the user sees it: the
/// stored `turn_items` keep the model's own content.
pub const CUT_OFF_NOTE: &str = "\n\n(Reply cut off: it reached the output limit.)";
/// Error message for a reply that stopped at `max_tokens` before any text.
pub const OUTPUT_LIMIT_BEFORE_TEXT: &str =
    "the reply reached the max_tokens output limit before any answer text";
pub const PROVIDER_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const PROVIDER_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Idle time before TCP keep-alive probes start. A non-streamed reply sends
/// nothing until it is complete, and Anthropic's guidance for direct API
/// integrations is a TCP keep-alive, so that networks which drop idle
/// connections do not drop a long request (checked 2026-11-16 against
/// platform.claude.com, API errors, long requests).
pub const PROVIDER_TCP_KEEPALIVE: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub struct Anthropic {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    max_effort: Option<String>,
    off_effort: Option<String>,
    max_tokens: Option<u32>,
    request_timeout: Duration,
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
            off_effort: None,
            max_tokens: None,
            request_timeout: PROVIDER_REQUEST_TIMEOUT,
        }
    }

    /// Sets the effort value sent for `ThinkingLevel::Max`, from the model's
    /// `max_effort` config. `None` keeps [`DEFAULT_MAX_EFFORT`].
    pub fn with_max_effort(mut self, max_effort: Option<String>) -> Self {
        self.max_effort = max_effort;
        self
    }

    /// Sets the effort value sent for `ThinkingLevel::Off`, from the model's
    /// `off_effort` config. `None` sends no effort, so the model's own
    /// default applies.
    pub fn with_off_effort(mut self, off_effort: Option<String>) -> Self {
        self.off_effort = off_effort;
        self
    }

    /// Sets `max_tokens`, from the model's `max_output_tokens` config. `None`
    /// keeps [`DEFAULT_MAX_TOKENS`].
    pub fn with_max_tokens(mut self, max_tokens: Option<u32>) -> Self {
        self.max_tokens = max_tokens;
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
                .tcp_keepalive(PROVIDER_TCP_KEEPALIVE)
                .build()
                .expect("valid reqwest client configuration"),
            base_url: trim_trailing_slash(base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string())),
            api_key: api_key.into(),
            max_effort: None,
            off_effort: None,
            max_tokens: None,
            request_timeout,
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
                .tcp_keepalive(PROVIDER_TCP_KEEPALIVE)
                .build()
                .expect("valid reqwest client configuration")
        })
        .clone()
}

/// How long one request with this output limit may run. tellm does not
/// stream, so the whole reply has to arrive inside a single HTTP response and
/// a larger limit needs a longer wait. The timeout grows with the limit at
/// the SDKs' ratio, [`MAX_TOKENS_PER_REQUEST_TIMEOUT`] tokens for each
/// `base`, and never drops below `base`: 16000 keeps 10 minutes and 64000
/// gets 30.
pub fn request_timeout_for(base: Duration, max_tokens: u32) -> Duration {
    let scale = f64::from(max_tokens) / f64::from(MAX_TOKENS_PER_REQUEST_TIMEOUT);
    base.mul_f64(scale.max(1.0))
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
        let mut usage: Option<Usage> = None;

        for continuation in 0..=MAX_PAUSE_TURN_CONTINUATIONS {
            let body = request_body(
                request,
                messages.clone(),
                self.off_effort.as_deref(),
                self.max_effort.as_deref(),
                self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            );
            let response_body = self.send_request(body).await?;
            if let Some(request_usage) = usage_from_response(&response_body) {
                match usage.as_mut() {
                    Some(total) => total.add(&request_usage),
                    None => usage = Some(request_usage),
                }
            }

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

            // Checked 2026-11-16 against platform.claude.com (stop reasons;
            // Claude Fable 5.2 migration guide): thinking tokens count toward
            // `max_tokens`, so a reply can stop there after a thinking block
            // and before any text. That is a failed turn: storing it would
            // keep a question whose answer nobody saw.
            let cut_off = stop_reason == "max_tokens";
            if cut_off && text_parts.is_empty() {
                return Err(ProviderError::Api {
                    status: 200,
                    message: OUTPUT_LIMIT_BEFORE_TEXT.to_string(),
                });
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

            let mut text = text_parts.join("");
            if cut_off {
                text.push_str(CUT_OFF_NOTE);
            }

            return Ok(ChatResponse {
                text,
                images: Vec::<GeneratedImage>::new(),
                turn_items,
                usage,
            });
        }

        unreachable!("pause_turn loop always returns")
    }
}

/// Reads one Messages response's `usage` block. Anthropic reports cache reads
/// and writes beside `input_tokens`, not inside it, so the three map straight
/// onto the disjoint [`Usage`] buckets. Thinking is billed as output and is
/// not reported separately.
pub fn usage_from_response(body: &Value) -> Option<Usage> {
    let usage = body.get("usage")?;
    Some(Usage {
        input_tokens: usage_count(usage, "/input_tokens"),
        cached_input_tokens: usage_count(usage, "/cache_read_input_tokens"),
        cache_write_tokens: usage_count(usage, "/cache_creation_input_tokens"),
        output_tokens: usage_count(usage, "/output_tokens"),
        reasoning_tokens: 0,
        web_searches: usage_count(usage, "/server_tool_use/web_search_requests"),
        reported_cost_usd: None,
    })
}

impl Anthropic {
    async fn send_request(&self, body: Value) -> Result<Value, ProviderError> {
        let response = self
            .http
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .timeout(request_timeout_for(
                self.request_timeout,
                self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            ))
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

fn request_body(
    request: &ChatRequest,
    messages: Vec<Value>,
    off_effort: Option<&str>,
    max_effort: Option<&str>,
    max_tokens: u32,
) -> Value {
    let mut body = json!({
        "model": request.model,
        "max_tokens": max_tokens,
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

    if let Some(effort) = anthropic_effort(request.thinking, off_effort, max_effort) {
        if request.thinking != ThinkingLevel::Off {
            // Checked 2026-07-04 against adaptive-thinking docs: adaptive mode
            // uses `thinking: {type: "adaptive"}` plus `output_config.effort`.
            body["thinking"] = json!({ "type": "adaptive" });
        }
        // `Off` with an `off_effort` still does not ask for thinking; it only
        // names the effort. Checked 2026-11-16 against platform.claude.com
        // (effort): effort applies to every output token, with or without
        // thinking, and it is the control for thinking depth on models whose
        // adaptive thinking is always on, such as Claude Fable 5.2.
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

fn anthropic_effort<'a>(
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
