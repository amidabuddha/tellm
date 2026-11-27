//! Core types shared by every tellm crate: the unified chat model and the
//! [`Provider`] trait that each wire-format crate implements.
//!
//! # The opaque-history rule
//!
//! Provider conversations carry state that providers require back **verbatim**
//! on later turns: Anthropic web-search results include `encrypted_content`
//! (missing or modified ⇒ 400 validation error), and OpenAI Responses
//! reasoning/function-call items must be replayed in stateless usage. A
//! "clean" unified history type would destroy that state and break multi-turn
//! behavior the moment server-side tools are involved.
//!
//! Therefore: **conversation history is stored as provider-native JSON items**
//! ([`ChatRequest::history`]), opaque to the runtime and owned by the provider
//! crate that produced them. The unified types are the *construction* API for
//! new user input ([`ChatRequest::input`]) and the *extraction* API for
//! display ([`ChatResponse::text`] / [`ChatResponse::images`]). Each provider
//! returns [`ChatResponse::turn_items`] — the full exchange in its own history
//! shape — which the runtime appends to the room's history verbatim. Switching
//! a room's wire format resets the opaque history.

use core::future::Future;

use serde::{Deserialize, Serialize};

/// Unified reasoning depth. Translated per wire format:
///
/// | Level  | Anthropic Messages                          | OpenAI/xAI Responses  | Chat completions      |
/// |--------|---------------------------------------------|-----------------------|-----------------------|
/// | Off    | omit `thinking`                             | omit `reasoning`      | omit `reasoning_effort` |
/// | Low..Max | `thinking: {type: "adaptive"}` + `output_config.effort` | `reasoning.effort`    | `reasoning_effort`    |
///
/// Gemini Interactions uses `generation_config.thinking_level`, and Off omits
/// the parameter.
///
/// Low, Medium and High carry the same value everywhere. The two ends differ
/// per model, so each provider client takes them from the model's config:
/// `max_effort` for `Max`, falling back to its wire format's default top
/// tier, and `off_effort` for `Off`, falling back to omitting the parameter
/// as the table shows.
///
/// Caveat on `Off`: without an `off_effort` it means "don't request
/// reasoning", implemented by omitting the parameter. On models where
/// adaptive thinking is on by default (and where an explicit disable is
/// rejected), the model may still think, at whatever its default is —
/// user-facing text must not promise "no thinking".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingLevel {
    Off,
    Low,
    #[default]
    Medium,
    High,
    Max,
}

/// One piece of new user input. Documents and images are passed through
/// natively to providers that accept them (Anthropic `document`/`image`
/// blocks, OpenAI `input_file`/`input_image`) — tellm never extracts or
/// converts file contents itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        media_type: String,
        base64: String,
    },
    Document {
        media_type: String,
        base64: String,
        name: Option<String>,
    },
}

/// The unified, provider-agnostic request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    /// Kept byte-stable by the runtime so Anthropic prompt caching works —
    /// never interpolate timestamps or per-request values here.
    pub system: Option<String>,
    /// Provider-native history: the concatenation of `turn_items` from prior
    /// [`ChatResponse`]s of the **same wire format** (and normally the same
    /// model). Opaque to the runtime; interpreted only by the provider crate,
    /// which is also the only one to rewrite a kept turn after the runtime
    /// pruned or left out earlier turns or changed the tool list.
    pub history: Vec<serde_json::Value>,
    /// The new user turn, unified.
    pub input: Vec<ContentPart>,
    pub thinking: ThinkingLevel,
    pub web_search: bool,
    pub image_generation: bool,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub media_type: String,
    pub base64: String,
}

/// What one reply was billed for, normalized into disjoint buckets so a
/// price table applies directly. Providers report usage in different shapes:
/// OpenAI-style APIs count cache reads and writes inside their input total,
/// while Anthropic reports them beside it. Each wire-format crate converts its
/// own shape, so a token is counted in exactly one input bucket.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Usage {
    /// Input tokens billed at the full input rate: neither read from nor
    /// written to the provider's prompt cache.
    pub input_tokens: u64,
    /// Input tokens read from the provider's prompt cache.
    pub cached_input_tokens: u64,
    /// Input tokens written to the provider's prompt cache.
    pub cache_write_tokens: u64,
    /// Output tokens, reasoning included; providers bill reasoning as output.
    pub output_tokens: u64,
    /// The part of `output_tokens` spent on reasoning, for display only.
    pub reasoning_tokens: u64,
    /// Provider-native web searches the reply ran.
    pub web_searches: u64,
    /// The cost the provider says it charged for the reply, in US dollars,
    /// when the provider reports one (xAI, OpenRouter).
    pub reported_cost_usd: Option<f64>,
    /// Tokens the conversation holds after this reply, when that is not its
    /// input plus output: a reply that took several requests is billed for
    /// all of them, but each request resent everything before it, so only
    /// the last one measures the conversation. `None` means input plus
    /// output; [`Usage::add`] sets it. Read it through [`Usage::context_size`].
    pub context_tokens: Option<u64>,
}

impl Usage {
    /// Adds the usage of another request that belongs to the same reply, such
    /// as an Anthropic `pause_turn` continuation. A reported cost survives
    /// only when both sides report one, since a partial sum would understate
    /// the reply.
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens = self.input_tokens.saturating_add(other.input_tokens);
        self.cached_input_tokens = self
            .cached_input_tokens
            .saturating_add(other.cached_input_tokens);
        self.cache_write_tokens = self
            .cache_write_tokens
            .saturating_add(other.cache_write_tokens);
        self.output_tokens = self.output_tokens.saturating_add(other.output_tokens);
        self.reasoning_tokens = self.reasoning_tokens.saturating_add(other.reasoning_tokens);
        self.web_searches = self.web_searches.saturating_add(other.web_searches);
        self.reported_cost_usd = match (self.reported_cost_usd, other.reported_cost_usd) {
            (Some(left), Some(right)) => Some(left + right),
            _ => None,
        };
        // `other` came later and resent everything before it.
        self.context_tokens = Some(other.context_size());
    }

    /// The conversation's size after this reply, in tokens.
    pub fn context_size(&self) -> u64 {
        self.context_tokens
            .unwrap_or_else(|| self.total_input_tokens().saturating_add(self.output_tokens))
    }

    /// Every input token the reply was billed for, in any bucket.
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cached_input_tokens)
            .saturating_add(self.cache_write_tokens)
    }
}

/// A completed model reply. No streaming: Telegram delivers whole messages,
/// so providers are called in non-streaming mode by design.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Display text, extracted for Telegram delivery.
    pub text: String,
    /// Generated images extracted from a provider-native response.
    pub images: Vec<GeneratedImage>,
    /// This full exchange (user turn + assistant turn) in the provider's own
    /// history shape, to be appended to the room history verbatim. Carries
    /// the opaque state (encrypted search results, reasoning items) the
    /// provider requires on later turns — do not filter or reshape it.
    pub turn_items: Vec<serde_json::Value>,
    /// What the reply was billed for, summed over every request it took;
    /// [`Usage::context_size`] gives the conversation's size after it.
    /// `None` when the provider sent no usage block.
    pub usage: Option<Usage>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api error {status}: {message}")]
    Api { status: u16, message: String },
    /// The provider said the request does not fit the model's context
    /// window. A wire-format crate returns this only for the error its
    /// provider documents for that; anything else stays [`Self::Api`].
    #[error("the request exceeds the model's context window (api error {status}: {message})")]
    ContextWindowExceeded { status: u16, message: String },
    #[error("model refused the request: {0}")]
    Refusal(String),
    #[error("not supported by this provider: {0}")]
    Unsupported(String),
}

/// Implemented by each wire-format crate (`tellm-anthropic`, `tellm-openai`,
/// `tellm-compat`, `tellm-gemini`). Dispatch is by enum in the binary — no
/// trait objects needed for four known variants.
pub trait Provider {
    fn chat(
        &self,
        request: &ChatRequest,
    ) -> impl Future<Output = Result<ChatResponse, ProviderError>> + Send;
}

/// A provider response whose body has been read: the JSON of a success, or
/// the status and JSON body of an error, which each wire-format crate turns
/// into its own [`ProviderError`].
#[derive(Debug)]
pub enum ProviderReply {
    Success(serde_json::Value),
    Error {
        status: u16,
        body: serde_json::Value,
    },
}

/// Longest stretch of a non-JSON error body kept in an error message.
const ERROR_TEXT_LIMIT: usize = 300;

/// Reads a provider response, looking at the status before the body. A
/// success must be JSON. An error body is often not: a proxy, gateway or
/// load balancer in front of the API answers with an HTML or plain-text
/// page. Decoding that as JSON would hide the status behind a parse error,
/// so such a body becomes [`ProviderError::Api`] with the status and a
/// short excerpt (an HTML page's title, when it has one).
pub async fn read_provider_response(
    response: reqwest::Response,
) -> Result<ProviderReply, ProviderError> {
    let status = response.status();
    let bytes = response.bytes().await?;
    let parsed = serde_json::from_slice::<serde_json::Value>(&bytes);
    if status.is_success() {
        return parsed
            .map(ProviderReply::Success)
            .map_err(|error| ProviderError::Api {
                status: status.as_u16(),
                message: format!("the response body is not JSON ({error})"),
            });
    }
    match parsed {
        Ok(body) => Ok(ProviderReply::Error {
            status: status.as_u16(),
            body,
        }),
        Err(_) => Err(ProviderError::Api {
            status: status.as_u16(),
            message: error_text(&bytes),
        }),
    }
}

/// A short, single-line description of a non-JSON error body.
fn error_text(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return "empty response body".to_string();
    }
    let excerpt = if trimmed.starts_with('<') {
        let lower = trimmed.to_ascii_lowercase();
        match (lower.find("<title>"), lower.find("</title>")) {
            (Some(start), Some(end)) if start + 7 <= end => &trimmed[start + 7..end],
            _ => return "an HTML error page".to_string(),
        }
    } else {
        trimmed
    };
    let mut line = excerpt.split_whitespace().collect::<Vec<_>>().join(" ");
    if line.len() > ERROR_TEXT_LIMIT {
        let mut cut = ERROR_TEXT_LIMIT;
        while !line.is_char_boundary(cut) {
            cut -= 1;
        }
        line.truncate(cut);
        line.push('…');
    }
    line
}

/// Refuses an input image whose media type `supported` rejects, before the
/// request is sent. Telegram passes a file's type through as the sender's
/// client named it, so an image sent as a file can be HEIC, TIFF or SVG; the
/// error names the type so the runtime can suggest sending a photo instead.
pub fn check_image_types(
    input: &[ContentPart],
    endpoint: &str,
    supported: impl Fn(&str) -> bool,
) -> Result<(), ProviderError> {
    for part in input {
        if let ContentPart::Image { media_type, .. } = part
            && !supported(media_type)
        {
            return Err(ProviderError::Unsupported(format!(
                "{endpoint} does not support image media type {media_type}"
            )));
        }
    }
    Ok(())
}

/// Appended to the text of a reply that stopped at the provider's output
/// limit, so a cut-off answer is not mistaken for a finished one. Only the
/// user sees it: the stored `turn_items` keep the model's own content.
pub const CUT_OFF_NOTE: &str = "\n\n(Reply cut off: it reached the output limit.)";
/// Error message for a reply that reached the provider's output limit before
/// any text. Storing such a turn would keep a question nobody saw answered.
pub const OUTPUT_LIMIT_BEFORE_TEXT: &str =
    "the reply reached the provider's output limit before any answer text";

/// Reads a non-negative integer token count from a provider usage object.
/// Missing, negative, or non-integer values count as zero, so a provider that
/// omits a field never fails the reply it belongs to.
pub fn usage_count(value: &serde_json::Value, pointer: &str) -> u64 {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_add_sums_tokens_and_keeps_a_cost_only_when_both_report_one() {
        let mut total = Usage {
            input_tokens: 10,
            cached_input_tokens: 1,
            cache_write_tokens: 2,
            output_tokens: 5,
            reasoning_tokens: 3,
            web_searches: 1,
            reported_cost_usd: Some(0.5),
            context_tokens: None,
        };
        total.add(&Usage {
            input_tokens: 4,
            cached_input_tokens: 6,
            cache_write_tokens: 0,
            output_tokens: 7,
            reasoning_tokens: 2,
            web_searches: 2,
            reported_cost_usd: Some(0.25),
            context_tokens: None,
        });
        assert_eq!(total.input_tokens, 14);
        assert_eq!(total.cached_input_tokens, 7);
        assert_eq!(total.cache_write_tokens, 2);
        assert_eq!(total.output_tokens, 12);
        assert_eq!(total.reasoning_tokens, 5);
        assert_eq!(total.web_searches, 3);
        assert_eq!(total.reported_cost_usd, Some(0.75));
        assert_eq!(total.total_input_tokens(), 23);

        total.add(&Usage::default());
        assert_eq!(total.reported_cost_usd, None);
    }

    #[test]
    fn context_size_is_the_last_requests_input_and_output() {
        let mut usage = Usage {
            input_tokens: 10,
            cached_input_tokens: 100,
            output_tokens: 5,
            ..Usage::default()
        };
        assert_eq!(usage.context_size(), 115, "one request: input plus output");

        usage.add(&Usage {
            input_tokens: 20,
            cached_input_tokens: 110,
            output_tokens: 7,
            ..Usage::default()
        });
        assert_eq!(usage.total_input_tokens(), 240, "billing sums both");
        assert_eq!(usage.context_size(), 137, "size comes from the later one");
    }

    #[test]
    fn non_json_error_bodies_become_a_short_line() {
        assert_eq!(
            error_text(b"<html><head><TITLE>502 Bad Gateway</TITLE></head><body>...</body></html>"),
            "502 Bad Gateway"
        );
        assert_eq!(
            error_text(b"<html><body>oops</body></html>"),
            "an HTML error page"
        );
        assert_eq!(
            error_text(b"  upstream\n  timed out \n"),
            "upstream timed out"
        );
        assert_eq!(error_text(b""), "empty response body");
        let long = error_text("é".repeat(400).as_bytes());
        assert!(long.ends_with('…') && long.len() <= ERROR_TEXT_LIMIT + '…'.len_utf8());
    }

    #[test]
    fn usage_count_treats_missing_and_malformed_values_as_zero() {
        let value = serde_json::json!({"a": {"b": 7}, "neg": -1, "text": "9"});
        assert_eq!(usage_count(&value, "/a/b"), 7);
        assert_eq!(usage_count(&value, "/a/missing"), 0);
        assert_eq!(usage_count(&value, "/neg"), 0);
        assert_eq!(usage_count(&value, "/text"), 0);
    }
}
