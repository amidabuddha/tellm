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
    /// What the reply was billed for, summed over every request it took.
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
        };
        total.add(&Usage {
            input_tokens: 4,
            cached_input_tokens: 6,
            cache_write_tokens: 0,
            output_tokens: 7,
            reasoning_tokens: 2,
            web_searches: 2,
            reported_cost_usd: Some(0.25),
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
    fn usage_count_treats_missing_and_malformed_values_as_zero() {
        let value = serde_json::json!({"a": {"b": 7}, "neg": -1, "text": "9"});
        assert_eq!(usage_count(&value, "/a/b"), 7);
        assert_eq!(usage_count(&value, "/a/missing"), 0);
        assert_eq!(usage_count(&value, "/neg"), 0);
        assert_eq!(usage_count(&value, "/text"), 0);
    }
}
