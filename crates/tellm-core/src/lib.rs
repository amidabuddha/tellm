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
/// Gemini Interactions uses `generation_config.thinking_level` for
/// Low/Medium/High; Max collapses to High, and Off omits the parameter.
///
/// Caveat on `Off`: it means "don't request reasoning", implemented by
/// omitting the parameter. On models where adaptive thinking is on by
/// default (and where an explicit disable is rejected), the model may still
/// think — user-facing text must not promise "no thinking", only "default".
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
    /// model). Opaque to the runtime; interpreted only by the provider crate.
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

/// A completed model reply. No streaming: Telegram delivers whole messages,
/// so providers are called in non-streaming mode by design.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Display text, extracted for Telegram delivery.
    pub text: String,
    /// Generated images (OpenAI Responses `image_generation` only, for now).
    pub images: Vec<GeneratedImage>,
    /// This full exchange (user turn + assistant turn) in the provider's own
    /// history shape, to be appended to the room history verbatim. Carries
    /// the opaque state (encrypted search results, reasoning items) the
    /// provider requires on later turns — do not filter or reshape it.
    pub turn_items: Vec<serde_json::Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("api error {status}: {message}")]
    Api { status: u16, message: String },
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
