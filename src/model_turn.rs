use std::collections::BTreeSet;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use tellm_anthropic::Anthropic;
use tellm_compat::Compat;
use tellm_config::{Config, ModelConfig, WireFormat, secrets};
use tellm_core::{ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider, ProviderError};
use tellm_gemini::Gemini;
use tellm_openai::Responses;
use tellm_telegram::{Document, IncomingMessage, PhotoSize, Telegram, TelegramError};
use tokio::task::spawn_blocking;

use crate::memory::ConversationMemory;
use crate::ollama;
use crate::rooms::{ChatMode, FailedTurnRollback, HistoryReset, RoomState};
use crate::summary;

const MAX_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;

pub(crate) struct PreparedChatRequest {
    pub(crate) model_config: ModelConfig,
    pub(crate) request: ChatRequest,
    pub(crate) rollback: Option<FailedTurnRollback>,
    pub(crate) generation: u64,
    pub(crate) reset_notice: Option<String>,
    /// The memory block this request's conversation uses; the room keeps it
    /// once the turn commits.
    pub(crate) memory: Option<String>,
    pub(crate) memory_notice: Option<String>,
}

/// `fresh_memory` is what a conversation starting now would get. It is used
/// only when this request sends no history; a running conversation keeps the
/// memory it started with.
pub(crate) fn prepare_chat_request(
    room: &mut RoomState,
    model_config: ModelConfig,
    input: Vec<ContentPart>,
    fresh_memory: ConversationMemory,
) -> PreparedChatRequest {
    let start = room.begin_turn(model_config.wire_format);
    let reset_notice = reset_notice(start.reset);
    let mut request = chat_request_from_room(room, &model_config, input);
    let (memory, memory_notice) = if request.history.is_empty() {
        (fresh_memory.prompt, fresh_memory.notice)
    } else {
        (room.memory.clone(), None)
    };
    let summary = room.summary.as_deref().map(summary::prompt_block);
    request.system = system_prompt(
        request.system.as_deref(),
        memory.as_deref(),
        summary.as_deref(),
    );
    let generation = room.generation();

    PreparedChatRequest {
        model_config,
        request,
        rollback: start.rollback,
        generation,
        reset_notice,
        memory,
        memory_notice,
    }
}

/// A request that stands alone, as in a shared group: the room's model, role
/// and settings, but no history, memory or carried summary, and the room's
/// conversation isn't touched, so there is nothing to roll back.
pub(crate) fn prepare_standalone_request(
    room: &RoomState,
    model_config: ModelConfig,
    input: Vec<ContentPart>,
) -> PreparedChatRequest {
    let mut request = chat_request_from_room(room, &model_config, input);
    request.history = Vec::new();
    PreparedChatRequest {
        model_config,
        request,
        rollback: None,
        generation: room.generation(),
        reset_notice: None,
        memory: None,
        memory_notice: None,
    }
}

/// The role comes first, then the memory block, then the summary the
/// conversation continues from, each after a blank line. Every part is fixed
/// for the whole conversation, so the prompt stays byte-stable.
fn system_prompt(
    role: Option<&str>,
    memory: Option<&str>,
    summary: Option<&str>,
) -> Option<String> {
    let blocks = [role, memory, summary]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    (!blocks.is_empty()).then(|| blocks.join("\n\n"))
}

fn chat_request_from_room(
    room: &RoomState,
    model_config: &ModelConfig,
    input: Vec<ContentPart>,
) -> ChatRequest {
    ChatRequest {
        model: model_config.model_name.clone(),
        system: room.settings.role.clone(),
        history: match room.settings.mode {
            ChatMode::Chat => room.history.clone(),
            ChatMode::Message => Vec::new(),
        },
        input,
        thinking: room.settings.thinking.unwrap_or(model_config.thinking),
        web_search: room.settings.web_search,
        image_generation: room.settings.image_generation,
    }
}

fn reset_notice(reset: HistoryReset) -> Option<String> {
    match reset {
        HistoryReset::WireFormatChanged {
            previous: Some(previous),
            new,
        } if previous != new => Some(format!(
            "Provider wire format changed from {previous:?} to {new:?}; chat history reset."
        )),
        HistoryReset::WireFormatChanged { .. } | HistoryReset::None => None,
    }
}

/// Why a model turn failed. A request the provider said is over the model's
/// context window is told apart, so the chat can suggest `/new summary` and
/// a summary request can be retried with fewer turns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TurnError {
    pub(crate) message: String,
    pub(crate) context_window: bool,
}

impl From<String> for TurnError {
    fn from(message: String) -> Self {
        Self {
            message,
            context_window: false,
        }
    }
}

impl From<ProviderError> for TurnError {
    fn from(error: ProviderError) -> Self {
        Self {
            context_window: matches!(error, ProviderError::ContextWindowExceeded { .. }),
            message: error.to_string(),
        }
    }
}

/// The room's model can change after a toggle was accepted, so a capability
/// error can still surface per-message; suggest the off switch. A reply that
/// used its whole output limit on thinking gets the two settings that help:
/// less thinking, or a higher limit. A chat the model can no longer take in
/// gets `/new summary`, which can leave the oldest turns out.
pub(crate) fn provider_error_reply(error: &TurnError) -> String {
    let message = error.message.as_str();
    // Provider error texts use both spellings ("image generation" from
    // Anthropic/compat, "image_generation" from the xAI backstop) — match
    // both.
    let mut reply = format!("Provider error: {message}");
    if error.context_window {
        reply.push_str(
            "\nTip: this chat no longer fits the model's context window; /new summary carries a summary of it into a new chat",
        );
    } else if message.contains("image media type") {
        reply.push_str(
            "\nTip: send the picture as a photo rather than a file; Telegram delivers photos as JPEG",
        );
    } else if message.contains("does not support documents") {
        // Only the chat-completions endpoints refuse documents; the
        // providers named here read PDFs natively (DESIGN.md § File upload).
        reply.push_str(
            "\nTip: this model can't read files; send a screenshot as a photo, or switch with /model to a Claude, OpenAI or Gemini model",
        );
    } else if message.contains("image generation") || message.contains("image_generation") {
        reply.push_str("\nTip: /imagegen off");
    } else if message.contains("web search") || message.contains("web_search") {
        reply.push_str("\nTip: /websearch off");
    } else if message.contains("max_tokens output limit") {
        // Anthropic: the only wire format that takes max_output_tokens.
        reply.push_str(
            "\nTip: /reasoning low, or raise max_output_tokens for this model in config.toml",
        );
    } else if message.contains("output limit") {
        reply.push_str("\nTip: /reasoning low, or /new summary if this chat is long");
    }
    reply
}

/// Error text for a provider reply that holds neither text nor an image.
pub(crate) const EMPTY_REPLY_ERROR: &str =
    "the model returned no text or image, so this turn was not saved";

/// A reply with nothing to show is a failed turn, not history. Stored, it
/// would be replayed as a question nobody saw an answer to, and a
/// reasoning-only turn can make every later request invalid: Meta answers a
/// reasoning item that sits directly before a user message with HTTP 400
/// (checked 2026-11-16 against dev.meta.ai).
pub(crate) fn require_reply_content(response: ChatResponse) -> Result<ChatResponse, TurnError> {
    if response.text.trim().is_empty() && response.images.is_empty() {
        return Err(EMPTY_REPLY_ERROR.to_string().into());
    }
    Ok(response)
}

pub(crate) async fn dispatch_provider(
    model: &ModelConfig,
    request: &ChatRequest,
) -> Result<ChatResponse, TurnError> {
    match model.wire_format {
        WireFormat::Anthropic => {
            let api_key = required_api_key(model).await?;
            Anthropic::new(api_key, model.base_url.clone())
                .with_max_effort(model.max_effort.clone())
                .with_off_effort(model.off_effort.clone())
                .with_max_tokens(model.max_output_tokens)
                .chat(request)
                .await
                .map_err(TurnError::from)
        }
        WireFormat::Responses => {
            let api_key = required_api_key(model).await?;
            Responses::new(api_key, model.base_url.clone())
                .with_max_effort(model.max_effort.clone())
                .with_off_effort(model.off_effort.clone())
                .chat(request)
                .await
                .map_err(TurnError::from)
        }
        WireFormat::Compat => {
            let base_url = model
                .base_url
                .clone()
                .ok_or_else(|| TurnError::from("compat model is missing base_url".to_string()))?;
            ollama::ensure_ready(&base_url).await?;
            let requested_model = request.model.clone();
            if request
                .input
                .iter()
                .any(|part| matches!(part, ContentPart::Image { .. }))
            {
                ollama::require_vision_capability(&base_url, &requested_model).await?;
            }
            let api_key = compat_api_key(model).await?;
            // Register before the request: an aborted task can still leave
            // Ollama loading the model after the HTTP future is dropped.
            ollama::remember_model(&base_url, &requested_model);
            Compat::new(api_key, base_url)
                .with_max_effort(model.max_effort.clone())
                .with_off_effort(model.off_effort.clone())
                .chat(request)
                .await
                .map_err(TurnError::from)
        }
        WireFormat::Gemini => {
            let api_key = required_api_key(model).await?;
            Gemini::new(api_key, model.base_url.clone())
                .with_max_effort(model.max_effort.clone())
                .with_off_effort(model.off_effort.clone())
                .chat(request)
                .await
                .map_err(TurnError::from)
        }
    }
}

pub(crate) async fn content_parts_from_message(
    telegram: &Telegram,
    message: &IncomingMessage,
) -> Result<Vec<ContentPart>, String> {
    let mut parts = Vec::new();
    if let Some(text) = message_text(message)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        parts.push(ContentPart::Text {
            text: text.to_string(),
        });
    }

    for item in std::iter::once(message).chain(&message.album) {
        push_attachment_parts(telegram, item, &mut parts).await?;
    }

    Ok(parts)
}

/// The photo or file one Telegram message carries, as content parts. An
/// album's items each go through here, in order, after the caption.
async fn push_attachment_parts(
    telegram: &Telegram,
    message: &IncomingMessage,
    parts: &mut Vec<ContentPart>,
) -> Result<(), String> {
    if let Some(photo) = largest_photo(message.photo.as_deref()) {
        let bytes = download_attachment(telegram, &photo.file_id, photo.file_size, "photo").await?;
        parts.push(ContentPart::Image {
            media_type: "image/jpeg".to_string(),
            base64: BASE64.encode(bytes),
        });
    }

    if let Some(document) = &message.document {
        let bytes =
            download_attachment(telegram, &document.file_id, document.file_size, "document")
                .await?;
        let media_type = document
            .mime_type
            .clone()
            .unwrap_or_else(|| "application/octet-stream".to_string());
        if is_text_document(document, &media_type) {
            let text = String::from_utf8(bytes)
                .map_err(|_| "text document was not valid UTF-8".to_string())?;
            parts.push(ContentPart::Text { text });
        } else if media_type.starts_with("image/") {
            parts.push(ContentPart::Image {
                media_type,
                base64: BASE64.encode(bytes),
            });
        } else {
            parts.push(ContentPart::Document {
                media_type,
                base64: BASE64.encode(bytes),
                name: document.file_name.clone(),
            });
        }
    }

    Ok(())
}

async fn download_attachment(
    telegram: &Telegram,
    file_id: &str,
    announced_size: Option<i64>,
    kind: &str,
) -> Result<Vec<u8>, String> {
    if let Some(size) = announced_size
        && size >= 0
    {
        validate_attachment_size(kind, size as usize)?;
    }

    let bytes = telegram
        .get_file_bytes(file_id, MAX_ATTACHMENT_BYTES)
        .await
        .map_err(|error| match error {
            TelegramError::FileTooLarge { size, .. } => attachment_too_large_error(kind, size),
            error => error.to_string(),
        })?;
    validate_attachment_size(kind, bytes.len())?;
    Ok(bytes)
}

fn validate_attachment_size(kind: &str, size: usize) -> Result<(), String> {
    if size > MAX_ATTACHMENT_BYTES {
        Err(attachment_too_large_error(kind, size))
    } else {
        Ok(())
    }
}

fn attachment_too_large_error(kind: &str, size: usize) -> String {
    format!(
        "{kind} is too large ({:.1} MiB); maximum attachment size is {} MiB",
        size as f64 / (1024.0 * 1024.0),
        MAX_ATTACHMENT_BYTES / (1024 * 1024)
    )
}

/// Delivers a reply. `cost_line` closes the text when there is any; a reply
/// that is only images gets it as its own message after them, so it still
/// arrives last.
pub(crate) async fn send_response(
    telegram: &Telegram,
    chat_id: i64,
    response: ChatResponse,
    cost_line: Option<String>,
) -> Result<(), String> {
    let has_text = !response.text.trim().is_empty();
    let mut separate_cost_line = cost_line;
    if has_text || response.images.is_empty() {
        let text = match separate_cost_line.take() {
            Some(line) if has_text => with_cost_line(&response.text, &line),
            Some(line) => line,
            None => response.text.clone(),
        };
        telegram
            .send_message(chat_id, &text)
            .await
            .map_err(|error| error.to_string())?;
    }

    for image in response.images {
        let (media_type, bytes) = decode_image(image)?;
        telegram
            .send_photo(chat_id, bytes, &media_type)
            .await
            .map_err(|error| error.to_string())?;
    }

    if let Some(line) = separate_cost_line {
        telegram
            .send_message(chat_id, &line)
            .await
            .map_err(|error| error.to_string())?;
    }

    Ok(())
}

/// The reply text with its cost line below a blank line. Only the delivered
/// copy carries it: the stored turn keeps the model's own content, so the
/// model never sees its own bill.
pub(crate) fn with_cost_line(text: &str, line: &str) -> String {
    format!("{}\n\n{line}", text.trim_end())
}

pub(crate) fn message_text(message: &IncomingMessage) -> Option<&str> {
    message.text.as_deref().or(message.caption.as_deref())
}

pub(crate) fn message_has_input(message: &IncomingMessage) -> bool {
    message_text(message).is_some_and(|text| !text.trim().is_empty())
        || std::iter::once(message)
            .chain(&message.album)
            .any(has_attachment)
}

pub(crate) fn has_photo(message: &IncomingMessage) -> bool {
    message
        .photo
        .as_ref()
        .is_some_and(|photos| !photos.is_empty())
}

fn has_attachment(message: &IncomingMessage) -> bool {
    has_photo(message) || message.document.is_some()
}

fn largest_photo(photos: Option<&[PhotoSize]>) -> Option<&PhotoSize> {
    // Widen to i128 before multiplying: `width * height` in i64 can overflow
    // for extreme dimensions and panic in debug builds. Any product of two
    // i64 values fits in i128, so this is exact for all inputs.
    photos?
        .iter()
        .max_by_key(|photo| photo.width as i128 * photo.height as i128)
}

fn is_text_document(document: &Document, media_type: &str) -> bool {
    media_type == "text/plain"
        || document
            .file_name
            .as_deref()
            .is_some_and(|name| name.to_ascii_lowercase().ends_with(".txt"))
}

fn decode_image(image: GeneratedImage) -> Result<(String, Vec<u8>), String> {
    let bytes = BASE64
        .decode(image.base64)
        .map_err(|error| format!("generated image was not valid base64: {error}"))?;
    Ok((image.media_type, bytes))
}

pub(crate) fn warm_configured_provider_secrets(config: &Config) {
    let secret_names = configured_provider_secret_names(config);
    if secret_names.is_empty() {
        return;
    }

    log::info!(
        target: "tellm::secrets",
        "checking configured provider secrets count={} keys=[{}]",
        secret_names.len(),
        secret_names.join(","),
    );

    for secret_name in secret_names {
        if secrets::get(&secret_name).is_none() {
            log::warn!(
                target: "tellm::secrets",
                "provider secret unavailable key={secret_name} impact=\"dependent model calls will fail until it is stored\""
            );
        }
    }
}

fn configured_provider_secret_names(config: &Config) -> Vec<String> {
    config
        .models
        .values()
        .filter_map(|model| model.api_key_secret.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

async fn required_api_key(model: &ModelConfig) -> Result<String, String> {
    let secret_name = model
        .api_key_secret
        .clone()
        .ok_or_else(|| format!("model {} has no api_key_secret", model.model_name))?;
    fetch_secret(secret_name).await
}

async fn compat_api_key(model: &ModelConfig) -> Result<String, String> {
    let Some(secret_name) = model.api_key_secret.clone() else {
        return Ok(String::new());
    };
    fetch_secret(secret_name).await
}

/// OS keychain access blocks (and can prompt on macOS); keep the per-request
/// secret read off the async worker threads.
async fn fetch_secret(secret_name: String) -> Result<String, String> {
    spawn_blocking(move || {
        secrets::get(&secret_name).ok_or_else(|| missing_provider_secret_error(&secret_name))
    })
    .await
    .map_err(|error| format!("secret lookup task failed: {error}"))?
}

fn missing_provider_secret_error(secret_name: &str) -> String {
    format!(
        "missing provider secret \"{secret_name}\" — run tellm secret set {secret_name} \
         in a separate terminal, not the one running the bot; the bot uses the key from \
         the next message without a restart"
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::rooms::RoomSettings;

    #[test]
    fn reset_notice_only_mentions_real_wire_format_switches() {
        assert_eq!(reset_notice(HistoryReset::None), None);
        assert_eq!(
            reset_notice(HistoryReset::WireFormatChanged {
                previous: None,
                new: WireFormat::Anthropic,
            }),
            None
        );
        assert!(
            reset_notice(HistoryReset::WireFormatChanged {
                previous: Some(WireFormat::Anthropic),
                new: WireFormat::Responses,
            })
            .unwrap()
            .contains("Anthropic")
        );
    }

    fn reply_to(message: &str) -> String {
        provider_error_reply(&TurnError::from(message.to_string()))
    }

    #[test]
    fn provider_error_reply_matches_both_capability_spellings() {
        // The xAI backstop error uses the underscore form.
        let xai = reply_to("xAI Responses does not support OpenAI image_generation");
        assert!(xai.contains("Tip: /imagegen off"), "{xai}");
        let anthropic = reply_to("Anthropic Messages does not support image generation");
        assert!(anthropic.contains("Tip: /imagegen off"), "{anthropic}");
        let compat =
            reply_to("chat completions compat does not support provider-native web search");
        assert!(compat.contains("Tip: /websearch off"), "{compat}");
        let plain = reply_to("api error 500: boom");
        assert!(!plain.contains("Tip:"), "{plain}");
    }

    #[test]
    fn provider_error_reply_offers_a_photo_or_another_model_for_documents() {
        let reply = reply_to("chat completions compat does not support documents");
        assert!(reply.contains("send a screenshot as a photo"), "{reply}");
        assert!(reply.contains("/model"), "{reply}");
    }

    #[test]
    fn provider_error_reply_suggests_new_summary_when_the_chat_outgrew_the_window() {
        let error = TurnError::from(ProviderError::ContextWindowExceeded {
            status: 400,
            message: "prompt is too long: 1048577 tokens > 1048576 maximum".to_string(),
        });
        assert!(error.context_window);

        let reply = provider_error_reply(&error);

        assert!(
            reply.starts_with("Provider error: the request exceeds the model's context window"),
            "{reply}"
        );
        assert!(reply.contains("Tip: this chat no longer fits"), "{reply}");
        assert!(reply.contains("/new summary"), "{reply}");
        assert!(
            !TurnError::from(ProviderError::Api {
                status: 400,
                message: "context".to_string()
            })
            .context_window
        );
    }

    #[test]
    fn provider_error_reply_suggests_less_reasoning_when_the_output_limit_ate_the_reply() {
        let error = TurnError::from(tellm_core::ProviderError::Api {
            status: 200,
            message: tellm_anthropic::OUTPUT_LIMIT_BEFORE_TEXT.to_string(),
        });

        let reply = provider_error_reply(&error);

        assert!(reply.contains("Tip: /reasoning low"), "{reply}");
        assert!(reply.contains("max_output_tokens"), "{reply}");
    }

    #[test]
    fn unsupported_image_types_suggest_sending_a_photo() {
        let error = TurnError::from(tellm_core::ProviderError::Unsupported(
            "Anthropic Messages does not support image media type image/heic".to_string(),
        ));

        let reply = provider_error_reply(&error);

        assert!(reply.contains("send the picture as a photo"), "{reply}");
    }

    #[test]
    fn output_limit_tip_names_max_output_tokens_only_for_anthropic() {
        let error = TurnError::from(tellm_core::ProviderError::Api {
            status: 200,
            message: tellm_core::OUTPUT_LIMIT_BEFORE_TEXT.to_string(),
        });

        let reply = provider_error_reply(&error);

        assert!(reply.contains("Tip: /reasoning low"), "{reply}");
        assert!(
            !reply.contains("max_output_tokens"),
            "only the anthropic wire format accepts that setting: {reply}"
        );
    }

    #[test]
    fn replies_with_nothing_to_show_are_rejected() {
        let reply = |text: &str, images: Vec<GeneratedImage>| ChatResponse {
            text: text.to_string(),
            images,
            turn_items: vec![serde_json::json!({ "type": "reasoning", "id": "rs_1" })],
            usage: None,
        };
        let image = || GeneratedImage {
            media_type: "image/png".to_string(),
            base64: "aGVsbG8=".to_string(),
        };

        assert_eq!(
            require_reply_content(reply("", Vec::new()))
                .unwrap_err()
                .message,
            EMPTY_REPLY_ERROR
        );
        assert_eq!(
            require_reply_content(reply(" \n", Vec::new()))
                .unwrap_err()
                .message,
            EMPTY_REPLY_ERROR
        );
        assert_eq!(
            require_reply_content(reply("Answer.", Vec::new()))
                .unwrap()
                .text,
            "Answer."
        );
        assert_eq!(
            require_reply_content(reply("", vec![image()]))
                .unwrap()
                .images
                .len(),
            1,
            "an image-only reply is a real reply"
        );
    }

    #[test]
    fn cost_line_follows_the_reply_text_after_a_blank_line() {
        assert_eq!(
            with_cost_line("Answer.\n\n", "Cost ≈ $0.010 · this chat ≈ $0.020"),
            "Answer.\n\nCost ≈ $0.010 · this chat ≈ $0.020"
        );
    }

    #[test]
    fn largest_photo_prefers_pixel_area() {
        let photos = vec![
            PhotoSize {
                file_id: "small".to_string(),
                width: 10,
                height: 10,
                file_size: None,
            },
            PhotoSize {
                file_id: "wide".to_string(),
                width: 50,
                height: 20,
                file_size: None,
            },
        ];

        assert_eq!(largest_photo(Some(&photos)).unwrap().file_id, "wide");
    }

    #[test]
    fn largest_photo_compares_overflowing_pixel_areas() {
        let overflowing = PhotoSize {
            file_id: "overflowing".to_string(),
            width: i64::MAX,
            height: i64::MAX,
            file_size: None,
        };
        let plain = PhotoSize {
            file_id: "plain".to_string(),
            width: 1_000,
            height: 1_000,
            file_size: None,
        };

        // `i64::MAX * i64::MAX` overflows i64, yet its mathematical area is
        // far larger than 1_000 * 1_000; selection must not panic or wrap.
        let photos = vec![plain.clone(), overflowing.clone()];
        assert_eq!(largest_photo(Some(&photos)).unwrap().file_id, "overflowing");
        let photos = vec![overflowing, plain];
        assert_eq!(largest_photo(Some(&photos)).unwrap().file_id, "overflowing");
    }

    #[test]
    fn attachment_size_limit_rejects_declared_or_downloaded_oversize_payloads() {
        assert!(validate_attachment_size("photo", MAX_ATTACHMENT_BYTES).is_ok());
        let error = validate_attachment_size("document", MAX_ATTACHMENT_BYTES + 1).unwrap_err();
        assert!(error.contains("document is too large"), "{error}");
        assert!(error.contains("20 MiB"), "{error}");
    }

    #[test]
    fn text_document_detection_accepts_mime_or_extension() {
        let mut document = Document {
            file_id: "f".to_string(),
            file_name: Some("notes.TXT".to_string()),
            mime_type: None,
            file_size: None,
        };

        assert!(is_text_document(&document, "application/octet-stream"));
        document.file_name = Some("notes.pdf".to_string());
        assert!(is_text_document(&document, "text/plain"));
        assert!(!is_text_document(&document, "application/pdf"));
    }

    #[test]
    fn missing_secret_error_points_outside_the_bots_terminal() {
        // The running bot's terminal only takes reset/exit/quit, so the hint
        // must send the user to another shell for the subcommand.
        let error = missing_provider_secret_error("gemini_api_key");
        assert!(error.contains("tellm secret set gemini_api_key"), "{error}");
        assert!(error.contains("separate terminal"), "{error}");
        assert!(!error.contains("tellm console"), "{error}");
    }

    #[test]
    fn message_model_input_detection_ignores_blank_messages() {
        let mut message = IncomingMessage {
            message_id: 1,
            chat: tellm_telegram::Chat {
                id: 42,
                title: None,
            },
            from: None,
            date: 1000,
            text: Some(" \n ".to_string()),
            caption: None,
            photo: None,
            document: None,
            media_group_id: None,
            album: Vec::new(),
            split_parts: 0,
            reply_to_message: None,
            new_chat_members: Vec::new(),
        };

        assert!(!message_has_input(&message));
        message.text = Some("hello".to_string());
        assert!(message_has_input(&message));
        message.text = None;
        message.photo = Some(vec![PhotoSize {
            file_id: "p".to_string(),
            width: 1,
            height: 1,
            file_size: None,
        }]);
        assert!(message_has_input(&message));
    }

    #[test]
    fn chat_request_uses_room_image_generation_toggle() {
        let room = RoomState::new(RoomSettings {
            image_generation: true,
            web_search: true,
            ..RoomSettings::default()
        });
        let mut model = model(WireFormat::Responses);
        model.model_name = "gpt-5.5".to_string();

        let request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "draw this".to_string(),
            }],
        );

        assert!(request.image_generation);
        assert!(request.web_search);
        assert_eq!(request.model, "gpt-5.5");
        assert_eq!(request.thinking, tellm_core::ThinkingLevel::High);
    }

    #[test]
    fn chat_request_uses_room_thinking_override_when_present() {
        let room = RoomState::new(RoomSettings {
            thinking: Some(tellm_core::ThinkingLevel::Low),
            ..RoomSettings::default()
        });
        let model = model(WireFormat::Responses);

        let request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "think less".to_string(),
            }],
        );

        assert_eq!(request.thinking, tellm_core::ThinkingLevel::Low);
    }

    #[test]
    fn message_mode_request_omits_retained_latest_turn() {
        let mut room = RoomState::new(RoomSettings {
            mode: ChatMode::Message,
            ..RoomSettings::default()
        });
        room.append_turn(
            WireFormat::Compat,
            vec![serde_json::json!({ "role": "assistant", "content": "previous" })],
        );
        let mut model = model(WireFormat::Compat);
        model.base_url = Some("http://localhost:11434/v1".to_string());
        model.api_key_secret = None;

        let message_request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "fresh".to_string(),
            }],
        );
        assert!(message_request.history.is_empty());

        room.settings.mode = ChatMode::Chat;
        let chat_request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "continue".to_string(),
            }],
        );
        assert_eq!(chat_request.history.len(), 1);
    }

    fn fresh(prompt: &str, notice: Option<&str>) -> ConversationMemory {
        ConversationMemory {
            prompt: Some(prompt.to_string()),
            notice: notice.map(str::to_string),
        }
    }

    fn text(text: &str) -> Vec<ContentPart> {
        vec![ContentPart::Text {
            text: text.to_string(),
        }]
    }

    #[test]
    fn a_new_conversation_starts_with_the_current_memory_after_the_role() {
        let mut room = RoomState::new(RoomSettings {
            role: Some("Be brief.".to_string()),
            ..RoomSettings::default()
        });

        let prepared = prepare_chat_request(
            &mut room,
            model(WireFormat::Responses),
            text("hi"),
            fresh("Facts v1", Some("note")),
        );

        assert_eq!(
            prepared.request.system.as_deref(),
            Some("Be brief.\n\nFacts v1")
        );
        assert_eq!(prepared.memory.as_deref(), Some("Facts v1"));
        assert_eq!(prepared.memory_notice.as_deref(), Some("note"));
    }

    #[test]
    fn a_running_conversation_keeps_the_memory_it_started_with() {
        let mut room = RoomState::new(RoomSettings::default());
        room.begin_turn(WireFormat::Responses);
        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);
        room.memory = Some("Facts v1".to_string());

        let prepared = prepare_chat_request(
            &mut room,
            model(WireFormat::Responses),
            text("and then?"),
            fresh("Facts v2", Some("note")),
        );

        assert_eq!(prepared.request.system.as_deref(), Some("Facts v1"));
        assert_eq!(prepared.memory.as_deref(), Some("Facts v1"));
        assert_eq!(prepared.memory_notice, None);
    }

    #[test]
    fn message_mode_takes_the_current_memory_every_time() {
        let mut room = RoomState::new(RoomSettings {
            mode: ChatMode::Message,
            ..RoomSettings::default()
        });
        room.begin_turn(WireFormat::Responses);
        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);
        room.memory = Some("Facts v1".to_string());

        let prepared = prepare_chat_request(
            &mut room,
            model(WireFormat::Responses),
            text("one-off"),
            fresh("Facts v2", None),
        );

        assert_eq!(prepared.request.system.as_deref(), Some("Facts v2"));
    }

    #[test]
    fn system_prompt_puts_the_role_first_then_memory_then_the_summary() {
        assert_eq!(
            system_prompt(Some("Be brief."), Some("Facts"), Some("Summary")).as_deref(),
            Some("Be brief.\n\nFacts\n\nSummary")
        );
        assert_eq!(
            system_prompt(Some("Be brief."), None, None).as_deref(),
            Some("Be brief.")
        );
        assert_eq!(
            system_prompt(None, Some("Facts"), None).as_deref(),
            Some("Facts")
        );
        assert_eq!(
            system_prompt(None, None, Some("Summary")).as_deref(),
            Some("Summary")
        );
        assert_eq!(system_prompt(None, None, None), None);
    }

    #[test]
    fn a_conversation_from_a_summary_carries_it_on_every_turn() {
        let mut room = RoomState::new(RoomSettings {
            role: Some("Be brief.".to_string()),
            ..RoomSettings::default()
        });
        room.continue_from_summary("Moving in May.".to_string());
        let expected = "Be brief.\n\nFacts v1\n\n\
            Summary of the earlier part of this conversation:\nMoving in May.";

        let first = prepare_chat_request(
            &mut room,
            model(WireFormat::Responses),
            text("hi"),
            fresh("Facts v1", None),
        );
        assert_eq!(first.request.system.as_deref(), Some(expected));
        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);
        room.memory = first.memory;

        let later = prepare_chat_request(
            &mut room,
            model(WireFormat::Responses),
            text("and then?"),
            fresh("Facts v2", None),
        );
        assert_eq!(later.request.system.as_deref(), Some(expected));
    }

    #[test]
    fn a_room_without_memory_sends_only_its_role() {
        let mut room = RoomState::new(RoomSettings {
            role: Some("Be brief.".to_string()),
            ..RoomSettings::default()
        });

        let prepared = prepare_chat_request(
            &mut room,
            model(WireFormat::Anthropic),
            text("hi"),
            ConversationMemory::default(),
        );

        assert_eq!(prepared.request.system.as_deref(), Some("Be brief."));
        assert_eq!(prepared.memory, None);
    }

    #[test]
    fn provider_secret_startup_warmup_uses_unique_configured_secret_names() {
        let mut models = BTreeMap::new();
        let mut claude = model(WireFormat::Anthropic);
        claude.api_key_secret = Some("shared_api_key".to_string());
        models.insert("claude".to_string(), claude);

        let mut grok = model(WireFormat::Responses);
        grok.api_key_secret = Some("shared_api_key".to_string());
        models.insert("grok".to_string(), grok);

        let mut gemini = model(WireFormat::Gemini);
        gemini.api_key_secret = Some("gemini_api_key".to_string());
        models.insert("gemini".to_string(), gemini);

        let mut ollama = model(WireFormat::Compat);
        ollama.api_key_secret = None;
        models.insert("ollama".to_string(), ollama);

        let config = Config {
            default_model: "claude".to_string(),
            models,
            telegram: tellm_config::TelegramConfig::default(),
        };

        assert_eq!(
            configured_provider_secret_names(&config),
            vec!["gemini_api_key".to_string(), "shared_api_key".to_string()]
        );
    }

    fn photo_message(file_id: &str) -> IncomingMessage {
        IncomingMessage {
            message_id: 1,
            chat: tellm_telegram::Chat {
                id: 42,
                title: None,
            },
            from: None,
            date: 1000,
            text: None,
            caption: None,
            photo: Some(vec![PhotoSize {
                file_id: file_id.to_string(),
                width: 1,
                height: 1,
                file_size: Some(5),
            }]),
            document: None,
            media_group_id: Some("album".to_string()),
            album: Vec::new(),
            split_parts: 0,
            reply_to_message: None,
            new_chat_members: Vec::new(),
        }
    }

    #[tokio::test]
    async fn an_album_is_its_caption_then_every_item_in_order() {
        use tellm_test_support::{MockHttpServer, MockResponse};
        let file = |path: &str| {
            MockResponse::json(
                200,
                serde_json::json!({ "ok": true, "result": { "file_path": path } }),
            )
        };
        let server = MockHttpServer::start(vec![
            file("photos/a.jpg"),
            MockResponse::bytes(200, "image/jpeg", b"first".to_vec()),
            file("photos/b.jpg"),
            MockResponse::bytes(200, "image/jpeg", b"other".to_vec()),
        ]);
        let telegram = Telegram::with_base_urls("token", server.base_url(), server.base_url());
        let mut album = photo_message("a");
        album.caption = Some("two shots".to_string());
        album.album = vec![photo_message("b")];

        assert!(message_has_input(&album));
        let parts = content_parts_from_message(&telegram, &album).await.unwrap();

        assert!(
            matches!(&parts[0], ContentPart::Text { text } if text == "two shots"),
            "{parts:?}"
        );
        let images: Vec<Vec<u8>> = parts[1..]
            .iter()
            .map(|part| match part {
                ContentPart::Image { base64, .. } => BASE64.decode(base64).unwrap(),
                other => panic!("expected an image, got {other:?}"),
            })
            .collect();
        assert_eq!(images, vec![b"first".to_vec(), b"other".to_vec()]);
    }

    #[tokio::test]
    async fn compat_api_key_distinguishes_keyless_from_missing_secret() {
        let mut keyless = model(WireFormat::Compat);
        keyless.api_key_secret = None;
        assert_eq!(compat_api_key(&keyless).await.unwrap(), "");

        let mut keyed = model(WireFormat::Compat);
        keyed.api_key_secret = Some("definitely_missing_tellm_test_secret".to_string());
        let error = compat_api_key(&keyed).await.unwrap_err();
        assert!(
            error.contains("definitely_missing_tellm_test_secret"),
            "{error}"
        );
    }

    fn model(wire_format: WireFormat) -> ModelConfig {
        ModelConfig {
            wire_format,
            model_name: "model".to_string(),
            base_url: None,
            allow_insecure_http: false,
            api_key_secret: Some("secret".to_string()),
            telegram_chat_ids: Vec::new(),
            thinking: tellm_core::ThinkingLevel::High,
            max_effort: None,
            off_effort: None,
            max_output_tokens: None,
            price: None,
        }
    }
}
