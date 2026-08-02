use std::collections::BTreeSet;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use tellm_anthropic::Anthropic;
use tellm_compat::Compat;
use tellm_config::{Config, ModelConfig, WireFormat, secrets};
use tellm_core::{ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider};
use tellm_gemini::Gemini;
use tellm_openai::Responses;
use tellm_telegram::{Document, IncomingMessage, PhotoSize, Telegram, TelegramError};
use tokio::task::spawn_blocking;

use crate::ollama;
use crate::rooms::{ChatMode, FailedTurnRollback, HistoryReset, RoomState};

const MAX_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;

pub(crate) struct PreparedChatRequest {
    pub(crate) model_config: ModelConfig,
    pub(crate) request: ChatRequest,
    pub(crate) rollback: Option<FailedTurnRollback>,
    pub(crate) generation: u64,
    pub(crate) reset_notice: Option<String>,
}

pub(crate) fn prepare_chat_request(
    room: &mut RoomState,
    model_config: ModelConfig,
    input: Vec<ContentPart>,
) -> PreparedChatRequest {
    let start = room.begin_turn(model_config.wire_format);
    let reset_notice = reset_notice(start.reset);
    let request = chat_request_from_room(room, &model_config, input);
    let generation = room.generation();

    PreparedChatRequest {
        model_config,
        request,
        rollback: start.rollback,
        generation,
        reset_notice,
    }
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

/// The room's model can change after a toggle was accepted, so a capability
/// error can still surface per-message; suggest the off switch.
pub(crate) fn provider_error_reply(error: &str) -> String {
    // Provider error texts use both spellings ("image generation" from
    // Anthropic/compat, "image_generation" from the xAI backstop) — match
    // both.
    let mut reply = format!("Provider error: {error}");
    if error.contains("image generation") || error.contains("image_generation") {
        reply.push_str("\nTip: /imagegen off");
    } else if error.contains("web search") || error.contains("web_search") {
        reply.push_str("\nTip: /websearch off");
    }
    reply
}

pub(crate) async fn dispatch_provider(
    model: &ModelConfig,
    request: &ChatRequest,
) -> Result<ChatResponse, String> {
    match model.wire_format {
        WireFormat::Anthropic => {
            let api_key = required_api_key(model).await?;
            Anthropic::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Responses => {
            let api_key = required_api_key(model).await?;
            Responses::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Compat => {
            let base_url = model
                .base_url
                .clone()
                .ok_or_else(|| "compat model is missing base_url".to_string())?;
            ollama::ensure_ready(&base_url).await?;
            let requested_model = request.model.clone();
            let api_key = compat_api_key(model).await?;
            // Register before the request: an aborted task can still leave
            // Ollama loading the model after the HTTP future is dropped.
            ollama::remember_model(&base_url, &requested_model);
            Compat::new(api_key, base_url)
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Gemini => {
            let api_key = required_api_key(model).await?;
            Gemini::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
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

    Ok(parts)
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

pub(crate) async fn send_response(
    telegram: &Telegram,
    chat_id: i64,
    response: ChatResponse,
) -> Result<(), String> {
    if !response.text.trim().is_empty() || response.images.is_empty() {
        telegram
            .send_message(chat_id, &response.text)
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

    Ok(())
}

pub(crate) fn message_text(message: &IncomingMessage) -> Option<&str> {
    message.text.as_deref().or(message.caption.as_deref())
}

pub(crate) fn message_has_input(message: &IncomingMessage) -> bool {
    message_text(message).is_some_and(|text| !text.trim().is_empty())
        || message
            .photo
            .as_ref()
            .is_some_and(|photos| !photos.is_empty())
        || message.document.is_some()
}

fn largest_photo(photos: Option<&[PhotoSize]>) -> Option<&PhotoSize> {
    photos?
        .iter()
        .max_by_key(|photo| photo.width * photo.height)
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
        "missing provider secret \"{secret_name}\" — set it in the tellm console with: \
         tellm secret set {secret_name}"
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

    #[test]
    fn provider_error_reply_matches_both_capability_spellings() {
        // The xAI backstop error uses the underscore form.
        let xai = provider_error_reply("xAI Responses does not support OpenAI image_generation");
        assert!(xai.contains("Tip: /imagegen off"), "{xai}");
        let anthropic =
            provider_error_reply("Anthropic Messages does not support image generation");
        assert!(anthropic.contains("Tip: /imagegen off"), "{anthropic}");
        let compat = provider_error_reply(
            "chat completions compat does not support provider-native web search",
        );
        assert!(compat.contains("Tip: /websearch off"), "{compat}");
        let plain = provider_error_reply("api error 500: boom");
        assert!(!plain.contains("Tip:"), "{plain}");
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
    fn message_model_input_detection_ignores_blank_messages() {
        let mut message = IncomingMessage {
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
        }
    }
}
