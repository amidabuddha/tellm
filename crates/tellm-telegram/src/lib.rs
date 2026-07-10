//! Minimal Telegram Bot API client — raw HTTP via reqwest, no framework.
//!
//! The delivery path ports the battle-tested fallback chain from
//! console-chat-gpt (`console_gpt/telegram_bot.py`):
//! `sendRichMessage` (Bot API 10.1 markdown) → `sendMessage` with HTML
//! entities → plain text. The pure functions (`chunk_text`,
//! `markdown_to_html`) must be golden-tested against the Python reference
//! implementation before the runtime goes live — see DESIGN.md § Porting
//! method.

use std::time::Duration;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Value, json};

const API_BASE_URL: &str = "https://api.telegram.org";
const FILE_BASE_URL: &str = "https://api.telegram.org/file";
pub const TELEGRAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const TELEGRAM_REQUEST_TIMEOUT: Duration = Duration::from_secs(90);
pub const TELEGRAM_LONG_POLL_GRACE: Duration = Duration::from_secs(30);

/// Chunk limit for plain/HTML `sendMessage` (Telegram caps at 4096).
/// Caveat: `chunk_text` counts chars while Telegram's cap counts UTF-16 code
/// units, so astral-plane-heavy text (emoji) can still exceed the cap and
/// fail delivery. Inherited from the Python reference — the golden vectors
/// pin char-counting semantics, so any fix belongs there first.
pub const TEXT_CHUNK_SIZE: usize = 3900;
/// Chunk limit for `sendRichMessage`.
pub const RICH_CHUNK_SIZE: usize = 32000;

#[derive(Debug, thiserror::Error)]
pub enum TelegramError {
    #[error("http error: {0}")]
    Http(String),
    #[error("telegram api error {code}: {description}")]
    Api { code: i64, description: String },
    #[error("telegram api returned invalid response: {0}")]
    InvalidResponse(String),
    #[error("telegram getFile response did not include file_path")]
    MissingFilePath,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Update {
    pub update_id: i64,
    pub message: Option<IncomingMessage>,
    pub edited_message: Option<IncomingMessage>,
    /// The bot's own membership changed in a chat (added/removed). This is
    /// how tellm notices being added to a group — checked 2026-07-05 against
    /// core.telegram.org/bots/api#chatmemberupdated.
    pub my_chat_member: Option<ChatMemberUpdated>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMemberUpdated {
    pub chat: Chat,
    /// The user who performed the membership change (e.g. added the bot).
    pub from: Option<User>,
    pub date: i64,
    pub new_chat_member: ChatMember,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMember {
    pub status: String,
}

/// The subset of Telegram's `Message` that tellm consumes.
#[derive(Debug, Clone, Deserialize)]
pub struct IncomingMessage {
    pub chat: Chat,
    pub from: Option<User>,
    pub date: i64,
    pub text: Option<String>,
    pub caption: Option<String>,
    pub photo: Option<Vec<PhotoSize>>,
    pub document: Option<Document>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Chat {
    pub id: i64,
    pub title: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
}

impl Chat {
    /// Human label for console messages: group title when present, else id.
    pub fn label(&self) -> String {
        match &self.title {
            Some(title) => format!("'{title}' ({})", self.id),
            None => self.id.to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct User {
    pub id: i64,
    pub is_bot: bool,
    pub first_name: String,
    pub username: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PhotoSize {
    pub file_id: String,
    pub file_unique_id: Option<String>,
    pub width: i64,
    pub height: i64,
    pub file_size: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Document {
    pub file_id: String,
    pub file_unique_id: Option<String>,
    pub file_name: Option<String>,
    pub mime_type: Option<String>,
    pub file_size: Option<i64>,
}

#[derive(Clone)]
pub struct Telegram {
    token: String,
    http: reqwest::Client,
    api_base_url: String,
    file_base_url: String,
    request_timeout: Duration,
}

impl Telegram {
    pub fn new(token: impl Into<String>) -> Self {
        Self::with_base_urls(token, API_BASE_URL, FILE_BASE_URL)
    }

    pub fn with_base_urls(
        token: impl Into<String>,
        api_base_url: impl Into<String>,
        file_base_url: impl Into<String>,
    ) -> Self {
        Self::with_base_urls_and_timeout(
            token,
            api_base_url,
            file_base_url,
            TELEGRAM_REQUEST_TIMEOUT,
        )
    }

    pub fn with_base_urls_and_timeout(
        token: impl Into<String>,
        api_base_url: impl Into<String>,
        file_base_url: impl Into<String>,
        request_timeout: Duration,
    ) -> Self {
        Self {
            token: token.into(),
            http: reqwest::Client::builder()
                .connect_timeout(TELEGRAM_CONNECT_TIMEOUT.min(request_timeout))
                .timeout(request_timeout)
                .build()
                .expect("valid reqwest client configuration"),
            api_base_url: trim_trailing_slash(api_base_url.into()),
            file_base_url: trim_trailing_slash(file_base_url.into()),
            request_timeout,
        }
    }

    /// Long-poll for updates. `timeout_s` stays modest so terminal controls
    /// remain responsive (same trade-off as the Python runtime).
    pub async fn get_updates(
        &self,
        offset: i64,
        timeout_s: u32,
    ) -> Result<Vec<Update>, TelegramError> {
        // Checked 2026-07-04 against core.telegram.org/bots/api#getupdates.
        self.post_json_with_timeout(
            "getUpdates",
            &json!({
                "offset": offset,
                "timeout": timeout_s,
                "allowed_updates": ["message", "my_chat_member"],
            }),
            Duration::from_secs(u64::from(timeout_s)) + TELEGRAM_LONG_POLL_GRACE,
        )
        .await
    }

    /// Validate the bot token and fetch bot metadata.
    pub async fn get_me(&self) -> Result<User, TelegramError> {
        // Checked 2026-07-04 against core.telegram.org/bots/api#getme.
        self.post_json("getMe", &json!({})).await
    }

    /// Deliver text via the rich → HTML → plain fallback chain, chunked.
    pub async fn send_message(&self, chat_id: i64, text: &str) -> Result<(), TelegramError> {
        let text = text.trim();
        let text = if text.is_empty() {
            "(empty response)"
        } else {
            text
        };

        for part in chunk_text(text, RICH_CHUNK_SIZE) {
            // Checked 2026-07-04 against core.telegram.org/bots/api#sendrichmessage.
            let result = self
                .post_json::<Value>(
                    "sendRichMessage",
                    &json!({
                        "chat_id": chat_id,
                        "rich_message": { "markdown": part },
                    }),
                )
                .await;

            match result {
                Ok(_) => {}
                Err(error) if should_fallback_from_rich_message_error(&error) => {
                    for legacy_part in chunk_text(&part, TEXT_CHUNK_SIZE) {
                        self.send_legacy_message_part(chat_id, &legacy_part).await?;
                    }
                }
                Err(error) => return Err(error),
            }
        }

        Ok(())
    }

    pub async fn send_chat_action(&self, chat_id: i64) -> Result<(), TelegramError> {
        // Checked 2026-07-04 against core.telegram.org/bots/api#sendchataction.
        self.post_json::<Value>(
            "sendChatAction",
            &json!({
                "chat_id": chat_id,
                "action": "typing",
            }),
        )
        .await?;
        Ok(())
    }

    pub async fn send_photo(
        &self,
        chat_id: i64,
        image_bytes: Vec<u8>,
    ) -> Result<(), TelegramError> {
        // Checked 2026-07-04 against core.telegram.org/bots/api#sendphoto.
        let (content_type, body) = photo_multipart_body(chat_id, &image_bytes);
        self.post_bytes::<Value>("sendPhoto", content_type, body)
            .await?;
        Ok(())
    }

    /// Fetch a user-sent file (photo or document) for native passthrough to
    /// the provider.
    pub async fn get_file_bytes(&self, file_id: &str) -> Result<Vec<u8>, TelegramError> {
        // Checked 2026-07-04 against core.telegram.org/bots/api#getfile.
        let file: TelegramFile = self
            .post_json("getFile", &json!({ "file_id": file_id }))
            .await?;
        let file_path = file.file_path.ok_or(TelegramError::MissingFilePath)?;
        let url = format!(
            "{}/bot{}/{}",
            self.file_base_url,
            self.token,
            file_path.trim_start_matches('/')
        );
        let response = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|error| self.http_error(error))?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| self.http_error(error))?;
        if !status.is_success() {
            return Err(TelegramError::Api {
                code: i64::from(status.as_u16()),
                description: self.sanitize_error_text(&String::from_utf8_lossy(&bytes)),
            });
        }
        Ok(bytes.to_vec())
    }

    async fn send_legacy_message_part(
        &self,
        chat_id: i64,
        part: &str,
    ) -> Result<(), TelegramError> {
        // Checked 2026-07-04 against core.telegram.org/bots/api#sendmessage and
        // LinkPreviewOptions on the same page.
        let html_result = self
            .post_json::<Value>(
                "sendMessage",
                &json!({
                    "chat_id": chat_id,
                    "text": markdown_to_html(part),
                    "parse_mode": "HTML",
                    "link_preview_options": { "is_disabled": true },
                }),
            )
            .await;

        match html_result {
            Ok(_) => Ok(()),
            Err(error) if should_fallback_from_html_message_error(&error) => {
                self.post_json::<Value>(
                    "sendMessage",
                    &json!({
                        "chat_id": chat_id,
                        "text": part,
                    }),
                )
                .await?;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    async fn post_json<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &impl Serialize,
    ) -> Result<T, TelegramError> {
        self.post_json_with_timeout(method, payload, self.request_timeout)
            .await
    }

    async fn post_json_with_timeout<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &impl Serialize,
        timeout: Duration,
    ) -> Result<T, TelegramError> {
        let response = self
            .http
            .post(self.method_url(method))
            .json(payload)
            .timeout(timeout)
            .send()
            .await
            .map_err(|error| self.http_error(error))?;
        self.parse_api_response(method, response).await
    }

    async fn post_bytes<T: DeserializeOwned>(
        &self,
        method: &str,
        content_type: String,
        body: Vec<u8>,
    ) -> Result<T, TelegramError> {
        let response = self
            .http
            .post(self.method_url(method))
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(body)
            .timeout(self.request_timeout)
            .send()
            .await
            .map_err(|error| self.http_error(error))?;
        self.parse_api_response(method, response).await
    }

    async fn parse_api_response<T: DeserializeOwned>(
        &self,
        method: &str,
        response: reqwest::Response,
    ) -> Result<T, TelegramError> {
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| self.http_error(error))?;
        if !status.is_success() {
            return Err(TelegramError::Api {
                code: i64::from(status.as_u16()),
                description: self
                    .sanitize_error_text(&format!("HTTP status from {method}: {text}")),
            });
        }

        let response: ApiResponse<T> = serde_json::from_str(&text).map_err(|error| {
            TelegramError::InvalidResponse(
                self.sanitize_error_text(&format!("{method}: non-JSON or malformed JSON: {error}")),
            )
        })?;

        if response.ok {
            response.result.ok_or_else(|| {
                TelegramError::InvalidResponse(
                    self.sanitize_error_text(&format!("{method}: ok response missing result")),
                )
            })
        } else {
            Err(TelegramError::Api {
                code: response.error_code.unwrap_or(0),
                description: self.sanitize_error_text(
                    &response
                        .description
                        .unwrap_or_else(|| "unknown error".to_string()),
                ),
            })
        }
    }

    fn http_error(&self, error: reqwest::Error) -> TelegramError {
        // reqwest attaches the request URL to transport and body errors. Bot
        // API URLs contain the authentication token, so discard the URL
        // before formatting and redact defensively before storing the error.
        TelegramError::Http(self.sanitize_error_text(&error.without_url().to_string()))
    }

    fn sanitize_error_text(&self, text: &str) -> String {
        if self.token.is_empty() {
            text.to_string()
        } else {
            text.replace(&self.token, "[REDACTED]")
        }
    }

    fn method_url(&self, method: &str) -> String {
        format!("{}/bot{}/{}", self.api_base_url, self.token, method)
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    ok: bool,
    result: Option<T>,
    error_code: Option<i64>,
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TelegramFile {
    file_path: Option<String>,
}

fn trim_trailing_slash(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}

fn should_fallback_from_rich_message_error(error: &TelegramError) -> bool {
    let error_text = error.to_string().to_lowercase();
    [
        "api error 404",
        "method not found",
        "can't parse",
        "cannot parse",
        "failed to parse",
        "can't find end",
        "unsupported start tag",
        "unsupported tag",
    ]
    .iter()
    .any(|marker| error_text.contains(marker))
}

fn should_fallback_from_html_message_error(error: &TelegramError) -> bool {
    let error_text = error.to_string().to_lowercase();
    error_text.contains("parse entities") || error_text.contains("can't parse entities")
}

fn photo_multipart_body(chat_id: i64, image_bytes: &[u8]) -> (String, Vec<u8>) {
    let boundary = "tellm-telegram-photo-boundary-20260704";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"chat_id\"\r\n\r\n");
    body.extend_from_slice(chat_id.to_string().as_bytes());
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"photo\"; filename=\"image.png\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: image/png\r\n\r\n");
    body.extend_from_slice(image_bytes);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// Split `text` into chunks of at most `chunk_size`, preferring to break at
/// newlines. Reference: `_chunk_text` in console-chat-gpt.
pub fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    assert!(chunk_size > 0, "chunk_size must be greater than zero");

    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = (start + chunk_size).min(chars.len());
        if end < chars.len()
            && let Some(split) = (start..end).rev().find(|&idx| chars[idx] == '\n')
            && split > start
        {
            end = split + 1;
        }

        chunks.push(chars[start..end].iter().collect());
        start = end;
    }

    chunks
}

/// Convert a subset of model-emitted markdown to Telegram-safe HTML
/// (fenced code, inline code, bold, headings, links). Reference:
/// `_telegram_markdown_to_html` in console-chat-gpt.
pub fn markdown_to_html(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let (without_fenced, fenced_blocks) = capture_fenced_blocks(text);
    let escaped_text = escape_html(&without_fenced);

    let lines = escaped_text
        .split('\n')
        .map(|line| {
            if let Some(heading) = markdown_heading(line) {
                format!("<b>{}</b>", heading.trim())
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>();

    let mut transformed = lines.join("\n");
    transformed = replace_links(&transformed);
    transformed = replace_delimited(&transformed, "**", "b");
    transformed = replace_delimited(&transformed, "__", "b");
    transformed = replace_delimited(&transformed, "`", "code");

    for (idx, block) in fenced_blocks.iter().enumerate() {
        let placeholder = format!("@@TG_CODEBLOCK_{idx}@@");
        transformed = transformed.replace(&placeholder, block);
    }

    transformed
}

fn capture_fenced_blocks(text: &str) -> (String, Vec<String>) {
    let mut output = String::with_capacity(text.len());
    let mut fenced_blocks = Vec::new();
    let mut search_start = 0;

    while let Some(open_relative) = text[search_start..].find("```") {
        let open = search_start + open_relative;
        let after_open = open + 3;

        let Some(content_start) = fenced_content_start(text, after_open) else {
            output.push_str(&text[search_start..open + 1]);
            search_start = open + 1;
            continue;
        };

        let Some(close_relative) = text[content_start..].find("```") else {
            output.push_str(&text[search_start..open + 1]);
            search_start = open + 1;
            continue;
        };

        let close = content_start + close_relative;
        output.push_str(&text[search_start..open]);
        let escaped = escape_html(text[content_start..close].trim_matches('\n'));
        let idx = fenced_blocks.len();
        fenced_blocks.push(format!("<pre><code>{escaped}</code></pre>"));
        output.push_str(&format!("@@TG_CODEBLOCK_{idx}@@"));
        search_start = close + 3;
    }

    output.push_str(&text[search_start..]);
    (output, fenced_blocks)
}

fn fenced_content_start(text: &str, after_open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut idx = after_open;
    while idx < bytes.len() {
        match bytes[idx] {
            b'\n' => return Some(idx + 1),
            b'`' => return None,
            _ => idx += 1,
        }
    }
    None
}

fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn markdown_heading(line: &str) -> Option<&str> {
    let mut idx = 0;
    for _ in 0..3 {
        let ch = line[idx..].chars().next()?;
        if !ch.is_whitespace() {
            break;
        }
        idx += ch.len_utf8();
    }

    let hash_start = idx;
    while idx < line.len() && line.as_bytes()[idx] == b'#' && idx - hash_start < 6 {
        idx += 1;
    }

    let hash_count = idx - hash_start;
    if hash_count == 0 {
        return None;
    }

    let mut whitespace_count = 0;
    let mut last_whitespace = idx;
    while idx < line.len() {
        let ch = line[idx..].chars().next()?;
        if !ch.is_whitespace() {
            break;
        }
        last_whitespace = idx;
        idx += ch.len_utf8();
        whitespace_count += 1;
    }

    if whitespace_count == 0 {
        return None;
    }

    if idx < line.len() {
        Some(&line[idx..])
    } else {
        Some(&line[last_whitespace..])
    }
}

fn replace_links(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut search_start = 0;

    while let Some(open_relative) = input[search_start..].find('[') {
        let open = search_start + open_relative;
        let label_start = open + 1;

        let Some(label_end_relative) = input[label_start..].find(']') else {
            output.push_str(&input[search_start..label_start]);
            search_start = label_start;
            continue;
        };

        let label_end = label_start + label_end_relative;
        if !input[label_end..].starts_with("](") {
            output.push_str(&input[search_start..label_start]);
            search_start = label_start;
            continue;
        }

        let url_start = label_end + 2;
        let Some(url_end_relative) = input[url_start..].find(')') else {
            output.push_str(&input[search_start..label_start]);
            search_start = label_start;
            continue;
        };

        let url_end = url_start + url_end_relative;
        let label = &input[label_start..label_end];
        let url = &input[url_start..url_end];

        if !label.is_empty()
            && !url.is_empty()
            && (url.starts_with("http://") || url.starts_with("https://"))
            && !url.chars().any(char::is_whitespace)
        {
            output.push_str(&input[search_start..open]);
            output.push_str(&format!("<a href=\"{url}\">{label}</a>"));
            search_start = url_end + 1;
        } else {
            output.push_str(&input[search_start..label_start]);
            search_start = label_start;
        }
    }

    output.push_str(&input[search_start..]);
    output
}

fn replace_delimited(input: &str, delimiter: &str, tag: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut search_start = 0;

    while let Some(open_relative) = input[search_start..].find(delimiter) {
        let open = search_start + open_relative;
        let content_start = open + delimiter.len();
        let content = &input[content_start..];
        let newline = content.find('\n');

        if let Some(close_relative) = content.find(delimiter)
            && close_relative > 0
            && newline.is_none_or(|newline| close_relative < newline)
        {
            let close = content_start + close_relative;
            output.push_str(&input[search_start..open]);
            output.push_str(&format!("<{tag}>{}</{tag}>", &input[content_start..close]));
            search_start = close + delimiter.len();
        } else {
            output.push_str(&input[search_start..content_start]);
            search_start = content_start;
        }
    }

    output.push_str(&input[search_start..]);
    output
}
