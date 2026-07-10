mod support;

use std::time::Duration;

use serde_json::json;
use support::{MockResponse, MockTelegram, body_contains};
use tellm_telegram::{TEXT_CHUNK_SIZE, Telegram, TelegramError};
use tokio::time::timeout;

const TOKEN: &str = "123:ABC";

fn client(mock: &MockTelegram) -> Telegram {
    Telegram::with_base_urls(TOKEN, mock.api_base_url(), mock.file_base_url())
}

fn short_timeout_client(mock: &MockTelegram) -> Telegram {
    Telegram::with_base_urls_and_timeout(
        TOKEN,
        mock.api_base_url(),
        mock.file_base_url(),
        Duration::from_millis(50),
    )
}

#[tokio::test]
async fn get_me_posts_empty_payload_and_parses_bot_user() {
    let mock = MockTelegram::start(vec![MockResponse::json_ok(json!({
        "id": 123,
        "is_bot": true,
        "first_name": "Tellm",
        "username": "tellm_bot"
    }))]);

    let bot = client(&mock).get_me().await.unwrap();

    assert_eq!(bot.id, 123);
    assert!(bot.is_bot);
    assert_eq!(bot.first_name, "Tellm");
    assert_eq!(bot.username.as_deref(), Some("tellm_bot"));

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, format!("/bot{TOKEN}/getMe"));
    assert_eq!(requests[0].json_body(), json!({}));
}

#[tokio::test]
async fn get_updates_posts_long_poll_payload_and_parses_messages() {
    let mock = MockTelegram::start(vec![MockResponse::json_ok(json!([
        {
            "update_id": 11,
            "message": {
                "chat": { "id": 42 },
                "date": 1000,
                "text": "hello"
            }
        },
        {
            "update_id": 12,
            "edited_message": {
                "chat": { "id": 43 },
                "date": 1001,
                "caption": "caption",
                "photo": [
                    { "file_id": "small", "width": 10, "height": 10 },
                    { "file_id": "large", "width": 100, "height": 100 }
                ]
            }
        },
        {
            "update_id": 13,
            "my_chat_member": {
                "chat": { "id": -900, "title": "Grok Chat", "type": "group" },
                "date": 1002,
                "from": { "id": 7, "is_bot": false, "first_name": "Owner" },
                "old_chat_member": { "status": "left" },
                "new_chat_member": { "status": "member" }
            }
        }
    ]))]);

    let updates = client(&mock).get_updates(99, 5).await.unwrap();

    assert_eq!(updates.len(), 3);
    assert_eq!(updates[0].update_id, 11);
    assert_eq!(updates[0].message.as_ref().unwrap().chat.id, 42);
    assert_eq!(
        updates[1]
            .edited_message
            .as_ref()
            .unwrap()
            .photo
            .as_ref()
            .unwrap()[1]
            .file_id,
        "large"
    );
    let membership = updates[2].my_chat_member.as_ref().unwrap();
    assert_eq!(membership.chat.id, -900);
    assert_eq!(membership.chat.label(), "'Grok Chat' (-900)");
    assert_eq!(membership.new_chat_member.status, "member");

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, format!("/bot{TOKEN}/getUpdates"));
    assert_eq!(
        requests[0].json_body(),
        json!({
            "offset": 99,
            "timeout": 5,
            "allowed_updates": ["message", "my_chat_member"]
        })
    );
}

#[tokio::test]
async fn send_message_uses_rich_message_happy_path() {
    let mock = MockTelegram::start(vec![MockResponse::json_ok(json!({ "message_id": 1 }))]);

    client(&mock)
        .send_message(42, " hello **there** ")
        .await
        .unwrap();

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].path, format!("/bot{TOKEN}/sendRichMessage"));
    assert_eq!(
        requests[0].json_body(),
        json!({
            "chat_id": 42,
            "rich_message": { "markdown": "hello **there**" }
        })
    );
}

#[tokio::test]
async fn help_bullet_list_uses_rich_message_without_raw_angle_placeholders() {
    let mock = MockTelegram::start(vec![MockResponse::json_ok(json!({ "message_id": 1 }))]);
    let help = "- /new - reset this chat\n- /pair CODE - pair a new chat\n- /help - show commands";

    client(&mock).send_message(42, help).await.unwrap();

    let body = mock.requests()[0].json_body();
    assert_eq!(
        body,
        json!({
            "chat_id": 42,
            "rich_message": { "markdown": help }
        })
    );
    let markdown = body["rich_message"]["markdown"].as_str().unwrap();
    assert!(!markdown.contains('<'));
    assert!(!markdown.contains('>'));
}

#[tokio::test]
async fn empty_send_message_uses_reference_placeholder() {
    let mock = MockTelegram::start(vec![MockResponse::json_ok(json!({ "message_id": 1 }))]);

    client(&mock).send_message(42, " \n ").await.unwrap();

    assert_eq!(
        mock.requests()[0].json_body(),
        json!({
            "chat_id": 42,
            "rich_message": { "markdown": "(empty response)" }
        })
    );
}

#[tokio::test]
async fn rich_404_falls_back_to_html_send_message() {
    let mock = MockTelegram::start(vec![
        MockResponse::json_error(404, "Not Found: method not found"),
        MockResponse::json_ok(json!({ "message_id": 2 })),
    ]);

    client(&mock).send_message(42, "**bold**").await.unwrap();

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].path, format!("/bot{TOKEN}/sendRichMessage"));
    assert_eq!(requests[1].path, format!("/bot{TOKEN}/sendMessage"));
    assert_eq!(
        requests[1].json_body(),
        json!({
            "chat_id": 42,
            "text": "<b>bold</b>",
            "parse_mode": "HTML",
            "link_preview_options": { "is_disabled": true }
        })
    );
}

#[tokio::test]
async fn html_entity_parse_error_falls_back_to_plain_text() {
    let mock = MockTelegram::start(vec![
        MockResponse::json_error(400, "Bad Request: failed to parse rich message"),
        MockResponse::json_error(400, "Bad Request: can't parse entities"),
        MockResponse::json_ok(json!({ "message_id": 3 })),
    ]);

    client(&mock)
        .send_message(42, "Run `cargo test`")
        .await
        .unwrap();

    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[1].json_body()["text"],
        json!("Run <code>cargo test</code>")
    );
    assert_eq!(
        requests[2].json_body(),
        json!({ "chat_id": 42, "text": "Run `cargo test`" })
    );
}

#[tokio::test]
async fn rich_fallback_chunks_legacy_send_message_parts() {
    let long_text = "A".repeat(TEXT_CHUNK_SIZE + 1);
    let mock = MockTelegram::start(vec![
        MockResponse::json_error(404, "Not Found: method not found"),
        MockResponse::json_ok(json!({ "message_id": 4 })),
        MockResponse::json_ok(json!({ "message_id": 5 })),
    ]);

    client(&mock).send_message(42, &long_text).await.unwrap();

    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        requests[1].json_body()["text"].as_str().unwrap().len(),
        TEXT_CHUNK_SIZE
    );
    assert_eq!(requests[2].json_body()["text"], json!("A"));
}

#[tokio::test]
async fn non_fallback_error_propagates_without_retry() {
    // "chat not found" matches no fallback marker: the error must surface,
    // and no sendMessage fallback request may be attempted.
    let mock = MockTelegram::start(vec![MockResponse::json_error(
        400,
        "Bad Request: chat not found",
    )]);

    let error = client(&mock).send_message(42, "hello").await.unwrap_err();

    assert!(error.to_string().contains("chat not found"), "{error}");
    let requests = mock.requests();
    assert_eq!(requests.len(), 1, "no fallback request expected");
    assert_eq!(requests[0].path, format!("/bot{TOKEN}/sendRichMessage"));
}

#[tokio::test]
async fn send_chat_action_posts_typing() {
    let mock = MockTelegram::start(vec![MockResponse::json_ok(json!(true))]);

    client(&mock).send_chat_action(42).await.unwrap();

    assert_eq!(
        mock.requests()[0].json_body(),
        json!({ "chat_id": 42, "action": "typing" })
    );
}

#[tokio::test]
async fn stalled_api_request_times_out_instead_of_hanging() {
    let mock = MockTelegram::start_stalled();

    let error = timeout(
        Duration::from_secs(2),
        short_timeout_client(&mock).send_chat_action(42),
    )
    .await
    .expect("client future should complete within test timeout")
    .unwrap_err();

    assert!(matches!(error, TelegramError::Http(_)), "{error}");
    let rendered = error.to_string();
    assert!(!rendered.contains(TOKEN), "token leaked in: {rendered}");
    assert!(
        !rendered.contains(&format!("/bot{TOKEN}/sendChatAction")),
        "token-bearing URL leaked in: {rendered}"
    );
    assert_eq!(
        mock.requests()[0].path,
        format!("/bot{TOKEN}/sendChatAction")
    );
}

#[tokio::test]
async fn telegram_api_error_body_cannot_echo_the_bot_token() {
    let mock = MockTelegram::start(vec![MockResponse::json_error(
        500,
        &format!(
            "upstream failed at {}/bot{TOKEN}/getMe",
            "https://api.telegram.org"
        ),
    )]);

    let error = client(&mock).get_me().await.unwrap_err();
    let rendered = error.to_string();

    assert!(!rendered.contains(TOKEN), "token leaked in: {rendered}");
    assert!(
        !rendered.contains(&format!("/bot{TOKEN}/getMe")),
        "token-bearing URL leaked in: {rendered}"
    );
    assert!(rendered.contains("[REDACTED]"), "{rendered}");
}

#[tokio::test]
async fn get_file_bytes_fetches_file_path_from_file_base_url() {
    let mock = MockTelegram::start(vec![
        MockResponse::json_ok(json!({
            "file_id": "file-1",
            "file_path": "photos/photo.jpg"
        })),
        MockResponse::bytes(200, "application/octet-stream", b"image-bytes".to_vec()),
    ]);

    let bytes = client(&mock).get_file_bytes("file-1", 1024).await.unwrap();

    assert_eq!(bytes, b"image-bytes");
    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, format!("/bot{TOKEN}/getFile"));
    assert_eq!(requests[0].json_body(), json!({ "file_id": "file-1" }));
    assert_eq!(requests[1].method, "GET");
    assert_eq!(
        requests[1].path,
        format!("/file/bot{TOKEN}/photos/photo.jpg")
    );
}

#[tokio::test]
async fn get_file_bytes_rejects_declared_oversize_before_buffering() {
    let mock = MockTelegram::start(vec![
        MockResponse::json_ok(json!({
            "file_id": "file-1",
            "file_path": "documents/large.bin"
        })),
        MockResponse::bytes(200, "application/octet-stream", b"too-large".to_vec()),
    ]);

    let error = client(&mock).get_file_bytes("file-1", 4).await.unwrap_err();

    assert!(matches!(
        error,
        TelegramError::FileTooLarge {
            size: 9,
            max_bytes: 4
        }
    ));
}

#[tokio::test]
async fn get_file_bytes_stops_an_unknown_length_body_at_the_limit() {
    let mock = MockTelegram::start(vec![
        MockResponse::json_ok(json!({
            "file_id": "file-1",
            "file_path": "documents/chunked.bin"
        })),
        MockResponse::bytes_without_content_length(
            200,
            "application/octet-stream",
            b"too-large".to_vec(),
        ),
    ]);

    let error = client(&mock).get_file_bytes("file-1", 4).await.unwrap_err();

    assert!(matches!(
        error,
        TelegramError::FileTooLarge {
            size: 9,
            max_bytes: 4
        }
    ));
}

#[tokio::test]
async fn send_photo_preserves_supported_media_types_in_multipart_body() {
    let mock = MockTelegram::start(vec![
        MockResponse::json_ok(json!({ "message_id": 6 })),
        MockResponse::json_ok(json!({ "message_id": 7 })),
        MockResponse::json_ok(json!({ "message_id": 8 })),
    ]);
    let cases = [
        ("image/png", "image.png", b"PNGDATA".as_slice()),
        ("image/jpeg", "image.jpg", b"JPEGDATA".as_slice()),
        ("image/webp", "image.webp", b"WEBPDATA".as_slice()),
    ];

    for (media_type, _, bytes) in cases {
        client(&mock)
            .send_photo(42, bytes.to_vec(), media_type)
            .await
            .unwrap();
    }

    let requests = mock.requests();
    assert_eq!(requests.len(), cases.len());
    for (request, (media_type, filename, bytes)) in requests.iter().zip(cases) {
        assert_eq!(request.path, format!("/bot{TOKEN}/sendPhoto"));
        assert!(
            request
                .header("content-type")
                .unwrap()
                .starts_with("multipart/form-data; boundary=")
        );
        assert!(body_contains(
            &request.body,
            b"Content-Disposition: form-data; name=\"chat_id\""
        ));
        assert!(body_contains(
            &request.body,
            format!("Content-Disposition: form-data; name=\"photo\"; filename=\"{filename}\"")
                .as_bytes()
        ));
        assert!(body_contains(
            &request.body,
            format!("Content-Type: {media_type}").as_bytes()
        ));
        assert!(body_contains(&request.body, bytes));
    }
}

#[tokio::test]
async fn send_photo_rejects_unsupported_media_type_without_an_http_request() {
    let mock = MockTelegram::start(Vec::new());

    let error = client(&mock)
        .send_photo(42, b"GIFDATA".to_vec(), "image/gif")
        .await
        .unwrap_err();

    assert!(matches!(error, TelegramError::UnsupportedPhotoMediaType));
    assert!(mock.requests().is_empty());
}
