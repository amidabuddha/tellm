use std::time::Duration;

use serde_json::{Value, json};
use tellm_compat::Compat;
use tellm_core::{ChatRequest, ContentPart, Provider, ProviderError, ThinkingLevel};
use tellm_test_support::{MockHttpServer as MockCompat, MockResponse};
use tokio::time::timeout;

fn request() -> ChatRequest {
    ChatRequest {
        model: "llama3.2".to_string(),
        system: Some("Be concise.".to_string()),
        history: vec![
            json!({ "role": "user", "content": "Earlier question" }),
            json!({
                "role": "assistant",
                "content": "Earlier answer",
                "tool_calls": [
                    {
                        "id": "call_prev",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"q\":\"earlier\"}" }
                    }
                ]
            }),
        ],
        input: vec![
            ContentPart::Image {
                media_type: "image/png".to_string(),
                base64: "iVBORw0KGgo=".to_string(),
            },
            ContentPart::Text {
                text: "What is in this image?".to_string(),
            },
        ],
        thinking: ThinkingLevel::High,
        web_search: false,
        image_generation: false,
    }
}

fn client(mock: &MockCompat) -> Compat {
    Compat::new("test-key", mock.base_url())
}

fn short_timeout_client(mock: &MockCompat) -> Compat {
    Compat::with_base_url_and_timeout("test-key", mock.base_url(), Duration::from_millis(50))
}

#[tokio::test]
async fn chat_maps_request_and_preserves_assistant_message_verbatim() {
    let assistant_message = json!({
        "role": "assistant",
        "content": "The image shows a chart.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": { "name": "describe", "arguments": "{}" }
            }
        ]
    });
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "id": "chatcmpl_1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": assistant_message
                }
            ]
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.text, "The image shows a chart.");
    assert!(response.images.is_empty());
    assert_eq!(response.turn_items.len(), 2);
    assert_eq!(response.turn_items[0]["role"], "user");
    assert_eq!(response.turn_items[0]["content"][0]["type"], "image_url");
    assert_eq!(response.turn_items[1], assistant_message);

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, "/chat/completions");
    assert_eq!(requests[0].header("authorization"), Some("Bearer test-key"));

    let body = requests[0].json_body();
    assert_eq!(body["model"], "llama3.2");
    assert_eq!(body["stream"], false);
    assert_missing(&body, "max_tokens");
    assert_eq!(body["reasoning_effort"], "high");
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[0],
        json!({ "role": "system", "content": "Be concise." })
    );
    assert_eq!(
        messages[2]["tool_calls"][0]["function"]["arguments"], "{\"q\":\"earlier\"}",
        "history must be sent back verbatim"
    );
    assert_eq!(
        messages[3],
        json!({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": { "url": "data:image/png;base64,iVBORw0KGgo=" }
                },
                { "type": "text", "text": "What is in this image?" }
            ]
        })
    );
}

#[tokio::test]
async fn text_only_input_uses_plain_string_content_and_omits_optional_fields() {
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            { "type": "text", "text": "hello" },
                            { "type": "text", "text": " there" }
                        ]
                    }
                }
            ]
        }),
    )]);
    let mut req = request();
    req.system = None;
    req.history = Vec::new();
    req.input = vec![ContentPart::Text {
        text: "Hi".to_string(),
    }];
    req.thinking = ThinkingLevel::Off;

    let response = client(&mock).chat(&req).await.unwrap();

    assert_eq!(response.text, "hello there");
    let body = mock.requests()[0].json_body();
    assert_eq!(
        body["messages"],
        json!([{ "role": "user", "content": "Hi" }])
    );
    assert_missing(&body, "reasoning_effort");
    assert_missing(&body, "max_tokens");
}

#[tokio::test]
async fn keyless_client_omits_authorization_header_entirely() {
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "choices": [{ "message": { "role": "assistant", "content": "local" } }]
        }),
    )]);

    let response = Compat::new("   ", mock.base_url())
        .chat(&request())
        .await
        .unwrap();

    assert_eq!(response.text, "local");
    assert_eq!(mock.requests()[0].header("authorization"), None);
}

#[tokio::test]
async fn unsupported_features_are_reported_before_network() {
    let client = Compat::new("test-key", "http://127.0.0.1:9");

    let mut req = request();
    req.web_search = true;
    let error = client.chat(&req).await.unwrap_err();
    assert!(matches!(error, ProviderError::Unsupported(message) if message.contains("web search")));

    let mut req = request();
    req.image_generation = true;
    let error = client.chat(&req).await.unwrap_err();
    assert!(
        matches!(error, ProviderError::Unsupported(message) if message.contains("image generation"))
    );

    let mut req = request();
    req.input.push(ContentPart::Document {
        media_type: "application/pdf".to_string(),
        base64: "JVBERi0x".to_string(),
        name: Some("paper.pdf".to_string()),
    });
    let error = client.chat(&req).await.unwrap_err();
    assert!(matches!(error, ProviderError::Unsupported(message) if message.contains("documents")));
}

#[tokio::test]
async fn refusal_field_returns_refusal_error() {
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "refusal": "I can't help with that."
                    }
                }
            ]
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Refusal(message) if message == "I can't help with that.")
    );
}

#[tokio::test]
async fn api_error_uses_provider_message() {
    let mock = MockCompat::start(vec![MockResponse::json(
        400,
        json!({
            "error": {
                "message": "model not found",
                "type": "invalid_request_error"
            }
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Api { status: 400, message } if message == "model not found")
    );
}

#[tokio::test]
async fn stalled_provider_request_times_out_instead_of_hanging() {
    let mock = MockCompat::start_stalled();

    let error = timeout(
        Duration::from_secs(2),
        short_timeout_client(&mock).chat(&request()),
    )
    .await
    .expect("client future should complete within test timeout")
    .unwrap_err();

    assert!(matches!(error, ProviderError::Http(_)), "{error}");
    assert_eq!(mock.requests()[0].path, "/chat/completions");
}

fn assert_missing(value: &Value, key: &str) {
    assert!(
        value.get(key).is_none(),
        "expected {key} to be omitted, got {value}"
    );
}
