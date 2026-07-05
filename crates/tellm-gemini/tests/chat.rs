mod support;

use std::time::Duration;

use serde_json::{Value, json};
use support::{MockGemini, MockResponse};
use tellm_core::{ChatRequest, ContentPart, Provider, ProviderError, ThinkingLevel};
use tellm_gemini::Gemini;
use tokio::time::timeout;

fn request() -> ChatRequest {
    ChatRequest {
        model: "gemini-3.5-flash".to_string(),
        system: Some("Be concise.".to_string()),
        history: vec![
            json!({
                "type": "user_input",
                "content": [{ "type": "text", "text": "Earlier question" }]
            }),
            json!({
                "type": "thought",
                "signature": "opaque-previous-thought"
            }),
            json!({
                "type": "google_search_result",
                "call_id": "search-1",
                "result": {
                    "search_suggestions": "previous",
                    "signature": "opaque-search-result"
                }
            }),
            json!({
                "type": "model_output",
                "content": [{ "type": "text", "text": "Earlier answer" }]
            }),
        ],
        input: vec![
            ContentPart::Image {
                media_type: "image/png".to_string(),
                base64: "iVBORw0KGgo=".to_string(),
            },
            ContentPart::Document {
                media_type: "application/pdf".to_string(),
                base64: "JVBERi0x".to_string(),
                name: Some("paper.pdf".to_string()),
            },
            ContentPart::Text {
                text: "Answer with search.".to_string(),
            },
        ],
        thinking: ThinkingLevel::High,
        web_search: true,
        image_generation: false,
        max_tokens: Some(456),
    }
}

fn client(mock: &MockGemini) -> Gemini {
    Gemini::new("test-key", Some(mock.base_url().to_string()))
}

fn short_timeout_client(mock: &MockGemini) -> Gemini {
    Gemini::with_base_url_and_timeout(
        "test-key",
        Some(mock.base_url().to_string()),
        Duration::from_millis(50),
    )
}

#[tokio::test]
async fn chat_maps_request_and_preserves_steps_verbatim() {
    let steps = json!([
        {
            "type": "thought",
            "signature": "opaque-new-thought",
            "summary": []
        },
        {
            "type": "google_search_call",
            "call_id": "search-2",
            "arguments": {
                "queries": ["latest"],
                "signature": "opaque-search-call"
            }
        },
        {
            "type": "google_search_result",
            "call_id": "search-2",
            "result": {
                "search_suggestions": "suggestion",
                "signature": "opaque-search-result-new"
            }
        },
        {
            "type": "model_output",
            "content": [
                { "type": "text", "text": "First " },
                { "type": "text", "text": "second." }
            ]
        }
    ]);
    let mock = MockGemini::start(vec![MockResponse::json(
        200,
        json!({
            "id": "interaction-1",
            "model": "gemini-3.5-flash",
            "status": "completed",
            "steps": steps
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.text, "First second.");
    assert!(response.images.is_empty());
    assert_eq!(response.turn_items.len(), 5);
    assert_eq!(response.turn_items[0]["type"], "user_input");
    assert_eq!(response.turn_items[1..], steps.as_array().unwrap()[..]);

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, "/v1beta/interactions");
    assert_eq!(requests[0].header("x-goog-api-key"), Some("test-key"));

    let body = requests[0].json_body();
    assert_eq!(body["model"], "gemini-3.5-flash");
    assert_eq!(body["system_instruction"], "Be concise.");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], false);
    assert_eq!(
        body["generation_config"],
        json!({
            "thinking_level": "high",
            "max_output_tokens": 456
        })
    );
    assert_eq!(
        body["tools"],
        json!([
            { "type": "google_search", "search_types": ["web_search"] }
        ])
    );

    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 5);
    assert_eq!(
        input[1]["signature"], "opaque-previous-thought",
        "history thought signatures must be sent back verbatim"
    );
    assert_eq!(
        input[2]["result"]["signature"], "opaque-search-result",
        "search result signatures must be sent back verbatim"
    );
    assert_eq!(
        input[4],
        json!({
            "type": "user_input",
            "content": [
                {
                    "type": "image",
                    "data": "iVBORw0KGgo=",
                    "mime_type": "image/png"
                },
                {
                    "type": "document",
                    "data": "JVBERi0x",
                    "mime_type": "application/pdf"
                },
                {
                    "type": "text",
                    "text": "Answer with search."
                }
            ]
        })
    );
}

#[tokio::test]
async fn off_reasoning_and_no_options_omit_optional_fields() {
    let mock = MockGemini::start(vec![MockResponse::json(
        200,
        json!({
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{ "type": "text", "text": "ok" }]
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
    req.web_search = false;
    req.image_generation = false;
    req.max_tokens = None;

    let response = client(&mock).chat(&req).await.unwrap();

    assert_eq!(response.text, "ok");
    let body = mock.requests()[0].json_body();
    assert_eq!(body["store"], false);
    assert_missing(&body, "system_instruction");
    assert_missing(&body, "generation_config");
    assert_missing(&body, "tools");
    assert_missing(&body, "response_format");
}

#[tokio::test]
async fn image_generation_extracts_image_content() {
    let mock = MockGemini::start(vec![MockResponse::json(
        200,
        json!({
            "status": "completed",
            "steps": [
                {
                    "type": "model_output",
                    "content": [
                        { "type": "text", "text": "created" },
                        { "type": "image", "mime_type": "image/png", "data": "BASE64PNG" }
                    ]
                }
            ]
        }),
    )]);
    let mut req = request();
    req.image_generation = true;
    req.model = "gemini-3.1-flash-image".to_string();

    let response = client(&mock).chat(&req).await.unwrap();

    assert_eq!(response.text, "created");
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.images[0].media_type, "image/png");
    assert_eq!(response.images[0].base64, "BASE64PNG");
    assert_eq!(
        mock.requests()[0].json_body()["response_format"],
        json!({ "type": "image" })
    );
}

#[tokio::test]
async fn image_generation_requires_image_model_before_network() {
    let mut req = request();
    req.image_generation = true;
    let error = Gemini::new("test-key", Some("http://127.0.0.1:9".to_string()))
        .chat(&req)
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("requires an image-capable model"),
        "{error}"
    );
}

#[tokio::test]
async fn unsupported_document_type_is_reported_before_network() {
    let mut req = request();
    req.input = vec![ContentPart::Document {
        media_type: "application/msword".to_string(),
        base64: "DOC".to_string(),
        name: Some("doc.doc".to_string()),
    }];

    let error = Gemini::new("test-key", Some("http://127.0.0.1:9".to_string()))
        .chat(&req)
        .await
        .unwrap_err();

    assert!(
        matches!(error, ProviderError::Unsupported(message) if message.contains("application/msword"))
    );
}

#[tokio::test]
async fn api_error_uses_gemini_error_message() {
    let mock = MockGemini::start(vec![MockResponse::json(
        400,
        json!({
            "error": {
                "code": 400,
                "message": "invalid thought signature",
                "status": "INVALID_ARGUMENT"
            }
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Api { status: 400, message } if message == "invalid thought signature")
    );
}

#[tokio::test]
async fn non_completed_status_is_reported_unsupported() {
    let mock = MockGemini::start(vec![MockResponse::json(
        200,
        json!({
            "status": "requires_action",
            "steps": []
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Unsupported(message) if message.contains("requires_action"))
    );
}

#[tokio::test]
async fn stalled_provider_request_times_out_instead_of_hanging() {
    let mock = MockGemini::start_stalled();

    let error = timeout(
        Duration::from_secs(2),
        short_timeout_client(&mock).chat(&request()),
    )
    .await
    .expect("client future should complete within test timeout")
    .unwrap_err();

    assert!(matches!(error, ProviderError::Http(_)), "{error}");
    assert_eq!(mock.requests()[0].path, "/v1beta/interactions");
}

fn assert_missing(value: &Value, key: &str) {
    assert!(
        value.get(key).is_none(),
        "expected {key} to be absent from {value}"
    );
}
