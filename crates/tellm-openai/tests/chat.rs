mod support;

use std::time::Duration;

use serde_json::{Value, json};
use support::{MockOpenAi, MockResponse};
use tellm_core::{ChatRequest, ContentPart, Provider, ProviderError, ThinkingLevel};
use tellm_openai::Responses;
use tokio::time::timeout;

fn request() -> ChatRequest {
    ChatRequest {
        model: "gpt-5.5".to_string(),
        system: Some("Be concise.".to_string()),
        history: vec![
            json!({
                "role": "user",
                "content": [{ "type": "input_text", "text": "Earlier question" }]
            }),
            json!({
                "type": "reasoning",
                "id": "rs_prev",
                "encrypted_content": "previous-opaque-reasoning"
            }),
            json!({
                "type": "function_call",
                "id": "fc_prev",
                "call_id": "call_prev",
                "name": "lookup",
                "arguments": "{\"q\":\"earlier\"}"
            }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Earlier answer" }]
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
                text: "Answer with an image.".to_string(),
            },
        ],
        thinking: ThinkingLevel::High,
        web_search: true,
        image_generation: true,
        max_tokens: Some(456),
    }
}

fn client(mock: &MockOpenAi) -> Responses {
    Responses::new("test-key", Some(mock.base_url().to_string()))
}

fn short_timeout_client(mock: &MockOpenAi) -> Responses {
    Responses::with_base_url_and_timeout(
        "test-key",
        Some(mock.base_url().to_string()),
        Duration::from_millis(50),
    )
}

#[tokio::test]
async fn chat_maps_request_and_preserves_output_items_verbatim() {
    let output = json!([
        {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "new-opaque-reasoning",
            "summary": []
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{\"q\":\"latest\"}"
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                { "type": "output_text", "text": "First " },
                { "type": "output_text", "text": "second." }
            ]
        },
        {
            "type": "image_generation_call",
            "id": "ig_1",
            "result": "BASE64PNG"
        }
    ]);
    let mock = MockOpenAi::start(vec![MockResponse::json(
        200,
        json!({
            "id": "resp_1",
            "object": "response",
            "output": output
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.text, "First second.");
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.images[0].media_type, "image/png");
    assert_eq!(response.images[0].base64, "BASE64PNG");
    assert_eq!(response.turn_items.len(), 5);
    assert_eq!(response.turn_items[0]["role"], "user");
    assert_eq!(response.turn_items[0]["content"][0]["type"], "input_image");
    assert_eq!(response.turn_items[0]["content"][1]["type"], "input_file");
    assert_eq!(response.turn_items[1..], output.as_array().unwrap()[..]);

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, "/responses");
    assert_eq!(requests[0].header("authorization"), Some("Bearer test-key"));

    let body = requests[0].json_body();
    assert_eq!(body["model"], "gpt-5.5");
    assert_eq!(body["instructions"], "Be concise.");
    assert_eq!(body["store"], false);
    assert_eq!(body["max_output_tokens"], 456);
    assert_eq!(body["reasoning"], json!({ "effort": "high" }));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(
        body["tools"],
        json!([
            { "type": "web_search" },
            { "type": "image_generation" }
        ])
    );

    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 5);
    assert_eq!(
        input[1]["encrypted_content"], "previous-opaque-reasoning",
        "history must be sent back verbatim"
    );
    assert_eq!(
        input[4],
        json!({
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0KGgo="
                },
                {
                    "type": "input_file",
                    "filename": "paper.pdf",
                    "file_data": "data:application/pdf;base64,JVBERi0x"
                },
                {
                    "type": "input_text",
                    "text": "Answer with an image."
                }
            ]
        })
    );
}

#[tokio::test]
async fn xai_uses_input_system_message_and_search_tools_without_instructions() {
    let mock = MockOpenAi::start(vec![MockResponse::json(
        200,
        json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "ok" }]
                }
            ]
        }),
    )]);
    let mut req = request();
    req.model = "grok-4.3".to_string();
    req.history = Vec::new();
    req.input = vec![ContentPart::Text {
        text: "Search X.".to_string(),
    }];
    req.thinking = ThinkingLevel::Max;
    req.web_search = true;
    req.image_generation = false;
    req.max_tokens = None;

    let response = client(&mock).chat(&req).await.unwrap();

    assert_eq!(response.text, "ok");
    let body = mock.requests()[0].json_body();
    assert_missing(&body, "instructions");
    assert_missing(&body, "max_output_tokens");
    // grok-4.3 has no xhigh effort: Max must clamp to high on xAI requests.
    assert_eq!(body["reasoning"], json!({ "effort": "high" }));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(
        body["tools"],
        json!([{ "type": "web_search" }, { "type": "x_search" }])
    );
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "system",
                "content": [{ "type": "input_text", "text": "Be concise." }]
            },
            {
                "role": "user",
                "content": [{ "type": "input_text", "text": "Search X." }]
            }
        ])
    );
}

#[tokio::test]
async fn meta_model_api_uses_responses_with_xhigh_search_and_no_generation() {
    let mock = MockOpenAi::start(vec![MockResponse::json(
        200,
        json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "ok" }]
                }
            ]
        }),
    )]);
    let mut req = request();
    req.model = "muse-spark-1.1".to_string();
    req.history = Vec::new();
    req.input = vec![ContentPart::Text {
        text: "Search the web.".to_string(),
    }];
    req.thinking = ThinkingLevel::Max;
    req.web_search = true;
    req.image_generation = false;
    req.max_tokens = None;

    let response = client(&mock).chat(&req).await.unwrap();

    assert_eq!(response.text, "ok");
    let body = mock.requests()[0].json_body();
    assert_eq!(body["instructions"], "Be concise.");
    assert_eq!(body["reasoning"], json!({ "effort": "xhigh" }));
    assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(body["tools"], json!([{ "type": "web_search" }]));
    assert_eq!(
        body["input"],
        json!([
            {
                "role": "user",
                "content": [{ "type": "input_text", "text": "Search the web." }]
            }
        ])
    );
}

#[tokio::test]
async fn off_reasoning_and_no_options_omit_optional_fields() {
    let mock = MockOpenAi::start(vec![MockResponse::json(
        200,
        json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "output_text", "text": "ok" }]
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
    assert_missing(&body, "instructions");
    assert_missing(&body, "max_output_tokens");
    assert_missing(&body, "reasoning");
    assert_missing(&body, "include");
    assert_missing(&body, "tools");
}

#[tokio::test]
async fn refusal_content_returns_refusal_error() {
    let mock = MockOpenAi::start(vec![MockResponse::json(
        200,
        json!({
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "refusal", "refusal": "I can't help with that." }]
                }
            ]
        }),
    )]);
    let mut req = request();
    req.image_generation = false;

    let error = client(&mock).chat(&req).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Refusal(message) if message == "I can't help with that.")
    );
}

#[tokio::test]
async fn api_error_uses_responses_error_message() {
    let mock = MockOpenAi::start(vec![MockResponse::json(
        400,
        json!({
            "error": {
                "message": "invalid encrypted_content",
                "type": "invalid_request_error"
            }
        }),
    )]);
    let mut req = request();
    req.image_generation = false;

    let error = client(&mock).chat(&req).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Api { status: 400, message } if message == "invalid encrypted_content")
    );
}

#[tokio::test]
async fn xai_image_generation_is_reported_unsupported_before_network() {
    let mut req = request();
    req.model = "grok-4.3".to_string();
    req.image_generation = true;

    let error = Responses::new("test-key", Some("http://127.0.0.1:9".to_string()))
        .chat(&req)
        .await
        .unwrap_err();

    assert!(
        matches!(error, ProviderError::Unsupported(message) if message.contains("image_generation"))
    );
}

#[tokio::test]
async fn meta_image_generation_is_reported_unsupported_before_network() {
    let mut req = request();
    req.model = "muse-spark-1.1".to_string();
    req.image_generation = true;

    let error = Responses::new("test-key", Some("http://127.0.0.1:9".to_string()))
        .chat(&req)
        .await
        .unwrap_err();

    assert!(
        matches!(error, ProviderError::Unsupported(message) if message.contains("image_generation"))
    );
}

#[tokio::test]
async fn stalled_provider_request_times_out_instead_of_hanging() {
    let mock = MockOpenAi::start_stalled();
    let mut req = request();
    req.image_generation = false;

    let error = timeout(
        Duration::from_secs(2),
        short_timeout_client(&mock).chat(&req),
    )
    .await
    .expect("client future should complete within test timeout")
    .unwrap_err();

    assert!(matches!(error, ProviderError::Http(_)), "{error}");
    assert_eq!(mock.requests()[0].path, "/responses");
}

fn assert_missing(value: &Value, key: &str) {
    assert!(
        value.get(key).is_none(),
        "expected {key} to be omitted, got {value}"
    );
}
