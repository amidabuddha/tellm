mod support;

use std::time::Duration;

use serde_json::{Value, json};
use support::{MockAnthropic, MockResponse};
use tellm_anthropic::{API_VERSION, Anthropic, WEB_SEARCH_MAX_USES, WEB_SEARCH_TOOL_TYPE};
use tellm_core::{ChatRequest, ContentPart, Provider, ProviderError, ThinkingLevel};
use tokio::time::timeout;

fn request() -> ChatRequest {
    ChatRequest {
        model: "claude-opus-4-8".to_string(),
        system: Some("Be precise.".to_string()),
        history: vec![
            json!({
                "role": "user",
                "content": [{ "type": "text", "text": "Earlier question" }]
            }),
            json!({
                "role": "assistant",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Source",
                                "url": "https://example.com",
                                "encrypted_content": "opaque-ciphertext"
                            }
                        ]
                    },
                    { "type": "text", "text": "Earlier answer" }
                ]
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
                text: "Summarize this.".to_string(),
            },
        ],
        thinking: ThinkingLevel::High,
        web_search: true,
        image_generation: false,
        max_tokens: Some(1234),
    }
}

fn client(mock: &MockAnthropic) -> Anthropic {
    Anthropic::new("test-key", Some(mock.base_url().to_string()))
}

fn short_timeout_client(mock: &MockAnthropic) -> Anthropic {
    Anthropic::with_base_url_and_timeout(
        "test-key",
        Some(mock.base_url().to_string()),
        Duration::from_millis(50),
    )
}

#[tokio::test]
async fn chat_maps_request_and_preserves_turn_items_verbatim() {
    let assistant_content = json!([
        { "type": "text", "text": "First " },
        {
            "type": "server_tool_use",
            "id": "srvtoolu_2",
            "name": "web_search",
            "input": { "query": "latest" }
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_2",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com/latest",
                    "title": "Latest",
                    "encrypted_content": "must-stay-verbatim",
                    "page_age": "July 4, 2026"
                }
            ]
        },
        { "type": "text", "text": "second." }
    ]);
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": assistant_content,
            "model": "claude-opus-4-8",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": { "input_tokens": 10, "output_tokens": 20 }
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.text, "First second.");
    assert!(response.images.is_empty());
    assert_eq!(response.turn_items.len(), 2);
    assert_eq!(response.turn_items[0]["role"], "user");
    assert_eq!(response.turn_items[0]["content"][0]["type"], "image");
    assert_eq!(response.turn_items[0]["content"][1]["type"], "document");
    assert_eq!(
        response.turn_items[0]["content"][2]["text"],
        "Summarize this."
    );
    assert_eq!(
        response.turn_items[1],
        json!({ "role": "assistant", "content": assistant_content })
    );

    let requests = mock.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, "/v1/messages");
    assert_eq!(requests[0].header("x-api-key"), Some("test-key"));
    assert_eq!(requests[0].header("anthropic-version"), Some(API_VERSION));

    let body = requests[0].json_body();
    assert_eq!(body["model"], "claude-opus-4-8");
    assert_eq!(body["max_tokens"], 1234);
    assert_eq!(
        body["system"],
        json!([
            {
                "type": "text",
                "text": "Be precise.",
                "cache_control": { "type": "ephemeral" }
            }
        ])
    );
    assert_eq!(
        body["thinking"],
        json!({ "type": "adaptive" }),
        "adaptive thinking should not use budget_tokens"
    );
    assert_eq!(body["output_config"], json!({ "effort": "high" }));
    assert_eq!(
        body["tools"],
        json!([
            {
                "type": WEB_SEARCH_TOOL_TYPE,
                "name": "web_search",
                "allowed_callers": ["direct"],
                "max_uses": WEB_SEARCH_MAX_USES
            }
        ])
    );

    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(
        messages[1]["content"][0]["content"][0]["encrypted_content"], "opaque-ciphertext",
        "history must be sent back verbatim"
    );
    assert_eq!(
        messages[2],
        json!({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgo="
                    }
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "JVBERi0x"
                    }
                },
                { "type": "text", "text": "Summarize this." }
            ]
        })
    );
}

#[tokio::test]
async fn pause_turn_continues_with_paused_assistant_message() {
    let paused_assistant_content = json!([
        {
            "type": "server_tool_use",
            "id": "srvtoolu_pause",
            "name": "web_search",
            "input": { "query": "long search" }
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_pause",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com/search",
                    "title": "Search",
                    "encrypted_content": "pause-turn-ciphertext"
                }
            ]
        }
    ]);
    let final_assistant_content = json!([
        { "type": "text", "text": "Final answer." }
    ]);
    let mock = MockAnthropic::start(vec![
        MockResponse::json(
            200,
            json!({
                "content": paused_assistant_content,
                "stop_reason": "pause_turn"
            }),
        ),
        MockResponse::json(
            200,
            json!({
                "content": final_assistant_content,
                "stop_reason": "end_turn"
            }),
        ),
    ]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.text, "Final answer.");
    assert_eq!(response.turn_items.len(), 3);
    assert_eq!(
        response.turn_items[1],
        json!({ "role": "assistant", "content": paused_assistant_content })
    );
    assert_eq!(
        response.turn_items[2],
        json!({ "role": "assistant", "content": final_assistant_content })
    );

    let requests = mock.requests();
    assert_eq!(requests.len(), 2);
    let second_body = requests[1].json_body();
    let second_messages = second_body["messages"].as_array().unwrap();
    assert_eq!(second_messages.len(), 4);
    assert_eq!(
        second_messages[3],
        json!({ "role": "assistant", "content": paused_assistant_content }),
        "pause_turn continuation must resend the paused assistant message"
    );
}

#[tokio::test]
async fn off_reasoning_and_no_system_or_search_omit_optional_fields() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn"
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
    req.max_tokens = None;

    let response = client(&mock).chat(&req).await.unwrap();

    assert_eq!(response.text, "ok");
    let body = mock.requests()[0].json_body();
    assert_eq!(body["max_tokens"], tellm_anthropic::DEFAULT_MAX_TOKENS);
    assert_missing(&body, "system");
    assert_missing(&body, "thinking");
    assert_missing(&body, "output_config");
    assert_missing(&body, "tools");
}

#[tokio::test]
async fn refusal_stop_reason_returns_refusal_error() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": [{ "type": "text", "text": "I can't help with that." }],
            "stop_reason": "refusal"
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Refusal(message) if message == "I can't help with that.")
    );
}

#[tokio::test]
async fn pre_output_refusal_uses_stop_details_message() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": [],
            "stop_reason": "refusal",
            "stop_details": {
                "category": "safety",
                "explanation": "The request was declined by safety policy."
            }
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Refusal(message) if message == "The request was declined by safety policy.")
    );
}

#[tokio::test]
async fn api_error_uses_anthropic_error_message() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        400,
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "encrypted_content is invalid"
            }
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Api { status: 400, message } if message == "encrypted_content is invalid")
    );
}

#[tokio::test]
async fn image_generation_is_reported_unsupported_before_network() {
    let mut req = request();
    req.image_generation = true;

    let error = Anthropic::new("test-key", Some("http://127.0.0.1:9".to_string()))
        .chat(&req)
        .await
        .unwrap_err();

    assert!(
        matches!(error, ProviderError::Unsupported(message) if message.contains("image generation"))
    );
}

#[tokio::test]
async fn stalled_provider_request_times_out_instead_of_hanging() {
    let mock = MockAnthropic::start_stalled();

    let error = timeout(
        Duration::from_secs(2),
        short_timeout_client(&mock).chat(&request()),
    )
    .await
    .expect("client future should complete within test timeout")
    .unwrap_err();

    assert!(matches!(error, ProviderError::Http(_)), "{error}");
    assert_eq!(mock.requests()[0].path, "/v1/messages");
}

fn assert_missing(value: &Value, key: &str) {
    assert!(
        value.get(key).is_none(),
        "expected {key} to be omitted, got {value}"
    );
}
