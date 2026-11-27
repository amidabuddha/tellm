use std::time::Duration;

use serde_json::{Value, json};
use tellm_compat::Compat;
use tellm_core::{
    CUT_OFF_NOTE, ChatRequest, ContentPart, OUTPUT_LIMIT_BEFORE_TEXT, Provider, ProviderError,
    ThinkingLevel,
};
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
async fn length_finish_marks_the_reply_as_cut_off() {
    let message = json!({ "role": "assistant", "content": "The first half" });
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({ "choices": [{ "message": message, "finish_reason": "length" }] }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.text, format!("The first half{CUT_OFF_NOTE}"));
    assert_eq!(response.turn_items[1], message);
}

#[tokio::test]
async fn length_finish_before_any_text_is_an_error() {
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "choices": [{
                "message": { "role": "assistant", "content": "", "reasoning": "Thinking..." },
                "finish_reason": "length"
            }]
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(&error, ProviderError::Api { status: 200, message } if message == OUTPUT_LIMIT_BEFORE_TEXT),
        "{error:?}"
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

#[tokio::test]
async fn max_reasoning_sends_high_unless_another_tier_is_configured() {
    for (configured, expected) in [(None, "high"), (Some("max"), "max")] {
        let mock = MockCompat::start(vec![MockResponse::json(
            200,
            json!({
                "choices": [
                    {
                        "message": { "role": "assistant", "content": "ok" },
                        "finish_reason": "stop"
                    }
                ]
            }),
        )]);
        let mut req = request();
        req.history = Vec::new();
        req.thinking = ThinkingLevel::Max;

        client(&mock)
            .with_max_effort(configured.map(str::to_string))
            .chat(&req)
            .await
            .unwrap();

        let body = mock.requests()[0].json_body();
        assert_eq!(
            body["reasoning_effort"], expected,
            "configured max_effort: {configured:?}"
        );
    }
}

#[tokio::test]
async fn off_reasoning_sends_the_configured_off_effort() {
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "choices": [
                {
                    "message": { "role": "assistant", "content": "ok" },
                    "finish_reason": "stop"
                }
            ]
        }),
    )]);
    let mut req = request();
    req.history = Vec::new();
    req.thinking = ThinkingLevel::Off;

    client(&mock)
        .with_off_effort(Some("none".to_string()))
        .with_max_effort(Some("max".to_string()))
        .chat(&req)
        .await
        .unwrap();

    let body = mock.requests()[0].json_body();
    assert_eq!(body["reasoning_effort"], "none");
}

fn assert_missing(value: &Value, key: &str) {
    assert!(
        value.get(key).is_none(),
        "expected {key} to be omitted, got {value}"
    );
}

#[tokio::test]
async fn usage_separates_cached_input_and_ignores_a_cost_from_other_endpoints() {
    let mock = MockCompat::start(vec![MockResponse::json(
        200,
        json!({
            "choices": [{
                "message": { "role": "assistant", "content": "A cat." },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1500,
                "prompt_tokens_details": { "cached_tokens": 1200 },
                "completion_tokens": 500,
                "completion_tokens_details": { "reasoning_tokens": 300 },
                "total_tokens": 2000,
                "cost": 0.5
            }
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 300);
    assert_eq!(usage.cached_input_tokens, 1200);
    assert_eq!(usage.output_tokens, 500);
    assert_eq!(usage.reasoning_tokens, 300);
    assert_eq!(
        usage.reported_cost_usd, None,
        "only OpenRouter's usage.cost is known to be US dollars"
    );
}

#[test]
fn usage_reads_deepseek_cache_hits_and_the_openrouter_cost() {
    let deepseek = json!({
        "usage": {
            "prompt_tokens": 18234,
            "prompt_cache_hit_tokens": 16000,
            "prompt_cache_miss_tokens": 2234,
            "completion_tokens": 412
        }
    });
    let usage = tellm_compat::usage_from_response(&deepseek, false).unwrap();
    assert_eq!(usage.input_tokens, 2234);
    assert_eq!(usage.cached_input_tokens, 16000);
    assert_eq!(usage.output_tokens, 412);

    let openrouter = json!({
        "usage": {
            "prompt_tokens": 1000,
            "prompt_tokens_details": { "cached_tokens": 200, "cache_write_tokens": 300 },
            "completion_tokens": 50,
            "cost": 0.0123
        }
    });
    let usage = tellm_compat::usage_from_response(&openrouter, true).unwrap();
    assert_eq!(usage.input_tokens, 500);
    assert_eq!(usage.cache_write_tokens, 300);
    assert_eq!(usage.reported_cost_usd, Some(0.0123));

    assert!(tellm_compat::is_openrouter_url(
        "https://openrouter.ai/api/v1/chat/completions"
    ));
    assert!(!tellm_compat::is_openrouter_url(
        "https://openrouter.ai.example.com/v1/chat/completions"
    ));
    assert!(!tellm_compat::is_openrouter_url(
        "http://localhost:11434/v1/chat/completions"
    ));
}
