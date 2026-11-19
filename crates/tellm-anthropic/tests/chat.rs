use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tellm_anthropic::{
    API_VERSION, Anthropic, CUT_OFF_NOTE, DEFAULT_MAX_TOKENS, MAX_TOKENS_PER_REQUEST_TIMEOUT,
    OUTPUT_LIMIT_BEFORE_TEXT, WEB_SEARCH_MAX_USES, WEB_SEARCH_TOOL_TYPE, request_timeout_for,
};
use tellm_core::{ChatRequest, ContentPart, Provider, ProviderError, ThinkingLevel};
use tellm_test_support::{MockHttpServer as MockAnthropic, MockResponse};
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
    assert_eq!(body["max_tokens"], tellm_anthropic::DEFAULT_MAX_TOKENS);
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
async fn max_tokens_stop_marks_the_reply_as_cut_off() {
    let assistant_content = json!([
        { "type": "thinking", "thinking": "", "signature": "sig-1" },
        { "type": "text", "text": "The first half of the answer" }
    ]);
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": assistant_content,
            "stop_reason": "max_tokens"
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(
        response.text,
        format!("The first half of the answer{CUT_OFF_NOTE}")
    );
    assert_eq!(
        response.turn_items[1],
        json!({ "role": "assistant", "content": assistant_content }),
        "the note is for the user; stored history keeps the model's own content"
    );
}

#[tokio::test]
async fn max_tokens_stop_before_any_text_is_an_error_not_an_empty_turn() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": [{ "type": "thinking", "thinking": "", "signature": "sig-1" }],
            "stop_reason": "max_tokens"
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Api { status: 200, message } if message == OUTPUT_LIMIT_BEFORE_TEXT)
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
async fn a_prompt_over_the_context_window_is_its_own_error() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        400,
        json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 1048577 tokens > 1048576 maximum"
            }
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(
            &error,
            ProviderError::ContextWindowExceeded { status: 400, message }
                if message.starts_with("prompt is too long")
        ),
        "{error:?}"
    );
}

#[tokio::test]
async fn other_errors_mentioning_a_long_prompt_stay_ordinary_api_errors() {
    // Only the documented 400 invalid_request_error means the context window.
    let mock = MockAnthropic::start(vec![MockResponse::json(
        529,
        json!({
            "type": "error",
            "error": { "type": "overloaded_error", "message": "prompt is too long to queue" }
        }),
    )]);

    let error = client(&mock).chat(&request()).await.unwrap_err();

    assert!(
        matches!(error, ProviderError::Api { status: 529, .. }),
        "{error:?}"
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

#[tokio::test]
async fn max_reasoning_sends_max_unless_another_tier_is_configured() {
    for (configured, expected) in [(None, "max"), (Some("xhigh"), "xhigh")] {
        let mock = MockAnthropic::start(vec![MockResponse::json(
            200,
            json!({
                "content": [{ "type": "text", "text": "ok" }],
                "stop_reason": "end_turn"
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
        assert_eq!(body["thinking"], json!({ "type": "adaptive" }));
        assert_eq!(
            body["output_config"],
            json!({ "effort": expected }),
            "configured max_effort: {configured:?}"
        );
    }
}

#[tokio::test]
async fn off_reasoning_sends_the_configured_off_effort_without_asking_for_thinking() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn"
        }),
    )]);
    let mut req = request();
    req.history = Vec::new();
    req.thinking = ThinkingLevel::Off;

    client(&mock)
        .with_off_effort(Some("low".to_string()))
        .with_max_effort(Some("xhigh".to_string()))
        .chat(&req)
        .await
        .unwrap();

    let body = mock.requests()[0].json_body();
    assert_eq!(body["output_config"], json!({ "effort": "low" }));
    assert_missing(&body, "thinking");
}

#[tokio::test]
async fn max_tokens_is_16000_unless_another_limit_is_configured() {
    for (configured, expected) in [(None, 16_000), (Some(21_000), 21_000)] {
        let mock = MockAnthropic::start(vec![MockResponse::json(
            200,
            json!({
                "content": [{ "type": "text", "text": "ok" }],
                "stop_reason": "end_turn"
            }),
        )]);

        client(&mock)
            .with_max_tokens(configured)
            .chat(&request())
            .await
            .unwrap();

        let body = mock.requests()[0].json_body();
        assert_eq!(
            body["max_tokens"], expected,
            "configured max_output_tokens: {configured:?}"
        );
    }
}

#[test]
fn request_timeout_grows_with_the_output_limit_and_never_shrinks() {
    let base = Duration::from_secs(600);

    assert_eq!(request_timeout_for(base, 4_096), base);
    assert_eq!(request_timeout_for(base, DEFAULT_MAX_TOKENS), base);
    assert_eq!(
        request_timeout_for(base, MAX_TOKENS_PER_REQUEST_TIMEOUT),
        base
    );
    let at_64k = request_timeout_for(base, 64_000);
    assert!(
        (Duration::from_secs(1800)..Duration::from_secs(1801)).contains(&at_64k),
        "{at_64k:?}"
    );
}

#[tokio::test]
async fn a_larger_output_limit_waits_longer_before_timing_out() {
    let mock = MockAnthropic::start_stalled();
    let started = Instant::now();

    let error = timeout(
        Duration::from_secs(5),
        short_timeout_client(&mock)
            .with_max_tokens(Some(64_000))
            .chat(&request()),
    )
    .await
    .expect("client future should complete within test timeout")
    .unwrap_err();

    assert!(matches!(error, ProviderError::Http(_)), "{error}");
    // The 50 ms base is tripled for a 64000-token limit.
    let waited = started.elapsed();
    assert!(waited >= Duration::from_millis(140), "{waited:?}");
}

fn assert_missing(value: &Value, key: &str) {
    assert!(
        value.get(key).is_none(),
        "expected {key} to be omitted, got {value}"
    );
}

#[tokio::test]
async fn usage_keeps_cache_buckets_apart_and_sums_pause_turn_requests() {
    let mock = MockAnthropic::start(vec![
        MockResponse::json(
            200,
            json!({
                "content": [{ "type": "text", "text": "Part one. " }],
                "stop_reason": "pause_turn",
                "usage": {
                    "input_tokens": 21,
                    "cache_creation_input_tokens": 188086,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 393,
                    "server_tool_use": { "web_search_requests": 2 }
                }
            }),
        ),
        MockResponse::json(
            200,
            json!({
                "content": [{ "type": "text", "text": "Part two." }],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 30,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 188086,
                    "output_tokens": 100
                }
            }),
        ),
    ]);

    let response = client(&mock).chat(&request()).await.unwrap();

    let usage = response.usage.expect("usage from both requests");
    assert_eq!(usage.input_tokens, 51);
    assert_eq!(usage.cache_write_tokens, 188086);
    assert_eq!(usage.cached_input_tokens, 188086);
    assert_eq!(usage.output_tokens, 493);
    assert_eq!(usage.web_searches, 2);
    assert_eq!(usage.reported_cost_usd, None);
}

#[tokio::test]
async fn a_reply_without_a_usage_block_reports_no_usage() {
    let mock = MockAnthropic::start(vec![MockResponse::json(
        200,
        json!({
            "content": [{ "type": "text", "text": "Answer." }],
            "stop_reason": "end_turn"
        }),
    )]);

    let response = client(&mock).chat(&request()).await.unwrap();

    assert_eq!(response.usage, None);
}
