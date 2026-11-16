use serde_json::{Value, json};
use tellm_anthropic::without_thinking_blocks;

fn user(text: &str) -> Value {
    json!({ "role": "user", "content": [{ "type": "text", "text": text }] })
}

fn assistant(content: Value) -> Value {
    json!({ "role": "assistant", "content": content })
}

fn thinking(text: &str) -> Value {
    json!({ "type": "thinking", "thinking": text, "signature": "opaque-signature" })
}

fn text(text: &str) -> Value {
    json!({ "type": "text", "text": text })
}

#[test]
fn removes_every_thinking_block_and_keeps_the_rest_verbatim() {
    let search_use = json!({
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "web_search",
        "input": { "query": "latest" }
    });
    let search_result = json!({
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": [
            {
                "type": "web_search_result",
                "url": "https://example.com/latest",
                "title": "Latest",
                "encrypted_content": "must-stay-verbatim"
            }
        ]
    });
    let turn = vec![
        user("What changed?"),
        assistant(json!([
            thinking("Plan the search."),
            search_use.clone(),
            search_result.clone(),
            { "type": "redacted_thinking", "data": "opaque-ciphertext" },
            thinking(""),
            text("Here is what changed.")
        ])),
    ];

    let rewritten = without_thinking_blocks(turn);

    assert_eq!(
        rewritten,
        vec![
            user("What changed?"),
            assistant(json!([
                search_use,
                search_result,
                text("Here is what changed.")
            ])),
        ],
        "thinking goes as a whole; search state must stay verbatim"
    );
}

#[test]
fn turn_without_thinking_is_returned_unchanged() {
    let turn = vec![user("Hi"), assistant(json!([text("Hello.")]))];

    assert_eq!(without_thinking_blocks(turn.clone()), turn);
}

#[test]
fn paused_assistant_message_with_only_thinking_is_removed_from_the_turn() {
    let turn = vec![
        user("Search for it."),
        assistant(json!([thinking("Still working.")])),
        assistant(json!([thinking("Done."), text("Final answer.")])),
    ];

    assert_eq!(
        without_thinking_blocks(turn),
        vec![
            user("Search for it."),
            assistant(json!([text("Final answer.")])),
        ]
    );
}

#[test]
fn turn_whose_reply_was_only_thinking_is_dropped_whole() {
    let turn = vec![
        user("Hard question."),
        assistant(json!([thinking("Cut off mid-thought.")])),
    ];

    assert!(
        without_thinking_blocks(turn).is_empty(),
        "a question must not be kept without its answer"
    );
}

#[test]
fn items_of_unknown_shape_pass_through() {
    let turn = vec![
        json!({ "turn": 1, "marker": "opaque" }),
        json!({ "role": "assistant", "content": "plain string content" }),
        assistant(json!([])),
    ];

    assert_eq!(without_thinking_blocks(turn.clone()), turn);
}
