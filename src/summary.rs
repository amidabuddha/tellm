//! Carrying a conversation into a new one as a summary: `/new summary` asks
//! the room's model to summarize the chat, then starts a new conversation
//! whose system prompt holds the summary.
//!
//! The summary request is an ordinary turn of the conversation: same model,
//! system prompt, tools and effort, and the full history. Only the user
//! message differs, so the provider can serve everything before it from its
//! prompt cache.

/// The user message of a summary request.
pub const SUMMARY_REQUEST: &str = "Summarize our conversation so far so it can continue in a new \
    chat without the full history. Keep facts about my situation, decisions made, options \
    considered and why, open questions and agreed next steps. Write compact notes in the \
    language we have been using. Do not search the web.";

const PROMPT_HEADER: &str = "Summary of the earlier part of this conversation:";

/// Reply when the chat has no conversation to summarize.
pub const NOTHING_TO_SUMMARIZE: &str =
    "Nothing to summarize: this chat has no conversation yet. /new starts a new one.";

/// Reply in message mode, where every message stands alone.
pub const NOTHING_TO_SUMMARIZE_IN_MESSAGE_MODE: &str = "Nothing to summarize: in message mode \
    each message stands alone. /mode chat starts a conversation.";

/// Error text for a summary reply without text.
pub const EMPTY_SUMMARY_ERROR: &str = "the model returned no summary text";

/// The system-prompt block of a conversation that continues from a summary.
pub fn prompt_block(summary: &str) -> String {
    format!("{PROMPT_HEADER}\n{summary}")
}

/// What the chat gets once the summary has been carried over.
pub fn carried_reply(summary: &str) -> String {
    format!("Summary carried into a new chat:\n\n{summary}")
}

/// What the chat gets when the summary request failed. The conversation is
/// still the one it was.
pub fn failed_reply(error: &str) -> String {
    format!("Couldn't summarize this chat, so it continues unchanged.\nProvider error: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_block_puts_the_summary_under_a_header() {
        assert_eq!(
            prompt_block("- Moving to Sofia in May."),
            "Summary of the earlier part of this conversation:\n- Moving to Sofia in May."
        );
    }

    #[test]
    fn the_request_asks_for_portable_notes_without_searching() {
        assert!(SUMMARY_REQUEST.contains("continue in a new chat"));
        assert!(SUMMARY_REQUEST.contains("Do not search the web."));
        assert!(!SUMMARY_REQUEST.contains("  "), "{SUMMARY_REQUEST}");
    }

    #[test]
    fn replies_show_the_summary_or_why_there_is_none() {
        assert_eq!(
            carried_reply("Notes."),
            "Summary carried into a new chat:\n\nNotes."
        );
        let failed = failed_reply("api error 500: boom");
        assert!(failed.contains("continues unchanged"), "{failed}");
        assert!(failed.contains("api error 500: boom"), "{failed}");
    }
}
