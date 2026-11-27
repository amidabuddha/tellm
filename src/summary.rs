//! Carrying a conversation into a new one as a summary: `/new summary`, or
//! tellm on its own once a chat reaches its token budget, asks the room's
//! model to summarize the chat, then starts a new conversation whose system
//! prompt holds the summary.
//!
//! The summary request is an ordinary turn of the conversation: same model,
//! system prompt, tools and effort, and the full history. Only the user
//! message differs, so the provider can serve everything before it from its
//! prompt cache.

use crate::cost::format_tokens;
use crate::rooms::MAX_HISTORY_BYTES;

/// A conversation's token budget, the same for every provider: once a reply
/// shows the chat's context at this size, tellm summarizes the chat into a
/// new one right after delivering that reply, while the provider's cache
/// still holds it. Chosen from what the providers document (checked
/// 2026-11-19):
/// - Anthropic's server-side compaction starts at 150,000 input tokens by
///   default (platform.claude.com, compaction at a token threshold).
/// - OpenAI's compaction examples use `compact_threshold: 200000`
///   (developers.openai.com, compaction).
/// - The lowest long-context price step among the presets is Grok 4.7's:
///   a prompt of 200K tokens or more bills the whole request at the higher
///   rate (docs.x.ai, Grok 4.7). GPT-6.2 Astra's starts above 272K input
///   tokens (developers.openai.com, GPT-6.2 Astra), and Claude Fable 5.2 has
///   no long-prompt premium (platform.claude.com, pricing).
/// - Context windows: Gemini 4 Carbon up to 1M tokens (deepmind.google model
///   card), Muse Wonder 1,048,576 (dev.meta.ai, models), Grok 4.7 500,000.
///
/// 150,000 matches the one documented compaction default, leaves room for
/// the next message below every preset's first price step, and is far
/// inside every preset's window.
pub const CONTEXT_SUMMARY_TOKENS: u64 = 150_000;

/// Failed automatic summaries after which a conversation is no longer
/// summarized on its own. Each attempt resends the whole conversation, so a
/// provider that keeps failing would otherwise bill a full context after
/// every reply. Two allows one retry for a passing outage.
pub const MAX_AUTOMATIC_SUMMARY_FAILURES: u8 = 2;

/// Why a conversation is being summarized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
    /// The user sent `/new summary`.
    Command,
    /// The chat's context reached [`CONTEXT_SUMMARY_TOKENS`].
    TokenBudget,
    /// The stored history passed [`MAX_HISTORY_BYTES`].
    HistoryBytes,
}

impl Trigger {
    /// What the chat is told before tellm summarizes it on its own.
    pub fn notice(self) -> Option<String> {
        let reached = match self {
            Self::Command => return None,
            Self::TokenBudget => format!(
                "This chat reached {} tokens",
                format_tokens(CONTEXT_SUMMARY_TOKENS)
            ),
            Self::HistoryBytes => format!(
                "This chat's history reached {} MiB",
                MAX_HISTORY_BYTES / (1024 * 1024)
            ),
        };
        Some(format!(
            "{reached}, so I'm summarizing it and continuing in a fresh chat."
        ))
    }
}

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

/// How many turns the next summary request sends after the provider said
/// `sent` turns are over the model's context window: the older half is left
/// out. `None` once a single turn was too much, since nothing is left to cut.
pub fn fewer_turns(sent: usize) -> Option<usize> {
    (sent > 1).then(|| sent - sent / 2)
}

/// What the chat gets once the summary has been carried over. `left_out`
/// counts the oldest turns that had to stay out of the summary request.
pub fn carried_reply(summary: &str, left_out: usize) -> String {
    if left_out == 0 {
        return format!("Summary carried into a new chat:\n\n{summary}");
    }
    format!(
        "Summary carried into a new chat. The oldest {} didn't fit the model's context window, \
         so the summary leaves them out:\n\n{summary}",
        turns(left_out)
    )
}

/// What the chat gets when the summary request failed. The conversation is
/// still the one it was; an automatic summary is tried again after the next
/// reply.
/// `will_retry` says whether an automatic summary is tried again after the
/// next reply; it is ignored for `/new summary`.
pub fn failed_reply(trigger: Trigger, error: &str, will_retry: bool) -> String {
    let next = match trigger {
        Trigger::Command => "so it continues unchanged",
        Trigger::TokenBudget | Trigger::HistoryBytes if will_retry => {
            "so it continues and I'll try again after the next reply"
        }
        Trigger::TokenBudget | Trigger::HistoryBytes => {
            "so it continues unchanged. I won't try again on my own: /new summary tries now, /new starts a fresh chat"
        }
    };
    format!("Couldn't summarize this chat, {next}.\nProvider error: {error}")
}

/// What the chat gets when the last-resort pruning dropped turns.
pub fn pruned_notice(dropped: usize) -> String {
    format!(
        "To keep this chat's history under {} MiB, I dropped its oldest {}; the model no longer sees them.",
        MAX_HISTORY_BYTES / (1024 * 1024),
        turns(dropped)
    )
}

fn turns(count: usize) -> String {
    match count {
        1 => "turn".to_string(),
        count => format!("{count} turns"),
    }
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
            carried_reply("Notes.", 0),
            "Summary carried into a new chat:\n\nNotes."
        );
        assert_eq!(
            carried_reply("Notes.", 3),
            "Summary carried into a new chat. The oldest 3 turns didn't fit the model's \
             context window, so the summary leaves them out:\n\nNotes."
        );
        let failed = failed_reply(Trigger::Command, "api error 500: boom", true);
        assert!(failed.contains("continues unchanged"), "{failed}");
        assert!(failed.contains("api error 500: boom"), "{failed}");
        let automatic = failed_reply(Trigger::TokenBudget, "boom", true);
        assert!(
            automatic.contains("try again after the next reply"),
            "{automatic}"
        );
        let last = failed_reply(Trigger::HistoryBytes, "boom", false);
        assert!(last.contains("won't try again"), "{last}");
        assert!(last.contains("/new summary"), "{last}");
        assert_eq!(
            pruned_notice(1),
            "To keep this chat's history under 4 MiB, I dropped its oldest turn; the model no longer sees them."
        );
        assert!(pruned_notice(2).contains("its oldest 2 turns;"));
    }

    #[test]
    fn automatic_summaries_say_why_before_they_start() {
        assert_eq!(Trigger::Command.notice(), None);
        assert_eq!(
            Trigger::TokenBudget.notice().as_deref(),
            Some(
                "This chat reached 150k tokens, so I'm summarizing it and continuing in a fresh chat."
            )
        );
        assert_eq!(
            Trigger::HistoryBytes.notice().as_deref(),
            Some(
                "This chat's history reached 4 MiB, so I'm summarizing it and continuing in a fresh chat."
            )
        );
    }

    #[test]
    fn each_retry_leaves_the_older_half_out_until_one_turn_is_left() {
        let mut sent = 5;
        let mut sequence = vec![sent];
        while let Some(next) = fewer_turns(sent) {
            sent = next;
            sequence.push(sent);
        }
        assert_eq!(sequence, vec![5, 3, 2, 1]);
        assert_eq!(fewer_turns(2), Some(1));
        assert_eq!(fewer_turns(1), None);
        assert_eq!(fewer_turns(0), None);
    }
}
