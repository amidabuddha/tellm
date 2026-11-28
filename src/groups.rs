//! Shared groups: every group an owner hasn't marked private. There tellm
//! answers only messages addressed to it, each on its own, and only owners
//! use commands, so a group with other people never runs a conversation on
//! the owner's keys, memory and history. See DESIGN.md § Shared groups.

use std::ops::Range;

use tellm_config::Config;
use tellm_telegram::{IncomingMessage, User};

/// Who besides the bot can read a chat, as far as tellm is told.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatKind {
    /// A direct chat with one person; always private.
    Direct,
    /// A group an owner marked private with `/group private`.
    PrivateGroup,
    /// Any other group.
    SharedGroup,
}

/// Bot API user dialog ids are the user's positive id and group ids are
/// negative (checked 2026-11-18 against core.telegram.org/api/bots/ids).
pub fn chat_kind(config: &Config, chat_id: i64) -> ChatKind {
    if chat_id > 0 {
        ChatKind::Direct
    } else if config.telegram.private_group_ids.contains(&chat_id) {
        ChatKind::PrivateGroup
    } else {
        ChatKind::SharedGroup
    }
}

/// The bot as Telegram's getMe describes it, to recognize what is meant for
/// it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bot<'a> {
    pub user_id: Option<i64>,
    pub username: Option<&'a str>,
}

/// Whether a shared-group message is meant for the bot: its text or caption
/// mentions the bot's username, or it replies to one of the bot's messages.
pub fn is_addressed(message: &IncomingMessage, bot: Bot<'_>) -> bool {
    let mentions = || {
        std::iter::once(message)
            .chain(&message.album)
            .filter_map(|item| item.text.as_deref().or(item.caption.as_deref()))
            .any(|text| !mention_spans(text, bot).is_empty())
    };
    replies_to_bot(message, bot) || mentions()
}

fn replies_to_bot(message: &IncomingMessage, bot: Bot<'_>) -> bool {
    let author = message
        .reply_to_message
        .as_ref()
        .and_then(|reply| reply.from.as_ref());
    matches!((author, bot.user_id), (Some(author), Some(bot)) if author.id == bot)
}

/// What the model gets for an addressed message: its text with the mentions
/// of the bot removed, after the text of the message it replies to, if that
/// has any. `None` when there is nothing left to send as text.
pub fn addressed_text(message: &IncomingMessage, bot: Bot<'_>) -> Option<String> {
    let own = message
        .text
        .as_deref()
        .or(message.caption.as_deref())
        .map(|text| without_mentions(text, bot))
        .filter(|text| !text.is_empty());
    let context = message.reply_to_message.as_deref().and_then(|reply| {
        let quoted = reply
            .text
            .as_deref()
            .or(reply.caption.as_deref())
            .map(str::trim)
            .filter(|text| !text.is_empty())?;
        Some(reply_context(reply.from.as_ref(), quoted, bot))
    });
    match (context, own) {
        (Some(context), Some(own)) => Some(format!("{context}\n\n{own}")),
        (Some(context), None) => Some(context),
        (None, own) => own,
    }
}

/// The message as a shared-group turn sends it: [`addressed_text`] in place
/// of its text, or of its caption when it carries a photo or file.
pub fn standalone(mut message: IncomingMessage, bot: Bot<'_>) -> IncomingMessage {
    let text = addressed_text(&message, bot);
    if message.text.is_some() {
        message.text = text;
    } else {
        message.caption = text;
    }
    message
}

/// The replied-to message, quoted, under who wrote it. Only its text or
/// caption: the model never sees that message's photo or file.
fn reply_context(author: Option<&User>, quoted: &str, bot: Bot<'_>) -> String {
    let heading = match author {
        Some(author) if Some(author.id) == bot.user_id => {
            "Replying to your earlier message:".to_string()
        }
        Some(author) => format!("Replying to {}:", author.first_name),
        None => "Replying to this message:".to_string(),
    };
    let quote = quoted
        .lines()
        .map(|line| format!("> {line}").trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    format!("{heading}\n{quote}")
}

/// The text without its mentions of the bot, each with the space after it.
fn without_mentions(text: &str, bot: Bot<'_>) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut from = 0;
    for span in mention_spans(text, bot) {
        kept.push_str(&text[from..span.start]);
        from = span.end;
        if text[from..].starts_with(' ') {
            from += 1;
        }
    }
    kept.push_str(&text[from..]);
    kept.trim().to_string()
}

/// Where the text mentions the bot: `@username`, in any letter case, as a
/// whole word. Telegram usernames are ASCII letters, digits and
/// underscores, compared without case (checked 2026-11-28 against
/// core.telegram.org/bots/api#user and telegram.org/faq#q-what-can-i-use-as-my-username).
/// Matching the text, not Telegram's entity offsets, keeps working after
/// tellm joins album captions or the parts of a split text.
fn mention_spans(text: &str, bot: Bot<'_>) -> Vec<Range<usize>> {
    let Some(username) = bot.username.filter(|username| !username.is_empty()) else {
        return Vec::new();
    };
    let bytes = text.as_bytes();
    let mut spans = Vec::new();
    for (at, _) in text.match_indices('@') {
        let start = at + 1;
        let end = start + username.len();
        let matches = bytes
            .get(start..end)
            .is_some_and(|name| name.eq_ignore_ascii_case(username.as_bytes()));
        let ends_word = bytes.get(end).is_none_or(|next| !is_username_byte(*next));
        let starts_word = text[..at]
            .chars()
            .next_back()
            .is_none_or(|before| !before.is_alphanumeric() && before != '_');
        if matches && ends_word && starts_word {
            spans.push(at..end);
        }
    }
    spans
}

fn is_username_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// The members a service message says joined that the owner should hear
/// about: anyone but the bot itself and its owners.
pub fn newcomers<'a>(message: &'a IncomingMessage, bot: Bot<'_>, owners: &[i64]) -> Vec<&'a str> {
    message
        .new_chat_members
        .iter()
        .filter(|member| Some(member.id) != bot.user_id && !owners.contains(&member.id))
        .map(|member| member.first_name.as_str())
        .collect()
}

/// Sent when someone who isn't an owner joins a group marked private.
pub fn join_notice(names: &[&str]) -> String {
    let who = match names {
        [one] => (*one).to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
        [] => "Someone".to_string(),
    };
    format!(
        "{who} joined. This group is marked private, so I answer every message here as one \
         conversation. If others will use it, an owner can send /group shared and I'll answer \
         only @mentions and replies."
    )
}

/// The `/group` reply, and the one after `/group private|shared`.
pub fn status_reply(kind: ChatKind, changed: bool) -> String {
    let now = if changed { " now" } else { "" };
    match kind {
        ChatKind::Direct => "This is a direct chat, which is always private.".to_string(),
        ChatKind::PrivateGroup => format!(
            "This group is{now} private: I answer every message here as one conversation. \
             /group shared limits me to @mentions and replies if others use it."
        ),
        ChatKind::SharedGroup => format!(
            "This group is{now} shared: I answer only messages that @mention me or reply to \
             me, each on its own, and only the bot's owners can use commands. If it's just \
             you, /group private makes it a private room."
        ),
    }
}

/// Added to the room-setup reply when a new group is approved: every group
/// starts shared.
pub const NEW_GROUP_NOTE: &str = "This group is shared: I answer only messages that @mention me \
    or reply to me, each on its own, and only the bot's owners can use commands. If it's just \
    you, send /group private.";

#[cfg(test)]
mod tests {
    use tellm_telegram::Chat;

    use super::*;

    const BOT: Bot<'static> = Bot {
        user_id: Some(99),
        username: Some("tellm_bot"),
    };

    fn user(id: i64, name: &str) -> User {
        User {
            id,
            is_bot: id == 99,
            first_name: name.to_string(),
            username: None,
        }
    }

    fn message(from: i64, text: &str) -> IncomingMessage {
        IncomingMessage {
            message_id: 1,
            chat: Chat {
                id: -5,
                title: None,
            },
            from: Some(user(from, "Ana")),
            date: 1000,
            text: Some(text.to_string()),
            caption: None,
            photo: None,
            document: None,
            media_group_id: None,
            reply_to_message: None,
            new_chat_members: Vec::new(),
            album: Vec::new(),
            split_parts: 0,
        }
    }

    fn reply(from: i64, name: &str, quoted: &str, text: &str) -> IncomingMessage {
        let mut original = message(from, quoted);
        original.from = Some(user(from, name));
        let mut reply = message(7, text);
        reply.reply_to_message = Some(Box::new(original));
        reply
    }

    #[test]
    fn groups_are_shared_unless_marked_private_and_direct_chats_never_are() {
        let config = Config {
            default_model: String::new(),
            models: Default::default(),
            telegram: tellm_config::TelegramConfig {
                private_group_ids: vec![-1],
                ..Default::default()
            },
        };
        assert_eq!(chat_kind(&config, 7), ChatKind::Direct);
        assert_eq!(chat_kind(&config, -1), ChatKind::PrivateGroup);
        assert_eq!(chat_kind(&config, -2), ChatKind::SharedGroup);
        assert_eq!(chat_kind(&config, -1_001_234), ChatKind::SharedGroup);
    }

    #[test]
    fn a_mention_is_the_bots_username_as_a_whole_word_in_any_case() {
        for text in [
            "@tellm_bot what now?",
            "what now, @TeLLm_Bot?",
            "(@tellm_bot)",
            "line one\n@tellm_bot",
        ] {
            assert!(is_addressed(&message(7, text), BOT), "{text}");
        }
        for text in [
            "what now?",
            "@tellm_bot_old what now?",
            "mail@tellm_bot",
            "@other_bot what now?",
            "tellm_bot what now?",
        ] {
            assert!(!is_addressed(&message(7, text), BOT), "{text}");
        }
        let nameless = Bot {
            user_id: Some(99),
            username: None,
        };
        assert!(!is_addressed(&message(7, "@tellm_bot hi"), nameless));
    }

    #[test]
    fn a_reply_to_the_bot_is_addressed_and_a_reply_to_a_member_is_not() {
        assert!(is_addressed(&reply(99, "tellm", "Option B.", "why?"), BOT));
        assert!(!is_addressed(&reply(8, "Ivo", "Option B.", "why?"), BOT));
        assert!(is_addressed(
            &reply(8, "Ivo", "Option B.", "@tellm_bot thoughts?"),
            BOT
        ));
    }

    #[test]
    fn an_album_is_addressed_by_a_mention_in_any_caption() {
        let mut album = message(7, "");
        album.text = None;
        let mut second = message(7, "");
        second.text = None;
        second.caption = Some("@tellm_bot what is this?".to_string());
        album.album = vec![second];
        assert!(is_addressed(&album, BOT));
    }

    #[test]
    fn the_model_gets_the_text_without_the_mention() {
        assert_eq!(
            addressed_text(&message(7, "@tellm_bot what now?"), BOT).as_deref(),
            Some("what now?")
        );
        assert_eq!(
            addressed_text(&message(7, "Привет 👋 @TELLM_BOT, what now?"), BOT).as_deref(),
            Some("Привет 👋 , what now?")
        );
        assert_eq!(addressed_text(&message(7, "@tellm_bot"), BOT), None);
    }

    #[test]
    fn a_reply_carries_the_quoted_message_under_who_wrote_it() {
        assert_eq!(
            addressed_text(
                &reply(99, "tellm", "Option B.\nIt is cheaper.", "why?"),
                BOT
            )
            .as_deref(),
            Some("Replying to your earlier message:\n> Option B.\n> It is cheaper.\n\nwhy?")
        );
        assert_eq!(
            addressed_text(&reply(8, "Ivo", "Shall we go?", "@tellm_bot"), BOT).as_deref(),
            Some("Replying to Ivo:\n> Shall we go?")
        );
        let mut photo = reply(8, "Ivo", "", "@tellm_bot what is this?");
        photo.reply_to_message.as_mut().unwrap().text.take();
        assert_eq!(
            addressed_text(&photo, BOT).as_deref(),
            Some("what is this?"),
            "a reply to a message without text adds nothing"
        );
    }

    #[test]
    fn only_newcomers_who_arent_the_bot_or_an_owner_are_announced() {
        let mut joined = message(7, "");
        joined.text = None;
        joined.new_chat_members = vec![user(99, "tellm"), user(7, "Owner"), user(8, "Ivo")];
        assert_eq!(newcomers(&joined, BOT, &[7]), vec!["Ivo"]);
        assert_eq!(
            join_notice(&["Ivo"]),
            "Ivo joined. This group is marked private, so I answer every message here as one \
             conversation. If others will use it, an owner can send /group shared and I'll \
             answer only @mentions and replies."
        );
        assert!(join_notice(&["Ivo", "Mia", "Lea"]).starts_with("Ivo, Mia and Lea joined."));
    }
}
