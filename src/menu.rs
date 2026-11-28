//! The commands a chat's members can use there. One list feeds both the
//! Telegram command menu (the "/" button), set for each chat and for each
//! owner in a group, and `/help`, so the two never disagree. A command left
//! out still works when typed; the list only shows what applies.

use crate::groups::ChatKind;
use crate::rooms::ChatMode;

/// What decides the list: the room and who is looking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    pub kind: ChatKind,
    /// The viewer is one of the bot's owners.
    pub owner: bool,
    /// The model the room is pinned to, if any.
    pub pinned: Option<String>,
    pub mode: ChatMode,
    /// The room's new conversations get the owner's memory.
    pub memory: bool,
    /// The room's model can search the web.
    pub web_search: bool,
    /// The room's model can generate images.
    pub image_generation: bool,
    /// The room's model runs on the local Ollama that tellm manages.
    pub ollama: bool,
}

/// One menu entry and its `/help` lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Command {
    /// The name, without the slash.
    pub name: &'static str,
    /// The menu's short description.
    pub menu: &'static str,
    /// The `/help` lines: usage, then what it does.
    pub help: Vec<&'static str>,
}

/// The commands that apply in the room for the viewer, in menu order.
///
/// - Other members of a shared group get only `/id` and `/help`; everything
///   else there is an owner's.
/// - A shared group has no conversation, so no `/new`, `/mode` or memory.
/// - Web search, image generation and `/ollama unload` show only when the
///   room's model has them; `/remember` and `/forget` only where memory is
///   on, and `/new summary` only in chat mode.
pub fn commands(ctx: &Context) -> Vec<Command> {
    let shared = ctx.kind == ChatKind::SharedGroup;
    let group = ctx.kind != ChatKind::Direct;
    let mut list = Vec::new();
    let mut add = |name, menu, help: &[&'static str]| {
        list.push(Command {
            name,
            menu,
            help: help.to_vec(),
        });
    };

    if shared && !ctx.owner {
        add(
            "id",
            "Show this chat's id",
            &["/id - show this Telegram chat id"],
        );
        add(
            "help",
            "Show what each command does",
            &["/help - show commands"],
        );
        return list;
    }

    if !shared {
        if ctx.mode == ChatMode::Chat {
            add(
                "new",
                "Start a new chat, or carry a summary of this one into it",
                &["/new [summary] - start a new chat, or carry a summary of this one into it"],
            );
        } else {
            add("new", "Start a new chat", &["/new - start a new chat"]);
        }
    }
    add(
        "id",
        "Show this chat's id",
        &["/id - show this Telegram chat id"],
    );
    if !shared {
        add(
            "mode",
            "Show or set chat or message mode",
            &["/mode chat|message - show or set conversation mode"],
        );
    }
    match (ctx.pinned.is_some(), ctx.owner) {
        (true, true) => add(
            "model",
            "Show the pinned model, unpin it, or add a provider",
            &[
                "/model - show this room's pinned model",
                "/model unpin - release the pin",
                "/model add [KEY] - list the provider catalog or add a preset",
            ],
        ),
        (true, false) => add(
            "model",
            "Show this room's model",
            &["/model - show this room's pinned model"],
        ),
        (false, true) => add(
            "model",
            "Show or switch the model, pin it, or add a provider",
            &[
                "/model [KEY] - show or switch the model",
                "/model pin KEY | /model unpin - lock or release this room's model",
                "/model add [KEY] - list the provider catalog or add a preset",
            ],
        ),
        (false, false) => add(
            "model",
            "Show or switch the model",
            &["/model [KEY] - show or switch the model"],
        ),
    }
    add(
        "role",
        "Show, set or clear the system role",
        &["/role TEXT|clear - show, set, or clear the system role"],
    );
    add(
        "reasoning",
        "Show or set the reasoning level",
        &["/reasoning default|off|low|medium|high|max - show or set reasoning level"],
    );
    if ctx.web_search {
        add(
            "websearch",
            "Turn web search on or off",
            &["/websearch on|off|status - toggle, set, or show web search"],
        );
    }
    if ctx.image_generation {
        add(
            "imagegen",
            "Turn image generation on or off",
            &["/imagegen on|off|status - toggle, set, or show image generation"],
        );
    }
    add(
        "cost",
        "Show this chat's cost, or the cost line on or off",
        &["/cost on|off - show this chat's cost, or show or hide the cost line under each reply"],
    );
    if ctx.owner && !shared {
        add(
            "memory",
            "List saved facts, or use them here or not",
            &["/memory on|off - list your saved facts, or use them in this chat or not"],
        );
        if ctx.memory {
            add(
                "remember",
                "Save a fact for new chats",
                &["/remember TEXT - save a fact for new chats"],
            );
            add(
                "forget",
                "Remove a saved fact",
                &["/forget N - remove saved fact N"],
            );
        }
    }
    if ctx.owner && ctx.ollama {
        add(
            "ollama",
            "Unload the local Ollama models",
            &["/ollama unload - unload local Ollama models used by this session"],
        );
    }
    if group && ctx.owner {
        add(
            "group",
            "Show or set who uses this group",
            &["/group private|shared - show or set whether this group is a private room or shared"],
        );
    } else if group {
        add(
            "group",
            "Show whether this group is private or shared",
            &["/group - show whether this group is a private room or shared"],
        );
    }
    if ctx.owner {
        add(
            "allow",
            "Approve a chat by its id",
            &["/allow CHAT_ID - allow a chat"],
        );
        add(
            "deny",
            "Revoke a chat and clear its room",
            &["/deny CHAT_ID - deny a chat and clear its room state"],
        );
        add("shutdown", "Stop tellm", &["/shutdown - stop tellm"]);
    }
    add(
        "help",
        "Show what each command does",
        &["/help - show commands"],
    );
    list
}

/// The `/help` reply: the same commands as the menu, with their usage.
pub fn help_text(ctx: &Context) -> String {
    let mut lines = commands(ctx)
        .into_iter()
        .flat_map(|command| command.help)
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>();
    if let Some(model_key) = &ctx.pinned {
        lines.push(format!(
            "- This room is pinned to {model_key}; /model changes are disabled"
        ));
    }
    lines.join("\n")
}

/// The menu for chats tellm hasn't approved: pairing is all they can do.
pub const UNAPPROVED: [(&str, &str); 1] = [(
    "pair",
    "Pair this chat with the code from the tellm terminal",
)];

#[cfg(test)]
mod tests {
    use super::*;

    fn owner_in_direct_chat() -> Context {
        Context {
            kind: ChatKind::Direct,
            owner: true,
            pinned: None,
            mode: ChatMode::Chat,
            memory: true,
            web_search: true,
            image_generation: true,
            ollama: false,
        }
    }

    fn names(ctx: &Context) -> Vec<&'static str> {
        commands(ctx)
            .into_iter()
            .map(|command| command.name)
            .collect()
    }

    #[test]
    fn an_owner_in_a_direct_chat_sees_what_the_room_can_do() {
        let ctx = owner_in_direct_chat();
        assert_eq!(
            names(&ctx),
            [
                "new",
                "id",
                "mode",
                "model",
                "role",
                "reasoning",
                "websearch",
                "imagegen",
                "cost",
                "memory",
                "remember",
                "forget",
                "allow",
                "deny",
                "shutdown",
                "help"
            ]
        );

        let local = Context {
            web_search: false,
            image_generation: false,
            ollama: true,
            memory: false,
            mode: ChatMode::Message,
            ..owner_in_direct_chat()
        };
        let names = names(&local);
        for hidden in ["websearch", "imagegen", "remember", "forget"] {
            assert!(!names.contains(&hidden), "{hidden}");
        }
        assert!(names.contains(&"ollama"));
        assert!(names.contains(&"memory"), "memory can still be turned on");
        assert!(help_text(&local).contains("- /new - start a new chat"));
        assert!(
            !help_text(&local).contains("summary"),
            "no summary in message mode"
        );
    }

    #[test]
    fn a_shared_group_shows_members_only_id_and_help_and_owners_no_conversation() {
        let member = Context {
            kind: ChatKind::SharedGroup,
            owner: false,
            ..owner_in_direct_chat()
        };
        assert_eq!(names(&member), ["id", "help"]);

        let owner = Context {
            owner: true,
            ..member
        };
        let names = names(&owner);
        for hidden in ["new", "mode", "memory", "remember", "forget"] {
            assert!(!names.contains(&hidden), "{hidden}");
        }
        assert!(names.contains(&"group"));
        assert!(names.contains(&"model"));
    }

    #[test]
    fn a_private_groups_other_members_see_no_owner_commands() {
        let ctx = Context {
            kind: ChatKind::PrivateGroup,
            owner: false,
            pinned: Some("gpt".to_string()),
            ..owner_in_direct_chat()
        };
        let names = names(&ctx);
        for hidden in ["memory", "allow", "deny", "shutdown", "ollama"] {
            assert!(!names.contains(&hidden), "{hidden}");
        }
        let help = help_text(&ctx);
        assert!(help.contains("- /group - show whether"), "{help}");
        assert!(help.contains("- /model - show this room's pinned model"));
        assert!(help.ends_with("- This room is pinned to gpt; /model changes are disabled"));
    }

    #[test]
    fn menu_entries_fit_telegrams_rules() {
        // Checked 2026-11-28 against core.telegram.org/bots/api#botcommand:
        // 1-32 lowercase letters, digits or underscores; a 1-256 character
        // description.
        for command in commands(&owner_in_direct_chat())
            .iter()
            .chain(&commands(&Context {
                kind: ChatKind::PrivateGroup,
                ollama: true,
                ..owner_in_direct_chat()
            }))
        {
            assert!((1..=32).contains(&command.name.len()), "{}", command.name);
            assert!(
                command
                    .name
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_'),
                "{}",
                command.name
            );
            assert!((1..=256).contains(&command.menu.chars().count()));
            assert!(
                command
                    .help
                    .iter()
                    .all(|line| line.starts_with(&format!("/{}", command.name))),
                "{:?}",
                command.help
            );
            assert!(command.help.iter().all(|line| !line.contains('<')));
        }
    }
}
