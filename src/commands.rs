use std::collections::BTreeSet;

use tellm_core::ThinkingLevel;

use crate::{access, rooms};

#[derive(Debug, Clone)]
pub struct CommandContext<'a> {
    pub settings: &'a rooms::RoomSettings,
    pub default_model: &'a str,
    pub model_keys: &'a BTreeSet<String>,
    pub pinned_model_key: Option<&'a str>,
    pub model_thinking: ThinkingLevel,
    pub shutdown_access: access::ShutdownAccess,
    pub capabilities: RoomCapabilities,
}

/// What the room's effective model can honor, statically known from its
/// wire format plus provider endpoint checks. Toggling a capability ON in a
/// room that can never honor it is refused at the toggle instead of arming
/// a per-message failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomCapabilities {
    pub web_search: bool,
    pub image_generation: bool,
    /// Effective model key, for refusal messages.
    pub model_key: String,
    /// Human label for the wire format/endpoint, for refusal messages.
    pub endpoint: String,
}

impl RoomCapabilities {
    /// Permissive fallback for contexts where no model is resolvable; provider
    /// dispatch still rejects a missing model before using a capability.
    pub fn permissive() -> Self {
        Self {
            web_search: true,
            image_generation: true,
            model_key: String::new(),
            endpoint: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    Command(CommandAction),
    UserMessage,
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedRoute<'a> {
    Command {
        command: KnownCommand,
        args: &'a str,
    },
    UserMessage,
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandAction {
    ResetHistory,
    ShowChatId,
    ShowMode {
        current: rooms::ChatMode,
    },
    SetMode {
        mode: rooms::ChatMode,
    },
    ShowModel {
        selected: Option<String>,
        effective: String,
        pinned: Option<String>,
        available: Vec<String>,
    },
    SetModel {
        model_key: String,
    },
    PinModel {
        model_key: String,
    },
    UnpinModel,
    ShowModelCatalog,
    AddModel {
        preset_key: String,
    },
    ShowRole {
        current: Option<String>,
    },
    SetRole {
        role: Option<String>,
    },
    ShowReasoning {
        override_level: Option<ThinkingLevel>,
        model_default: ThinkingLevel,
    },
    SetReasoning {
        thinking: Option<ThinkingLevel>,
    },
    ShowWebSearch {
        enabled: bool,
    },
    SetWebSearch {
        enabled: bool,
    },
    ShowImageGeneration {
        enabled: bool,
    },
    SetImageGeneration {
        enabled: bool,
    },
    AllowChat {
        chat_id: i64,
    },
    DenyChat {
        chat_id: i64,
    },
    Pair {
        code: String,
    },
    UnloadOllama,
    Shutdown,
    Help {
        pinned_model_key: Option<String>,
    },
    Reject {
        command: KnownCommand,
        reason: CommandReject,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnownCommand {
    New,
    Id,
    Mode,
    Model,
    Role,
    Reasoning,
    WebSearch,
    ImageGeneration,
    Allow,
    Deny,
    Pair,
    Ollama,
    Shutdown,
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandReject {
    MissingPairingCode,
    UnknownMode {
        value: String,
    },
    UnknownModel {
        value: String,
        available: Vec<String>,
    },
    UnknownReasoning {
        value: String,
    },
    UnknownBoolean {
        value: String,
    },
    MissingChatId {
        command: KnownCommand,
    },
    InvalidChatId {
        value: String,
    },
    UnknownOllamaAction {
        value: Option<String>,
    },
    PinnedModel {
        model_key: String,
    },
    AdminNotAllowed,
    AdminStale,
    ShutdownNotAdmin,
    ShutdownStale,
    CapabilityUnsupported {
        feature: String,
        model_key: String,
        endpoint: String,
    },
}

pub fn parse<'a>(text: &'a str, bot_username: Option<&str>) -> ParsedRoute<'a> {
    let Some(parsed) = parse_command(text, bot_username) else {
        return ParsedRoute::UserMessage;
    };
    if parsed.for_other_bot {
        return ParsedRoute::Ignore;
    }
    let Some(command) = parse_known_command(parsed.name) else {
        return ParsedRoute::UserMessage;
    };
    ParsedRoute::Command {
        command,
        args: parsed.args,
    }
}

pub fn resolve(command: KnownCommand, args: &str, context: &CommandContext<'_>) -> CommandAction {
    match command {
        KnownCommand::New => CommandAction::ResetHistory,
        KnownCommand::Id => CommandAction::ShowChatId,
        KnownCommand::Mode => route_mode(args, context.settings),
        KnownCommand::Model => route_model(args, context),
        KnownCommand::Role => route_role(args, context.settings),
        KnownCommand::Reasoning => route_reasoning(args, context.settings, context.model_thinking),
        KnownCommand::WebSearch => route_web_search(args, context.settings, &context.capabilities),
        KnownCommand::ImageGeneration => {
            route_image_generation(args, context.settings, &context.capabilities)
        }
        KnownCommand::Allow => route_chat_approval(command, args, context.shutdown_access),
        KnownCommand::Deny => route_chat_approval(command, args, context.shutdown_access),
        KnownCommand::Pair => route_pair(args),
        KnownCommand::Ollama => route_ollama(args, context.shutdown_access),
        KnownCommand::Shutdown => route_shutdown(context.shutdown_access),
        KnownCommand::Help => CommandAction::Help {
            pinned_model_key: context.pinned_model_key.map(str::to_string),
        },
    }
}

#[cfg(test)]
pub fn route(text: &str, bot_username: Option<&str>, context: &CommandContext<'_>) -> Route {
    match parse(text, bot_username) {
        ParsedRoute::Command { command, args } => Route::Command(resolve(command, args, context)),
        ParsedRoute::UserMessage => Route::UserMessage,
        ParsedRoute::Ignore => Route::Ignore,
    }
}

pub fn pair_code<'a>(text: &'a str, bot_username: Option<&str>) -> Option<&'a str> {
    match parse(text, bot_username) {
        ParsedRoute::Command {
            command: KnownCommand::Pair,
            args,
        } => first_arg(args),
        _ => None,
    }
}

fn route_mode(args: &str, settings: &rooms::RoomSettings) -> CommandAction {
    let Some(value) = first_arg(args) else {
        return CommandAction::ShowMode {
            current: settings.mode,
        };
    };

    match value.to_ascii_lowercase().as_str() {
        "chat" => CommandAction::SetMode {
            mode: rooms::ChatMode::Chat,
        },
        "message" => CommandAction::SetMode {
            mode: rooms::ChatMode::Message,
        },
        _ => CommandAction::Reject {
            command: KnownCommand::Mode,
            reason: CommandReject::UnknownMode {
                value: value.to_string(),
            },
        },
    }
}

fn route_model(args: &str, context: &CommandContext<'_>) -> CommandAction {
    let Some(value) = first_arg(args) else {
        let selected = context.settings.model_key.clone();
        let pinned = context.pinned_model_key.map(str::to_string);
        return CommandAction::ShowModel {
            effective: pinned
                .clone()
                .or_else(|| selected.clone())
                .unwrap_or_else(|| context.default_model.to_string()),
            selected,
            pinned,
            available: available_models(context.model_keys),
        };
    };

    // pin/unpin are admin operations and must work in pinned rooms too, so
    // they are routed before the pinned-room rejection.
    match value.to_ascii_lowercase().as_str() {
        "pin" => return route_model_pin(args, context),
        "unpin" => {
            return match admin_gate(KnownCommand::Model, context.shutdown_access) {
                Some(reject) => reject,
                None => CommandAction::UnpinModel,
            };
        }
        "add" => {
            return match admin_gate(KnownCommand::Model, context.shutdown_access) {
                Some(reject) => reject,
                None => match args.split_whitespace().nth(1) {
                    Some(preset_key) => CommandAction::AddModel {
                        preset_key: preset_key.to_string(),
                    },
                    None => CommandAction::ShowModelCatalog,
                },
            };
        }
        _ => {}
    }

    if let Some(model_key) = context.pinned_model_key {
        return CommandAction::Reject {
            command: KnownCommand::Model,
            reason: CommandReject::PinnedModel {
                model_key: model_key.to_string(),
            },
        };
    }

    if context.model_keys.contains(value) {
        return CommandAction::SetModel {
            model_key: value.to_string(),
        };
    }

    CommandAction::Reject {
        command: KnownCommand::Model,
        reason: CommandReject::UnknownModel {
            value: value.to_string(),
            available: available_models(context.model_keys),
        },
    }
}

fn route_role(args: &str, settings: &rooms::RoomSettings) -> CommandAction {
    let args = args.trim();
    if args.is_empty() {
        return CommandAction::ShowRole {
            current: settings.role.clone(),
        };
    }

    match args.to_ascii_lowercase().as_str() {
        "clear" | "off" | "none" | "reset" => CommandAction::SetRole { role: None },
        _ => CommandAction::SetRole {
            role: Some(args.to_string()),
        },
    }
}

fn route_reasoning(
    args: &str,
    settings: &rooms::RoomSettings,
    model_default: ThinkingLevel,
) -> CommandAction {
    let Some(value) = first_arg(args) else {
        return CommandAction::ShowReasoning {
            override_level: settings.thinking,
            model_default,
        };
    };

    match parse_thinking_setting(value) {
        Some(thinking) => CommandAction::SetReasoning { thinking },
        None => CommandAction::Reject {
            command: KnownCommand::Reasoning,
            reason: CommandReject::UnknownReasoning {
                value: value.to_string(),
            },
        },
    }
}

fn route_web_search(
    args: &str,
    settings: &rooms::RoomSettings,
    capabilities: &RoomCapabilities,
) -> CommandAction {
    let gate = |enabled: bool| {
        if enabled && !capabilities.web_search {
            capability_reject(KnownCommand::WebSearch, "Web search", capabilities)
        } else {
            CommandAction::SetWebSearch { enabled }
        }
    };

    let Some(value) = first_arg(args) else {
        return gate(!settings.web_search);
    };

    match value.to_ascii_lowercase().as_str() {
        "status" => CommandAction::ShowWebSearch {
            enabled: settings.web_search,
        },
        _ => match parse_bool(value) {
            Some(enabled) => gate(enabled),
            None => CommandAction::Reject {
                command: KnownCommand::WebSearch,
                reason: CommandReject::UnknownBoolean {
                    value: value.to_string(),
                },
            },
        },
    }
}

fn route_image_generation(
    args: &str,
    settings: &rooms::RoomSettings,
    capabilities: &RoomCapabilities,
) -> CommandAction {
    let gate = |enabled: bool| {
        if enabled && !capabilities.image_generation {
            capability_reject(
                KnownCommand::ImageGeneration,
                "Image generation",
                capabilities,
            )
        } else {
            CommandAction::SetImageGeneration { enabled }
        }
    };

    let Some(value) = first_arg(args) else {
        return gate(!settings.image_generation);
    };

    match value.to_ascii_lowercase().as_str() {
        "status" => CommandAction::ShowImageGeneration {
            enabled: settings.image_generation,
        },
        _ => match parse_bool(value) {
            Some(enabled) => gate(enabled),
            None => CommandAction::Reject {
                command: KnownCommand::ImageGeneration,
                reason: CommandReject::UnknownBoolean {
                    value: value.to_string(),
                },
            },
        },
    }
}

fn capability_reject(
    command: KnownCommand,
    feature: &str,
    capabilities: &RoomCapabilities,
) -> CommandAction {
    CommandAction::Reject {
        command,
        reason: CommandReject::CapabilityUnsupported {
            feature: feature.to_string(),
            model_key: capabilities.model_key.clone(),
            endpoint: capabilities.endpoint.clone(),
        },
    }
}

/// Admin gate shared by /allow, /deny, and /model pin|unpin. Returns the
/// rejection when the caller isn't an allowed, fresh admin message.
fn admin_gate(command: KnownCommand, access: access::ShutdownAccess) -> Option<CommandAction> {
    match access {
        access::ShutdownAccess::NotAdmin => Some(CommandAction::Reject {
            command,
            reason: CommandReject::AdminNotAllowed,
        }),
        access::ShutdownAccess::Stale => Some(CommandAction::Reject {
            command,
            reason: CommandReject::AdminStale,
        }),
        access::ShutdownAccess::Allowed => None,
    }
}

/// `/model pin [KEY]` — pin this room to KEY (or its current effective
/// model when KEY is omitted).
fn route_model_pin(args: &str, context: &CommandContext<'_>) -> CommandAction {
    if let Some(reject) = admin_gate(KnownCommand::Model, context.shutdown_access) {
        return reject;
    }

    let key_arg = args
        .split_whitespace()
        .nth(1)
        .map(str::to_string)
        .or_else(|| context.pinned_model_key.map(str::to_string))
        .or_else(|| context.settings.model_key.clone())
        .unwrap_or_else(|| context.default_model.to_string());

    if context.model_keys.contains(&key_arg) {
        CommandAction::PinModel { model_key: key_arg }
    } else {
        CommandAction::Reject {
            command: KnownCommand::Model,
            reason: CommandReject::UnknownModel {
                value: key_arg,
                available: available_models(context.model_keys),
            },
        }
    }
}

fn route_chat_approval(
    command: KnownCommand,
    args: &str,
    access: access::ShutdownAccess,
) -> CommandAction {
    match access {
        access::ShutdownAccess::NotAdmin => {
            return CommandAction::Reject {
                command,
                reason: CommandReject::AdminNotAllowed,
            };
        }
        access::ShutdownAccess::Stale => {
            return CommandAction::Reject {
                command,
                reason: CommandReject::AdminStale,
            };
        }
        access::ShutdownAccess::Allowed => {}
    }

    let Some(value) = first_arg(args) else {
        return CommandAction::Reject {
            command,
            reason: CommandReject::MissingChatId { command },
        };
    };
    let Ok(chat_id) = value.parse::<i64>() else {
        return CommandAction::Reject {
            command,
            reason: CommandReject::InvalidChatId {
                value: value.to_string(),
            },
        };
    };

    match command {
        KnownCommand::Allow => CommandAction::AllowChat { chat_id },
        KnownCommand::Deny => CommandAction::DenyChat { chat_id },
        _ => unreachable!("route_chat_approval only handles allow/deny"),
    }
}

fn route_pair(args: &str) -> CommandAction {
    match first_arg(args) {
        Some(code) => CommandAction::Pair {
            code: code.to_string(),
        },
        None => CommandAction::Reject {
            command: KnownCommand::Pair,
            reason: CommandReject::MissingPairingCode,
        },
    }
}

fn route_ollama(args: &str, access: access::ShutdownAccess) -> CommandAction {
    if let Some(reject) = admin_gate(KnownCommand::Ollama, access) {
        return reject;
    }

    match first_arg(args) {
        Some(action) if action.eq_ignore_ascii_case("unload") => CommandAction::UnloadOllama,
        value => CommandAction::Reject {
            command: KnownCommand::Ollama,
            reason: CommandReject::UnknownOllamaAction {
                value: value.map(str::to_string),
            },
        },
    }
}

fn route_shutdown(access: access::ShutdownAccess) -> CommandAction {
    match access {
        access::ShutdownAccess::Allowed => CommandAction::Shutdown,
        access::ShutdownAccess::NotAdmin => CommandAction::Reject {
            command: KnownCommand::Shutdown,
            reason: CommandReject::ShutdownNotAdmin,
        },
        access::ShutdownAccess::Stale => CommandAction::Reject {
            command: KnownCommand::Shutdown,
            reason: CommandReject::ShutdownStale,
        },
    }
}

struct ParsedCommand<'a> {
    name: &'a str,
    args: &'a str,
    for_other_bot: bool,
}

fn parse_command<'a>(text: &'a str, bot_username: Option<&str>) -> Option<ParsedCommand<'a>> {
    let text = text.trim_start();
    if !text.starts_with('/') {
        return None;
    }

    let token_end = text.find(char::is_whitespace).unwrap_or(text.len());
    let token = &text[..token_end];
    let args = text[token_end..].trim();
    let command = token.strip_prefix('/')?;
    if command.is_empty() {
        return None;
    }

    let (name, target) = command
        .split_once('@')
        .map_or((command, None), |(name, target)| (name, Some(target)));
    if name.is_empty() {
        return None;
    }

    let for_other_bot = match (target, bot_username) {
        (Some(target), Some(bot_username)) => !target.eq_ignore_ascii_case(bot_username),
        _ => false,
    };

    Some(ParsedCommand {
        name,
        args,
        for_other_bot,
    })
}

fn parse_known_command(name: &str) -> Option<KnownCommand> {
    match name.to_ascii_lowercase().as_str() {
        "new" => Some(KnownCommand::New),
        "id" => Some(KnownCommand::Id),
        "mode" => Some(KnownCommand::Mode),
        "model" => Some(KnownCommand::Model),
        "role" => Some(KnownCommand::Role),
        "reasoning" => Some(KnownCommand::Reasoning),
        "websearch" => Some(KnownCommand::WebSearch),
        "imagegen" => Some(KnownCommand::ImageGeneration),
        "allow" => Some(KnownCommand::Allow),
        "deny" => Some(KnownCommand::Deny),
        "pair" => Some(KnownCommand::Pair),
        "ollama" => Some(KnownCommand::Ollama),
        "shutdown" => Some(KnownCommand::Shutdown),
        "help" => Some(KnownCommand::Help),
        _ => None,
    }
}

fn parse_thinking_setting(value: &str) -> Option<Option<ThinkingLevel>> {
    match value.to_ascii_lowercase().as_str() {
        "default" | "model" | "reset" => Some(None),
        "off" => Some(Some(ThinkingLevel::Off)),
        "low" => Some(Some(ThinkingLevel::Low)),
        "medium" => Some(Some(ThinkingLevel::Medium)),
        "high" => Some(Some(ThinkingLevel::High)),
        "max" => Some(Some(ThinkingLevel::Max)),
        _ => None,
    }
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "enable" | "enabled" => Some(true),
        "off" | "false" | "no" | "disable" | "disabled" => Some(false),
        _ => None,
    }
}

fn first_arg(args: &str) -> Option<&str> {
    args.split_whitespace().next()
}

fn available_models(model_keys: &BTreeSet<String>) -> Vec<String> {
    model_keys.iter().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access;
    use tellm_core::ThinkingLevel;

    #[test]
    fn non_commands_and_unknown_commands_route_to_model_input() {
        assert_eq!(route_command("hello"), Route::UserMessage);
        assert_eq!(route_command("/unknown value"), Route::UserMessage);
        assert_eq!(route_command("/"), Route::UserMessage);
    }

    #[test]
    fn group_command_suffix_accepts_this_bot_and_ignores_other_bots() {
        assert_eq!(
            route_command_with(
                "/help@TellmBot",
                rooms::RoomSettings::default(),
                Some("tellmbot"),
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Help {
                pinned_model_key: None,
            })
        );
        assert_eq!(
            route_command_with(
                "/help@OtherBot",
                rooms::RoomSettings::default(),
                Some("tellmbot"),
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Ignore
        );
    }

    #[test]
    fn new_id_and_help_commands_dispatch() {
        assert_eq!(
            route_command("/new"),
            Route::Command(CommandAction::ResetHistory)
        );
        assert_eq!(
            route_command("/id"),
            Route::Command(CommandAction::ShowChatId)
        );
        assert_eq!(
            route_command("/help"),
            Route::Command(CommandAction::Help {
                pinned_model_key: None,
            })
        );
    }

    #[test]
    fn mode_command_shows_sets_and_rejects_unknown_modes() {
        let mut settings = rooms::RoomSettings {
            mode: rooms::ChatMode::Message,
            ..rooms::RoomSettings::default()
        };
        assert_eq!(
            route_command_with_default_settings("/mode", settings.clone()),
            Route::Command(CommandAction::ShowMode {
                current: rooms::ChatMode::Message,
            })
        );
        assert_eq!(
            route_command_with_default_settings("/mode chat", settings.clone()),
            Route::Command(CommandAction::SetMode {
                mode: rooms::ChatMode::Chat,
            })
        );

        settings.mode = rooms::ChatMode::Chat;
        assert_eq!(
            route_command_with_default_settings("/mode nope", settings),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Mode,
                reason: CommandReject::UnknownMode {
                    value: "nope".to_string(),
                },
            })
        );
    }

    #[test]
    fn model_command_shows_sets_and_rejects_unknown_models() {
        let settings = rooms::RoomSettings {
            model_key: Some("gpt".to_string()),
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with(
                "/model",
                settings.clone(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::ShowModel {
                selected: Some("gpt".to_string()),
                effective: "gpt".to_string(),
                pinned: None,
                available: vec!["claude".to_string(), "gpt".to_string()],
            })
        );
        assert_eq!(
            route_command_with(
                "/model claude",
                settings.clone(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::SetModel {
                model_key: "claude".to_string(),
            })
        );
        assert_eq!(
            route_command_with(
                "/model missing",
                settings,
                None,
                access::ShutdownAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Model,
                reason: CommandReject::UnknownModel {
                    value: "missing".to_string(),
                    available: vec!["claude".to_string(), "gpt".to_string()],
                },
            })
        );
    }

    #[test]
    fn model_command_shows_and_enforces_room_pin() {
        let settings = rooms::RoomSettings {
            model_key: Some("claude".to_string()),
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with_pinned(
                "/model",
                settings.clone(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude", "gpt"],
                Some("gpt"),
            ),
            Route::Command(CommandAction::ShowModel {
                selected: Some("claude".to_string()),
                effective: "gpt".to_string(),
                pinned: Some("gpt".to_string()),
                available: vec!["claude".to_string(), "gpt".to_string()],
            })
        );
        assert_eq!(
            route_command_with_pinned(
                "/model claude",
                settings.clone(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude", "gpt"],
                Some("gpt"),
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Model,
                reason: CommandReject::PinnedModel {
                    model_key: "gpt".to_string(),
                },
            })
        );
        assert_eq!(
            route_command_with_pinned(
                "/help",
                settings,
                None,
                access::ShutdownAccess::Allowed,
                &["claude", "gpt"],
                Some("gpt"),
            ),
            Route::Command(CommandAction::Help {
                pinned_model_key: Some("gpt".to_string()),
            })
        );
    }

    #[test]
    fn model_add_preserves_configured_model_key_case() {
        assert_eq!(
            route_command_with(
                "/model add Mistral",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["Mistral"],
            ),
            Route::Command(CommandAction::AddModel {
                preset_key: "Mistral".to_string(),
            })
        );
    }

    #[test]
    fn model_add_does_not_select_configured_model() {
        assert_eq!(
            route_command_with(
                "/model add claude2",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude2"],
            ),
            Route::Command(CommandAction::AddModel {
                preset_key: "claude2".to_string(),
            })
        );
        assert_eq!(
            route_command_with(
                "/model claude2",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude2"],
            ),
            Route::Command(CommandAction::SetModel {
                model_key: "claude2".to_string(),
            })
        );
    }

    #[test]
    fn role_command_shows_sets_and_clears_full_text_roles() {
        let settings = rooms::RoomSettings {
            role: Some("terse".to_string()),
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with_default_settings("/role", settings.clone()),
            Route::Command(CommandAction::ShowRole {
                current: Some("terse".to_string()),
            })
        );
        assert_eq!(
            route_command_with_default_settings("/role You are concise.", settings.clone()),
            Route::Command(CommandAction::SetRole {
                role: Some("You are concise.".to_string()),
            })
        );
        assert_eq!(
            route_command_with_default_settings("/role clear", settings),
            Route::Command(CommandAction::SetRole { role: None })
        );
    }

    #[test]
    fn reasoning_command_shows_sets_and_rejects_unknown_levels() {
        let settings = rooms::RoomSettings {
            thinking: Some(ThinkingLevel::High),
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with_default_settings("/reasoning", settings.clone()),
            Route::Command(CommandAction::ShowReasoning {
                override_level: Some(ThinkingLevel::High),
                model_default: ThinkingLevel::Medium,
            })
        );
        assert_eq!(
            route_command_with_default_settings("/reasoning max", settings.clone()),
            Route::Command(CommandAction::SetReasoning {
                thinking: Some(ThinkingLevel::Max),
            })
        );
        assert_eq!(
            route_command_with_default_settings("/reasoning default", settings),
            Route::Command(CommandAction::SetReasoning { thinking: None })
        );
        assert_eq!(
            route_command_with_default_settings("/reasoning nope", rooms::RoomSettings::default()),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Reasoning,
                reason: CommandReject::UnknownReasoning {
                    value: "nope".to_string(),
                },
            })
        );
    }

    #[test]
    fn websearch_command_toggles_sets_and_shows_status() {
        let disabled = rooms::RoomSettings {
            web_search: false,
            ..rooms::RoomSettings::default()
        };
        let enabled = rooms::RoomSettings {
            web_search: true,
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with_default_settings("/websearch", disabled),
            Route::Command(CommandAction::SetWebSearch { enabled: true })
        );
        assert_eq!(
            route_command_with_default_settings("/websearch off", enabled.clone()),
            Route::Command(CommandAction::SetWebSearch { enabled: false })
        );
        assert_eq!(
            route_command_with_default_settings("/websearch status", enabled.clone()),
            Route::Command(CommandAction::ShowWebSearch { enabled: true })
        );
        assert_eq!(
            route_command_with_default_settings("/websearch maybe", enabled),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::WebSearch,
                reason: CommandReject::UnknownBoolean {
                    value: "maybe".to_string(),
                },
            })
        );
    }

    #[test]
    fn imagegen_command_toggles_sets_and_shows_status() {
        let disabled = rooms::RoomSettings {
            image_generation: false,
            ..rooms::RoomSettings::default()
        };
        let enabled = rooms::RoomSettings {
            image_generation: true,
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with_default_settings("/imagegen", disabled),
            Route::Command(CommandAction::SetImageGeneration { enabled: true })
        );
        assert_eq!(
            route_command_with_default_settings("/imagegen off", enabled.clone()),
            Route::Command(CommandAction::SetImageGeneration { enabled: false })
        );
        assert_eq!(
            route_command_with_default_settings("/imagegen status", enabled.clone()),
            Route::Command(CommandAction::ShowImageGeneration { enabled: true })
        );
        assert_eq!(
            route_command_with_default_settings("/imagegen maybe", enabled),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::ImageGeneration,
                reason: CommandReject::UnknownBoolean {
                    value: "maybe".to_string(),
                },
            })
        );
    }

    #[test]
    fn pair_command_extracts_code_and_requires_one() {
        assert_eq!(
            route_command("/pair 123456"),
            Route::Command(CommandAction::Pair {
                code: "123456".to_string(),
            })
        );
        assert_eq!(
            route_command("/pair"),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Pair,
                reason: CommandReject::MissingPairingCode,
            })
        );
        assert_eq!(pair_code("/pair 123456", None), Some("123456"));
        assert_eq!(
            pair_code("/pair@TellmBot 654321", Some("tellmbot")),
            Some("654321")
        );
        assert_eq!(pair_code("/pair@OtherBot 123456", Some("tellmbot")), None);
        assert_eq!(pair_code("/pairing 123456", None), None);
        assert_eq!(pair_code("/pair", None), None);
    }

    #[test]
    fn allow_and_deny_commands_require_admin_and_chat_id() {
        assert_eq!(
            route_command_with(
                "/allow -100",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::AllowChat { chat_id: -100 })
        );
        assert_eq!(
            route_command_with(
                "/deny 42",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::DenyChat { chat_id: 42 })
        );
        assert_eq!(
            route_command_with(
                "/allow",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Allow,
                reason: CommandReject::MissingChatId {
                    command: KnownCommand::Allow,
                },
            })
        );
        assert_eq!(
            route_command_with(
                "/deny nope",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Deny,
                reason: CommandReject::InvalidChatId {
                    value: "nope".to_string(),
                },
            })
        );
        assert_eq!(
            route_command_with(
                "/allow 42",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::NotAdmin,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Allow,
                reason: CommandReject::AdminNotAllowed,
            })
        );
        assert_eq!(
            route_command_with(
                "/deny 42",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Stale,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Deny,
                reason: CommandReject::AdminStale,
            })
        );
    }

    #[test]
    fn shutdown_command_uses_access_decision() {
        assert_eq!(
            route_command_with(
                "/shutdown",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Shutdown)
        );
        assert_eq!(
            route_command_with(
                "/shutdown",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::NotAdmin,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Shutdown,
                reason: CommandReject::ShutdownNotAdmin,
            })
        );
        assert_eq!(
            route_command_with(
                "/shutdown",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Stale,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Shutdown,
                reason: CommandReject::ShutdownStale,
            })
        );
    }

    #[test]
    fn ollama_unload_command_uses_admin_gate() {
        assert_eq!(
            route_command_with(
                "/ollama unload",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::UnloadOllama)
        );
        assert_eq!(
            route_command_with(
                "/ollama",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Ollama,
                reason: CommandReject::UnknownOllamaAction { value: None },
            })
        );
        assert_eq!(
            route_command_with(
                "/ollama unload",
                rooms::RoomSettings::default(),
                None,
                access::ShutdownAccess::NotAdmin,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                command: KnownCommand::Ollama,
                reason: CommandReject::AdminNotAllowed,
            })
        );
    }

    fn route_command(text: &str) -> Route {
        route_command_with_default_settings(text, rooms::RoomSettings::default())
    }

    fn route_command_with_default_settings(text: &str, settings: rooms::RoomSettings) -> Route {
        route_command_with(
            text,
            settings,
            None,
            access::ShutdownAccess::Allowed,
            &["claude"],
        )
    }

    fn route_command_with(
        text: &str,
        settings: rooms::RoomSettings,
        bot_username: Option<&str>,
        shutdown_access: access::ShutdownAccess,
        models: &[&str],
    ) -> Route {
        route_command_with_pinned(text, settings, bot_username, shutdown_access, models, None)
    }

    fn route_command_with_pinned(
        text: &str,
        settings: rooms::RoomSettings,
        bot_username: Option<&str>,
        shutdown_access: access::ShutdownAccess,
        models: &[&str],
        pinned_model_key: Option<&str>,
    ) -> Route {
        let model_keys = models
            .iter()
            .map(|model| model.to_string())
            .collect::<BTreeSet<_>>();
        let context = CommandContext {
            settings: &settings,
            default_model: "claude",
            model_keys: &model_keys,
            pinned_model_key,
            model_thinking: ThinkingLevel::Medium,
            shutdown_access,
            capabilities: RoomCapabilities::permissive(),
        };
        route(text, bot_username, &context)
    }

    fn route_with_capabilities(
        text: &str,
        settings: rooms::RoomSettings,
        capabilities: RoomCapabilities,
    ) -> Route {
        let model_keys = BTreeSet::from(["claude".to_string()]);
        let context = CommandContext {
            settings: &settings,
            default_model: "claude",
            model_keys: &model_keys,
            pinned_model_key: None,
            model_thinking: ThinkingLevel::Medium,
            shutdown_access: access::ShutdownAccess::NotAdmin,
            capabilities,
        };
        route(text, None, &context)
    }

    fn incapable_room() -> RoomCapabilities {
        RoomCapabilities {
            web_search: false,
            image_generation: false,
            model_key: "claude".to_string(),
            endpoint: "Anthropic Messages".to_string(),
        }
    }

    #[test]
    fn toggles_on_are_rejected_when_capability_is_unsupported() {
        for command in ["/imagegen on", "/imagegen", "/websearch on"] {
            let route =
                route_with_capabilities(command, rooms::RoomSettings::default(), incapable_room());
            assert!(
                matches!(
                    route,
                    Route::Command(CommandAction::Reject {
                        reason: CommandReject::CapabilityUnsupported { .. },
                        ..
                    })
                ),
                "{command} should be rejected, got {route:?}"
            );
        }
    }

    #[test]
    fn toggles_off_and_status_bypass_capability_gate() {
        let enabled = rooms::RoomSettings {
            image_generation: true,
            web_search: true,
            ..Default::default()
        };
        for (command, expected) in [
            (
                "/imagegen off",
                CommandAction::SetImageGeneration { enabled: false },
            ),
            (
                "/imagegen status",
                CommandAction::ShowImageGeneration { enabled: true },
            ),
            (
                "/websearch off",
                CommandAction::SetWebSearch { enabled: false },
            ),
        ] {
            assert_eq!(
                route_with_capabilities(command, enabled.clone(), incapable_room()),
                Route::Command(expected),
                "{command}"
            );
        }
    }

    #[test]
    fn toggles_on_pass_when_capability_is_supported() {
        assert_eq!(
            route_with_capabilities(
                "/imagegen on",
                rooms::RoomSettings::default(),
                RoomCapabilities::permissive(),
            ),
            Route::Command(CommandAction::SetImageGeneration { enabled: true })
        );
    }
}
