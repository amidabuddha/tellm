use std::collections::BTreeSet;

use tellm_core::ThinkingLevel;

use crate::groups::ChatKind;
use crate::{access, rooms};

#[derive(Debug, Clone)]
pub struct CommandContext<'a> {
    pub settings: &'a rooms::RoomSettings,
    pub default_model: &'a str,
    pub model_keys: &'a BTreeSet<String>,
    pub pinned_model_key: Option<&'a str>,
    pub model_thinking: ThinkingLevel,
    pub privileged_access: access::PrivilegedAccess,
    pub capabilities: RoomCapabilities,
    pub chat_kind: ChatKind,
}

impl CommandContext<'_> {
    /// The room's own model selection, ignored once its model is no longer
    /// configured, as the runtime ignores it when it picks the model.
    fn selected_model_key(&self) -> Option<String> {
        self.settings
            .model_key
            .clone()
            .filter(|key| self.model_keys.contains(key))
    }
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
    /// A well-formed command name tellm doesn't have, such as `/start` or a
    /// typo. It gets a short reply instead of a billed model call.
    Unknown {
        name: &'a str,
    },
    UserMessage,
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandAction {
    ResetHistory,
    /// `/new summary`: summarize the conversation and carry the summary into
    /// a new one.
    SummarizeChat,
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
    /// `/model KEY` for the model the room already uses: switching would
    /// only end the conversation, so nothing changes.
    ModelUnchanged {
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
    ShowCost,
    SetCostLine {
        enabled: bool,
    },
    ShowMemory,
    SetMemory {
        enabled: bool,
    },
    Remember {
        fact: String,
    },
    Forget {
        number: usize,
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
    /// `/group`: say whether this chat is a shared or a private group.
    ShowGroup {
        kind: ChatKind,
    },
    /// `/group private|shared`: mark this group.
    SetGroupPrivate {
        private: bool,
    },
    /// `/help`: the commands that apply in the room for its sender.
    Help {
        owner: bool,
    },
    Reject {
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
    Cost,
    Memory,
    Remember,
    Forget,
    Allow,
    Deny,
    Pair,
    Ollama,
    Shutdown,
    Group,
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandReject {
    /// Anyone but an owner, in a shared group, for any command except
    /// `/help` and `/id`.
    SharedGroupOwnerOnly,
    /// `/mode` changes and `/new summary` in a shared group, where every
    /// message stands alone.
    StandaloneInSharedGroup,
    /// `/memory`, `/remember` and `/forget` in a shared group: memory is off
    /// there, and listing or echoing facts would show them to everyone.
    MemoryInSharedGroup,
    UnknownGroupOption {
        value: String,
    },
    UnknownCommand {
        name: String,
    },
    UnknownNewOption {
        value: String,
    },
    MissingPairingCode,
    MissingMemoryFact,
    InvalidMemoryNumber {
        value: Option<String>,
    },
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
    OwnerNotAllowed,
    OwnerStale,
    ShutdownNotOwner,
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
        if is_command_name(parsed.name) {
            return ParsedRoute::Unknown { name: parsed.name };
        }
        // Text that only starts like a command, such as a path
        // ("/etc/hosts: what is this?"), is a normal message.
        return ParsedRoute::UserMessage;
    };
    ParsedRoute::Command {
        command,
        args: parsed.args,
    }
}

pub fn resolve(command: KnownCommand, args: &str, context: &CommandContext<'_>) -> CommandAction {
    let shared = context.chat_kind == ChatKind::SharedGroup;
    // In a shared group the other members may read and ask, but only an
    // owner changes what the bot does or what it costs.
    if shared
        && matches!(
            context.privileged_access,
            access::PrivilegedAccess::NotOwner
        )
        && !matches!(command, KnownCommand::Help | KnownCommand::Id)
    {
        return CommandAction::Reject {
            reason: CommandReject::SharedGroupOwnerOnly,
        };
    }
    match command {
        KnownCommand::New => route_new(args, shared),
        KnownCommand::Id => CommandAction::ShowChatId,
        KnownCommand::Mode => route_mode(args, context.settings, shared),
        KnownCommand::Model => route_model(args, context),
        KnownCommand::Role => route_role(args, context.settings),
        KnownCommand::Reasoning => route_reasoning(args, context.settings, context.model_thinking),
        KnownCommand::WebSearch => route_web_search(args, context.settings, &context.capabilities),
        KnownCommand::ImageGeneration => {
            route_image_generation(args, context.settings, &context.capabilities)
        }
        KnownCommand::Cost => route_cost(args),
        KnownCommand::Memory | KnownCommand::Remember | KnownCommand::Forget if shared => {
            CommandAction::Reject {
                reason: CommandReject::MemoryInSharedGroup,
            }
        }
        KnownCommand::Memory => route_memory(args, context.privileged_access),
        KnownCommand::Remember => route_remember(args, context.privileged_access),
        KnownCommand::Forget => route_forget(args, context.privileged_access),
        KnownCommand::Allow => route_chat_approval(command, args, context.privileged_access),
        KnownCommand::Deny => route_chat_approval(command, args, context.privileged_access),
        KnownCommand::Pair => route_pair(args),
        KnownCommand::Ollama => route_ollama(args, context.privileged_access),
        KnownCommand::Shutdown => route_shutdown(context.privileged_access),
        KnownCommand::Group => route_group(args, context),
        KnownCommand::Help => CommandAction::Help {
            owner: !matches!(
                context.privileged_access,
                access::PrivilegedAccess::NotOwner
            ),
        },
    }
}

#[cfg(test)]
pub fn route(text: &str, bot_username: Option<&str>, context: &CommandContext<'_>) -> Route {
    match parse(text, bot_username) {
        ParsedRoute::Command { command, args } => Route::Command(resolve(command, args, context)),
        ParsedRoute::Unknown { name } => Route::Command(unknown_command(name)),
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

/// `/new` starts over and `/new summary` carries a summary along. Anything
/// else is refused rather than read as a plain `/new`, so a mistyped
/// `summary` never throws the conversation away.
/// In a shared group `/new summary` is refused: there is no conversation
/// there, and a summary of the room's earlier private one would be posted for
/// everyone to read.
fn route_new(args: &str, shared: bool) -> CommandAction {
    match first_arg(args) {
        None => CommandAction::ResetHistory,
        Some(value) if value.eq_ignore_ascii_case("summary") && shared => CommandAction::Reject {
            reason: CommandReject::StandaloneInSharedGroup,
        },
        Some(value) if value.eq_ignore_ascii_case("summary") => CommandAction::SummarizeChat,
        Some(value) => CommandAction::Reject {
            reason: CommandReject::UnknownNewOption {
                value: value.to_string(),
            },
        },
    }
}

/// A shared group is always in message mode; the room's own setting waits
/// for the group to be marked private.
fn route_mode(args: &str, settings: &rooms::RoomSettings, shared: bool) -> CommandAction {
    let Some(value) = first_arg(args) else {
        return CommandAction::ShowMode {
            current: if shared {
                rooms::ChatMode::Message
            } else {
                settings.mode
            },
        };
    };
    if shared {
        return CommandAction::Reject {
            reason: CommandReject::StandaloneInSharedGroup,
        };
    }

    match value.to_ascii_lowercase().as_str() {
        "chat" => CommandAction::SetMode {
            mode: rooms::ChatMode::Chat,
        },
        "message" => CommandAction::SetMode {
            mode: rooms::ChatMode::Message,
        },
        _ => CommandAction::Reject {
            reason: CommandReject::UnknownMode {
                value: value.to_string(),
            },
        },
    }
}

fn route_model(args: &str, context: &CommandContext<'_>) -> CommandAction {
    let Some(value) = first_arg(args) else {
        let selected = context.selected_model_key();
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

    // pin/unpin are owner-only operations and must work in pinned rooms too, so
    // they are routed before the pinned-room rejection.
    match value.to_ascii_lowercase().as_str() {
        "pin" => return route_model_pin(args, context),
        "unpin" => {
            return match owner_gate(context.privileged_access) {
                Some(reject) => reject,
                None => CommandAction::UnpinModel,
            };
        }
        "add" => {
            return match owner_gate(context.privileged_access) {
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
            reason: CommandReject::PinnedModel {
                model_key: model_key.to_string(),
            },
        };
    }

    if context.model_keys.contains(value) {
        let current = context
            .selected_model_key()
            .unwrap_or_else(|| context.default_model.to_string());
        if current == value {
            return CommandAction::ModelUnchanged {
                model_key: value.to_string(),
            };
        }
        return CommandAction::SetModel {
            model_key: value.to_string(),
        };
    }

    CommandAction::Reject {
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
            capability_reject("Web search", capabilities)
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
            capability_reject("Image generation", capabilities)
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
                reason: CommandReject::UnknownBoolean {
                    value: value.to_string(),
                },
            },
        },
    }
}

/// `/cost` shows the conversation's running cost; `/cost on|off` shows or
/// hides the cost line under each reply. Unlike the capability toggles, a
/// bare `/cost` reports instead of toggling, since the totals are what it is
/// for.
fn route_cost(args: &str) -> CommandAction {
    let Some(value) = first_arg(args) else {
        return CommandAction::ShowCost;
    };
    if value.eq_ignore_ascii_case("status") {
        return CommandAction::ShowCost;
    }
    match parse_bool(value) {
        Some(enabled) => CommandAction::SetCostLine { enabled },
        None => CommandAction::Reject {
            reason: CommandReject::UnknownBoolean {
                value: value.to_string(),
            },
        },
    }
}

/// The memory commands are owner-only: the facts are the owner's, and
/// listing them in a shared room would show them to everyone there.
fn route_memory(args: &str, access: access::PrivilegedAccess) -> CommandAction {
    if let Some(reject) = owner_gate(access) {
        return reject;
    }
    let Some(value) = first_arg(args) else {
        return CommandAction::ShowMemory;
    };
    if value.eq_ignore_ascii_case("status") {
        return CommandAction::ShowMemory;
    }
    match parse_bool(value) {
        Some(enabled) => CommandAction::SetMemory { enabled },
        None => CommandAction::Reject {
            reason: CommandReject::UnknownBoolean {
                value: value.to_string(),
            },
        },
    }
}

/// `/group` shows whether this group is shared or private; `/group private`
/// and `/group shared` are an owner's to set. A direct chat is always
/// private.
fn route_group(args: &str, context: &CommandContext<'_>) -> CommandAction {
    let value = first_arg(args).map(str::to_ascii_lowercase);
    let private = match value.as_deref() {
        None | Some("status") => {
            return CommandAction::ShowGroup {
                kind: context.chat_kind,
            };
        }
        Some("private") => true,
        Some("shared") => false,
        Some(value) => {
            return CommandAction::Reject {
                reason: CommandReject::UnknownGroupOption {
                    value: value.to_string(),
                },
            };
        }
    };
    if let Some(reject) = owner_gate(context.privileged_access) {
        return reject;
    }
    if context.chat_kind == ChatKind::Direct {
        return CommandAction::ShowGroup {
            kind: ChatKind::Direct,
        };
    }
    CommandAction::SetGroupPrivate { private }
}

fn route_remember(args: &str, access: access::PrivilegedAccess) -> CommandAction {
    if let Some(reject) = owner_gate(access) {
        return reject;
    }
    let fact = crate::memory::normalize_fact(args);
    if fact.is_empty() {
        return CommandAction::Reject {
            reason: CommandReject::MissingMemoryFact,
        };
    }
    CommandAction::Remember { fact }
}

fn route_forget(args: &str, access: access::PrivilegedAccess) -> CommandAction {
    if let Some(reject) = owner_gate(access) {
        return reject;
    }
    let value = first_arg(args);
    match value
        .map(|value| value.trim_start_matches('#'))
        .and_then(|value| value.parse::<usize>().ok())
    {
        Some(number) if number > 0 => CommandAction::Forget { number },
        _ => CommandAction::Reject {
            reason: CommandReject::InvalidMemoryNumber {
                value: value.map(str::to_string),
            },
        },
    }
}

fn capability_reject(feature: &str, capabilities: &RoomCapabilities) -> CommandAction {
    CommandAction::Reject {
        reason: CommandReject::CapabilityUnsupported {
            feature: feature.to_string(),
            model_key: capabilities.model_key.clone(),
            endpoint: capabilities.endpoint.clone(),
        },
    }
}

/// Owner gate shared by /allow, /deny, and /model pin|unpin. Returns the
/// rejection when the caller isn't a registered owner using a fresh message.
fn owner_gate(access: access::PrivilegedAccess) -> Option<CommandAction> {
    match access {
        access::PrivilegedAccess::NotOwner => Some(CommandAction::Reject {
            reason: CommandReject::OwnerNotAllowed,
        }),
        access::PrivilegedAccess::Stale => Some(CommandAction::Reject {
            reason: CommandReject::OwnerStale,
        }),
        access::PrivilegedAccess::Allowed => None,
    }
}

/// `/model pin [KEY]` — pin this room to KEY (or its current effective
/// model when KEY is omitted).
fn route_model_pin(args: &str, context: &CommandContext<'_>) -> CommandAction {
    if let Some(reject) = owner_gate(context.privileged_access) {
        return reject;
    }

    let key_arg = args
        .split_whitespace()
        .nth(1)
        .map(str::to_string)
        .or_else(|| context.pinned_model_key.map(str::to_string))
        .or_else(|| context.selected_model_key())
        .unwrap_or_else(|| context.default_model.to_string());

    if context.model_keys.contains(&key_arg) {
        CommandAction::PinModel { model_key: key_arg }
    } else {
        CommandAction::Reject {
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
    access: access::PrivilegedAccess,
) -> CommandAction {
    if let Some(reject) = owner_gate(access) {
        return reject;
    }

    let Some(value) = first_arg(args) else {
        return CommandAction::Reject {
            reason: CommandReject::MissingChatId { command },
        };
    };
    let Ok(chat_id) = value.parse::<i64>() else {
        return CommandAction::Reject {
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
            reason: CommandReject::MissingPairingCode,
        },
    }
}

fn route_ollama(args: &str, access: access::PrivilegedAccess) -> CommandAction {
    if let Some(reject) = owner_gate(access) {
        return reject;
    }

    match first_arg(args) {
        Some(action) if action.eq_ignore_ascii_case("unload") => CommandAction::UnloadOllama,
        value => CommandAction::Reject {
            reason: CommandReject::UnknownOllamaAction {
                value: value.map(str::to_string),
            },
        },
    }
}

fn route_shutdown(access: access::PrivilegedAccess) -> CommandAction {
    match access {
        access::PrivilegedAccess::Allowed => CommandAction::Shutdown,
        access::PrivilegedAccess::NotOwner => CommandAction::Reject {
            reason: CommandReject::ShutdownNotOwner,
        },
        access::PrivilegedAccess::Stale => CommandAction::Reject {
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

/// Telegram's command shape: 1–32 Latin letters, digits and underscores
/// (checked 2026-11-27 against core.telegram.org/bots/features#commands).
fn is_command_name(name: &str) -> bool {
    (1..=32).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

pub fn unknown_command(name: &str) -> CommandAction {
    CommandAction::Reject {
        reason: CommandReject::UnknownCommand {
            name: name.to_string(),
        },
    }
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
        "cost" => Some(KnownCommand::Cost),
        "memory" => Some(KnownCommand::Memory),
        "remember" => Some(KnownCommand::Remember),
        "forget" => Some(KnownCommand::Forget),
        "allow" => Some(KnownCommand::Allow),
        "deny" => Some(KnownCommand::Deny),
        "pair" => Some(KnownCommand::Pair),
        "ollama" => Some(KnownCommand::Ollama),
        "shutdown" => Some(KnownCommand::Shutdown),
        "group" => Some(KnownCommand::Group),
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
    fn non_commands_and_command_lookalikes_route_to_model_input() {
        assert_eq!(route_command("hello"), Route::UserMessage);
        assert_eq!(route_command("/"), Route::UserMessage);
        assert_eq!(
            route_command("/etc/hosts: what is this file?"),
            Route::UserMessage
        );
        assert_eq!(route_command("/día"), Route::UserMessage);
        assert_eq!(
            route_command(&format!("/{}", "a".repeat(33))),
            Route::UserMessage
        );
    }

    #[test]
    fn unknown_commands_are_refused_instead_of_sent_to_the_model() {
        for (text, name) in [
            ("/start", "start"),
            ("/unknown value", "unknown"),
            ("/web_search on", "web_search"),
            ("/Models", "Models"),
        ] {
            assert_eq!(
                route_command(text),
                Route::Command(CommandAction::Reject {
                    reason: CommandReject::UnknownCommand {
                        name: name.to_string()
                    },
                }),
                "{text}"
            );
        }
        assert_eq!(
            route_command_with(
                "/start@OtherBot",
                rooms::RoomSettings::default(),
                Some("tellmbot"),
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Ignore,
            "another bot's command stays ignored"
        );
    }

    #[test]
    fn group_command_suffix_accepts_this_bot_and_ignores_other_bots() {
        assert_eq!(
            route_command_with(
                "/help@TellmBot",
                rooms::RoomSettings::default(),
                Some("tellmbot"),
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Help { owner: true })
        );
        assert_eq!(
            route_command_with(
                "/help@OtherBot",
                rooms::RoomSettings::default(),
                Some("tellmbot"),
                access::PrivilegedAccess::Allowed,
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
            route_command("/new Summary"),
            Route::Command(CommandAction::SummarizeChat)
        );
        // Like /new, it is not owner-gated.
        assert_eq!(
            route_command_with(
                "/new summary",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::NotOwner,
                &["claude"],
            ),
            Route::Command(CommandAction::SummarizeChat)
        );
        assert_eq!(
            route_command("/new summery"),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::UnknownNewOption {
                    value: "summery".to_string(),
                },
            })
        );
        assert_eq!(
            route_command("/id"),
            Route::Command(CommandAction::ShowChatId)
        );
        assert_eq!(
            route_command("/help"),
            Route::Command(CommandAction::Help { owner: true })
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
                reason: CommandReject::UnknownMode {
                    value: "nope".to_string(),
                },
            })
        );
    }

    #[test]
    fn model_command_shows_the_default_when_the_selected_model_is_gone() {
        let settings = rooms::RoomSettings {
            model_key: Some("openai".to_string()),
            ..rooms::RoomSettings::default()
        };

        assert_eq!(
            route_command_with(
                "/model",
                settings.clone(),
                None,
                access::PrivilegedAccess::Allowed,
                &["astra", "claude"],
            ),
            Route::Command(CommandAction::ShowModel {
                selected: None,
                effective: "claude".to_string(),
                pinned: None,
                available: vec!["astra".to_string(), "claude".to_string()],
            })
        );
        assert_eq!(
            route_command_with(
                "/model pin",
                settings,
                None,
                access::PrivilegedAccess::Allowed,
                &["astra", "claude"],
            ),
            Route::Command(CommandAction::PinModel {
                model_key: "claude".to_string(),
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
                access::PrivilegedAccess::Allowed,
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
                access::PrivilegedAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::SetModel {
                model_key: "claude".to_string(),
            })
        );
        assert_eq!(
            route_command_with(
                "/model gpt",
                settings.clone(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::ModelUnchanged {
                model_key: "gpt".to_string(),
            }),
            "the room's own model"
        );
        assert_eq!(
            route_command_with(
                "/model claude",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::ModelUnchanged {
                model_key: "claude".to_string(),
            }),
            "the default model, when the room has no selection"
        );
        assert_eq!(
            route_command_with(
                "/model missing",
                settings,
                None,
                access::PrivilegedAccess::Allowed,
                &["claude", "gpt"],
            ),
            Route::Command(CommandAction::Reject {
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
                access::PrivilegedAccess::Allowed,
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
                access::PrivilegedAccess::Allowed,
                &["claude", "gpt"],
                Some("gpt"),
            ),
            Route::Command(CommandAction::Reject {
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
                access::PrivilegedAccess::Allowed,
                &["claude", "gpt"],
                Some("gpt"),
            ),
            Route::Command(CommandAction::Help { owner: true })
        );
    }

    #[test]
    fn model_add_preserves_configured_model_key_case() {
        assert_eq!(
            route_command_with(
                "/model add Mistral",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
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
                access::PrivilegedAccess::Allowed,
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
                access::PrivilegedAccess::Allowed,
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
                reason: CommandReject::UnknownBoolean {
                    value: "maybe".to_string(),
                },
            })
        );
    }

    #[test]
    fn cost_command_shows_totals_and_sets_the_cost_line() {
        let settings = rooms::RoomSettings::default();
        assert_eq!(
            route_command_with_default_settings("/cost", settings.clone()),
            Route::Command(CommandAction::ShowCost)
        );
        assert_eq!(
            route_command_with_default_settings("/cost status", settings.clone()),
            Route::Command(CommandAction::ShowCost)
        );
        assert_eq!(
            route_command_with_default_settings("/cost off", settings.clone()),
            Route::Command(CommandAction::SetCostLine { enabled: false })
        );
        assert_eq!(
            route_command_with_default_settings("/COST On", settings.clone()),
            Route::Command(CommandAction::SetCostLine { enabled: true })
        );
        assert_eq!(
            route_command_with_default_settings("/cost maybe", settings),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::UnknownBoolean {
                    value: "maybe".to_string()
                }
            })
        );
    }

    #[test]
    fn memory_commands_show_toggle_add_and_remove_facts() {
        assert_eq!(
            route_command("/memory"),
            Route::Command(CommandAction::ShowMemory)
        );
        assert_eq!(
            route_command("/memory status"),
            Route::Command(CommandAction::ShowMemory)
        );
        assert_eq!(
            route_command("/memory OFF"),
            Route::Command(CommandAction::SetMemory { enabled: false })
        );
        assert_eq!(
            route_command("/memory sometimes"),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::UnknownBoolean {
                    value: "sometimes".to_string()
                }
            })
        );
        assert_eq!(
            route_command("/remember I have two kids,\n  aged 7 and 10"),
            Route::Command(CommandAction::Remember {
                fact: "I have two kids, aged 7 and 10".to_string()
            })
        );
        assert_eq!(
            route_command("/remember   "),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::MissingMemoryFact
            })
        );
        assert_eq!(
            route_command("/forget 3"),
            Route::Command(CommandAction::Forget { number: 3 })
        );
        assert_eq!(
            route_command("/forget #2"),
            Route::Command(CommandAction::Forget { number: 2 })
        );
        for (text, value) in [
            ("/forget", None),
            ("/forget 0", Some("0")),
            ("/forget kids", Some("kids")),
        ] {
            assert_eq!(
                route_command(text),
                Route::Command(CommandAction::Reject {
                    reason: CommandReject::InvalidMemoryNumber {
                        value: value.map(str::to_string)
                    }
                }),
                "{text}"
            );
        }
    }

    #[test]
    fn memory_commands_are_owner_only() {
        for text in ["/memory", "/memory on", "/remember I am 41", "/forget 1"] {
            for (access, reason) in [
                (
                    access::PrivilegedAccess::NotOwner,
                    CommandReject::OwnerNotAllowed,
                ),
                (access::PrivilegedAccess::Stale, CommandReject::OwnerStale),
            ] {
                assert_eq!(
                    route_command_with(
                        text,
                        rooms::RoomSettings::default(),
                        None,
                        access,
                        &["claude"],
                    ),
                    Route::Command(CommandAction::Reject {
                        reason: reason.clone()
                    }),
                    "{text}"
                );
            }
        }
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
    fn allow_and_deny_commands_require_owner_and_chat_id() {
        assert_eq!(
            route_command_with(
                "/allow -100",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::AllowChat { chat_id: -100 })
        );
        assert_eq!(
            route_command_with(
                "/deny 42",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::DenyChat { chat_id: 42 })
        );
        assert_eq!(
            route_command_with(
                "/allow",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
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
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
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
                access::PrivilegedAccess::NotOwner,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::OwnerNotAllowed,
            })
        );
        assert_eq!(
            route_command_with(
                "/deny 42",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Stale,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::OwnerStale,
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
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Shutdown)
        );
        assert_eq!(
            route_command_with(
                "/shutdown",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::NotOwner,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::ShutdownNotOwner,
            })
        );
        assert_eq!(
            route_command_with(
                "/shutdown",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Stale,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::ShutdownStale,
            })
        );
    }

    #[test]
    fn ollama_unload_command_uses_owner_gate() {
        assert_eq!(
            route_command_with(
                "/ollama unload",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::UnloadOllama)
        );
        assert_eq!(
            route_command_with(
                "/ollama",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::Allowed,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::UnknownOllamaAction { value: None },
            })
        );
        assert_eq!(
            route_command_with(
                "/ollama unload",
                rooms::RoomSettings::default(),
                None,
                access::PrivilegedAccess::NotOwner,
                &["claude"],
            ),
            Route::Command(CommandAction::Reject {
                reason: CommandReject::OwnerNotAllowed,
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
            access::PrivilegedAccess::Allowed,
            &["claude"],
        )
    }

    fn route_command_with(
        text: &str,
        settings: rooms::RoomSettings,
        bot_username: Option<&str>,
        privileged_access: access::PrivilegedAccess,
        models: &[&str],
    ) -> Route {
        route_command_with_pinned(
            text,
            settings,
            bot_username,
            privileged_access,
            models,
            None,
        )
    }

    fn route_command_with_pinned(
        text: &str,
        settings: rooms::RoomSettings,
        bot_username: Option<&str>,
        privileged_access: access::PrivilegedAccess,
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
            privileged_access,
            capabilities: RoomCapabilities::permissive(),
            chat_kind: ChatKind::Direct,
        };
        route(text, bot_username, &context)
    }

    fn route_in(
        chat_kind: ChatKind,
        text: &str,
        privileged_access: access::PrivilegedAccess,
    ) -> Route {
        let settings = rooms::RoomSettings::default();
        let model_keys = BTreeSet::from(["claude".to_string()]);
        let context = CommandContext {
            settings: &settings,
            default_model: "claude",
            model_keys: &model_keys,
            pinned_model_key: None,
            model_thinking: ThinkingLevel::Medium,
            privileged_access,
            capabilities: RoomCapabilities::permissive(),
            chat_kind,
        };
        route(text, None, &context)
    }

    fn rejected(reason: CommandReject) -> Route {
        Route::Command(CommandAction::Reject { reason })
    }

    #[test]
    fn in_a_shared_group_only_owners_use_commands_but_help_and_id() {
        use access::PrivilegedAccess::{Allowed, NotOwner, Stale};
        let shared = ChatKind::SharedGroup;
        for text in [
            "/new",
            "/role be brief",
            "/model claude",
            "/cost",
            "/group",
            "/pair 1",
        ] {
            assert_eq!(
                route_in(shared, text, NotOwner),
                rejected(CommandReject::SharedGroupOwnerOnly),
                "{text}"
            );
        }
        assert_eq!(
            route_in(shared, "/id", NotOwner),
            Route::Command(CommandAction::ShowChatId)
        );
        assert!(matches!(
            route_in(shared, "/help", NotOwner),
            Route::Command(CommandAction::Help { .. })
        ));
        // An owner's settings commands aren't checked for staleness, as in
        // any other room.
        for access in [Allowed, Stale] {
            assert_eq!(
                route_in(shared, "/new", access),
                Route::Command(CommandAction::ResetHistory)
            );
        }
        for kind in [ChatKind::Direct, ChatKind::PrivateGroup] {
            assert_eq!(
                route_in(kind, "/new", NotOwner),
                Route::Command(CommandAction::ResetHistory),
                "other members still use a private room's commands"
            );
        }
    }

    #[test]
    fn a_shared_group_has_no_conversation_mode_summary_or_memory() {
        use access::PrivilegedAccess::Allowed;
        let shared = ChatKind::SharedGroup;
        assert_eq!(
            route_in(shared, "/mode", Allowed),
            Route::Command(CommandAction::ShowMode {
                current: rooms::ChatMode::Message,
            })
        );
        for text in ["/mode chat", "/mode message", "/new summary"] {
            assert_eq!(
                route_in(shared, text, Allowed),
                rejected(CommandReject::StandaloneInSharedGroup),
                "{text}"
            );
        }
        for text in ["/memory", "/memory on", "/remember I am 41", "/forget 1"] {
            assert_eq!(
                route_in(shared, text, Allowed),
                rejected(CommandReject::MemoryInSharedGroup),
                "{text}"
            );
        }
        assert_eq!(
            route_in(ChatKind::PrivateGroup, "/new summary", Allowed),
            Route::Command(CommandAction::SummarizeChat)
        );
    }

    #[test]
    fn group_command_shows_to_anyone_and_sets_for_owners_in_groups_only() {
        use access::PrivilegedAccess::{Allowed, NotOwner, Stale};
        assert_eq!(
            route_in(ChatKind::PrivateGroup, "/group", NotOwner),
            Route::Command(CommandAction::ShowGroup {
                kind: ChatKind::PrivateGroup,
            })
        );
        assert_eq!(
            route_in(ChatKind::SharedGroup, "/group Private", Allowed),
            Route::Command(CommandAction::SetGroupPrivate { private: true })
        );
        assert_eq!(
            route_in(ChatKind::PrivateGroup, "/group shared", Allowed),
            Route::Command(CommandAction::SetGroupPrivate { private: false })
        );
        assert_eq!(
            route_in(ChatKind::PrivateGroup, "/group shared", NotOwner),
            rejected(CommandReject::OwnerNotAllowed)
        );
        assert_eq!(
            route_in(ChatKind::PrivateGroup, "/group shared", Stale),
            rejected(CommandReject::OwnerStale)
        );
        assert_eq!(
            route_in(ChatKind::Direct, "/group shared", Allowed),
            Route::Command(CommandAction::ShowGroup {
                kind: ChatKind::Direct,
            }),
            "a direct chat stays private"
        );
        assert_eq!(
            route_in(ChatKind::PrivateGroup, "/group public", Allowed),
            rejected(CommandReject::UnknownGroupOption {
                value: "public".to_string(),
            })
        );
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
            privileged_access: access::PrivilegedAccess::NotOwner,
            capabilities,
            chat_kind: ChatKind::Direct,
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
