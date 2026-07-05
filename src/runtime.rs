use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use tellm_anthropic::Anthropic;
use tellm_compat::Compat;
use tellm_config::{Config, ConfigError, ModelConfig, WireFormat, secrets};
use tellm_core::{ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider};
use tellm_gemini::Gemini;
use tellm_openai::Responses;
use tellm_telegram::{Document, IncomingMessage, PhotoSize, Telegram, TelegramError};
use tokio::sync::{Mutex, mpsc};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::{Instant as TokioInstant, sleep, timeout};

use crate::access::{AccessConfig, AccessControl, AccessTime, ChatAccess, PairingAttempt};
use crate::commands::{self, CommandAction, CommandContext, CommandReject, KnownCommand, Route};
use crate::rooms::{self, ChatMode, HistoryReset, RoomState, RoomStates};

const LONG_POLL_TIMEOUT_S: u32 = 20;
const CHAT_QUEUE_SIZE: usize = 32;
const CHAT_TASK_IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const TYPING_INTERVAL: Duration = Duration::from_secs(4);
const OLLAMA_CONNECT_TIMEOUT: Duration = Duration::from_millis(300);
const OLLAMA_START_WAIT: Duration = Duration::from_secs(15);
const OLLAMA_READY_POLL: Duration = Duration::from_millis(250);
const OLLAMA_UNLOAD_TIMEOUT: Duration = Duration::from_secs(5);
const TERMINAL_SECRET_PROMPT_TIMEOUT: Duration = Duration::from_secs(5 * 60);
static OLLAMA_START_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static OLLAMA_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();
static OLLAMA_LOADED_MODELS: OnceLock<Mutex<BTreeSet<OllamaLoadedModel>>> = OnceLock::new();

pub struct Runtime {
    telegram: Telegram,
    config: Arc<Mutex<Config>>,
    rooms: Arc<Mutex<RoomStates>>,
    access: Arc<Mutex<AccessControl>>,
    terminal_prompts: TerminalSecretPrompts,
    terminal_rx: mpsc::Receiver<TerminalCommand>,
    shutdown_tx: mpsc::Sender<ShutdownReason>,
    shutdown_rx: mpsc::Receiver<ShutdownReason>,
}

#[derive(Clone)]
struct RuntimeHandles {
    telegram: Telegram,
    config: Arc<Mutex<Config>>,
    rooms: Arc<Mutex<RoomStates>>,
    access: Arc<Mutex<AccessControl>>,
    terminal_prompts: TerminalSecretPrompts,
    bot_username: Option<String>,
    shutdown_tx: mpsc::Sender<ShutdownReason>,
}

struct ChatDispatcher {
    sender: mpsc::Sender<DispatchMessage>,
    handle: JoinHandle<()>,
}

#[derive(Debug, Clone)]
struct DispatchMessage {
    kind: UpdateKind,
    message: IncomingMessage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateKind {
    Message,
    EditedMessage,
}

impl UpdateKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::EditedMessage => "edited_message",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UpdateRoute {
    Command,
    Model,
    Ignored,
}

impl UpdateRoute {
    fn as_str(self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::Model => "model",
            Self::Ignored => "ignored",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalCommand {
    Reset,
    Shutdown,
}

type TerminalSecretPrompts = Arc<StdMutex<Option<TerminalSecretPrompt>>>;

#[derive(Clone)]
struct TerminalSecretPrompt {
    secret_name: String,
    line_tx: mpsc::UnboundedSender<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OllamaLoadedModel {
    base_url: String,
    model: String,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct OllamaUnloadSummary {
    attempted: usize,
    unloaded: Vec<String>,
    failed: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShutdownReason {
    Telegram,
}

impl Runtime {
    pub fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        let token = secrets::get(secrets::TELEGRAM_BOT_TOKEN).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "missing Telegram bot token secret \"{}\"",
                    secrets::TELEGRAM_BOT_TOKEN
                ),
            )
        })?;
        let room_settings = rooms::load_settings()?;
        let access_config = AccessConfig::from_config(&config);
        let group_chat_ids = allowed_group_chat_ids(&access_config);
        let access = AccessControl::new(access_config, now_access_time());
        warm_configured_provider_secrets(&config);
        print_startup_notice(access.startup_notice());
        print_group_privacy_hints(&group_chat_ids);

        let (shutdown_tx, shutdown_rx) = mpsc::channel(4);
        let terminal_prompts = Arc::new(StdMutex::new(None));

        Ok(Self {
            telegram: Telegram::new(token),
            config: Arc::new(Mutex::new(config)),
            rooms: Arc::new(Mutex::new(RoomStates::from_settings(room_settings))),
            access: Arc::new(Mutex::new(access)),
            terminal_prompts: Arc::clone(&terminal_prompts),
            terminal_rx: spawn_terminal_controls(terminal_prompts),
            shutdown_tx,
            shutdown_rx,
        })
    }

    pub async fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let bot = self.telegram.get_me().await?;
        let bot_username = bot.username;
        if let Some(username) = &bot_username {
            eprintln!(
                "tellm {} running as @{username}. Terminal commands: reset, exit, quit.",
                env!("CARGO_PKG_VERSION")
            );
        } else {
            eprintln!(
                "tellm {} running. Terminal commands: reset, exit, quit.",
                env!("CARGO_PKG_VERSION")
            );
        }

        let handles = RuntimeHandles {
            telegram: self.telegram.clone(),
            config: Arc::clone(&self.config),
            rooms: Arc::clone(&self.rooms),
            access: Arc::clone(&self.access),
            terminal_prompts: Arc::clone(&self.terminal_prompts),
            bot_username,
            shutdown_tx: self.shutdown_tx.clone(),
        };
        let mut dispatchers = BTreeMap::new();
        let mut offset = 0_i64;

        loop {
            tokio::select! {
                command = self.terminal_rx.recv() => {
                    match command {
                        Some(TerminalCommand::Reset) => {
                            self.rooms.lock().await.reset_all_history();
                            eprintln!("All in-memory chat histories cleared; room settings kept.");
                        }
                        Some(TerminalCommand::Shutdown) | None => {
                            eprintln!("Shutdown requested from terminal.");
                            break;
                        }
                    }
                }
                reason = self.shutdown_rx.recv() => {
                    match reason {
                        Some(ShutdownReason::Telegram) => eprintln!("Shutdown requested from Telegram."),
                        None => eprintln!("Shutdown requested."),
                    }
                    break;
                }
                updates = self.telegram.get_updates(offset, LONG_POLL_TIMEOUT_S) => {
                    match updates {
                        Ok(updates) => {
                            for update in updates {
                                offset = offset.max(update.update_id + 1);
                                if let Some(membership) = update.my_chat_member {
                                    self.handle_membership_change(membership, &handles).await;
                                } else if let Some(message) = update.message {
                                    self.handle_update(UpdateKind::Message, message, &handles, &mut dispatchers).await;
                                } else if let Some(message) = update.edited_message {
                                    self.handle_update(UpdateKind::EditedMessage, message, &handles, &mut dispatchers).await;
                                }
                            }
                        }
                        Err(error) => {
                            eprintln!("Telegram getUpdates failed: {error}");
                            sleep(Duration::from_secs(2)).await;
                        }
                    }
                }
            }

            dispatchers.retain(|_, dispatcher| {
                !dispatcher.handle.is_finished() && !dispatcher.sender.is_closed()
            });
        }

        drop(dispatchers);
        stop_started_ollama().await;
        Ok(())
    }

    /// The bot's own membership changed in a chat: being added to an unknown
    /// group arms a pairing code and announces it on the console unless a
    /// proven owner did the adding.
    async fn handle_membership_change(
        &self,
        membership: tellm_telegram::ChatMemberUpdated,
        handles: &RuntimeHandles,
    ) {
        let chat_id = membership.chat.id;
        let label = membership.chat.label();
        match membership.new_chat_member.status.as_str() {
            "member" | "administrator" | "restricted" => {
                let adder = membership.from.as_ref().map(|user| user.id);
                let trusted_adder = match adder {
                    Some(user_id) => {
                        let config = self.config.lock().await;
                        config.telegram.owner_user_ids.contains(&user_id)
                    }
                    None => false,
                };
                if let Some(user_id) = adder.filter(|_| trusted_adder) {
                    let changed = {
                        let mut config = self.config.lock().await;
                        let changed = allow_chat_in_config(&mut config, chat_id);
                        if changed && let Err(error) = tellm_config::save(&config) {
                            eprintln!("failed to persist auto-approved chat {chat_id}: {error}");
                        }
                        changed
                    };
                    {
                        let mut access = self.access.lock().await;
                        access.allow_chat(chat_id);
                    }
                    eprintln!(
                        "Auto-approved chat {label}: added by owner user {user_id}{}.",
                        if changed { "" } else { " (already allowed)" }
                    );
                    if chat_id < 0 {
                        eprintln!("{}", group_privacy_hint(chat_id));
                    }
                    let setup = room_setup_reply(chat_id, false, handles).await;
                    let _ = handles.telegram.send_message(chat_id, &setup).await;
                    return;
                }

                let pairing = {
                    let mut access = self.access.lock().await;
                    access.arm_room(chat_id, now_access_time())
                };
                match pairing {
                    Some(pairing) => {
                        eprintln!(
                            "Added to chat {label}. Pairing code: {} — send /pair {} in that \
                             chat to approve it (or an owner sends /allow {chat_id}).",
                            pairing.code, pairing.code
                        );
                        if chat_id < 0 {
                            eprintln!("{}", group_privacy_hint(chat_id));
                        }
                    }
                    None => eprintln!("Added to already-allowed chat {label}."),
                }
            }
            "left" | "kicked" => {
                eprintln!(
                    "Removed from chat {label}. An owner can use /deny {chat_id} to also \
                     clear its access and room state."
                );
            }
            _ => {}
        }
    }

    async fn handle_update(
        &self,
        kind: UpdateKind,
        message: IncomingMessage,
        handles: &RuntimeHandles,
        dispatchers: &mut BTreeMap<i64, ChatDispatcher>,
    ) {
        let chat_id = message.chat.id;
        let access = {
            let mut access = self.access.lock().await;
            access.check_chat(chat_id)
        };

        match access {
            ChatAccess::Allowed => {
                send_to_chat_worker(chat_id, kind, message, handles, dispatchers).await;
            }
            ChatAccess::Unknown { send_hint } => {
                // Arm (or refresh) this room's pairing code on any contact;
                // print it to the console only when newly issued.
                {
                    let mut access = self.access.lock().await;
                    if let Some(pairing) = access.arm_room(chat_id, now_access_time())
                        && pairing.newly_issued
                    {
                        eprintln!(
                            "Pairing code for chat {chat_id}: {} — send /pair {} in that chat to approve it.",
                            pairing.code, pairing.code
                        );
                    }
                }
                if let Some(code) = pair_code_from_message(&message, handles).await {
                    log_update_route(chat_id, kind, UpdateRoute::Command);
                    let pairer = message.from.as_ref().map(|user| user.id);
                    if let Err(error) = handle_pair_attempt(chat_id, code, pairer, handles).await {
                        eprintln!("pairing attempt for chat {chat_id} failed: {error}");
                    }
                } else if send_hint {
                    log_update_route(chat_id, kind, UpdateRoute::Ignored);
                    let _ = self
                        .telegram
                        .send_message(chat_id, &unknown_chat_hint(chat_id))
                        .await;
                } else {
                    log_update_route(chat_id, kind, UpdateRoute::Ignored);
                }
            }
        }
    }
}

async fn send_to_chat_worker(
    chat_id: i64,
    kind: UpdateKind,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    dispatchers: &mut BTreeMap<i64, ChatDispatcher>,
) {
    dispatchers
        .entry(chat_id)
        .or_insert_with(|| spawn_chat_worker(chat_id, handles.clone()));

    let send_result = dispatchers
        .get(&chat_id)
        .expect("dispatcher inserted")
        .sender
        .send(DispatchMessage {
            kind,
            message: message.clone(),
        })
        .await;

    if send_result.is_err() {
        dispatchers.remove(&chat_id);
        dispatchers.insert(chat_id, spawn_chat_worker(chat_id, handles.clone()));
        let _ = dispatchers
            .get(&chat_id)
            .expect("dispatcher respawned")
            .sender
            .send(DispatchMessage { kind, message })
            .await;
    }
}

fn spawn_chat_worker(chat_id: i64, handles: RuntimeHandles) -> ChatDispatcher {
    let (sender, mut receiver) = mpsc::channel::<DispatchMessage>(CHAT_QUEUE_SIZE);
    let handle = tokio::spawn(async move {
        loop {
            match timeout(CHAT_TASK_IDLE_TIMEOUT, receiver.recv()).await {
                Ok(Some(dispatch)) => {
                    if let Err(error) =
                        handle_allowed_message(chat_id, dispatch.kind, dispatch.message, &handles)
                            .await
                    {
                        eprintln!("chat {chat_id} dispatch failed: {error}");
                        let _ = handles
                            .telegram
                            .send_message(chat_id, &format!("tellm error: {error}"))
                            .await;
                    }
                }
                Ok(None) => break,
                Err(_) => {
                    eprintln!("chat {chat_id} idle; reaping dispatcher task.");
                    break;
                }
            }
        }
    });

    ChatDispatcher { sender, handle }
}

async fn handle_allowed_message(
    chat_id: i64,
    kind: UpdateKind,
    message: IncomingMessage,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    if let Some(text) = message_text(&message) {
        match route_command(
            chat_id,
            text,
            message.date,
            message.from.as_ref().map(|user| user.id),
            handles,
        )
        .await?
        {
            Route::Command(action) => {
                log_update_route(chat_id, kind, UpdateRoute::Command);
                return handle_command(chat_id, action, handles).await;
            }
            Route::Ignore => {
                log_update_route(chat_id, kind, UpdateRoute::Ignored);
                return Ok(());
            }
            Route::UserMessage => {}
        }
    }

    if !message_has_model_input(&message) {
        log_update_route(chat_id, kind, UpdateRoute::Ignored);
        return Ok(());
    }
    log_update_route(chat_id, kind, UpdateRoute::Model);
    handle_model_message(chat_id, message, handles).await
}

async fn route_command(
    chat_id: i64,
    text: &str,
    message_date: i64,
    sender_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<Route, String> {
    let (room, default_model, model_keys, pinned_model_key, capabilities) = {
        let config = handles.config.lock().await;
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id).clone();
        let capabilities = room_capabilities(&config, &room, chat_id);
        (
            room,
            config.default_model.clone(),
            config.models.keys().cloned().collect::<BTreeSet<_>>(),
            pinned_model_key(&config, chat_id).map(str::to_string),
            capabilities,
        )
    };
    let shutdown_access = {
        let access = handles.access.lock().await;
        access.check_privileged(
            sender_user_id,
            u64::try_from(message_date).unwrap_or_default(),
            now_access_time(),
        )
    };
    let context = CommandContext {
        bot_username: handles.bot_username.as_deref(),
        room: &room,
        default_model: &default_model,
        model_keys: &model_keys,
        pinned_model_key: pinned_model_key.as_deref(),
        shutdown_access,
        capabilities,
    };
    Ok(commands::route(text, &context))
}

async fn handle_command(
    chat_id: i64,
    action: CommandAction,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    match action {
        CommandAction::ResetHistory => {
            let mut rooms = handles.rooms.lock().await;
            rooms.get_or_default(chat_id).reset_history();
            handles
                .telegram
                .send_message(chat_id, "Started a new chat.")
                .await
                .map_err(|error| error.to_string())
        }
        CommandAction::ShowChatId => {
            send_command_reply(handles, chat_id, &chat_id_reply(chat_id)).await
        }
        CommandAction::ShowMode { current } => {
            send_command_reply(handles, chat_id, &format!("Mode: {}", mode_name(current))).await
        }
        CommandAction::SetMode { mode } => {
            mutate_room(handles, chat_id, |room| {
                room.settings.mode = mode;
                if mode == ChatMode::Message {
                    room.reset_history();
                }
            })
            .await?;
            send_command_reply(
                handles,
                chat_id,
                &format!("Mode set to {}.", mode_name(mode)),
            )
            .await
        }
        CommandAction::ShowModel {
            selected,
            effective,
            pinned,
            available,
        } => {
            let reply = format_model_status(selected, effective, pinned, available);
            send_command_reply(handles, chat_id, &reply).await
        }
        CommandAction::SetModel { model_key } => {
            mutate_room(handles, chat_id, |room| {
                room.settings.model_key = Some(model_key.clone());
                room.wire_format = None;
                room.reset_history();
            })
            .await?;
            send_command_reply(handles, chat_id, &model_set_reply(&model_key)).await
        }
        CommandAction::PinModel { model_key } => {
            {
                let mut config = handles.config.lock().await;
                for model in config.models.values_mut() {
                    model.telegram_chat_ids.retain(|pinned| *pinned != chat_id);
                }
                if let Some(model) = config.models.get_mut(&model_key) {
                    model.telegram_chat_ids.push(chat_id);
                }
                tellm_config::save(&config).map_err(|error| error.to_string())?;
            }
            mutate_room(handles, chat_id, |room| {
                room.wire_format = None;
                room.reset_history();
            })
            .await?;
            send_command_reply(
                handles,
                chat_id,
                &format!(
                    "Room pinned to {model_key}. Chat history reset. /model unpin releases it."
                ),
            )
            .await
        }
        CommandAction::UnpinModel => {
            let was_pinned = {
                let mut config = handles.config.lock().await;
                let mut was_pinned = false;
                for model in config.models.values_mut() {
                    let before = model.telegram_chat_ids.len();
                    model.telegram_chat_ids.retain(|pinned| *pinned != chat_id);
                    was_pinned |= model.telegram_chat_ids.len() != before;
                }
                // A chat allowed only via its pin must stay allowed after
                // unpinning, or it silently loses access on restart.
                if was_pinned && !config.telegram.allowed_chat_ids.contains(&chat_id) {
                    config.telegram.allowed_chat_ids.push(chat_id);
                }
                if was_pinned {
                    tellm_config::save(&config).map_err(|error| error.to_string())?;
                }
                was_pinned
            };
            let reply = if was_pinned {
                "Room unpinned. /model KEY now switches models here."
            } else {
                "This room isn't pinned."
            };
            send_command_reply(handles, chat_id, reply).await
        }
        CommandAction::ShowModelCatalog => {
            let configured: BTreeSet<String> = {
                let config = handles.config.lock().await;
                config.models.keys().cloned().collect()
            };
            send_command_reply(handles, chat_id, &model_catalog_reply(&configured)).await
        }
        CommandAction::AddModel { preset_key } => {
            if let Some(model_key) = {
                let config = handles.config.lock().await;
                configured_model_key(&config, &preset_key)
            } {
                return handle_configured_model_secret(chat_id, model_key, handles).await;
            }

            let Some(preset) = crate::wizard::preset_by_key(&preset_key) else {
                return handle_configured_model_secret(chat_id, preset_key, handles).await;
            };

            let already = {
                let mut config = handles.config.lock().await;
                if config.models.contains_key(preset.key) {
                    true
                } else {
                    config.models.insert(
                        preset.key.to_string(),
                        crate::wizard::model_config_from_preset(preset),
                    );
                    tellm_config::save(&config).map_err(|error| error.to_string())?;
                    false
                }
            };

            let base_reply = model_add_base_reply(already, preset);
            if secrets::get(preset.api_key_secret).is_some() {
                let reply = model_add_key_ready_reply(&base_reply, preset);
                send_command_reply(handles, chat_id, &reply).await
            } else {
                prompt_for_model_secret(handles, chat_id, preset, &base_reply).await
            }
        }
        CommandAction::ShowRole { current } => {
            let role = current.unwrap_or_else(|| "(none)".to_string());
            send_command_reply(handles, chat_id, &format!("Role: {role}")).await
        }
        CommandAction::SetRole { role } => {
            let cleared = role.is_none();
            mutate_room(handles, chat_id, |room| {
                room.settings.role = role.clone();
                room.reset_history();
            })
            .await?;
            let reply = if cleared {
                "Role cleared. Chat history reset.".to_string()
            } else {
                "Role updated. Chat history reset.".to_string()
            };
            send_command_reply(handles, chat_id, &reply).await
        }
        CommandAction::ShowReasoning { current } => {
            send_command_reply(handles, chat_id, &format!("Reasoning: {current:?}.")).await
        }
        CommandAction::SetReasoning { thinking } => {
            mutate_room(handles, chat_id, |room| {
                room.settings.thinking = thinking;
            })
            .await?;
            send_command_reply(handles, chat_id, &format!("Reasoning set to {thinking:?}.")).await
        }
        CommandAction::ShowWebSearch { enabled } => {
            let state = if enabled { "on" } else { "off" };
            send_command_reply(handles, chat_id, &format!("Web search: {state}.")).await
        }
        CommandAction::SetWebSearch { enabled } => {
            mutate_room(handles, chat_id, |room| {
                room.settings.web_search = enabled;
            })
            .await?;
            let state = if enabled { "on" } else { "off" };
            send_command_reply(handles, chat_id, &format!("Web search set to {state}.")).await
        }
        CommandAction::ShowImageGeneration { enabled } => {
            let state = if enabled { "on" } else { "off" };
            send_command_reply(handles, chat_id, &format!("Image generation: {state}.")).await
        }
        CommandAction::SetImageGeneration { enabled } => {
            mutate_room(handles, chat_id, |room| {
                room.settings.image_generation = enabled;
            })
            .await?;
            let state = if enabled { "on" } else { "off" };
            send_command_reply(
                handles,
                chat_id,
                &format!("Image generation set to {state}."),
            )
            .await
        }
        CommandAction::AllowChat {
            chat_id: target_chat_id,
        } => handle_allow_chat(chat_id, target_chat_id, handles).await,
        CommandAction::DenyChat {
            chat_id: target_chat_id,
        } => handle_deny_chat(chat_id, target_chat_id, handles).await,
        CommandAction::Pair { code } => handle_pair_attempt(chat_id, code, None, handles).await,
        CommandAction::UnloadOllama => {
            let summary = unload_tracked_ollama_models().await;
            send_command_reply(handles, chat_id, &ollama_unload_reply(&summary)).await
        }
        CommandAction::Shutdown => {
            send_command_reply(handles, chat_id, "Shutting down.").await?;
            handles
                .shutdown_tx
                .send(ShutdownReason::Telegram)
                .await
                .map_err(|error| error.to_string())
        }
        CommandAction::Help { pinned_model_key } => {
            let help = help_text(pinned_model_key.as_deref());
            send_command_reply(handles, chat_id, &help).await
        }
        CommandAction::Reject { reason, .. } => {
            send_command_reply(handles, chat_id, &format_reject(reason)).await
        }
    }
}

async fn handle_model_message(
    chat_id: i64,
    message: IncomingMessage,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let input = content_parts_from_message(&handles.telegram, &message).await?;
    if input.is_empty() {
        return Ok(());
    }

    let (model_config, request, before_state, reset_notice) =
        build_chat_request(chat_id, input, handles).await?;
    if let Some(notice) = reset_notice {
        let _ = handles.telegram.send_message(chat_id, &notice).await;
    }

    let typing = spawn_typing_indicator(handles.telegram.clone(), chat_id);
    let response = dispatch_provider(&model_config, &request).await;
    typing.abort();

    match response {
        Ok(response) => {
            {
                let mut rooms = handles.rooms.lock().await;
                rooms
                    .get_or_default(chat_id)
                    .append_turn(model_config.wire_format, response.turn_items.clone());
            }
            send_model_response(&handles.telegram, chat_id, response).await
        }
        Err(error) => {
            {
                let mut rooms = handles.rooms.lock().await;
                *rooms.get_or_default(chat_id) = before_state;
            }
            let _ = handles
                .telegram
                .send_message(chat_id, &provider_error_reply(&error))
                .await;
            Err(error)
        }
    }
}

/// The room's model can change after a toggle was accepted, so a capability
/// error can still surface per-message; suggest the off switch.
fn provider_error_reply(error: &str) -> String {
    // Provider error texts use both spellings ("image generation" from
    // Anthropic/compat, "image_generation" from the xAI backstop) — match
    // both.
    let mut reply = format!("Provider error: {error}");
    if error.contains("image generation") || error.contains("image_generation") {
        reply.push_str("\nTip: /imagegen off");
    } else if error.contains("web search") || error.contains("web_search") {
        reply.push_str("\nTip: /websearch off");
    }
    reply
}

async fn build_chat_request(
    chat_id: i64,
    input: Vec<ContentPart>,
    handles: &RuntimeHandles,
) -> Result<(ModelConfig, ChatRequest, RoomState, Option<String>), String> {
    let config = handles.config.lock().await;
    let mut rooms = handles.rooms.lock().await;
    let room = rooms.get_or_default(chat_id);
    let model_key = selected_model_key(&config, room, chat_id);
    let model_config = config
        .models
        .get(&model_key)
        .cloned()
        .ok_or_else(|| format!("configured model \"{model_key}\" was not found"))?;
    let before_state = room.clone();
    let reset = room.begin_turn(model_config.wire_format);
    let reset_notice = reset_notice(reset);
    let request = chat_request_from_room(room, &model_config, input);

    Ok((model_config, request, before_state, reset_notice))
}

fn chat_request_from_room(
    room: &RoomState,
    model_config: &ModelConfig,
    input: Vec<ContentPart>,
) -> ChatRequest {
    ChatRequest {
        model: model_config.model_name.clone(),
        system: room.settings.role.clone(),
        history: match room.settings.mode {
            ChatMode::Chat => room.history.clone(),
            ChatMode::Message => Vec::new(),
        },
        input,
        thinking: room.settings.thinking,
        web_search: room.settings.web_search,
        image_generation: room.settings.image_generation,
        max_tokens: None,
    }
}

async fn dispatch_provider(
    model: &ModelConfig,
    request: &ChatRequest,
) -> Result<ChatResponse, String> {
    match model.wire_format {
        WireFormat::Anthropic => {
            let api_key = required_api_key(model)?;
            Anthropic::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Responses => {
            let api_key = required_api_key(model)?;
            Responses::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Compat => {
            let base_url = model
                .base_url
                .clone()
                .ok_or_else(|| "compat model is missing base_url".to_string())?;
            ensure_local_ollama_ready(&base_url).await?;
            remember_local_ollama_model(&base_url, &request.model).await;
            let api_key = compat_api_key(model)?;
            Compat::new(api_key, base_url)
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Gemini => {
            let api_key = required_api_key(model)?;
            Gemini::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
    }
}

async fn ensure_local_ollama_ready(base_url: &str) -> Result<(), String> {
    let Some(addr) = local_ollama_addr(base_url) else {
        return Ok(());
    };

    if tcp_connects(addr.clone()).await? {
        return Ok(());
    }

    let _start_guard = ollama_start_lock().lock().await;
    if tcp_connects(addr.clone()).await? {
        return Ok(());
    }

    eprintln!("local Ollama endpoint {base_url} is not reachable; starting `ollama serve`");
    start_ollama_serve().await?;

    let deadline = TokioInstant::now() + OLLAMA_START_WAIT;
    loop {
        sleep(OLLAMA_READY_POLL).await;
        if tcp_connects(addr.clone()).await? {
            eprintln!("local Ollama endpoint {base_url} is ready");
            return Ok(());
        }
        if TokioInstant::now() >= deadline {
            return Err(format!(
                "local Ollama endpoint {base_url} did not become reachable after {}s",
                OLLAMA_START_WAIT.as_secs()
            ));
        }
    }
}

fn ollama_start_lock() -> &'static Mutex<()> {
    OLLAMA_START_LOCK.get_or_init(|| Mutex::new(()))
}

fn ollama_child() -> &'static Mutex<Option<Child>> {
    OLLAMA_CHILD.get_or_init(|| Mutex::new(None))
}

fn ollama_loaded_models() -> &'static Mutex<BTreeSet<OllamaLoadedModel>> {
    OLLAMA_LOADED_MODELS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

async fn tcp_connects(addr: String) -> Result<bool, String> {
    spawn_blocking(move || {
        let addrs = addr
            .to_socket_addrs()
            .map_err(|error| format!("invalid Ollama listen address {addr}: {error}"))?;
        for socket_addr in addrs {
            if TcpStream::connect_timeout(&socket_addr, OLLAMA_CONNECT_TIMEOUT).is_ok() {
                return Ok(true);
            }
        }
        Ok(false)
    })
    .await
    .map_err(|error| format!("Ollama readiness check task failed: {error}"))?
}

async fn start_ollama_serve() -> Result<(), String> {
    let child = spawn_blocking(|| {
        ProcessCommand::new("ollama")
            .arg("serve")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    })
    .await
    .map_err(|error| format!("failed to start `ollama serve`: {error}"))?
    .map_err(|error| {
        format!("local Ollama is not running and `ollama serve` could not be started: {error}")
    })?;
    let pid = child.id();
    *ollama_child().lock().await = Some(child);
    eprintln!("started `ollama serve` with pid {pid}");
    Ok(())
}

async fn stop_started_ollama() {
    let Some(child) = ollama_child().lock().await.take() else {
        return;
    };
    unload_started_ollama_models().await;
    match spawn_blocking(move || stop_ollama_child(child)).await {
        Ok(Ok(message)) => eprintln!("{message}"),
        Ok(Err(error)) => eprintln!("failed to stop spawned Ollama process: {error}"),
        Err(error) => eprintln!("failed to join Ollama shutdown task: {error}"),
    }
}

async fn remember_local_ollama_model(base_url: &str, model: &str) {
    if local_ollama_addr(base_url).is_none() {
        return;
    }

    ollama_loaded_models()
        .lock()
        .await
        .insert(OllamaLoadedModel {
            base_url: base_url.to_string(),
            model: model.to_string(),
        });
}

async fn unload_started_ollama_models() {
    let summary = unload_tracked_ollama_models().await;
    for model in summary.unloaded {
        eprintln!("unloaded Ollama model {model}");
    }
    for (model, error) in summary.failed {
        eprintln!("failed to unload Ollama model {model}: {error}");
    }
}

async fn unload_tracked_ollama_models() -> OllamaUnloadSummary {
    let models = {
        let models = ollama_loaded_models().lock().await;
        models.iter().cloned().collect::<Vec<_>>()
    };
    let mut summary = OllamaUnloadSummary {
        attempted: models.len(),
        ..OllamaUnloadSummary::default()
    };

    for model in models {
        match unload_ollama_model(model.base_url.clone(), model.model.clone()).await {
            Ok(()) => {
                ollama_loaded_models().lock().await.remove(&model);
                summary.unloaded.push(model.model);
            }
            Err(error) => summary.failed.push((model.model, error)),
        }
    }

    summary
}

async fn unload_ollama_model(base_url: String, model: String) -> Result<(), String> {
    let addr = local_ollama_addr(&base_url)
        .ok_or_else(|| format!("not a local Ollama endpoint: {base_url}"))?;
    spawn_blocking(move || unload_ollama_model_blocking(&addr, &model))
        .await
        .map_err(|error| format!("Ollama unload task failed: {error}"))?
}

fn unload_ollama_model_blocking(addr: &str, model: &str) -> Result<(), String> {
    let mut stream = connect_ollama_tcp(addr)?;
    stream
        .set_read_timeout(Some(OLLAMA_UNLOAD_TIMEOUT))
        .map_err(|error| format!("could not set Ollama unload read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(OLLAMA_UNLOAD_TIMEOUT))
        .map_err(|error| format!("could not set Ollama unload write timeout: {error}"))?;

    let request = ollama_unload_request(addr, model);
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("could not send Ollama unload request: {error}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|error| format!("could not read Ollama unload response: {error}"))?;
    if http_response_is_success(&response) {
        return Ok(());
    }

    Err(format!(
        "Ollama unload returned {}",
        response.lines().next().unwrap_or("an empty HTTP response")
    ))
}

fn connect_ollama_tcp(addr: &str) -> Result<TcpStream, String> {
    let addrs = addr
        .to_socket_addrs()
        .map_err(|error| format!("invalid Ollama listen address {addr}: {error}"))?;
    for socket_addr in addrs {
        if let Ok(stream) = TcpStream::connect_timeout(&socket_addr, OLLAMA_CONNECT_TIMEOUT) {
            return Ok(stream);
        }
    }
    Err(format!("could not connect to local Ollama endpoint {addr}"))
}

fn ollama_unload_request(addr: &str, model: &str) -> String {
    // Checked 2026-07-05 against docs.ollama.com/api/generate:
    // keep_alive accepts 0 to unload a model immediately.
    let body = serde_json::json!({
        "model": model,
        "prompt": "",
        "stream": false,
        "keep_alive": 0,
    })
    .to_string();
    format!(
        "POST /api/generate HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    )
}

fn http_response_is_success(response: &str) -> bool {
    matches!(
        response.lines().next().and_then(|line| line.split(' ').nth(1)),
        Some(code) if code.starts_with('2')
    )
}

fn ollama_unload_reply(summary: &OllamaUnloadSummary) -> String {
    if summary.attempted == 0 {
        return "No local Ollama models have been used by this tellm session.".to_string();
    }

    let mut parts = Vec::new();
    if !summary.unloaded.is_empty() {
        parts.push(format!(
            "Unloaded local Ollama model{}: {}.",
            plural(summary.unloaded.len()),
            summary.unloaded.join(", ")
        ));
    }
    if !summary.failed.is_empty() {
        let failures = summary
            .failed
            .iter()
            .map(|(model, error)| format!("{model} ({error})"))
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!(
            "Failed to unload local Ollama model{}: {failures}.",
            plural(summary.failed.len())
        ));
    }

    parts.join(" ")
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn stop_ollama_child(mut child: Child) -> Result<String, String> {
    let pid = child.id();
    if let Some(status) = child
        .try_wait()
        .map_err(|error| format!("could not inspect pid {pid}: {error}"))?
    {
        return Ok(format!(
            "`ollama serve` pid {pid} already exited with {status}"
        ));
    }

    child
        .kill()
        .map_err(|error| format!("could not kill pid {pid}: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for pid {pid}: {error}"))?;
    Ok(format!(
        "stopped tellm-started `ollama serve` pid {pid} with {status}"
    ))
}

fn local_ollama_addr(base_url: &str) -> Option<String> {
    let rest = base_url.strip_prefix("http://")?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port) = split_host_port(authority)?;
    if port != "11434" {
        return None;
    }
    let normalized = host.trim_start_matches('[').trim_end_matches(']');
    match normalized {
        "localhost" | "127.0.0.1" | "::1" => Some(format!("{host}:{port}")),
        _ => None,
    }
}

fn split_host_port(authority: &str) -> Option<(&str, &str)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let closing = rest.find(']')?;
        let host = &authority[..closing + 2];
        let port = rest[closing + 1..].strip_prefix(':')?;
        return Some((host, port));
    }
    authority.rsplit_once(':')
}

async fn content_parts_from_message(
    telegram: &Telegram,
    message: &IncomingMessage,
) -> Result<Vec<ContentPart>, String> {
    let mut parts = Vec::new();
    if let Some(text) = message_text(message)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        parts.push(ContentPart::Text {
            text: text.to_string(),
        });
    }

    if let Some(photo) = largest_photo(message.photo.as_deref()) {
        let bytes = telegram
            .get_file_bytes(&photo.file_id)
            .await
            .map_err(|error| error.to_string())?;
        parts.push(ContentPart::Image {
            media_type: "image/jpeg".to_string(),
            base64: BASE64.encode(bytes),
        });
    }

    if let Some(document) = &message.document {
        let bytes = telegram
            .get_file_bytes(&document.file_id)
            .await
            .map_err(|error| error.to_string())?;
        let media_type = document
            .mime_type
            .clone()
            .unwrap_or_else(|| "application/octet-stream".to_string());
        if is_text_document(document, &media_type) {
            let text = String::from_utf8(bytes)
                .map_err(|_| "text document was not valid UTF-8".to_string())?;
            parts.push(ContentPart::Text { text });
        } else if media_type.starts_with("image/") {
            parts.push(ContentPart::Image {
                media_type,
                base64: BASE64.encode(bytes),
            });
        } else {
            parts.push(ContentPart::Document {
                media_type,
                base64: BASE64.encode(bytes),
                name: document.file_name.clone(),
            });
        }
    }

    Ok(parts)
}

async fn send_model_response(
    telegram: &Telegram,
    chat_id: i64,
    response: ChatResponse,
) -> Result<(), String> {
    if !response.text.trim().is_empty() || response.images.is_empty() {
        telegram
            .send_message(chat_id, &response.text)
            .await
            .map_err(|error| error.to_string())?;
    }

    for image in response.images {
        let bytes = decode_image(image)?;
        telegram
            .send_photo(chat_id, bytes)
            .await
            .map_err(|error| error.to_string())?;
    }

    Ok(())
}

async fn handle_allow_chat(
    admin_chat_id: i64,
    target_chat_id: i64,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let changed = {
        let mut config = handles.config.lock().await;
        let changed = allow_chat_in_config(&mut config, target_chat_id);
        if changed {
            tellm_config::save(&config).map_err(|error| error.to_string())?;
        }
        changed
    };

    {
        let mut access = handles.access.lock().await;
        access.allow_chat(target_chat_id);
    }

    if target_chat_id < 0 {
        eprintln!("{}", group_privacy_hint(target_chat_id));
    }

    // The approved room gets the setup prompt too — approval via /allow must
    // feel the same as approval via /pair (best-effort: the bot may not be a
    // member of the target chat yet).
    if changed && target_chat_id != admin_chat_id {
        let setup = room_setup_reply(target_chat_id, false, handles).await;
        let _ = handles.telegram.send_message(target_chat_id, &setup).await;
    }

    let reply = if changed {
        format!("Allowed chat {target_chat_id}.")
    } else {
        format!("Chat {target_chat_id} is already allowed.")
    };
    send_command_reply(handles, admin_chat_id, &reply).await
}

async fn handle_deny_chat(
    admin_chat_id: i64,
    target_chat_id: i64,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let config_result = {
        let mut config = handles.config.lock().await;
        let result = deny_chat_in_config(&mut config, target_chat_id);
        if result.changed() {
            tellm_config::save(&config).map_err(|error| error.to_string())?;
        }
        result
    };

    {
        let mut rooms = handles.rooms.lock().await;
        rooms.remove(target_chat_id);
        rooms::save_settings(&rooms.settings()).map_err(|error| error.to_string())?;
    }

    {
        let mut access = handles.access.lock().await;
        access.deny_chat(target_chat_id);
    }

    send_command_reply(
        handles,
        admin_chat_id,
        &format_deny_chat_reply(target_chat_id, &config_result),
    )
    .await
}

async fn handle_pair_attempt(
    chat_id: i64,
    code: String,
    pairer_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let attempt = {
        let mut access = handles.access.lock().await;
        access.attempt_pair(chat_id, &code, now_access_time())
    };

    match attempt {
        PairingAttempt::Paired => {
            let became_owner = persist_paired_chat(chat_id, pairer_user_id, handles)
                .await
                .map_err(|error| error.to_string())?;
            if let Some(user_id) = pairer_user_id.filter(|_| became_owner) {
                // Apply to the live access set too, or the new owner is
                // rejected until restart.
                let mut access = handles.access.lock().await;
                access.add_owner(user_id);
            }
            if chat_id < 0 {
                eprintln!("{}", group_privacy_hint(chat_id));
            }
            let setup = room_setup_reply(chat_id, became_owner, handles).await;
            send_command_reply(handles, chat_id, &setup).await
        }
        PairingAttempt::AlreadyAllowed => {
            send_command_reply(handles, chat_id, "This chat is already paired.").await
        }
        PairingAttempt::Rejected { attempts_remaining } => {
            let current_code = {
                let access = handles.access.lock().await;
                access.room_code(chat_id).map(str::to_string)
            };
            if let Some(code) = current_code {
                eprintln!("Current pairing code for chat {chat_id}: {code}");
            }
            send_command_reply(
                handles,
                chat_id,
                &format!("Pairing code rejected. Attempts remaining: {attempts_remaining}."),
            )
            .await
        }
        PairingAttempt::LockedOut { until } => {
            send_command_reply(
                handles,
                chat_id,
                &format!(
                    "Too many pairing attempts. Try again after Unix time {}.",
                    until.as_unix_seconds()
                ),
            )
            .await
        }
    }
}

async fn pair_code_from_message(
    message: &IncomingMessage,
    handles: &RuntimeHandles,
) -> Option<String> {
    let text = message_text(message)?;
    let config = handles.config.lock().await;
    let room = RoomState::new(Default::default());
    let model_keys = config.models.keys().cloned().collect::<BTreeSet<_>>();
    let context = CommandContext {
        bot_username: handles.bot_username.as_deref(),
        room: &room,
        default_model: &config.default_model,
        model_keys: &model_keys,
        pinned_model_key: None,
        shutdown_access: crate::access::ShutdownAccess::NotAdmin,
        capabilities: commands::RoomCapabilities::permissive(),
    };
    match commands::route(text, &context) {
        Route::Command(CommandAction::Pair { code }) => Some(code),
        _ => None,
    }
}

/// Compute what the room's effective model can honor. Statically
/// knowable from the wire format plus the xAI endpoint check; model-level
/// variation inside a capable format stays a request-time error.
fn room_capabilities(
    config: &Config,
    room: &RoomState,
    chat_id: i64,
) -> commands::RoomCapabilities {
    let model_key = selected_model_key(config, room, chat_id);
    let Some(model) = config.models.get(&model_key) else {
        return commands::RoomCapabilities::permissive();
    };

    let (web_search, image_generation, endpoint) = match model.wire_format {
        WireFormat::Anthropic => (true, false, "Anthropic Messages"),
        WireFormat::Responses => {
            if tellm_openai::is_xai_endpoint(model.base_url.as_deref(), &model.model_name) {
                (true, false, "xAI Responses")
            } else {
                (true, true, "OpenAI Responses")
            }
        }
        WireFormat::Compat => (false, false, "chat-completions endpoint"),
        WireFormat::Gemini => (
            true,
            tellm_gemini::is_image_generation_model(&model.model_name),
            "Gemini Interactions",
        ),
    };

    commands::RoomCapabilities {
        web_search,
        image_generation,
        model_key,
        endpoint: endpoint.to_string(),
    }
}

/// Persist a newly paired chat and register the pairing user as an owner.
/// Privileged commands are user-gated, so a fresh install gets a fully
/// empowered owner from the first successful /pair. Returns whether a new
/// owner was recorded.
async fn persist_paired_chat(
    chat_id: i64,
    pairer_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<bool, ConfigError> {
    let mut config = handles.config.lock().await;
    let mut changed = false;
    if !config.telegram.allowed_chat_ids.contains(&chat_id) {
        config.telegram.allowed_chat_ids.push(chat_id);
        changed = true;
    }
    // Code pairing proves console access: record the pairing USER as an
    // owner. Owners are the privilege concept — there is no
    // chat-scoped admin anymore.
    let mut became_owner = false;
    if let Some(user_id) = pairer_user_id
        && !config.telegram.owner_user_ids.contains(&user_id)
    {
        config.telegram.owner_user_ids.push(user_id);
        became_owner = true;
        changed = true;
    }
    if changed {
        tellm_config::save(&config)?;
    }
    Ok(became_owner)
}

/// The provider catalog for /model add: every preset as a tappable command,
/// with configured/key status.
fn model_catalog_reply(configured: &BTreeSet<String>) -> String {
    let mut reply = "Provider catalog:\n\n".to_string();
    for preset in crate::wizard::provider_presets() {
        let status = if configured.contains(preset.key) {
            "configured"
        } else if secrets::get(preset.api_key_secret).is_some() {
            "key ready"
        } else {
            "needs key"
        };
        reply.push_str(&format!(
            "- /model add {} — {} ({status})\n",
            preset.key, preset.label
        ));
    }
    reply.push_str(
        "\nUse /model add KEY to add a preset, or to enter the api_key_secret for a custom model already in config.toml. If its key is missing, tellm asks for it in this terminal.",
    );
    reply
}

async fn handle_configured_model_secret(
    chat_id: i64,
    model_key: String,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let configured_model_secret = {
        let config = handles.config.lock().await;
        config
            .models
            .get(&model_key)
            .map(|model| model.api_key_secret.clone())
    };

    match configured_model_secret {
        Some(Some(secret_name)) => {
            let base_reply = configured_model_base_reply(&model_key);
            if secrets::get(&secret_name).is_some() {
                let reply = configured_model_key_ready_reply(&base_reply, &model_key);
                send_command_reply(handles, chat_id, &reply).await
            } else {
                prompt_for_configured_model_secret(
                    handles,
                    chat_id,
                    model_key,
                    secret_name,
                    &base_reply,
                )
                .await
            }
        }
        Some(None) => {
            let reply = configured_model_has_no_secret_reply(&model_key);
            send_command_reply(handles, chat_id, &reply).await
        }
        None => {
            let configured: BTreeSet<String> = {
                let config = handles.config.lock().await;
                config.models.keys().cloned().collect()
            };
            let reply = format!(
                "Unknown provider preset or configured model \"{model_key}\".\n\n{}",
                model_catalog_reply(&configured)
            );
            send_command_reply(handles, chat_id, &reply).await
        }
    }
}

async fn prompt_for_model_secret(
    handles: &RuntimeHandles,
    chat_id: i64,
    preset: &'static crate::wizard::ProviderPreset,
    base_reply: &str,
) -> Result<(), String> {
    let prompt_notice = model_add_key_prompt_reply(base_reply, preset);

    let reply = match prompt_and_store_secret(
        handles,
        chat_id,
        &prompt_notice,
        preset.api_key_secret,
    )
    .await
    {
        Ok(Some(destination)) => model_add_key_stored_reply(preset, destination),
        Ok(None) => model_add_key_skipped_reply(preset),
        Err(error) => model_add_key_prompt_failed_reply(preset, &error),
    };
    send_command_reply(handles, chat_id, &reply).await
}

async fn prompt_for_configured_model_secret(
    handles: &RuntimeHandles,
    chat_id: i64,
    model_key: String,
    secret_name: String,
    base_reply: &str,
) -> Result<(), String> {
    let prompt_notice = configured_model_key_prompt_reply(base_reply, &secret_name);

    let reply = match prompt_and_store_secret(handles, chat_id, &prompt_notice, secret_name.clone())
        .await
    {
        Ok(Some(destination)) => configured_model_key_stored_reply(&model_key, destination),
        Ok(None) => configured_model_key_skipped_reply(&model_key),
        Err(error) => configured_model_key_prompt_failed_reply(&model_key, &secret_name, &error),
    };
    send_command_reply(handles, chat_id, &reply).await
}

async fn prompt_and_store_secret(
    handles: &RuntimeHandles,
    chat_id: i64,
    prompt_notice: &str,
    secret_name: impl Into<String>,
) -> Result<Option<secrets::SecretDestination>, String> {
    let secret_name = secret_name.into();
    let (mut lines, _guard) =
        reserve_terminal_secret_prompt(&handles.terminal_prompts, secret_name.clone())?;

    send_command_reply(handles, chat_id, prompt_notice).await?;
    eprintln!(
        "Telegram requested secret {secret_name}. Enter it in this terminal (visible, like first-run setup); press Enter to skip."
    );
    print_terminal_secret_prompt(&secret_name);

    loop {
        let line = match timeout(TERMINAL_SECRET_PROMPT_TIMEOUT, lines.recv()).await {
            Ok(Some(line)) => line,
            Ok(None) => {
                return Err("terminal input closed before a value was entered".to_string());
            }
            Err(_) => {
                return Err("terminal prompt expired; run /model add again".to_string());
            }
        };
        match store_prompted_secret(&secret_name, &line)? {
            PromptedSecret::Stored(destination) => return Ok(Some(destination)),
            PromptedSecret::Skipped => return Ok(None),
            PromptedSecret::Retry(message) => {
                eprintln!("{message}");
                print_terminal_secret_prompt(&secret_name);
            }
        }
    }
}

fn model_add_base_reply(already: bool, preset: &crate::wizard::ProviderPreset) -> String {
    if already {
        format!("{} is already configured.", preset.key)
    } else {
        format!("Added {} ({}).", preset.key, preset.label)
    }
}

fn model_add_key_ready_reply(base_reply: &str, preset: &crate::wizard::ProviderPreset) -> String {
    model_key_ready_reply(base_reply, preset.key)
}

fn model_add_key_prompt_reply(base_reply: &str, preset: &crate::wizard::ProviderPreset) -> String {
    model_key_prompt_reply(base_reply, preset.api_key_secret)
}

fn model_add_key_stored_reply(
    preset: &crate::wizard::ProviderPreset,
    destination: secrets::SecretDestination,
) -> String {
    model_key_stored_reply(preset.key, destination)
}

fn model_add_key_skipped_reply(preset: &crate::wizard::ProviderPreset) -> String {
    model_key_skipped_reply(preset.key)
}

fn model_add_key_prompt_failed_reply(
    preset: &crate::wizard::ProviderPreset,
    error: &str,
) -> String {
    model_key_prompt_failed_reply(preset.key, preset.api_key_secret, error)
}

fn configured_model_base_reply(model_key: &str) -> String {
    format!("{model_key} is already configured.")
}

fn configured_model_key_ready_reply(base_reply: &str, model_key: &str) -> String {
    model_key_ready_reply(base_reply, model_key)
}

fn configured_model_key_prompt_reply(base_reply: &str, secret_name: &str) -> String {
    model_key_prompt_reply(base_reply, secret_name)
}

fn configured_model_key_stored_reply(
    model_key: &str,
    destination: secrets::SecretDestination,
) -> String {
    model_key_stored_reply(model_key, destination)
}

fn configured_model_key_skipped_reply(model_key: &str) -> String {
    model_key_skipped_reply(model_key)
}

fn configured_model_key_prompt_failed_reply(
    model_key: &str,
    secret_name: &str,
    error: &str,
) -> String {
    model_key_prompt_failed_reply(model_key, secret_name, error)
}

fn configured_model_has_no_secret_reply(model_key: &str) -> String {
    format!(
        "{model_key} is already configured with no API key prompt. Select it here with /model {model_key}. If this endpoint needs a key, add api_key_secret = \"...\" under [models.{model_key}] in config.toml."
    )
}

fn model_key_ready_reply(base_reply: &str, model_key: &str) -> String {
    format!("{base_reply}\n\nAPI key found. Select it here with /model {model_key}.")
}

fn model_key_prompt_reply(base_reply: &str, secret_name: &str) -> String {
    format!(
        "{base_reply}\n\nAPI key missing. A terminal prompt is waiting for {secret_name}. Enter it there; it is visible locally and never goes through Telegram. Press Enter there to skip."
    )
}

fn model_key_stored_reply(model_key: &str, destination: secrets::SecretDestination) -> String {
    format!(
        "{model_key} key stored in {}. Select it here with /model {model_key}.",
        destination.location_label(),
    )
}

fn model_key_skipped_reply(model_key: &str) -> String {
    format!(
        "No key stored for {model_key}. Run /model add {model_key} again when you're ready to enter it in the tellm terminal."
    )
}

fn model_key_prompt_failed_reply(model_key: &str, secret_name: &str, error: &str) -> String {
    format!(
        "Could not complete the terminal prompt for {model_key}: {error}. Fallback: run tellm secret set {secret_name} in the tellm console."
    )
}

/// The in-room setup prompt after a room is approved (via /pair or /allow):
/// current model plus the actual picker, not just a pointer to it.
async fn room_setup_reply(chat_id: i64, became_owner: bool, handles: &RuntimeHandles) -> String {
    let (model_key, available) = {
        let config = handles.config.lock().await;
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id).clone();
        (
            selected_model_key(&config, &room, chat_id),
            config.models.keys().cloned().collect::<Vec<_>>(),
        )
    };
    // Blank lines matter: in Telegram's rich markdown a plain line after a
    // list item is absorbed into the item.
    let mut reply = "Room approved. Pick this room's model:\n\n".to_string();
    for key in &available {
        let marker = if *key == model_key { " (current)" } else { "" };
        reply.push_str(&format!("- /model {key}{marker}\n"));
    }
    reply.push_str(
        "\nLock it with /model pin KEY, add providers with /model add, or just start \
         chatting to use the current model.",
    );
    if became_owner {
        reply.push_str(
            "\n\nYou are registered as this bot's owner: /allow, /deny, /shutdown, and \
             /model pin work for you from any chat, and rooms you add the bot to are \
             approved automatically.",
        );
    }
    reply
}

async fn mutate_room(
    handles: &RuntimeHandles,
    chat_id: i64,
    mutate: impl FnOnce(&mut RoomState),
) -> Result<(), String> {
    let settings = {
        let mut rooms = handles.rooms.lock().await;
        mutate(rooms.get_or_default(chat_id));
        rooms.settings()
    };
    rooms::save_settings(&settings).map_err(|error| error.to_string())
}

async fn send_command_reply(
    handles: &RuntimeHandles,
    chat_id: i64,
    text: &str,
) -> Result<(), String> {
    handles
        .telegram
        .send_message(chat_id, text)
        .await
        .map_err(|error| error.to_string())
}

fn spawn_typing_indicator(telegram: Telegram, chat_id: i64) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let result: Result<(), TelegramError> = telegram.send_chat_action(chat_id).await;
            if let Err(error) = result {
                eprintln!("sendChatAction failed for chat {chat_id}: {error}");
            }
            sleep(TYPING_INTERVAL).await;
        }
    })
}

fn spawn_terminal_controls(
    secret_prompts: TerminalSecretPrompts,
) -> mpsc::Receiver<TerminalCommand> {
    let (sender, receiver) = mpsc::channel(8);
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(line) => line,
                Err(error) => {
                    eprintln!("terminal input read failed; ignoring line: {error}");
                    continue;
                }
            };
            if let Some(prompt) = current_terminal_secret_prompt(&secret_prompts) {
                if prompt.line_tx.send(line).is_err() {
                    clear_terminal_secret_prompt(&secret_prompts, &prompt.secret_name);
                }
                continue;
            }
            match line.trim().to_ascii_lowercase().as_str() {
                "reset" => {
                    let _ = sender.blocking_send(TerminalCommand::Reset);
                }
                "exit" | "quit" => {
                    let _ = sender.blocking_send(TerminalCommand::Shutdown);
                    break;
                }
                "" => {}
                _ => eprintln!("terminal commands: reset, exit, quit"),
            }
        }
        let _ = take_terminal_secret_prompt(&secret_prompts);
    });
    receiver
}

struct TerminalPromptGuard {
    secret_prompts: TerminalSecretPrompts,
    secret_name: String,
}

impl Drop for TerminalPromptGuard {
    fn drop(&mut self) {
        clear_terminal_secret_prompt(&self.secret_prompts, &self.secret_name);
    }
}

fn reserve_terminal_secret_prompt(
    secret_prompts: &TerminalSecretPrompts,
    secret_name: String,
) -> Result<(mpsc::UnboundedReceiver<String>, TerminalPromptGuard), String> {
    let (line_tx, line_rx) = mpsc::unbounded_channel();
    {
        let mut pending = secret_prompts
            .lock()
            .map_err(|_| "terminal prompt lock poisoned".to_string())?;
        if pending.is_some() {
            return Err(
                "another terminal prompt is already waiting; complete it first".to_string(),
            );
        }
        *pending = Some(TerminalSecretPrompt {
            secret_name: secret_name.clone(),
            line_tx,
        });
    }

    Ok((
        line_rx,
        TerminalPromptGuard {
            secret_prompts: Arc::clone(secret_prompts),
            secret_name,
        },
    ))
}

fn current_terminal_secret_prompt(
    secret_prompts: &TerminalSecretPrompts,
) -> Option<TerminalSecretPrompt> {
    secret_prompts.lock().ok()?.clone()
}

fn take_terminal_secret_prompt(
    secret_prompts: &TerminalSecretPrompts,
) -> Option<TerminalSecretPrompt> {
    secret_prompts.lock().ok()?.take()
}

fn clear_terminal_secret_prompt(secret_prompts: &TerminalSecretPrompts, secret_name: &str) {
    let Ok(mut pending) = secret_prompts.lock() else {
        return;
    };
    if pending
        .as_ref()
        .is_some_and(|prompt| prompt.secret_name == secret_name)
    {
        pending.take();
    }
}

enum PromptedSecret {
    Stored(secrets::SecretDestination),
    Skipped,
    Retry(&'static str),
}

fn store_prompted_secret(secret_name: &str, value: &str) -> Result<PromptedSecret, String> {
    let value = value.trim();
    if value.is_empty() {
        eprintln!("No value entered for {secret_name}; secret not changed.");
        return Ok(PromptedSecret::Skipped);
    }
    if is_terminal_command_word(value) {
        return Ok(PromptedSecret::Retry(
            "That looks like a terminal command, not an API key. It was not stored. Enter the key, or press Enter to skip.",
        ));
    }
    secrets::set_nonempty(secret_name, value)
        .map(|destination| destination.expect("trimmed non-empty secret should be stored"))
        .map(PromptedSecret::Stored)
        .map_err(|error| error.to_string())
}

fn is_terminal_command_word(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "reset" | "exit" | "quit"
    )
}

fn print_terminal_secret_prompt(secret_name: &str) {
    eprintln!("Value for {secret_name} (visible; blank skips):");
    let _ = std::io::stderr().flush();
}

fn print_startup_notice(notice: crate::access::StartupNotice) {
    match notice {
        crate::access::StartupNotice::PairingMode => {
            eprintln!(
                "No chats are allowed yet. A pairing code is printed here the moment a chat \
                 messages the bot or the bot is added to a group; approve with /pair CODE in \
                 that chat."
            );
        }
        crate::access::StartupNotice::Restricted { allowed_chat_count } => {
            eprintln!(
                "{allowed_chat_count} chat(s) are allowed (allowlist + model room pins). New \
                 chats get a pairing code printed here on first contact, or an owner \
                 can send /allow."
            );
        }
    }
}

fn print_group_privacy_hints(chat_ids: &BTreeSet<i64>) {
    for chat_id in chat_ids {
        eprintln!("{}", group_privacy_hint(*chat_id));
    }
}

fn group_privacy_hint(chat_id: i64) -> String {
    format!(
        "group chat detected (chat_id={chat_id}) - if the bot ignores plain text, disable privacy mode via BotFather (/setprivacy) and re-add the bot to the group"
    )
}

fn unknown_chat_hint(chat_id: i64) -> String {
    format!(
        "This bot is private. Chat id: {chat_id}. If you own it, send /pair CODE here with the \
         code printed in the tellm terminal, or send /allow {chat_id} from the admin chat."
    )
}

fn allowed_group_chat_ids(config: &AccessConfig) -> BTreeSet<i64> {
    config
        .allowed_chat_ids
        .union(&config.pinned_chat_ids)
        .copied()
        .filter(|chat_id| *chat_id < 0)
        .collect()
}

fn log_update_route(chat_id: i64, kind: UpdateKind, route: UpdateRoute) {
    eprintln!("{}", update_log_line(chat_id, kind, route));
}

fn update_log_line(chat_id: i64, kind: UpdateKind, route: UpdateRoute) -> String {
    format!(
        "telegram update: chat_id={chat_id} kind={} route={}",
        kind.as_str(),
        route.as_str()
    )
}

fn now_access_time() -> AccessTime {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    AccessTime::from_unix_seconds(seconds)
}

fn reset_notice(reset: HistoryReset) -> Option<String> {
    match reset {
        HistoryReset::WireFormatChanged {
            previous: Some(previous),
            new,
        } if previous != new => Some(format!(
            "Provider wire format changed from {previous:?} to {new:?}; chat history reset."
        )),
        HistoryReset::WireFormatChanged { .. } | HistoryReset::None => None,
    }
}

fn selected_model_key(config: &Config, room: &RoomState, chat_id: i64) -> String {
    pinned_model_key(config, chat_id)
        .map(str::to_string)
        .or_else(|| room.settings.model_key.clone())
        .unwrap_or_else(|| config.default_model.clone())
}

fn allow_chat_in_config(config: &mut Config, chat_id: i64) -> bool {
    if config.telegram.allowed_chat_ids.contains(&chat_id) {
        return false;
    }
    config.telegram.allowed_chat_ids.push(chat_id);
    true
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DenyChatConfigResult {
    allowed_removed: bool,
    removed_model_pins: Vec<String>,
}

impl DenyChatConfigResult {
    fn changed(&self) -> bool {
        self.allowed_removed || !self.removed_model_pins.is_empty()
    }
}

fn deny_chat_in_config(config: &mut Config, chat_id: i64) -> DenyChatConfigResult {
    let before_allowed = config.telegram.allowed_chat_ids.len();
    config
        .telegram
        .allowed_chat_ids
        .retain(|allowed| *allowed != chat_id);
    let allowed_removed = config.telegram.allowed_chat_ids.len() != before_allowed;

    let mut removed_model_pins = Vec::new();
    for (model_key, model) in &mut config.models {
        let before_pins = model.telegram_chat_ids.len();
        model.telegram_chat_ids.retain(|pinned| *pinned != chat_id);
        if model.telegram_chat_ids.len() != before_pins {
            removed_model_pins.push(model_key.clone());
        }
    }

    DenyChatConfigResult {
        allowed_removed,
        removed_model_pins,
    }
}

fn pinned_model_key(config: &Config, chat_id: i64) -> Option<&str> {
    config
        .models
        .iter()
        .find(|(_, model)| model.telegram_chat_ids.contains(&chat_id))
        .map(|(key, _)| key.as_str())
}

fn warm_configured_provider_secrets(config: &Config) {
    let secret_names = configured_provider_secret_names(config);
    if secret_names.is_empty() {
        return;
    }

    eprintln!(
        "Checking configured provider secrets on startup: {}",
        secret_names
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(", ")
    );

    for secret_name in secret_names {
        if secrets::get(&secret_name).is_none() {
            eprintln!(
                "Provider secret \"{secret_name}\" is not available yet; model calls that require it will fail until it is stored."
            );
        }
    }
}

fn configured_provider_secret_names(config: &Config) -> Vec<String> {
    config
        .models
        .values()
        .filter_map(|model| model.api_key_secret.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn configured_model_key(config: &Config, requested: &str) -> Option<String> {
    config
        .models
        .keys()
        .find(|key| key.eq_ignore_ascii_case(requested))
        .cloned()
}

fn required_api_key(model: &ModelConfig) -> Result<String, String> {
    let secret_name = model
        .api_key_secret
        .as_deref()
        .ok_or_else(|| format!("model {} has no api_key_secret", model.model_name))?;
    secrets::get(secret_name).ok_or_else(|| missing_provider_secret_error(secret_name))
}

fn compat_api_key(model: &ModelConfig) -> Result<String, String> {
    let Some(secret_name) = model.api_key_secret.as_deref() else {
        return Ok(String::new());
    };
    secrets::get(secret_name).ok_or_else(|| missing_provider_secret_error(secret_name))
}

fn missing_provider_secret_error(secret_name: &str) -> String {
    format!(
        "missing provider secret \"{secret_name}\" — set it in the tellm console with: \
         tellm secret set {secret_name}"
    )
}

fn message_text(message: &IncomingMessage) -> Option<&str> {
    message.text.as_deref().or(message.caption.as_deref())
}

fn message_has_model_input(message: &IncomingMessage) -> bool {
    message_text(message).is_some_and(|text| !text.trim().is_empty())
        || message
            .photo
            .as_ref()
            .is_some_and(|photos| !photos.is_empty())
        || message.document.is_some()
}

fn largest_photo(photos: Option<&[PhotoSize]>) -> Option<&PhotoSize> {
    photos?
        .iter()
        .max_by_key(|photo| photo.width * photo.height)
}

fn is_text_document(document: &Document, media_type: &str) -> bool {
    media_type == "text/plain"
        || document
            .file_name
            .as_deref()
            .is_some_and(|name| name.to_ascii_lowercase().ends_with(".txt"))
}

fn decode_image(image: GeneratedImage) -> Result<Vec<u8>, String> {
    BASE64
        .decode(image.base64)
        .map_err(|error| format!("generated image was not valid base64: {error}"))
}

fn mode_name(mode: ChatMode) -> &'static str {
    match mode {
        ChatMode::Chat => "chat",
        ChatMode::Message => "message",
    }
}

fn format_model_status(
    selected: Option<String>,
    effective: String,
    pinned: Option<String>,
    available: Vec<String>,
) -> String {
    let available = available.join(", ");
    match pinned {
        Some(pinned) => {
            let selected = selected.unwrap_or_else(|| "(default)".to_string());
            format!(
                "Model: {effective} (pinned room: {pinned}; selected: {selected} ignored). Available: {available}."
            )
        }
        None => match selected {
            Some(_) => format!("Model: {effective}. Available: {available}."),
            None => format!("Model: {effective} (default). Available: {available}."),
        },
    }
}

fn model_set_reply(model_key: &str) -> String {
    format!(
        "Model set to {model_key}. Chat history reset. This is this room's selection; owners can lock it into config.toml with /model pin {model_key}."
    )
}

fn chat_id_reply(chat_id: i64) -> String {
    format!("Chat id: {chat_id}.")
}

fn format_deny_chat_reply(chat_id: i64, result: &DenyChatConfigResult) -> String {
    if result.changed() {
        let mut reply = format!("Denied chat {chat_id}. Room state cleared.");
        if !result.removed_model_pins.is_empty() {
            reply.push_str(&format!(
                " Removed model pin(s): {}.",
                result.removed_model_pins.join(", ")
            ));
        }
        reply
    } else {
        format!("Chat {chat_id} was not allowed. Room state cleared.")
    }
}

fn format_reject(reason: CommandReject) -> String {
    match reason {
        CommandReject::MissingPairingCode => "Usage: /pair CODE.".to_string(),
        CommandReject::UnknownMode { value } => {
            format!("Unknown mode \"{value}\". Use chat or message.")
        }
        CommandReject::UnknownModel { value, available } => {
            format!(
                "Unknown model \"{value}\". Available: {}.",
                available.join(", ")
            )
        }
        CommandReject::UnknownReasoning { value } => {
            format!("Unknown reasoning level \"{value}\". Use off, low, medium, high, or max.")
        }
        CommandReject::UnknownBoolean { value } => {
            format!("Unknown on/off value \"{value}\". Use on, off, or status.")
        }
        CommandReject::MissingChatId { command } => {
            format!("Usage: /{} CHAT_ID.", command_name(command))
        }
        CommandReject::InvalidChatId { value } => {
            format!("Invalid chat id \"{value}\". Use a numeric Telegram chat id.")
        }
        CommandReject::UnknownOllamaAction { value } => match value {
            Some(value) => format!("Unknown Ollama action \"{value}\". Usage: /ollama unload."),
            None => "Usage: /ollama unload.".to_string(),
        },
        CommandReject::PinnedModel { model_key } => {
            format!("This room is pinned to {model_key}; /model changes are disabled.")
        }
        CommandReject::AdminNotAllowed => "Only the bot owner can use this command.".to_string(),
        CommandReject::AdminStale => "Ignoring stale owner command.".to_string(),
        CommandReject::ShutdownNotAdmin => "Only the bot owner can use /shutdown.".to_string(),
        CommandReject::ShutdownStale => "Ignoring stale /shutdown command.".to_string(),
        CommandReject::CapabilityUnsupported {
            feature,
            model_key,
            endpoint,
        } => {
            format!(
                "{feature} isn't supported by this room's model \"{model_key}\" ({endpoint}). \
                 Switch models with /model, or leave it off."
            )
        }
    }
}

fn command_name(command: KnownCommand) -> &'static str {
    match command {
        KnownCommand::New => "new",
        KnownCommand::Id => "id",
        KnownCommand::Mode => "mode",
        KnownCommand::Model => "model",
        KnownCommand::Role => "role",
        KnownCommand::Reasoning => "reasoning",
        KnownCommand::WebSearch => "websearch",
        KnownCommand::ImageGeneration => "imagegen",
        KnownCommand::Allow => "allow",
        KnownCommand::Deny => "deny",
        KnownCommand::Pair => "pair",
        KnownCommand::Ollama => "ollama",
        KnownCommand::Shutdown => "shutdown",
        KnownCommand::Help => "help",
    }
}

fn help_text(pinned_model_key: Option<&str>) -> String {
    let mut text = HELP_TEXT.to_string();
    if let Some(model_key) = pinned_model_key {
        text.push_str(&format!(
            "\n- This room is pinned to {model_key}; /model changes are disabled"
        ));
    }
    text
}

const HELP_TEXT: &str = "\
- /new - reset this chat
- /id - show this Telegram chat id
- /mode chat|message - show or set conversation mode
- /model KEY - show or set model
- /model pin KEY | /model unpin - lock or release this room's model (owner)
- /model add [KEY] - list the provider catalog or add a preset (owner)
- /role TEXT|clear - show, set, or clear the system role
- /reasoning off|low|medium|high|max - show or set reasoning level
- /websearch on|off|status - toggle, set, or show web search
- /imagegen on|off|status - toggle, set, or show image generation
- /allow CHAT_ID - allow a chat (owner)
- /deny CHAT_ID - deny a chat and clear its room state (owner)
- /pair CODE - pair a new chat
- /ollama unload - unload local Ollama models used by this session (owner)
- /shutdown - stop tellm (owner)
- /help - show commands";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_notice_only_mentions_real_wire_format_switches() {
        assert_eq!(reset_notice(HistoryReset::None), None);
        assert_eq!(
            reset_notice(HistoryReset::WireFormatChanged {
                previous: None,
                new: WireFormat::Anthropic,
            }),
            None
        );
        assert!(
            reset_notice(HistoryReset::WireFormatChanged {
                previous: Some(WireFormat::Anthropic),
                new: WireFormat::Responses,
            })
            .unwrap()
            .contains("Anthropic")
        );
    }

    #[test]
    fn pinned_chat_uses_pinned_model_over_room_selection() {
        let mut models = BTreeMap::new();
        models.insert("claude".to_string(), test_model(WireFormat::Anthropic, &[]));
        models.insert("gpt".to_string(), test_model(WireFormat::Responses, &[42]));
        let config = Config {
            default_model: "claude".to_string(),
            models,
            telegram: tellm_config::TelegramConfig::default(),
        };
        let room = RoomState::new(crate::rooms::RoomSettings {
            model_key: Some("claude".to_string()),
            ..crate::rooms::RoomSettings::default()
        });

        assert_eq!(pinned_model_key(&config, 42), Some("gpt"));
        assert_eq!(selected_model_key(&config, &room, 42), "gpt");
        assert_eq!(selected_model_key(&config, &room, 7), "claude");
    }

    #[test]
    fn local_ollama_addr_only_matches_local_default_ollama_port() {
        assert_eq!(
            local_ollama_addr("http://localhost:11434/v1").as_deref(),
            Some("localhost:11434")
        );
        assert_eq!(
            local_ollama_addr("http://127.0.0.1:11434/v1/").as_deref(),
            Some("127.0.0.1:11434")
        );
        assert_eq!(
            local_ollama_addr("http://[::1]:11434/v1").as_deref(),
            Some("[::1]:11434")
        );
        assert_eq!(local_ollama_addr("https://api.mistral.ai/v1"), None);
        assert_eq!(local_ollama_addr("http://localhost:8080/v1"), None);
        assert_eq!(local_ollama_addr("http://192.168.1.10:11434/v1"), None);
    }

    #[test]
    fn ollama_unload_request_uses_keep_alive_zero() {
        let request = ollama_unload_request("localhost:11434", "gemma4:31b-mlx");
        let (headers, body) = request.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("POST /api/generate HTTP/1.1"));
        assert!(headers.contains("Host: localhost:11434"));
        assert!(headers.contains(&format!("Content-Length: {}", body.len())));

        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["model"], "gemma4:31b-mlx");
        assert_eq!(body["prompt"], "");
        assert_eq!(body["stream"], false);
        assert_eq!(body["keep_alive"], 0);
    }

    #[test]
    fn ollama_unload_reply_reports_empty_success_and_failures() {
        assert_eq!(
            ollama_unload_reply(&OllamaUnloadSummary::default()),
            "No local Ollama models have been used by this tellm session."
        );
        assert_eq!(
            ollama_unload_reply(&OllamaUnloadSummary {
                attempted: 2,
                unloaded: vec!["llama3.3:70b".to_string(), "qwen3:32b".to_string()],
                failed: Vec::new(),
            }),
            "Unloaded local Ollama models: llama3.3:70b, qwen3:32b."
        );
        let partial = ollama_unload_reply(&OllamaUnloadSummary {
            attempted: 2,
            unloaded: vec!["llama3.3:70b".to_string()],
            failed: vec![("qwen3:32b".to_string(), "connection refused".to_string())],
        });
        assert!(
            partial.contains("Unloaded local Ollama model: llama3.3:70b."),
            "{partial}"
        );
        assert!(
            partial.contains("Failed to unload local Ollama model: qwen3:32b"),
            "{partial}"
        );
    }

    #[test]
    fn http_response_success_only_accepts_2xx() {
        assert!(http_response_is_success("HTTP/1.1 200 OK\r\n\r\n{}"));
        assert!(http_response_is_success("HTTP/1.1 204 No Content\r\n\r\n"));
        assert!(!http_response_is_success("HTTP/1.1 404 Not Found\r\n\r\n"));
        assert!(!http_response_is_success(""));
    }

    #[test]
    fn allow_and_deny_chat_update_persisted_config_shape() {
        let mut models = BTreeMap::new();
        models.insert(
            "claude".to_string(),
            test_model(WireFormat::Anthropic, &[42]),
        );
        models.insert(
            "gpt".to_string(),
            test_model(WireFormat::Responses, &[-100]),
        );
        let mut config = Config {
            default_model: "claude".to_string(),
            models,
            telegram: tellm_config::TelegramConfig {
                allowed_chat_ids: vec![1, 42],
                owner_user_ids: Vec::new(),
                max_concurrent_updates: None,
            },
        };

        assert!(allow_chat_in_config(&mut config, -100));
        assert!(!allow_chat_in_config(&mut config, -100));
        assert!(config.telegram.allowed_chat_ids.contains(&-100));

        let result = deny_chat_in_config(&mut config, -100);
        assert_eq!(
            result,
            DenyChatConfigResult {
                allowed_removed: true,
                removed_model_pins: vec!["gpt".to_string()],
            }
        );
        assert!(!config.telegram.allowed_chat_ids.contains(&-100));
        assert!(config.models["gpt"].telegram_chat_ids.is_empty());
        assert_eq!(
            format_deny_chat_reply(-100, &result),
            "Denied chat -100. Room state cleared. Removed model pin(s): gpt."
        );
    }

    #[test]
    fn denying_a_chat_leaves_owner_users_intact() {
        let mut config = Config {
            default_model: "gpt".to_string(),
            models: BTreeMap::new(),
            telegram: tellm_config::TelegramConfig {
                allowed_chat_ids: vec![7],
                owner_user_ids: vec![7],
                max_concurrent_updates: None,
            },
        };

        let result = deny_chat_in_config(&mut config, 7);
        assert!(result.allowed_removed);
        // Privilege belongs to users, so denying the chat with the same
        // numeric id cannot strand the bot without an owner.
        assert_eq!(config.telegram.owner_user_ids, vec![7]);
    }

    #[test]
    fn provider_error_reply_matches_both_capability_spellings() {
        // The xAI backstop error uses the underscore form.
        let xai = provider_error_reply("xAI Responses does not support OpenAI image_generation");
        assert!(xai.contains("Tip: /imagegen off"), "{xai}");
        let anthropic =
            provider_error_reply("Anthropic Messages does not support image generation");
        assert!(anthropic.contains("Tip: /imagegen off"), "{anthropic}");
        let compat = provider_error_reply(
            "chat completions compat does not support provider-native web search",
        );
        assert!(compat.contains("Tip: /websearch off"), "{compat}");
        let plain = provider_error_reply("api error 500: boom");
        assert!(!plain.contains("Tip:"), "{plain}");
    }

    #[test]
    fn model_set_reply_distinguishes_selection_from_pin() {
        let reply = model_set_reply("openai");

        assert!(reply.contains("room's selection"), "{reply}");
        assert!(reply.contains("/model pin openai"), "{reply}");
        assert!(reply.contains("config.toml"), "{reply}");
    }

    #[test]
    fn model_add_missing_key_reply_prompts_terminal_not_telegram_secret_entry() {
        let preset = crate::wizard::preset_by_key("openai").unwrap();
        let base = model_add_base_reply(false, preset);
        let prompt = model_add_key_prompt_reply(&base, preset);
        let skipped = model_add_key_skipped_reply(preset);
        let stored =
            model_add_key_stored_reply(preset, secrets::SecretDestination::CredentialsFile);

        assert!(prompt.contains("terminal prompt"), "{prompt}");
        assert!(prompt.contains("visible locally"), "{prompt}");
        assert!(prompt.contains(preset.api_key_secret), "{prompt}");
        assert!(!prompt.contains("tellm secret set"), "{prompt}");
        assert!(skipped.contains("/model add openai"), "{skipped}");
        assert!(stored.contains("/model openai"), "{stored}");
        assert!(stored.contains("credentials.toml"), "{stored}");
    }

    #[test]
    fn configured_model_key_reply_uses_model_secret_name() {
        let base = configured_model_base_reply("mistral");
        let prompt = configured_model_key_prompt_reply(&base, "mistral_api_key");
        let skipped = configured_model_key_skipped_reply("mistral");
        let stored = configured_model_key_stored_reply(
            "mistral",
            secrets::SecretDestination::CredentialsFile,
        );
        let no_secret = configured_model_has_no_secret_reply("ollama");

        assert!(base.contains("mistral is already configured"), "{base}");
        assert!(prompt.contains("mistral_api_key"), "{prompt}");
        assert!(prompt.contains("visible locally"), "{prompt}");
        assert!(skipped.contains("/model add mistral"), "{skipped}");
        assert!(stored.contains("/model mistral"), "{stored}");
        assert!(no_secret.contains("no API key prompt"), "{no_secret}");
        assert!(no_secret.contains("/model ollama"), "{no_secret}");
        assert!(no_secret.contains("api_key_secret"), "{no_secret}");
        assert!(no_secret.contains("[models.ollama]"), "{no_secret}");
    }

    #[test]
    fn empty_terminal_secret_input_skips_without_storing() {
        assert!(matches!(
            store_prompted_secret("openai_api_key", "").unwrap(),
            PromptedSecret::Skipped
        ));
        assert!(matches!(
            store_prompted_secret("openai_api_key", "   ").unwrap(),
            PromptedSecret::Skipped
        ));
    }

    #[test]
    fn terminal_commands_are_not_stored_as_prompted_secrets() {
        assert!(matches!(
            store_prompted_secret("openai_api_key", "exit").unwrap(),
            PromptedSecret::Retry(_)
        ));
        assert!(matches!(
            store_prompted_secret("openai_api_key", " reset ").unwrap(),
            PromptedSecret::Retry(_)
        ));
    }

    #[test]
    fn configured_model_lookup_is_case_insensitive() {
        let mut models = BTreeMap::new();
        models.insert("Mistral".to_string(), test_model(WireFormat::Compat, &[]));
        let config = Config {
            default_model: "Mistral".to_string(),
            models,
            telegram: tellm_config::TelegramConfig::default(),
        };

        assert_eq!(
            configured_model_key(&config, "mistral"),
            Some("Mistral".to_string())
        );
    }

    #[test]
    fn pinned_model_status_and_help_name_the_pin() {
        assert_eq!(
            format_model_status(
                Some("claude".to_string()),
                "gpt".to_string(),
                Some("gpt".to_string()),
                vec!["claude".to_string(), "gpt".to_string()],
            ),
            "Model: gpt (pinned room: gpt; selected: claude ignored). Available: claude, gpt."
        );
        assert!(
            help_text(Some("gpt"))
                .contains("This room is pinned to gpt; /model changes are disabled")
        );
        assert_eq!(
            format_reject(CommandReject::PinnedModel {
                model_key: "gpt".to_string()
            }),
            "This room is pinned to gpt; /model changes are disabled."
        );
    }

    #[test]
    fn largest_photo_prefers_pixel_area() {
        let photos = vec![
            PhotoSize {
                file_id: "small".to_string(),
                file_unique_id: None,
                width: 10,
                height: 10,
                file_size: None,
            },
            PhotoSize {
                file_id: "wide".to_string(),
                file_unique_id: None,
                width: 50,
                height: 20,
                file_size: None,
            },
        ];

        assert_eq!(largest_photo(Some(&photos)).unwrap().file_id, "wide");
    }

    #[test]
    fn text_document_detection_accepts_mime_or_extension() {
        let mut document = Document {
            file_id: "f".to_string(),
            file_unique_id: None,
            file_name: Some("notes.TXT".to_string()),
            mime_type: None,
            file_size: None,
        };

        assert!(is_text_document(&document, "application/octet-stream"));
        document.file_name = Some("notes.pdf".to_string());
        assert!(is_text_document(&document, "text/plain"));
        assert!(!is_text_document(&document, "application/pdf"));
    }

    #[test]
    fn update_log_line_contains_only_metadata() {
        assert_eq!(
            update_log_line(-100, UpdateKind::Message, UpdateRoute::Model),
            "telegram update: chat_id=-100 kind=message route=model"
        );
        assert_eq!(
            update_log_line(42, UpdateKind::EditedMessage, UpdateRoute::Ignored),
            "telegram update: chat_id=42 kind=edited_message route=ignored"
        );
    }

    #[test]
    fn chat_id_reply_reports_negative_group_ids() {
        assert_eq!(chat_id_reply(-100), "Chat id: -100.");
    }

    #[test]
    fn allowed_group_chat_ids_include_allowed_and_pinned_negative_ids() {
        let config = AccessConfig {
            allowed_chat_ids: [-100, 42].into_iter().collect(),
            owner_user_ids: BTreeSet::new(),
            pinned_chat_ids: [-300, 42].into_iter().collect(),
        };

        assert_eq!(
            allowed_group_chat_ids(&config),
            [-300, -100].into_iter().collect()
        );
    }

    #[test]
    fn group_privacy_hint_names_botfather_privacy_mode() {
        let hint = group_privacy_hint(-100);

        assert!(hint.contains("chat_id=-100"));
        assert!(hint.contains("/setprivacy"));
        assert!(hint.contains("re-add the bot"));
    }

    #[test]
    fn unknown_chat_hint_offers_pairing_and_allow_with_chat_id() {
        let hint = unknown_chat_hint(-100);
        assert!(hint.contains("Chat id: -100"));
        assert!(hint.contains("/pair CODE"));
        assert!(hint.contains("/allow -100"));
    }

    #[test]
    fn message_model_input_detection_ignores_blank_messages() {
        let mut message = IncomingMessage {
            chat: tellm_telegram::Chat {
                id: 42,
                title: None,
                kind: None,
            },
            from: None,
            date: 1000,
            text: Some(" \n ".to_string()),
            caption: None,
            photo: None,
            document: None,
        };

        assert!(!message_has_model_input(&message));
        message.text = Some("hello".to_string());
        assert!(message_has_model_input(&message));
        message.text = None;
        message.photo = Some(vec![PhotoSize {
            file_id: "p".to_string(),
            file_unique_id: None,
            width: 1,
            height: 1,
            file_size: None,
        }]);
        assert!(message_has_model_input(&message));
    }

    #[test]
    fn chat_request_uses_room_image_generation_toggle() {
        let room = RoomState::new(crate::rooms::RoomSettings {
            image_generation: true,
            web_search: true,
            ..crate::rooms::RoomSettings::default()
        });
        let model = ModelConfig {
            wire_format: WireFormat::Responses,
            model_name: "gpt-5.5".to_string(),
            base_url: None,
            api_key_secret: Some("openai_api_key".to_string()),
            telegram_chat_ids: Vec::new(),
            thinking: tellm_core::ThinkingLevel::High,
        };

        let request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "draw this".to_string(),
            }],
        );

        assert!(request.image_generation);
        assert!(request.web_search);
        assert_eq!(request.model, "gpt-5.5");
    }

    #[test]
    fn gemini_image_generation_capability_requires_image_model() {
        let room = RoomState::new(crate::rooms::RoomSettings {
            model_key: Some("gemini".to_string()),
            ..crate::rooms::RoomSettings::default()
        });
        let mut config = Config {
            default_model: "gemini".to_string(),
            models: BTreeMap::new(),
            telegram: tellm_config::TelegramConfig::default(),
        };

        let mut gemini = test_model(WireFormat::Gemini, &[]);
        gemini.model_name = "gemini-3.5-flash".to_string();
        config.models.insert("gemini".to_string(), gemini);
        let capabilities = room_capabilities(&config, &room, 42);
        assert!(capabilities.web_search);
        assert!(!capabilities.image_generation);

        config.models.get_mut("gemini").unwrap().model_name = "gemini-3.1-flash-image".to_string();
        let capabilities = room_capabilities(&config, &room, 42);
        assert!(capabilities.web_search);
        assert!(capabilities.image_generation);
    }

    #[test]
    fn message_mode_request_omits_retained_latest_turn() {
        let mut room = RoomState::new(crate::rooms::RoomSettings {
            mode: ChatMode::Message,
            ..crate::rooms::RoomSettings::default()
        });
        room.append_turn(
            WireFormat::Compat,
            vec![serde_json::json!({ "role": "assistant", "content": "previous" })],
        );
        let model = ModelConfig {
            wire_format: WireFormat::Compat,
            model_name: "local".to_string(),
            base_url: Some("http://localhost:11434/v1".to_string()),
            api_key_secret: None,
            telegram_chat_ids: Vec::new(),
            thinking: tellm_core::ThinkingLevel::default(),
        };

        let message_request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "fresh".to_string(),
            }],
        );
        assert!(message_request.history.is_empty());

        room.settings.mode = ChatMode::Chat;
        let chat_request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "continue".to_string(),
            }],
        );
        assert_eq!(chat_request.history.len(), 1);
    }

    #[test]
    fn provider_secret_startup_warmup_uses_unique_configured_secret_names() {
        let mut models = BTreeMap::new();
        let mut claude = test_model(WireFormat::Anthropic, &[]);
        claude.api_key_secret = Some("shared_api_key".to_string());
        models.insert("claude".to_string(), claude);

        let mut grok = test_model(WireFormat::Responses, &[]);
        grok.api_key_secret = Some("shared_api_key".to_string());
        models.insert("grok".to_string(), grok);

        let mut gemini = test_model(WireFormat::Gemini, &[]);
        gemini.api_key_secret = Some("gemini_api_key".to_string());
        models.insert("gemini".to_string(), gemini);

        let mut ollama = test_model(WireFormat::Compat, &[]);
        ollama.api_key_secret = None;
        models.insert("ollama".to_string(), ollama);

        let config = Config {
            default_model: "claude".to_string(),
            models,
            telegram: tellm_config::TelegramConfig::default(),
        };

        assert_eq!(
            configured_provider_secret_names(&config),
            vec!["gemini_api_key".to_string(), "shared_api_key".to_string()]
        );
    }

    #[test]
    fn compat_api_key_distinguishes_keyless_from_missing_secret() {
        let mut keyless = test_model(WireFormat::Compat, &[]);
        keyless.api_key_secret = None;
        assert_eq!(compat_api_key(&keyless).unwrap(), "");

        let mut keyed = test_model(WireFormat::Compat, &[]);
        keyed.api_key_secret = Some("definitely_missing_tellm_test_secret".to_string());
        let error = compat_api_key(&keyed).unwrap_err();
        assert!(
            error.contains("definitely_missing_tellm_test_secret"),
            "{error}"
        );
    }

    fn test_model(wire_format: WireFormat, chat_ids: &[i64]) -> ModelConfig {
        ModelConfig {
            wire_format,
            model_name: "model".to_string(),
            base_url: None,
            api_key_secret: Some("secret".to_string()),
            telegram_chat_ids: chat_ids.to_vec(),
            thinking: tellm_core::ThinkingLevel::default(),
        }
    }

    #[test]
    fn help_text_is_markdown_bullets_without_raw_angle_placeholders() {
        assert!(HELP_TEXT.lines().all(|line| line.starts_with("- /")));
        assert!(HELP_TEXT.contains("- /id - show this Telegram chat id"));
        assert!(HELP_TEXT.contains("- /allow CHAT_ID - allow a chat (owner)"));
        assert!(
            HELP_TEXT.contains("- /deny CHAT_ID - deny a chat and clear its room state (owner)")
        );
        assert!(HELP_TEXT.contains("- /pair CODE - pair a new chat"));
        assert!(HELP_TEXT.contains("- /ollama unload - unload local Ollama models"));
        assert!(!HELP_TEXT.contains('<'));
        assert!(!HELP_TEXT.contains('>'));
    }
}
