use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{Duration, Instant as StdInstant, SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use tellm_anthropic::Anthropic;
use tellm_compat::Compat;
use tellm_config::{Config, ModelConfig, WireFormat, secrets};
use tellm_core::{ChatRequest, ChatResponse, ContentPart, GeneratedImage, Provider};
use tellm_gemini::Gemini;
use tellm_openai::Responses;
use tellm_telegram::{Document, IncomingMessage, PhotoSize, Telegram, TelegramError};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, mpsc, oneshot};
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
const OLLAMA_TERMINATE_WAIT: Duration = Duration::from_secs(2);
const OLLAMA_TERMINATE_POLL: Duration = Duration::from_millis(100);
const TERMINAL_SECRET_PROMPT_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const MAX_ATTACHMENT_BYTES: usize = 20 * 1024 * 1024;
static OLLAMA_START_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static OLLAMA_CHILD: OnceLock<StdMutex<Option<ManagedOllamaChild>>> = OnceLock::new();
static OLLAMA_LOADED_MODELS: OnceLock<StdMutex<BTreeSet<OllamaLoadedModel>>> = OnceLock::new();
static NEXT_CHAT_WORKER_ID: AtomicU64 = AtomicU64::new(1);

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
    room_persist: Arc<Mutex<()>>,
    persistence: PersistenceWriter,
    workers: WorkerRegistry,
    queue_full_notices: Arc<StdMutex<BTreeSet<i64>>>,
}

#[derive(Clone)]
struct PersistenceWriter {
    sender: std_mpsc::Sender<PersistenceCommand>,
}

enum PersistenceCommand {
    SaveConfig {
        config: Config,
        reply: oneshot::Sender<Result<(), String>>,
    },
    SaveRooms {
        settings: BTreeMap<i64, rooms::RoomSettings>,
        reply: oneshot::Sender<Result<(), String>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

impl PersistenceWriter {
    async fn save_config(&self, config: Config) -> Result<(), String> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(PersistenceCommand::SaveConfig { config, reply })
            .map_err(|_| "config persistence writer stopped".to_string())?;
        result
            .await
            .map_err(|_| "config persistence writer stopped before replying".to_string())?
    }

    async fn save_rooms(&self, settings: BTreeMap<i64, rooms::RoomSettings>) -> Result<(), String> {
        let (reply, result) = oneshot::channel();
        self.sender
            .send(PersistenceCommand::SaveRooms { settings, reply })
            .map_err(|_| "room persistence writer stopped".to_string())?;
        result
            .await
            .map_err(|_| "room persistence writer stopped before replying".to_string())?
    }

    async fn shutdown(&self) -> Result<(), String> {
        let (reply, done) = oneshot::channel();
        self.sender
            .send(PersistenceCommand::Shutdown { reply })
            .map_err(|_| "persistence writer already stopped".to_string())?;
        done.await
            .map_err(|_| "persistence writer stopped before shutdown completed".to_string())
    }
}

fn spawn_persistence_writer() -> std::io::Result<(PersistenceWriter, std::thread::JoinHandle<()>)> {
    spawn_persistence_writer_with(
        |config| tellm_config::save(&config).map_err(|error| error.to_string()),
        |settings| rooms::save_settings(&settings).map_err(|error| error.to_string()),
    )
}

fn spawn_persistence_writer_with<SaveConfig, SaveRooms>(
    save_config: SaveConfig,
    save_rooms: SaveRooms,
) -> std::io::Result<(PersistenceWriter, std::thread::JoinHandle<()>)>
where
    SaveConfig: Fn(Config) -> Result<(), String> + Send + 'static,
    SaveRooms: Fn(BTreeMap<i64, rooms::RoomSettings>) -> Result<(), String> + Send + 'static,
{
    let (sender, receiver) = std_mpsc::channel();
    let thread = std::thread::Builder::new()
        .name("tellm-persistence".to_string())
        .spawn(move || {
            while let Ok(command) = receiver.recv() {
                match command {
                    PersistenceCommand::SaveConfig { config, reply } => {
                        let _ = reply.send(save_config(config));
                    }
                    PersistenceCommand::SaveRooms { settings, reply } => {
                        let _ = reply.send(save_rooms(settings));
                    }
                    PersistenceCommand::Shutdown { reply } => {
                        let _ = reply.send(());
                        break;
                    }
                }
            }
        })?;
    Ok((PersistenceWriter { sender }, thread))
}

struct ChatDispatcher {
    sender: mpsc::Sender<DispatchMessage>,
    handle: JoinHandle<()>,
    worker_id: u64,
    cancelled: Arc<AtomicBool>,
}

type WorkerRegistry = Arc<StdMutex<BTreeMap<i64, WorkerRegistration>>>;

struct WorkerRegistration {
    worker_id: u64,
    cancelled: Arc<AtomicBool>,
    abort_handle: tokio::task::AbortHandle,
}

struct WorkerRegistryGuard {
    chat_id: i64,
    worker_id: u64,
    workers: WorkerRegistry,
}

impl Drop for WorkerRegistryGuard {
    fn drop(&mut self) {
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if workers
            .get(&self.chat_id)
            .is_some_and(|worker| worker.worker_id == self.worker_id)
        {
            workers.remove(&self.chat_id);
        }
    }
}

struct AbortOnDrop(JoinHandle<()>);

impl AbortOnDrop {
    fn abort(&self) {
        self.0.abort();
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

struct QueueFullNoticeGuard {
    chat_id: i64,
    pending: Arc<StdMutex<BTreeSet<i64>>>,
}

impl Drop for QueueFullNoticeGuard {
    fn drop(&mut self) {
        self.pending
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&self.chat_id);
    }
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

struct ManagedOllamaChild {
    child: Option<Child>,
}

impl ManagedOllamaChild {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn id(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    fn stop(mut self) -> Result<String, String> {
        let child = self
            .child
            .take()
            .ok_or_else(|| "Ollama child was already stopped".to_string())?;
        stop_ollama_child(child)
    }
}

impl Drop for ManagedOllamaChild {
    fn drop(&mut self) {
        let Some(child) = self.child.take() else {
            return;
        };
        match stop_ollama_child(child) {
            Ok(message) => eprintln!("{message}"),
            Err(error) => eprintln!("failed to stop spawned Ollama process during drop: {error}"),
        }
    }
}

struct RuntimeOllamaCleanup;

impl Drop for RuntimeOllamaCleanup {
    fn drop(&mut self) {
        stop_started_ollama_blocking();
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct OllamaUnloadSummary {
    attempted: usize,
    unloaded: Vec<String>,
    not_loaded: Vec<String>,
    failed: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OllamaUnloadOutcome {
    Unloaded,
    NotLoaded,
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
        let _ollama_cleanup = RuntimeOllamaCleanup;
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

        let workers = Arc::new(StdMutex::new(BTreeMap::new()));
        let (persistence, persistence_thread) = spawn_persistence_writer()?;
        let handles = RuntimeHandles {
            telegram: self.telegram.clone(),
            config: Arc::clone(&self.config),
            rooms: Arc::clone(&self.rooms),
            access: Arc::clone(&self.access),
            terminal_prompts: Arc::clone(&self.terminal_prompts),
            bot_username,
            shutdown_tx: self.shutdown_tx.clone(),
            room_persist: Arc::new(Mutex::new(())),
            persistence,
            workers: Arc::clone(&workers),
            queue_full_notices: Arc::new(StdMutex::new(BTreeSet::new())),
        };
        let mut dispatchers = BTreeMap::new();
        let mut offset = 0_i64;
        let mut terminal_controls_open = true;
        let shutdown_signal = shutdown_signal();
        tokio::pin!(shutdown_signal);

        loop {
            tokio::select! {
                signal = &mut shutdown_signal => {
                    eprintln!("Shutdown requested from {signal}.");
                    break;
                }
                command = self.terminal_rx.recv(), if terminal_controls_open => {
                    match command {
                        Some(TerminalCommand::Reset) => {
                            self.rooms.lock().await.reset_all_history();
                            eprintln!("All in-memory chat histories cleared; room settings kept.");
                        }
                        Some(TerminalCommand::Shutdown) => {
                            eprintln!("Shutdown requested from terminal.");
                            break;
                        }
                        None => {
                            terminal_controls_open = false;
                            eprintln!("Terminal input closed; terminal controls disabled.");
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
                            for mut update in updates {
                                offset = offset.max(update.update_id + 1);
                                if let Some(membership) = update.my_chat_member.take() {
                                    self.handle_membership_change(membership, &handles).await;
                                } else if let Some((kind, message)) = dispatchable_message(&mut update) {
                                    self.handle_update(kind, message, &handles, &mut dispatchers).await;
                                } else if let Some(message) = update.edited_message {
                                    // Updates already queued before edited messages were removed
                                    // from allowed_updates can still arrive. Never turn an edit
                                    // into a second billed provider call.
                                    log_update_route(
                                        message.chat.id,
                                        UpdateKind::EditedMessage,
                                        UpdateRoute::Ignored,
                                    );
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

            dispatchers.retain(|chat_id, dispatcher| {
                let keep = !dispatcher.handle.is_finished() && !dispatcher.sender.is_closed();
                if !keep {
                    remove_worker_registration(&workers, *chat_id, dispatcher.worker_id);
                }
                keep
            });
        }

        stop_chat_workers(dispatchers, &workers).await;
        if let Err(error) = handles.persistence.shutdown().await {
            eprintln!("failed to flush persistence writer during shutdown: {error}");
        }
        match spawn_blocking(move || persistence_thread.join()).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => eprintln!("persistence writer panicked during shutdown"),
            Err(error) => eprintln!("failed to join persistence writer: {error}"),
        }
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
                    let persisted = {
                        let mut config = self.config.lock().await;
                        let before = config.clone();
                        let changed = allow_chat_in_config(&mut config, chat_id);
                        if changed && let Err(error) = save_config(handles, &config).await {
                            *config = before;
                            Err(error)
                        } else {
                            Ok(changed)
                        }
                    };
                    let changed = match persisted {
                        Ok(changed) => changed,
                        Err(error) => {
                            eprintln!("failed to persist auto-approved chat {chat_id}: {error}");
                            return;
                        }
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
                    let telegram = handles.telegram.clone();
                    tokio::spawn(async move {
                        let _ = telegram.send_message(chat_id, &setup).await;
                    });
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
                send_to_chat_worker(chat_id, kind, message, handles, dispatchers);
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
                    let telegram = self.telegram.clone();
                    tokio::spawn(async move {
                        let _ = telegram
                            .send_message(chat_id, &unknown_chat_hint(chat_id))
                            .await;
                    });
                } else {
                    log_update_route(chat_id, kind, UpdateRoute::Ignored);
                }
            }
        }
    }
}

fn dispatchable_message(
    update: &mut tellm_telegram::Update,
) -> Option<(UpdateKind, IncomingMessage)> {
    update
        .message
        .take()
        .map(|message| (UpdateKind::Message, message))
}

/// Queue the update for its chat worker without ever blocking the poll
/// loop: awaiting a full queue would stall dispatch for every chat behind
/// one slow room, so a full queue drops the message with a busy notice.
fn send_to_chat_worker(
    chat_id: i64,
    kind: UpdateKind,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    dispatchers: &mut BTreeMap<i64, ChatDispatcher>,
) {
    let dispatcher = dispatchers
        .entry(chat_id)
        .or_insert_with(|| spawn_chat_worker(chat_id, handles.clone()));

    let dispatch = match dispatcher
        .sender
        .try_send(DispatchMessage { kind, message })
    {
        Ok(()) => return,
        Err(TrySendError::Full(_)) => {
            eprintln!("chat {chat_id} queue is full; dropping message");
            let should_send_notice =
                reserve_queue_full_notice(&handles.queue_full_notices, chat_id);
            if should_send_notice {
                let telegram = handles.telegram.clone();
                let pending = Arc::clone(&handles.queue_full_notices);
                tokio::spawn(async move {
                    let _guard = QueueFullNoticeGuard { chat_id, pending };
                    let _ = telegram
                        .send_message(
                            chat_id,
                            "Too many queued messages in this chat; this one was dropped. \
                             Resend it after the current reply.",
                        )
                        .await;
                });
            }
            return;
        }
        // The worker reaped itself after idling; respawn and retry.
        Err(TrySendError::Closed(dispatch)) => dispatch,
    };

    let dispatcher = spawn_chat_worker(chat_id, handles.clone());
    if dispatcher.sender.try_send(dispatch).is_err() {
        eprintln!("chat {chat_id} dispatch failed: fresh worker rejected the message");
    }
    dispatchers.insert(chat_id, dispatcher);
}

fn reserve_queue_full_notice(pending: &Arc<StdMutex<BTreeSet<i64>>>, chat_id: i64) -> bool {
    pending
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(chat_id)
}

fn spawn_chat_worker(chat_id: i64, handles: RuntimeHandles) -> ChatDispatcher {
    let (sender, mut receiver) = mpsc::channel::<DispatchMessage>(CHAT_QUEUE_SIZE);
    let worker_id = NEXT_CHAT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let workers = Arc::clone(&handles.workers);
    let worker_guard_registry = Arc::clone(&workers);
    let handle = tokio::spawn(async move {
        let _registry_guard = WorkerRegistryGuard {
            chat_id,
            worker_id,
            workers: worker_guard_registry,
        };
        loop {
            // A self-deny does not abort the task so it can send the command
            // confirmation. Observe its flag before waiting for another item
            // instead of retaining the denied room until the idle timeout.
            if worker_cancelled.load(Ordering::Acquire) || !chat_is_allowed(chat_id, &handles).await
            {
                break;
            }
            match timeout(CHAT_TASK_IDLE_TIMEOUT, receiver.recv()).await {
                Ok(Some(dispatch)) => {
                    if worker_cancelled.load(Ordering::Acquire)
                        || !chat_is_allowed(chat_id, &handles).await
                    {
                        break;
                    }
                    if let Err(error) = handle_allowed_message(
                        chat_id,
                        dispatch.kind,
                        dispatch.message,
                        &handles,
                        &worker_cancelled,
                    )
                    .await
                    {
                        eprintln!("chat {chat_id} dispatch failed: {error}");
                        if worker_can_reply(chat_id, &handles, &worker_cancelled).await {
                            let _ = handles
                                .telegram
                                .send_message(chat_id, &format!("tellm error: {error}"))
                                .await;
                        }
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

    workers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(
            chat_id,
            WorkerRegistration {
                worker_id,
                cancelled: Arc::clone(&cancelled),
                abort_handle: handle.abort_handle(),
            },
        );

    ChatDispatcher {
        sender,
        handle,
        worker_id,
        cancelled,
    }
}

fn remove_worker_registration(workers: &WorkerRegistry, chat_id: i64, worker_id: u64) {
    let mut workers = workers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if workers
        .get(&chat_id)
        .is_some_and(|worker| worker.worker_id == worker_id)
    {
        workers.remove(&chat_id);
    }
}

fn cancel_chat_worker(workers: &WorkerRegistry, chat_id: i64, abort: bool) {
    let abort_handle = {
        let workers = workers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        workers.get(&chat_id).map(|worker| {
            worker.cancelled.store(true, Ordering::Release);
            worker.abort_handle.clone()
        })
    };
    if abort && let Some(abort_handle) = abort_handle {
        abort_handle.abort();
    }
}

fn reactivate_chat_worker(workers: &WorkerRegistry, chat_id: i64) {
    let workers = workers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(worker) = workers.get(&chat_id) {
        worker.cancelled.store(false, Ordering::Release);
    }
}

async fn stop_chat_workers(dispatchers: BTreeMap<i64, ChatDispatcher>, workers: &WorkerRegistry) {
    let mut handles = Vec::with_capacity(dispatchers.len());
    for (chat_id, dispatcher) in dispatchers {
        dispatcher.cancelled.store(true, Ordering::Release);
        dispatcher.handle.abort();
        remove_worker_registration(workers, chat_id, dispatcher.worker_id);
        handles.push((chat_id, dispatcher.handle));
    }
    for (chat_id, handle) in handles {
        if let Err(error) = handle.await
            && !error.is_cancelled()
        {
            eprintln!("chat {chat_id} worker join failed during shutdown: {error}");
        }
    }
}

async fn chat_is_allowed(chat_id: i64, handles: &RuntimeHandles) -> bool {
    let access = handles.access.lock().await;
    access.is_chat_allowed(chat_id)
}

async fn worker_can_reply(chat_id: i64, handles: &RuntimeHandles, cancelled: &AtomicBool) -> bool {
    !cancelled.load(Ordering::Acquire) && chat_is_allowed(chat_id, handles).await
}

async fn handle_allowed_message(
    chat_id: i64,
    kind: UpdateKind,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
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
    handle_model_message(chat_id, message, handles, cancelled).await
}

async fn route_command(
    chat_id: i64,
    text: &str,
    message_date: i64,
    sender_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<Route, String> {
    if !chat_is_allowed(chat_id, handles).await {
        return Err("chat access was revoked".to_string());
    }
    let (room, default_model, model_keys, pinned_model_key, model_thinking, capabilities) = {
        let config = handles.config.lock().await;
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id).clone();
        let model_key = selected_model_key(&config, &room, chat_id);
        let model_thinking = config
            .models
            .get(&model_key)
            .map(|model| model.thinking)
            .unwrap_or_default();
        let capabilities = room_capabilities(&config, &room, chat_id);
        (
            room,
            config.default_model.clone(),
            config.models.keys().cloned().collect::<BTreeSet<_>>(),
            pinned_model_key(&config, chat_id).map(str::to_string),
            model_thinking,
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
        model_thinking,
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
                room.settings.thinking = None;
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
                save_config(handles, &config).await?;
            }
            mutate_room(handles, chat_id, |room| {
                room.settings.thinking = None;
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
                    save_config(handles, &config).await?;
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
                    save_config(handles, &config).await?;
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
        CommandAction::ShowReasoning {
            override_level,
            model_default,
        } => {
            send_command_reply(
                handles,
                chat_id,
                &format_reasoning_status(override_level, model_default),
            )
            .await
        }
        CommandAction::SetReasoning { thinking } => {
            mutate_room(handles, chat_id, |room| {
                room.settings.thinking = thinking;
            })
            .await?;
            send_command_reply(handles, chat_id, &reasoning_set_reply(thinking)).await
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
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let input = content_parts_from_message(&handles.telegram, &message).await?;
    if input.is_empty() {
        return Ok(());
    }
    if !worker_can_reply(chat_id, handles, cancelled).await {
        return Ok(());
    }

    let prepared = build_chat_request(chat_id, input, handles).await?;
    if let Some(notice) = prepared.reset_notice.as_deref()
        && worker_can_reply(chat_id, handles, cancelled).await
    {
        let _ = handles.telegram.send_message(chat_id, notice).await;
    }

    let typing = AbortOnDrop(spawn_typing_indicator(handles.telegram.clone(), chat_id));
    let response = dispatch_provider(&prepared.model_config, &prepared.request).await;
    typing.abort();

    match response {
        Ok(mut response) => {
            if !worker_can_reply(chat_id, handles, cancelled).await {
                return Ok(());
            }
            let turn_items = std::mem::take(&mut response.turn_items);
            let committed = {
                let mut rooms = handles.rooms.lock().await;
                rooms.get_mut(chat_id).is_some_and(|room| {
                    room.append_turn_if_generation(
                        prepared.generation,
                        prepared.model_config.wire_format,
                        turn_items,
                    )
                })
            };
            if committed && worker_can_reply(chat_id, handles, cancelled).await {
                send_model_response(&handles.telegram, chat_id, response).await
            } else {
                Ok(())
            }
        }
        Err(error) => {
            let restored = {
                let mut rooms = handles.rooms.lock().await;
                if let Some(room) = rooms.get_mut(chat_id) {
                    room.restore_failed_turn(prepared.generation, prepared.before_state)
                } else {
                    false
                }
            };
            // This reply IS the error handling — returning Err here would
            // make the chat worker send a second "tellm error" message for
            // the same failure.
            eprintln!("chat {chat_id} model call failed: {error}");
            if restored && worker_can_reply(chat_id, handles, cancelled).await {
                let _ = handles
                    .telegram
                    .send_message(chat_id, &provider_error_reply(&error))
                    .await;
            }
            Ok(())
        }
    }
}

struct PreparedChatRequest {
    model_config: ModelConfig,
    request: ChatRequest,
    before_state: RoomState,
    generation: u64,
    reset_notice: Option<String>,
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
) -> Result<PreparedChatRequest, String> {
    if !chat_is_allowed(chat_id, handles).await {
        return Err("chat access was revoked".to_string());
    }
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
    let generation = room.generation();

    Ok(PreparedChatRequest {
        model_config,
        request,
        before_state,
        generation,
        reset_notice,
    })
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
        thinking: room.settings.thinking.unwrap_or(model_config.thinking),
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
            let api_key = required_api_key(model).await?;
            Anthropic::new(api_key, model.base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())
        }
        WireFormat::Responses => {
            let api_key = required_api_key(model).await?;
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
            let requested_model = request.model.clone();
            let api_key = compat_api_key(model).await?;
            // Register before the request: an aborted task can still leave
            // Ollama loading the model after the HTTP future is dropped.
            remember_local_ollama_model(&base_url, &requested_model);
            let response = Compat::new(api_key, base_url.clone())
                .chat(request)
                .await
                .map_err(|error| error.to_string())?;
            Ok(response)
        }
        WireFormat::Gemini => {
            let api_key = required_api_key(model).await?;
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

fn ollama_child() -> &'static StdMutex<Option<ManagedOllamaChild>> {
    OLLAMA_CHILD.get_or_init(|| StdMutex::new(None))
}

fn ollama_loaded_models() -> &'static StdMutex<BTreeSet<OllamaLoadedModel>> {
    OLLAMA_LOADED_MODELS.get_or_init(|| StdMutex::new(BTreeSet::new()))
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
    let child = ManagedOllamaChild::new(child);
    let pid = child.id().expect("newly spawned child has a pid");
    *ollama_child()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(child);
    eprintln!("started `ollama serve` with pid {pid}");
    Ok(())
}

async fn stop_started_ollama() {
    let Some(child) = take_started_ollama_child() else {
        return;
    };
    unload_started_ollama_models().await;
    match spawn_blocking(move || child.stop()).await {
        Ok(Ok(message)) => eprintln!("{message}"),
        Ok(Err(error)) => eprintln!("failed to stop spawned Ollama process: {error}"),
        Err(error) => eprintln!("failed to join Ollama shutdown task: {error}"),
    }
}

fn stop_started_ollama_blocking() {
    let Some(child) = take_started_ollama_child() else {
        return;
    };
    unload_started_ollama_models_blocking();
    match child.stop() {
        Ok(message) => eprintln!("{message}"),
        Err(error) => eprintln!("failed to stop spawned Ollama process: {error}"),
    }
}

fn take_started_ollama_child() -> Option<ManagedOllamaChild> {
    ollama_child()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
}

fn remember_local_ollama_model(base_url: &str, model: &str) {
    if local_ollama_addr(base_url).is_none() {
        return;
    }

    ollama_loaded_models()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(OllamaLoadedModel {
            base_url: base_url.to_string(),
            model: model.to_string(),
        });
}

async fn unload_started_ollama_models() {
    let summary = unload_tracked_ollama_models().await;
    log_ollama_unload_summary(summary);
}

fn unload_started_ollama_models_blocking() {
    let summary = unload_tracked_ollama_models_blocking();
    log_ollama_unload_summary(summary);
}

fn log_ollama_unload_summary(summary: OllamaUnloadSummary) {
    for model in summary.unloaded {
        eprintln!("unloaded Ollama model {model}");
    }
    for model in summary.not_loaded {
        eprintln!("Ollama model {model} was already not loaded");
    }
    for (model, error) in summary.failed {
        eprintln!("failed to unload Ollama model {model}: {error}");
    }
}

async fn unload_tracked_ollama_models() -> OllamaUnloadSummary {
    spawn_blocking(unload_tracked_ollama_models_blocking)
        .await
        .unwrap_or_else(|error| OllamaUnloadSummary {
            attempted: 1,
            failed: vec![(
                "tracked Ollama models".to_string(),
                format!("Ollama unload task failed: {error}"),
            )],
            ..OllamaUnloadSummary::default()
        })
}

fn unload_tracked_ollama_models_blocking() -> OllamaUnloadSummary {
    let models = {
        let models = ollama_loaded_models()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        models.iter().cloned().collect::<Vec<_>>()
    };
    let mut summary = OllamaUnloadSummary {
        attempted: models.len(),
        ..OllamaUnloadSummary::default()
    };

    for model in models {
        match unload_ollama_model(&model) {
            Ok(OllamaUnloadOutcome::Unloaded) => {
                ollama_loaded_models()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&model);
                summary.unloaded.push(model.model);
            }
            Ok(OllamaUnloadOutcome::NotLoaded) => {
                ollama_loaded_models()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&model);
                summary.not_loaded.push(model.model);
            }
            Err(error) => summary.failed.push((model.model, error)),
        }
    }

    summary
}

fn unload_ollama_model(model: &OllamaLoadedModel) -> Result<OllamaUnloadOutcome, String> {
    let addr = local_ollama_addr(&model.base_url)
        .ok_or_else(|| format!("not a local Ollama endpoint: {}", model.base_url))?;
    unload_ollama_model_blocking(&addr, &model.model)
}

fn unload_ollama_model_blocking(addr: &str, model: &str) -> Result<OllamaUnloadOutcome, String> {
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
    ollama_unload_response_outcome(&response)
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
    matches!(http_response_status_code(response), Some(code) if (200..300).contains(&code))
}

fn http_response_status_code(response: &str) -> Option<u16> {
    response
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
}

fn ollama_unload_response_outcome(response: &str) -> Result<OllamaUnloadOutcome, String> {
    if http_response_is_success(response) {
        return Ok(OllamaUnloadOutcome::Unloaded);
    }

    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or("");
    if http_response_status_code(response) == Some(404)
        || body.to_ascii_lowercase().contains("not found")
    {
        return Ok(OllamaUnloadOutcome::NotLoaded);
    }

    Err(format!(
        "Ollama unload returned {}",
        response.lines().next().unwrap_or("an empty HTTP response")
    ))
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
    if !summary.not_loaded.is_empty() {
        parts.push(format!(
            "Already not loaded local Ollama model{}: {}.",
            plural(summary.not_loaded.len()),
            summary.not_loaded.join(", ")
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

#[cfg(unix)]
async fn shutdown_signal() -> &'static str {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = sigint.recv() => "SIGINT",
        _ = sigterm.recv() => "SIGTERM",
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> &'static str {
    let _ = tokio::signal::ctrl_c().await;
    "Ctrl-C"
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

    let fallback_reason = match terminate_ollama_child(&mut child) {
        Ok(method) => {
            if let Some(status) = wait_for_ollama_exit(&mut child)? {
                return Ok(format!(
                    "stopped tellm-started `ollama serve` pid {pid} after {method} with {status}"
                ));
            }
            format!("{method} timeout")
        }
        Err(error) => format!("failed graceful stop: {error}"),
    };

    if let Some(status) = child
        .try_wait()
        .map_err(|error| format!("could not inspect pid {pid}: {error}"))?
    {
        return Ok(format!(
            "stopped tellm-started `ollama serve` pid {pid} after {fallback_reason} with {status}"
        ));
    }

    child
        .kill()
        .map_err(|error| format!("could not kill pid {pid} after {fallback_reason}: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for pid {pid}: {error}"))?;
    Ok(format!(
        "stopped tellm-started `ollama serve` pid {pid} with SIGKILL after {fallback_reason}: {status}"
    ))
}

#[cfg(unix)]
fn terminate_ollama_child(child: &mut Child) -> Result<&'static str, String> {
    let pid = child.id();
    let raw_pid = i32::try_from(pid).map_err(|_| format!("pid {pid} does not fit in pid_t"))?;
    let result = unsafe { libc::kill(raw_pid, libc::SIGTERM) };
    if result == 0 {
        Ok("SIGTERM")
    } else {
        Err(format!(
            "could not send SIGTERM to pid {pid}: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(not(unix))]
fn terminate_ollama_child(_child: &mut Child) -> Result<&'static str, String> {
    Err("graceful process termination is unsupported on this platform".to_string())
}

fn wait_for_ollama_exit(child: &mut Child) -> Result<Option<std::process::ExitStatus>, String> {
    let pid = child.id();
    let deadline = StdInstant::now() + OLLAMA_TERMINATE_WAIT;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("could not inspect pid {pid}: {error}"))?
        {
            return Ok(Some(status));
        }
        if StdInstant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(OLLAMA_TERMINATE_POLL);
    }
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
        let bytes = download_attachment(telegram, &photo.file_id, photo.file_size, "photo").await?;
        parts.push(ContentPart::Image {
            media_type: "image/jpeg".to_string(),
            base64: BASE64.encode(bytes),
        });
    }

    if let Some(document) = &message.document {
        let bytes =
            download_attachment(telegram, &document.file_id, document.file_size, "document")
                .await?;
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

async fn download_attachment(
    telegram: &Telegram,
    file_id: &str,
    announced_size: Option<i64>,
    kind: &str,
) -> Result<Vec<u8>, String> {
    if let Some(size) = announced_size
        && size >= 0
    {
        validate_attachment_size(kind, size as usize)?;
    }

    let bytes = telegram
        .get_file_bytes(file_id, MAX_ATTACHMENT_BYTES)
        .await
        .map_err(|error| match error {
            TelegramError::FileTooLarge { size, .. } => attachment_too_large_error(kind, size),
            error => error.to_string(),
        })?;
    validate_attachment_size(kind, bytes.len())?;
    Ok(bytes)
}

fn validate_attachment_size(kind: &str, size: usize) -> Result<(), String> {
    if size > MAX_ATTACHMENT_BYTES {
        Err(attachment_too_large_error(kind, size))
    } else {
        Ok(())
    }
}

fn attachment_too_large_error(kind: &str, size: usize) -> String {
    format!(
        "{kind} is too large ({:.1} MiB); maximum attachment size is {} MiB",
        size as f64 / (1024.0 * 1024.0),
        MAX_ATTACHMENT_BYTES / (1024 * 1024)
    )
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
        let before = config.clone();
        let changed = allow_chat_in_config(&mut config, target_chat_id);
        if changed {
            if let Err(error) = save_config(handles, &config).await {
                *config = before;
                return Err(format!(
                    "failed to persist allowed chat {target_chat_id}: {error}"
                ));
            }
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
        let before = config.clone();
        let result = deny_chat_in_config(&mut config, target_chat_id);

        // Revocation is a live safety boundary, not a consequence of disk
        // latency. Close it before waiting behind any older persistence work
        // so an in-flight provider result cannot commit or send in the gap.
        let was_allowed = {
            let mut access = handles.access.lock().await;
            let was_allowed = access.is_chat_allowed(target_chat_id);
            access.deny_chat(target_chat_id);
            was_allowed
        };
        // Flip the cancellation flag before aborting so even a provider future
        // racing its final commit sees revocation synchronously. A command that
        // denies its own chat is allowed to finish the confirmation reply; its
        // worker exits before dequeuing anything else.
        cancel_chat_worker(
            &handles.workers,
            target_chat_id,
            target_chat_id != admin_chat_id,
        );

        if result.changed() {
            if let Err(error) = save_config(handles, &config).await {
                *config = before;
                if was_allowed {
                    handles.access.lock().await.allow_chat(target_chat_id);
                }
                // Only a self-deny leaves the task alive. Restore its flag so
                // the normal worker error reply can report the failed save;
                // an aborted target worker and its queued work stay dropped.
                if target_chat_id == admin_chat_id {
                    reactivate_chat_worker(&handles.workers, target_chat_id);
                }
                return Err(format!(
                    "failed to persist denied chat {target_chat_id}: {error}"
                ));
            }
        }
        result
    };

    let _persist = handles.room_persist.lock().await;
    let settings = {
        let mut rooms = handles.rooms.lock().await;
        rooms.remove(target_chat_id);
        rooms.settings()
    };
    if let Err(error) = save_room_settings(handles, settings).await {
        // Access remains denied and the in-memory room remains absent. A
        // later settings save retries the complete snapshot; never recreate a
        // denied room merely because cleanup persistence failed.
        return Err(format!(
            "chat {target_chat_id} was denied, but its room cleanup could not be persisted: {error}"
        ));
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
            let became_owner = match persist_provisional_pair(chat_id, pairer_user_id, handles)
                .await
            {
                Ok(became_owner) => became_owner,
                Err(error) => {
                    let _ = handles
                        .telegram
                        .send_message(
                            chat_id,
                            "Pairing could not be saved, so this chat remains denied. Contact the bot owner after fixing local config storage.",
                        )
                        .await;
                    return Err(error);
                }
            };
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

async fn persist_provisional_pair(
    chat_id: i64,
    pairer_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<bool, String> {
    match persist_paired_chat(chat_id, pairer_user_id, handles).await {
        Ok(became_owner) => Ok(became_owner),
        Err(error) => {
            // attempt_pair grants live access after a constant-time code
            // match. Revoke that provisional grant if durability fails, or
            // the room would remain usable until restart.
            handles.access.lock().await.deny_chat(chat_id);
            Err(format!(
                "pairing matched but could not be persisted; access was revoked: {error}"
            ))
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
        model_thinking: config
            .models
            .get(&config.default_model)
            .map(|model| model.thinking)
            .unwrap_or_default(),
        shutdown_access: crate::access::ShutdownAccess::NotAdmin,
        capabilities: commands::RoomCapabilities::permissive(),
    };
    match commands::route(text, &context) {
        Route::Command(CommandAction::Pair { code }) => Some(code),
        _ => None,
    }
}

/// Compute what the room's effective model can honor. Statically
/// knowable from the wire format plus endpoint checks; model-level variation
/// inside a capable format stays a request-time error.
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
            } else if tellm_openai::is_meta_model_api_endpoint(
                model.base_url.as_deref(),
                &model.model_name,
            ) {
                (true, false, "Meta Model API Responses")
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
) -> Result<bool, String> {
    let mut config = handles.config.lock().await;
    let before = config.clone();
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
        if let Err(error) = save_config(handles, &config).await {
            *config = before;
            return Err(error);
        }
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
    let _persist = handles.room_persist.lock().await;
    let (before, settings) = {
        let mut rooms = handles.rooms.lock().await;
        let before = rooms.get(chat_id).cloned();
        let room = rooms.get_or_default(chat_id);
        mutate(room);
        (before, rooms.settings())
    };
    if let Err(error) = save_room_settings(handles, settings).await {
        let mut rooms = handles.rooms.lock().await;
        match before {
            Some(before) => {
                if let Some(current) = rooms.get_mut(chat_id) {
                    // Only settings are durable. Roll them back, but never
                    // resurrect history invalidated by this command or by a
                    // concurrent terminal reset.
                    current.settings = before.settings;
                } else {
                    rooms.insert(chat_id, before);
                }
            }
            None => {
                if rooms.get(chat_id).is_some() {
                    rooms.remove(chat_id);
                }
            }
        }
        return Err(format!(
            "failed to persist room {chat_id}; settings mutation rolled back: {error}"
        ));
    }
    Ok(())
}

async fn save_room_settings(
    handles: &RuntimeHandles,
    settings: BTreeMap<i64, rooms::RoomSettings>,
) -> Result<(), String> {
    handles.persistence.save_rooms(settings).await
}

async fn save_config(handles: &RuntimeHandles, config: &Config) -> Result<(), String> {
    handles.persistence.save_config(config.clone()).await
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

async fn required_api_key(model: &ModelConfig) -> Result<String, String> {
    let secret_name = model
        .api_key_secret
        .clone()
        .ok_or_else(|| format!("model {} has no api_key_secret", model.model_name))?;
    fetch_secret(secret_name).await
}

async fn compat_api_key(model: &ModelConfig) -> Result<String, String> {
    let Some(secret_name) = model.api_key_secret.clone() else {
        return Ok(String::new());
    };
    fetch_secret(secret_name).await
}

/// OS keychain access blocks (and can prompt on macOS); keep the per-request
/// secret read off the async worker threads.
async fn fetch_secret(secret_name: String) -> Result<String, String> {
    spawn_blocking(move || {
        secrets::get(&secret_name).ok_or_else(|| missing_provider_secret_error(&secret_name))
    })
    .await
    .map_err(|error| format!("secret lookup task failed: {error}"))?
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

fn format_reasoning_status(
    override_level: Option<tellm_core::ThinkingLevel>,
    model_default: tellm_core::ThinkingLevel,
) -> String {
    match override_level {
        Some(level) => {
            format!("Reasoning: {level:?} (room override; model default is {model_default:?}).")
        }
        None => format!("Reasoning: {model_default:?} (model default)."),
    }
}

fn reasoning_set_reply(thinking: Option<tellm_core::ThinkingLevel>) -> String {
    match thinking {
        Some(thinking) => format!("Reasoning set to {thinking:?} for this room."),
        None => "Reasoning reset to this model's configured default.".to_string(),
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
        "Model set to {model_key}. Chat history reset. This room now uses the model's configured reasoning default; /reasoning can override it. Owners can lock it into config.toml with /model pin {model_key}."
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
            format!(
                "Unknown reasoning level \"{value}\". Use default, off, low, medium, high, or max."
            )
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
- /reasoning default|off|low|medium|high|max - show or set reasoning level
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

    #[tokio::test]
    async fn persistence_writer_orders_an_abandoned_write_before_the_next_snapshot() {
        let observed = Arc::new(StdMutex::new(Vec::<Vec<i64>>::new()));
        let thread_observed = Arc::clone(&observed);
        let release_first = Arc::new(std::sync::Barrier::new(2));
        let thread_release = Arc::clone(&release_first);
        let (started_tx, started_rx) = std_mpsc::channel();
        let (writer, thread) = spawn_persistence_writer_with(
            |_config: Config| Ok(()),
            move |settings| {
                let ids = settings.keys().copied().collect::<Vec<_>>();
                if ids == [1] {
                    let _ = started_tx.send(());
                    thread_release.wait();
                }
                thread_observed
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .push(ids);
                Ok(())
            },
        )
        .expect("spawn persistence writer");

        let first_writer = writer.clone();
        let first = tokio::spawn(async move {
            first_writer
                .save_rooms(BTreeMap::from([(1, rooms::RoomSettings::default())]))
                .await
        });
        spawn_blocking(move || started_rx.recv())
            .await
            .expect("join start wait")
            .expect("first write should start");
        first.abort();

        let second_writer = writer.clone();
        let second = tokio::spawn(async move {
            second_writer
                .save_rooms(BTreeMap::from([(2, rooms::RoomSettings::default())]))
                .await
        });
        spawn_blocking(move || release_first.wait())
            .await
            .expect("release first write");

        second
            .await
            .expect("second caller should run")
            .expect("second write should succeed");
        writer.shutdown().await.expect("writer should shut down");
        spawn_blocking(move || thread.join())
            .await
            .expect("join persistence thread")
            .expect("persistence thread should not panic");

        assert_eq!(
            *observed
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
            vec![vec![1], vec![2]]
        );
    }

    #[tokio::test]
    async fn failed_pair_persistence_revokes_live_access_and_rearms_the_room() {
        let mut codes = ["123456", "654321"].into_iter();
        let mut access = AccessControl::new_with_generator(AccessConfig::default(), move || {
            codes.next().expect("enough test pairing codes").to_string()
        });
        let now = now_access_time();
        let first = access.arm_room(42, now).expect("room should arm");
        assert_eq!(
            access.attempt_pair(42, &first.code, now),
            PairingAttempt::Paired
        );
        assert_eq!(access.check_chat(42), ChatAccess::Allowed);

        let config = Config {
            default_model: "openai".to_string(),
            models: BTreeMap::from([(
                "openai".to_string(),
                test_model(WireFormat::Responses, &[]),
            )]),
            telegram: tellm_config::TelegramConfig::default(),
        };
        let config = Arc::new(Mutex::new(config));
        let (persistence, thread) = spawn_persistence_writer_with(
            |_config: Config| Err("injected config failure".to_string()),
            |_settings| Ok(()),
        )
        .expect("spawn persistence writer");
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);
        let handles = RuntimeHandles {
            telegram: Telegram::new("test-token"),
            config: Arc::clone(&config),
            rooms: Arc::new(Mutex::new(RoomStates::default())),
            access: Arc::new(Mutex::new(access)),
            terminal_prompts: Arc::new(StdMutex::new(None)),
            bot_username: None,
            shutdown_tx,
            room_persist: Arc::new(Mutex::new(())),
            persistence: persistence.clone(),
            workers: Arc::new(StdMutex::new(BTreeMap::new())),
            queue_full_notices: Arc::new(StdMutex::new(BTreeSet::new())),
        };

        let error = persist_provisional_pair(42, Some(7), &handles)
            .await
            .unwrap_err();

        assert!(error.contains("access was revoked"), "{error}");
        assert!(config.lock().await.telegram.allowed_chat_ids.is_empty());
        let mut access = handles.access.lock().await;
        assert!(matches!(access.check_chat(42), ChatAccess::Unknown { .. }));
        let rearmed = access
            .arm_room(42, now)
            .expect("failed persistence should permit a new pairing attempt");
        assert_eq!(rearmed.code, "654321");
        drop(access);

        persistence
            .shutdown()
            .await
            .expect("writer should shut down");
        spawn_blocking(move || thread.join())
            .await
            .expect("join persistence thread")
            .expect("persistence thread should not panic");
    }

    #[tokio::test]
    async fn cancellation_registry_marks_and_aborts_in_flight_worker() {
        let workers: WorkerRegistry = Arc::new(StdMutex::new(BTreeMap::new()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let handle = tokio::spawn(std::future::pending::<()>());
        workers.lock().unwrap().insert(
            42,
            WorkerRegistration {
                worker_id: 1,
                cancelled: Arc::clone(&cancelled),
                abort_handle: handle.abort_handle(),
            },
        );

        cancel_chat_worker(&workers, 42, true);

        assert!(cancelled.load(Ordering::Acquire));
        assert!(handle.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn deny_revokes_and_cancels_before_config_persistence_finishes() {
        let config = Config {
            default_model: "openai".to_string(),
            models: BTreeMap::from([(
                "openai".to_string(),
                test_model(WireFormat::Responses, &[]),
            )]),
            telegram: tellm_config::TelegramConfig {
                allowed_chat_ids: vec![42],
                owner_user_ids: vec![7],
            },
        };
        let access = AccessControl::new(AccessConfig::from_config(&config), now_access_time());
        let config = Arc::new(Mutex::new(config));
        let rooms = Arc::new(Mutex::new(RoomStates::default()));
        rooms.lock().await.get_or_default(42).settings.role = Some("preserve me".to_string());

        let release_save = Arc::new(std::sync::Barrier::new(2));
        let writer_release = Arc::clone(&release_save);
        let (save_started_tx, save_started_rx) = std_mpsc::channel();
        let (persistence, writer_thread) = spawn_persistence_writer_with(
            move |_config: Config| {
                let _ = save_started_tx.send(());
                writer_release.wait();
                Err("injected config failure".to_string())
            },
            |_settings| Ok(()),
        )
        .expect("spawn persistence writer");

        let workers: WorkerRegistry = Arc::new(StdMutex::new(BTreeMap::new()));
        let cancelled = Arc::new(AtomicBool::new(false));
        let target_worker = tokio::spawn(std::future::pending::<()>());
        workers.lock().unwrap().insert(
            42,
            WorkerRegistration {
                worker_id: 1,
                cancelled: Arc::clone(&cancelled),
                abort_handle: target_worker.abort_handle(),
            },
        );
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);
        let handles = RuntimeHandles {
            telegram: Telegram::new("test-token"),
            config: Arc::clone(&config),
            rooms: Arc::clone(&rooms),
            access: Arc::new(Mutex::new(access)),
            terminal_prompts: Arc::new(StdMutex::new(None)),
            bot_username: None,
            shutdown_tx,
            room_persist: Arc::new(Mutex::new(())),
            persistence: persistence.clone(),
            workers,
            queue_full_notices: Arc::new(StdMutex::new(BTreeSet::new())),
        };

        let task_handles = handles.clone();
        let deny = tokio::spawn(async move { handle_deny_chat(7, 42, &task_handles).await });
        spawn_blocking(move || save_started_rx.recv())
            .await
            .expect("join save-start wait")
            .expect("config save should start");

        assert!(cancelled.load(Ordering::Acquire));
        assert!(!handles.access.lock().await.is_chat_allowed(42));

        spawn_blocking(move || release_save.wait())
            .await
            .expect("release failed config save");
        let error = deny
            .await
            .expect("deny task should run")
            .expect_err("injected save failure should be returned");
        assert!(error.contains("injected config failure"), "{error}");

        // A failed config transaction restores durable/live policy, while
        // already-aborted provider work and its queue remain dropped.
        assert!(config.lock().await.telegram.allowed_chat_ids.contains(&42));
        assert!(handles.access.lock().await.is_chat_allowed(42));
        assert_eq!(
            rooms.lock().await.get(42).unwrap().settings.role.as_deref(),
            Some("preserve me")
        );
        assert!(target_worker.await.unwrap_err().is_cancelled());

        persistence
            .shutdown()
            .await
            .expect("writer should shut down");
        spawn_blocking(move || writer_thread.join())
            .await
            .expect("join persistence writer")
            .expect("persistence writer should not panic");
    }

    #[tokio::test]
    async fn abort_on_drop_cancels_owned_background_task() {
        struct NotifyOnDrop(Option<tokio::sync::oneshot::Sender<()>>);
        impl Drop for NotifyOnDrop {
            fn drop(&mut self) {
                if let Some(sender) = self.0.take() {
                    let _ = sender.send(());
                }
            }
        }

        let (dropped_tx, dropped_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let _notify = NotifyOnDrop(Some(dropped_tx));
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;

        drop(AbortOnDrop(handle));

        timeout(Duration::from_secs(1), dropped_rx)
            .await
            .expect("background task should be aborted")
            .expect("drop notification should arrive");
    }

    #[test]
    fn queue_full_notice_is_coalesced_per_room_until_send_finishes() {
        let pending = Arc::new(StdMutex::new(BTreeSet::new()));
        assert!(reserve_queue_full_notice(&pending, 42));
        assert!(!reserve_queue_full_notice(&pending, 42));
        assert!(reserve_queue_full_notice(&pending, 7));

        drop(QueueFullNoticeGuard {
            chat_id: 42,
            pending: Arc::clone(&pending),
        });

        assert!(reserve_queue_full_notice(&pending, 42));
    }

    #[test]
    fn edited_updates_are_never_dispatchable_as_model_or_command_messages() {
        let edited = IncomingMessage {
            chat: tellm_telegram::Chat {
                id: 42,
                title: None,
                kind: None,
            },
            from: None,
            date: 1000,
            text: Some("edited prompt".to_string()),
            caption: None,
            photo: None,
            document: None,
        };
        let mut update = tellm_telegram::Update {
            update_id: 7,
            message: None,
            edited_message: Some(edited),
            my_chat_member: None,
        };

        assert!(dispatchable_message(&mut update).is_none());
        assert!(update.edited_message.is_some());
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
                not_loaded: Vec::new(),
                failed: Vec::new(),
            }),
            "Unloaded local Ollama models: llama3.3:70b, qwen3:32b."
        );
        let partial = ollama_unload_reply(&OllamaUnloadSummary {
            attempted: 3,
            unloaded: vec!["llama3.3:70b".to_string()],
            not_loaded: vec!["gemma4:31b-mlx".to_string()],
            failed: vec![("qwen3:32b".to_string(), "connection refused".to_string())],
        });
        assert!(
            partial.contains("Unloaded local Ollama model: llama3.3:70b."),
            "{partial}"
        );
        assert!(
            partial.contains("Already not loaded local Ollama model: gemma4:31b-mlx."),
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
    fn ollama_unload_response_treats_missing_model_as_terminal() {
        assert_eq!(
            ollama_unload_response_outcome("HTTP/1.1 200 OK\r\n\r\n{}"),
            Ok(OllamaUnloadOutcome::Unloaded)
        );
        assert_eq!(
            ollama_unload_response_outcome("HTTP/1.1 404 Not Found\r\n\r\n{}"),
            Ok(OllamaUnloadOutcome::NotLoaded)
        );
        assert_eq!(
            ollama_unload_response_outcome(
                "HTTP/1.1 400 Bad Request\r\n\r\n{\"error\":\"model \\\"bad\\\" not found\"}"
            ),
            Ok(OllamaUnloadOutcome::NotLoaded)
        );

        let error = ollama_unload_response_outcome("HTTP/1.1 500 Server Error\r\n\r\n{}")
            .expect_err("500 remains a real unload failure");
        assert!(error.contains("HTTP/1.1 500 Server Error"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn stop_ollama_child_uses_sigterm_before_sigkill() {
        let child = ProcessCommand::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep test process");

        let message = stop_ollama_child(child).expect("stop child");
        assert!(message.contains("SIGTERM"), "{message}");
        assert!(!message.contains("SIGKILL"), "{message}");
    }

    #[cfg(unix)]
    #[test]
    fn runtime_ollama_cleanup_drop_stops_tracked_child() {
        drop(take_started_ollama_child());
        let child = ProcessCommand::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep test process");
        let pid = child.id();
        *ollama_child()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(ManagedOllamaChild::new(child));

        {
            let _guard = RuntimeOllamaCleanup;
        }

        assert!(take_started_ollama_child().is_none());
        assert!(!process_exists(pid), "pid {pid} should have exited");
    }

    #[cfg(unix)]
    fn process_exists(pid: u32) -> bool {
        let raw_pid = i32::try_from(pid).expect("test pid fits in pid_t");
        let result = unsafe { libc::kill(raw_pid, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
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

        assert!(reply.contains("configured reasoning default"), "{reply}");
        assert!(reply.contains("/reasoning can override"), "{reply}");
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
    fn attachment_size_limit_rejects_declared_or_downloaded_oversize_payloads() {
        assert!(validate_attachment_size("photo", MAX_ATTACHMENT_BYTES).is_ok());
        let error = validate_attachment_size("document", MAX_ATTACHMENT_BYTES + 1).unwrap_err();
        assert!(error.contains("document is too large"), "{error}");
        assert!(error.contains("20 MiB"), "{error}");
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
            allow_insecure_http: false,
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
        assert_eq!(request.thinking, tellm_core::ThinkingLevel::High);
    }

    #[test]
    fn chat_request_uses_room_thinking_override_when_present() {
        let room = RoomState::new(crate::rooms::RoomSettings {
            thinking: Some(tellm_core::ThinkingLevel::Low),
            ..crate::rooms::RoomSettings::default()
        });
        let model = ModelConfig {
            wire_format: WireFormat::Responses,
            model_name: "gpt-5.5".to_string(),
            base_url: None,
            allow_insecure_http: false,
            api_key_secret: Some("openai_api_key".to_string()),
            telegram_chat_ids: Vec::new(),
            thinking: tellm_core::ThinkingLevel::High,
        };

        let request = chat_request_from_room(
            &room,
            &model,
            vec![ContentPart::Text {
                text: "think less".to_string(),
            }],
        );

        assert_eq!(request.thinking, tellm_core::ThinkingLevel::Low);
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
    fn meta_model_api_responses_disable_image_generation_capability() {
        let room = RoomState::new(crate::rooms::RoomSettings {
            model_key: Some("meta".to_string()),
            ..crate::rooms::RoomSettings::default()
        });
        let mut config = Config {
            default_model: "meta".to_string(),
            models: BTreeMap::new(),
            telegram: tellm_config::TelegramConfig::default(),
        };

        let mut meta = test_model(WireFormat::Responses, &[]);
        meta.model_name = "muse-spark-1.1".to_string();
        meta.base_url = Some(tellm_openai::META_MODEL_API_BASE_URL.to_string());
        config.models.insert("meta".to_string(), meta);

        let capabilities = room_capabilities(&config, &room, 42);

        assert!(capabilities.web_search);
        assert!(!capabilities.image_generation);
        assert_eq!(capabilities.endpoint, "Meta Model API Responses");
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
            allow_insecure_http: false,
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

    #[tokio::test]
    async fn compat_api_key_distinguishes_keyless_from_missing_secret() {
        let mut keyless = test_model(WireFormat::Compat, &[]);
        keyless.api_key_secret = None;
        assert_eq!(compat_api_key(&keyless).await.unwrap(), "");

        let mut keyed = test_model(WireFormat::Compat, &[]);
        keyed.api_key_secret = Some("definitely_missing_tellm_test_secret".to_string());
        let error = compat_api_key(&keyed).await.unwrap_err();
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
            allow_insecure_http: false,
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
