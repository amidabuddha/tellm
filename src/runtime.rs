use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant as StdInstant, SystemTime, UNIX_EPOCH};

use tellm_config::{Config, WireFormat, secrets};
use tellm_core::ContentPart;
use tellm_telegram::{IncomingMessage, Telegram, TelegramError};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, mpsc};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::{sleep, timeout};

use crate::access::{AccessConfig, AccessControl, AccessTime, ChatAccess, PairingAttempt};
use crate::commands::{self, CommandAction, CommandContext, CommandReject, KnownCommand, Route};
use crate::model_turn::{self, PreparedChatRequest};
use crate::ollama;
use crate::persistence::{self, Persistence};
use crate::rooms::{self, ChatMode, RoomState, RoomStates};

const LONG_POLL_TIMEOUT_S: u32 = 20;
const GET_UPDATES_RETRY_DELAY: Duration = Duration::from_secs(2);
const GET_UPDATES_FAILURE_LOG_INTERVAL: Duration = Duration::from_secs(30);
const CHAT_QUEUE_SIZE: usize = 32;
const CHAT_TASK_IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const TYPING_INTERVAL: Duration = Duration::from_secs(4);
const TERMINAL_SECRET_PROMPT_TIMEOUT: Duration = Duration::from_secs(5 * 60);
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
    persistence: Persistence,
    workers: WorkerRegistry,
    queue_full_notices: Arc<StdMutex<BTreeSet<i64>>>,
}

struct ChatDispatcher {
    sender: mpsc::Sender<IncomingMessage>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShutdownReason {
    Telegram,
}

#[derive(Debug, Default)]
struct GetUpdatesLogState {
    consecutive_failures: u64,
    started_at: Option<StdInstant>,
    last_reported_at: Option<StdInstant>,
    last_reported_failure: u64,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GetUpdatesFailureReport {
    consecutive_failures: u64,
    elapsed: Duration,
    suppressed: u64,
    first: bool,
    error_changed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GetUpdatesRecovery {
    failures: u64,
    downtime: Duration,
}

impl GetUpdatesLogState {
    fn record_failure(&mut self, error: &str, now: StdInstant) -> Option<GetUpdatesFailureReport> {
        self.consecutive_failures += 1;
        let started_at = *self.started_at.get_or_insert(now);
        let first = self.consecutive_failures == 1;
        let error_changed = !first && self.last_error.as_deref() != Some(error);
        let report_interval_elapsed = self.last_reported_at.is_some_and(|last_reported| {
            now.duration_since(last_reported) >= GET_UPDATES_FAILURE_LOG_INTERVAL
        });
        self.last_error = Some(error.to_string());

        if !first && !error_changed && !report_interval_elapsed {
            return None;
        }

        let suppressed = self
            .consecutive_failures
            .saturating_sub(self.last_reported_failure)
            .saturating_sub(1);
        self.last_reported_at = Some(now);
        self.last_reported_failure = self.consecutive_failures;
        Some(GetUpdatesFailureReport {
            consecutive_failures: self.consecutive_failures,
            elapsed: now.duration_since(started_at),
            suppressed,
            first,
            error_changed,
        })
    }

    fn record_success(&mut self, now: StdInstant) -> Option<GetUpdatesRecovery> {
        if self.consecutive_failures == 0 {
            return None;
        }

        let recovery = GetUpdatesRecovery {
            failures: self.consecutive_failures,
            downtime: self
                .started_at
                .map_or(Duration::ZERO, |started_at| now.duration_since(started_at)),
        };
        *self = Self::default();
        Some(recovery)
    }
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
        let access = AccessControl::new(access_config);
        model_turn::warm_configured_provider_secrets(&config);
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
        let _ollama_cleanup = ollama::CleanupGuard;
        let bot = self.telegram.get_me().await?;
        let bot_username = bot.username;
        if let Some(username) = &bot_username {
            log::info!(
                target: "tellm",
                "version={} bot=@{username} status=running terminal_commands=reset,exit,quit",
                env!("CARGO_PKG_VERSION"),
            );
        } else {
            log::info!(
                target: "tellm",
                "version={} status=running terminal_commands=reset,exit,quit",
                env!("CARGO_PKG_VERSION"),
            );
        }

        let workers = Arc::new(StdMutex::new(BTreeMap::new()));
        let (persistence, persistence_thread) = persistence::spawn()?;
        let handles = RuntimeHandles {
            telegram: self.telegram.clone(),
            config: Arc::clone(&self.config),
            rooms: Arc::clone(&self.rooms),
            access: Arc::clone(&self.access),
            terminal_prompts: Arc::clone(&self.terminal_prompts),
            bot_username,
            shutdown_tx: self.shutdown_tx.clone(),
            persistence,
            workers: Arc::clone(&workers),
            queue_full_notices: Arc::new(StdMutex::new(BTreeSet::new())),
        };
        let mut dispatchers = BTreeMap::new();
        let mut offset = 0_i64;
        let mut get_updates_log = GetUpdatesLogState::default();
        let mut terminal_controls_open = true;
        let shutdown_signal = shutdown_signal();
        tokio::pin!(shutdown_signal);

        loop {
            tokio::select! {
                signal = &mut shutdown_signal => {
                    log::info!(target: "tellm", "shutdown requested source={signal}");
                    break;
                }
                command = self.terminal_rx.recv(), if terminal_controls_open => {
                    match command {
                        Some(TerminalCommand::Reset) => {
                            self.rooms.lock().await.reset_all_history();
                            log::info!(target: "tellm::terminal", "all in-memory chat histories cleared; room settings kept");
                        }
                        Some(TerminalCommand::Shutdown) => {
                            log::info!(target: "tellm", "shutdown requested source=terminal");
                            break;
                        }
                        None => {
                            terminal_controls_open = false;
                            log::warn!(target: "tellm::terminal", "input closed; terminal controls disabled");
                        }
                    }
                }
                reason = self.shutdown_rx.recv() => {
                    match reason {
                        Some(ShutdownReason::Telegram) => log::info!(target: "tellm", "shutdown requested source=telegram"),
                        None => log::info!(target: "tellm", "shutdown requested source=internal"),
                    }
                    break;
                }
                updates = self.telegram.get_updates(offset, LONG_POLL_TIMEOUT_S) => {
                    match updates {
                        Ok(updates) => {
                            if let Some(recovery) = get_updates_log.record_success(StdInstant::now()) {
                                log::info!(
                                    target: "tellm::telegram",
                                    "getUpdates recovered failures={} downtime={}",
                                    recovery.failures,
                                    format_log_duration(recovery.downtime),
                                );
                            }
                            for mut update in updates {
                                offset = offset.max(update.update_id + 1);
                                if let Some(membership) = update.my_chat_member.take() {
                                    self.handle_membership_change(membership, &handles).await;
                                } else if let Some(message) = dispatchable_message(&mut update) {
                                    self.handle_update(message, &handles, &mut dispatchers).await;
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
                            let error = error.to_string();
                            if let Some(report) = get_updates_log.record_failure(&error, StdInstant::now()) {
                                if report.first {
                                    log::warn!(
                                        target: "tellm::telegram",
                                        "getUpdates failed consecutive_failures=1 error={error:?} retry_in={}s",
                                        GET_UPDATES_RETRY_DELAY.as_secs(),
                                    );
                                } else {
                                    let status = if report.error_changed {
                                        "failure changed"
                                    } else {
                                        "still failing"
                                    };
                                    log::warn!(
                                        target: "tellm::telegram",
                                        "getUpdates {status} consecutive_failures={} elapsed={} suppressed={} error={error:?} retry_in={}s",
                                        report.consecutive_failures,
                                        format_log_duration(report.elapsed),
                                        report.suppressed,
                                        GET_UPDATES_RETRY_DELAY.as_secs(),
                                    );
                                }
                            }
                            sleep(GET_UPDATES_RETRY_DELAY).await;
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
            log::error!(target: "tellm::persistence", "shutdown flush failed error={error:?}");
        }
        match spawn_blocking(move || persistence_thread.join()).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => {
                log::error!(target: "tellm::persistence", "writer panicked during shutdown")
            }
            Err(error) => {
                log::error!(target: "tellm::persistence", "writer join failed error={error:?}")
            }
        }
        ollama::stop_started().await;
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
                        if changed
                            && let Err(error) = handles.persistence.save_config(&config).await
                        {
                            *config = before;
                            Err(error)
                        } else {
                            Ok(changed)
                        }
                    };
                    let changed = match persisted {
                        Ok(changed) => changed,
                        Err(error) => {
                            log::error!(
                                target: "tellm::persistence",
                                "auto-approved chat save failed chat_id={chat_id} error={error:?}"
                            );
                            return;
                        }
                    };
                    {
                        let mut access = self.access.lock().await;
                        access.allow_chat(chat_id);
                    }
                    log::info!(
                        target: "tellm::access",
                        "chat auto-approved chat={label:?} owner_user_id={user_id} changed={changed}"
                    );
                    if chat_id < 0 {
                        log::warn!(target: "tellm::telegram", "{}", group_privacy_hint(chat_id));
                    }
                    let setup = room_setup_reply(chat_id, false, handles).await;
                    let handles = handles.clone();
                    tokio::spawn(async move {
                        let _ = send_room_setup(chat_id, setup, &handles).await;
                    });
                    return;
                }

                let pairing = {
                    let mut access = self.access.lock().await;
                    access.arm_room(chat_id, now_access_time())
                };
                match pairing {
                    Some(pairing) => {
                        log::info!(
                            target: "tellm::access",
                            "added to chat chat={label:?} pairing_code={} action=\"send /pair {} in that chat, or /allow {chat_id} from an owner\"",
                            pairing.code,
                            pairing.code,
                        );
                        if chat_id < 0 {
                            log::warn!(target: "tellm::telegram", "{}", group_privacy_hint(chat_id));
                        }
                    }
                    None => log::info!(
                        target: "tellm::access",
                        "added to already-allowed chat chat={label:?}"
                    ),
                }
            }
            "left" | "kicked" => {
                log::info!(
                    target: "tellm::access",
                    "removed from chat chat={label:?} action=\"an owner can use /deny {chat_id} to clear access and room state\""
                );
            }
            _ => {}
        }
    }

    async fn handle_update(
        &self,
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
                send_to_chat_worker(chat_id, message, handles, dispatchers);
            }
            ChatAccess::Unknown { send_hint } => {
                // Arm (or refresh) this room's pairing code on any contact;
                // print it to the console only when newly issued.
                {
                    let mut access = self.access.lock().await;
                    if let Some(pairing) = access.arm_room(chat_id, now_access_time())
                        && pairing.newly_issued
                    {
                        log::info!(
                            target: "tellm::access",
                            "pairing code issued chat_id={chat_id} pairing_code={} action=\"send /pair {} in that chat\"",
                            pairing.code,
                            pairing.code,
                        );
                    }
                }
                if let Some(code) =
                    pair_code_from_message(&message, handles.bot_username.as_deref())
                {
                    log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Command);
                    let pairer = message.from.as_ref().map(|user| user.id);
                    if let Err(error) = handle_pair_attempt(chat_id, code, pairer, handles).await {
                        log::warn!(
                            target: "tellm::access",
                            "pairing attempt failed chat_id={chat_id} error={error:?}"
                        );
                    }
                } else if send_hint {
                    log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Ignored);
                    let telegram = self.telegram.clone();
                    tokio::spawn(async move {
                        let _ = telegram
                            .send_message(chat_id, &unknown_chat_hint(chat_id))
                            .await;
                    });
                } else {
                    log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Ignored);
                }
            }
        }
    }
}

fn dispatchable_message(update: &mut tellm_telegram::Update) -> Option<IncomingMessage> {
    update.message.take()
}

/// Queue the update for its chat worker without ever blocking the poll
/// loop: awaiting a full queue would stall dispatch for every chat behind
/// one slow room, so a full queue drops the message with a busy notice.
fn send_to_chat_worker(
    chat_id: i64,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    dispatchers: &mut BTreeMap<i64, ChatDispatcher>,
) {
    let dispatcher = dispatchers
        .entry(chat_id)
        .or_insert_with(|| spawn_chat_worker(chat_id, handles.clone()));

    let message = match dispatcher.sender.try_send(message) {
        Ok(()) => return,
        Err(TrySendError::Full(_)) => {
            log::warn!(
                target: "tellm::dispatcher",
                "chat queue full; dropping message chat_id={chat_id}"
            );
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
        Err(TrySendError::Closed(message)) => message,
    };

    let dispatcher = spawn_chat_worker(chat_id, handles.clone());
    if dispatcher.sender.try_send(message).is_err() {
        log::error!(
            target: "tellm::dispatcher",
            "fresh worker rejected message chat_id={chat_id}"
        );
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
    let (sender, mut receiver) = mpsc::channel::<IncomingMessage>(CHAT_QUEUE_SIZE);
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
                Ok(Some(message)) => {
                    if worker_cancelled.load(Ordering::Acquire)
                        || !chat_is_allowed(chat_id, &handles).await
                    {
                        break;
                    }
                    if let Err(error) =
                        handle_allowed_message(chat_id, message, &handles, &worker_cancelled).await
                    {
                        log::warn!(
                            target: "tellm::dispatcher",
                            "chat dispatch failed chat_id={chat_id} error={error:?}"
                        );
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
                    log::debug!(
                        target: "tellm::dispatcher",
                        "idle task reaped chat_id={chat_id}"
                    );
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
            log::error!(
                target: "tellm::dispatcher",
                "worker join failed during shutdown chat_id={chat_id} error={error:?}"
            );
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
    message: IncomingMessage,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    if let Some(text) = model_turn::message_text(&message) {
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
                log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Command);
                return handle_command(chat_id, action, handles).await;
            }
            Route::Ignore => {
                log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Ignored);
                return Ok(());
            }
            Route::UserMessage => {}
        }
    }

    if !model_turn::message_has_input(&message) {
        log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Ignored);
        return Ok(());
    }
    log_update_route(chat_id, UpdateKind::Message, UpdateRoute::Model);
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
    let (command, args) = match commands::parse(text, handles.bot_username.as_deref()) {
        commands::ParsedRoute::Command { command, args } => (command, args),
        commands::ParsedRoute::UserMessage => return Ok(Route::UserMessage),
        commands::ParsedRoute::Ignore => return Ok(Route::Ignore),
    };
    let privileged_access = {
        let access = handles.access.lock().await;
        access.check_privileged(
            sender_user_id,
            u64::try_from(message_date).unwrap_or_default(),
            now_access_time(),
        )
    };
    let action = {
        let config = handles.config.lock().await;
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id);
        let model_key = selected_model_key(&config, room, chat_id);
        let model_thinking = config
            .models
            .get(&model_key)
            .map(|model| model.thinking)
            .unwrap_or_default();
        let capabilities = room_capabilities(&config, room, chat_id);
        let model_keys = config.models.keys().cloned().collect::<BTreeSet<_>>();
        let context = CommandContext {
            settings: &room.settings,
            default_model: &config.default_model,
            model_keys: &model_keys,
            pinned_model_key: pinned_model_key(&config, chat_id),
            model_thinking,
            privileged_access,
            capabilities,
        };
        commands::resolve(command, args, &context)
    };
    Ok(Route::Command(action))
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
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
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
            let show_picker = pinned.is_none();
            let reply = format_model_status(selected, effective, pinned, available.clone());
            if show_picker {
                handles
                    .telegram
                    .send_model_picker(chat_id, &reply, &available)
                    .await
                    .map_err(|error| error.to_string())
            } else {
                send_command_reply(handles, chat_id, &reply).await
            }
        }
        CommandAction::SetModel { model_key } => {
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
                    room.settings.model_key = Some(model_key.clone());
                    reset_room_model_context(room);
                })
                .await?;
            send_command_reply(handles, chat_id, &model_set_reply(&model_key)).await
        }
        CommandAction::PinModel { model_key } => {
            {
                let mut config = handles.config.lock().await;
                let before = config.clone();
                for model in config.models.values_mut() {
                    model.telegram_chat_ids.retain(|pinned| *pinned != chat_id);
                }
                if let Some(model) = config.models.get_mut(&model_key) {
                    model.telegram_chat_ids.push(chat_id);
                }
                if let Err(error) = handles.persistence.save_config(&config).await {
                    *config = before;
                    return Err(format!("failed to persist model pin: {error}"));
                }
            }
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, reset_room_model_context)
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
                let before = config.clone();
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
                if was_pinned && let Err(error) = handles.persistence.save_config(&config).await {
                    *config = before;
                    return Err(format!("failed to persist model unpin: {error}"));
                }
                was_pinned
            };
            let reply = if was_pinned {
                handles
                    .persistence
                    .mutate_room(&handles.rooms, chat_id, reset_room_model_context)
                    .await?;
                "Room unpinned. Chat history reset. /model KEY now switches models here."
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
                    let before = config.clone();
                    config.models.insert(
                        preset.key.to_string(),
                        crate::wizard::model_config_from_preset(preset),
                    );
                    if let Err(error) = handles.persistence.save_config(&config).await {
                        *config = before;
                        return Err(format!("failed to persist added model: {error}"));
                    }
                    false
                }
            };

            let base_reply = model_add_base_reply(already, preset);
            if secrets::get(preset.api_key_secret).is_some() {
                let reply = model_key_ready_reply(&base_reply, preset.key);
                send_command_reply(handles, chat_id, &reply).await
            } else {
                prompt_for_model_secret(
                    handles,
                    chat_id,
                    preset.key,
                    preset.api_key_secret,
                    &base_reply,
                )
                .await
            }
        }
        CommandAction::ShowRole { current } => {
            let role = current.unwrap_or_else(|| "(none)".to_string());
            send_command_reply(handles, chat_id, &format!("Role: {role}")).await
        }
        CommandAction::SetRole { role } => {
            let cleared = role.is_none();
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
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
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
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
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
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
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
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
            let summary = ollama::unload_models().await;
            send_command_reply(handles, chat_id, &ollama::unload_reply(&summary)).await
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
        CommandAction::Reject { reason } => {
            send_command_reply(handles, chat_id, &format_reject(reason)).await
        }
    }
}

fn reset_room_model_context(room: &mut RoomState) {
    room.settings.thinking = None;
    room.wire_format = None;
    room.reset_history();
}

async fn handle_model_message(
    chat_id: i64,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let input = model_turn::content_parts_from_message(&handles.telegram, &message).await?;
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
    let response = model_turn::dispatch_provider(&prepared.model_config, &prepared.request).await;
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
                model_turn::send_response(&handles.telegram, chat_id, response).await
            } else {
                Ok(())
            }
        }
        Err(error) => {
            let restored = {
                let mut rooms = handles.rooms.lock().await;
                if let Some(room) = rooms.get_mut(chat_id) {
                    room.restore_failed_turn(prepared.generation, prepared.rollback)
                } else {
                    false
                }
            };
            // This reply IS the error handling — returning Err here would
            // make the chat worker send a second "tellm error" message for
            // the same failure.
            log::warn!(
                target: "tellm::model",
                "call failed chat_id={chat_id} error={error:?}"
            );
            if restored && worker_can_reply(chat_id, handles, cancelled).await {
                let _ = handles
                    .telegram
                    .send_message(chat_id, &model_turn::provider_error_reply(&error))
                    .await;
            }
            Ok(())
        }
    }
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
    Ok(model_turn::prepare_chat_request(room, model_config, input))
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

async fn handle_allow_chat(
    requester_chat_id: i64,
    target_chat_id: i64,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let changed = {
        let mut config = handles.config.lock().await;
        let before = config.clone();
        let changed = allow_chat_in_config(&mut config, target_chat_id);
        if changed && let Err(error) = handles.persistence.save_config(&config).await {
            *config = before;
            return Err(format!(
                "failed to persist allowed chat {target_chat_id}: {error}"
            ));
        }
        changed
    };

    {
        let mut access = handles.access.lock().await;
        access.allow_chat(target_chat_id);
    }

    if target_chat_id < 0 {
        log::warn!(
            target: "tellm::telegram",
            "{}",
            group_privacy_hint(target_chat_id)
        );
    }

    // The approved room gets the setup prompt too — approval via /allow must
    // feel the same as approval via /pair (best-effort: the bot may not be a
    // member of the target chat yet).
    if changed && target_chat_id != requester_chat_id {
        let setup = room_setup_reply(target_chat_id, false, handles).await;
        let _ = send_room_setup(target_chat_id, setup, handles).await;
    }

    let reply = if changed {
        format!("Allowed chat {target_chat_id}.")
    } else {
        format!("Chat {target_chat_id} is already allowed.")
    };
    send_command_reply(handles, requester_chat_id, &reply).await
}

async fn handle_deny_chat(
    requester_chat_id: i64,
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
            target_chat_id != requester_chat_id,
        );

        if result.changed()
            && let Err(error) = handles.persistence.save_config(&config).await
        {
            *config = before;
            if was_allowed {
                handles.access.lock().await.allow_chat(target_chat_id);
            }
            // Only a self-deny leaves the task alive. Restore its flag so
            // the normal worker error reply can report the failed save;
            // an aborted target worker and its queued work stay dropped.
            if target_chat_id == requester_chat_id {
                reactivate_chat_worker(&handles.workers, target_chat_id);
            }
            return Err(format!(
                "failed to persist denied chat {target_chat_id}: {error}"
            ));
        }
        result
    };

    if let Err(error) = handles
        .persistence
        .remove_room(&handles.rooms, target_chat_id)
        .await
    {
        // Access remains denied and the in-memory room remains absent. A
        // later settings save retries the complete snapshot; never recreate a
        // denied room merely because cleanup persistence failed.
        return Err(format!(
            "chat {target_chat_id} was denied, but its room cleanup could not be persisted: {error}"
        ));
    }

    send_command_reply(
        handles,
        requester_chat_id,
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
                log::warn!(target: "tellm::telegram", "{}", group_privacy_hint(chat_id));
            }
            let setup = room_setup_reply(chat_id, became_owner, handles).await;
            send_room_setup(chat_id, setup, handles).await
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
                log::info!(
                    target: "tellm::access",
                    "current pairing code chat_id={chat_id} pairing_code={code}"
                );
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

fn pair_code_from_message(message: &IncomingMessage, bot_username: Option<&str>) -> Option<String> {
    let text = model_turn::message_text(message)?;
    commands::pair_code(text, bot_username).map(str::to_string)
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
    // owner. Owners are the sole privilege concept; chats never confer it.
    let mut became_owner = false;
    if let Some(user_id) = pairer_user_id
        && !config.telegram.owner_user_ids.contains(&user_id)
    {
        config.telegram.owner_user_ids.push(user_id);
        became_owner = true;
        changed = true;
    }
    if changed && let Err(error) = handles.persistence.save_config(&config).await {
        *config = before;
        return Err(error);
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
                let reply = model_key_ready_reply(&base_reply, &model_key);
                send_command_reply(handles, chat_id, &reply).await
            } else {
                prompt_for_model_secret(handles, chat_id, &model_key, &secret_name, &base_reply)
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
    model_key: &str,
    secret_name: &str,
    base_reply: &str,
) -> Result<(), String> {
    let prompt_notice = model_key_prompt_reply(base_reply, secret_name);

    let reply = match prompt_and_store_secret(handles, chat_id, &prompt_notice, secret_name).await {
        Ok(Some(destination)) => model_key_stored_reply(model_key, destination),
        Ok(None) => model_key_skipped_reply(model_key),
        Err(error) => model_key_prompt_failed_reply(model_key, secret_name, &error),
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
    log::info!(
        target: "tellm::secrets",
        "secret requested from Telegram key={secret_name} action=\"enter it in this terminal; press Enter to skip\""
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
                log::warn!(target: "tellm::secrets", "{message}");
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

fn configured_model_base_reply(model_key: &str) -> String {
    format!("{model_key} is already configured.")
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

struct RoomSetupReply {
    text: String,
    model_keys: Option<Vec<String>>,
}

/// The in-room setup prompt after a room is approved (via /pair or /allow):
/// current model plus the actual picker, not just a pointer to it. Pinned
/// rooms have no picker because every alternative would be rejected.
async fn room_setup_reply(
    chat_id: i64,
    became_owner: bool,
    handles: &RuntimeHandles,
) -> RoomSetupReply {
    let (model_key, pinned, available) = {
        let config = handles.config.lock().await;
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id);
        (
            selected_model_key(&config, room, chat_id),
            pinned_model_key(&config, chat_id).map(str::to_string),
            config.models.keys().cloned().collect::<Vec<_>>(),
        )
    };
    build_room_setup_reply(&model_key, pinned.as_deref(), available, became_owner)
}

fn build_room_setup_reply(
    model_key: &str,
    pinned: Option<&str>,
    available: Vec<String>,
    became_owner: bool,
) -> RoomSetupReply {
    let mut reply = if let Some(pinned) = pinned {
        format!(
            "Room approved. This room is pinned to {pinned}. Use /model unpin before switching models."
        )
    } else {
        format!(
            "Room approved. Current model: {model_key}. Pick a model below, lock it with /model pin KEY, add providers with /model add, or just start chatting."
        )
    };
    if became_owner {
        reply.push_str(
            "\n\nYou are registered as this bot's owner: /allow, /deny, /shutdown, and \
             /model pin work for you from any chat, and rooms you add the bot to are \
             approved automatically.",
        );
    }
    RoomSetupReply {
        text: reply,
        model_keys: pinned.is_none().then_some(available),
    }
}

async fn send_room_setup(
    chat_id: i64,
    setup: RoomSetupReply,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    if !chat_is_allowed(chat_id, handles).await {
        return Ok(());
    }
    match setup.model_keys {
        Some(model_keys) => handles
            .telegram
            .send_model_picker(chat_id, &setup.text, &model_keys)
            .await
            .map_err(|error| error.to_string()),
        None => send_command_reply(handles, chat_id, &setup.text).await,
    }
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
                log::warn!(
                    target: "tellm::telegram",
                    "sendChatAction failed chat_id={chat_id} error={error:?}"
                );
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
                    log::warn!(
                        target: "tellm::terminal",
                        "input read failed; ignoring line error={error:?}"
                    );
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
                _ => log::info!(
                    target: "tellm::terminal",
                    "available commands: reset, exit, quit"
                ),
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
        log::info!(
            target: "tellm::secrets",
            "no value entered; secret unchanged key={secret_name}"
        );
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
            log::info!(
                target: "tellm::access",
                "allowed_chats=0 mode=pairing action=\"approve the console pairing code with /pair CODE in that chat\""
            );
        }
        crate::access::StartupNotice::Restricted { allowed_chat_count } => {
            log::info!(
                target: "tellm::access",
                "allowed_chats={allowed_chat_count} source=\"allowlist + model room pins\" new_chat_access=\"pairing code or owner /allow\""
            );
        }
    }
}

fn print_group_privacy_hints(chat_ids: &BTreeSet<i64>) {
    if chat_ids.is_empty() {
        return;
    }

    log::warn!(
        target: "tellm::telegram",
        "group chats detected count={} action=\"if plain text is ignored, disable privacy mode with BotFather /setprivacy and re-add the bot\"",
        chat_ids.len(),
    );
    log::debug!(
        target: "tellm::telegram",
        "group_chat_ids=[{}]",
        chat_ids
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
}

fn group_privacy_hint(chat_id: i64) -> String {
    format!(
        "group chat detected chat_id={chat_id} action=\"if plain text is ignored, disable privacy mode with BotFather /setprivacy and re-add the bot\""
    )
}

fn unknown_chat_hint(chat_id: i64) -> String {
    format!(
        "This bot is private. Chat id: {chat_id}. If you own it, send /pair CODE here with the \
         code printed in the tellm terminal, or ask a registered owner to send /allow {chat_id} \
         from any chat they are in."
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
    log::debug!(
        target: "tellm::telegram",
        "{}",
        update_log_line(chat_id, kind, route)
    );
}

fn update_log_line(chat_id: i64, kind: UpdateKind, route: UpdateRoute) -> String {
    format!(
        "update chat_id={chat_id} kind={} route={}",
        kind.as_str(),
        route.as_str()
    )
}

fn format_log_duration(duration: Duration) -> String {
    let millis = duration.as_millis();
    if millis < 1_000 {
        return format!("{millis}ms");
    }
    if millis.is_multiple_of(1_000) {
        return format!("{}s", millis / 1_000);
    }
    format!("{}.{:03}s", millis / 1_000, millis % 1_000)
}

fn now_access_time() -> AccessTime {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    AccessTime::from_unix_seconds(seconds)
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

fn configured_model_key(config: &Config, requested: &str) -> Option<String> {
    config
        .models
        .keys()
        .find(|key| key.eq_ignore_ascii_case(requested))
        .cloned()
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
        CommandReject::OwnerNotAllowed => "Only the bot owner can use this command.".to_string(),
        CommandReject::OwnerStale => "Ignoring stale owner command.".to_string(),
        CommandReject::ShutdownNotOwner => "Only the bot owner can use /shutdown.".to_string(),
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
    use std::sync::mpsc as std_mpsc;

    use tellm_config::ModelConfig;

    use super::*;
    use crate::rooms::HistoryReset;

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
        let (persistence, thread) = persistence::spawn_with(
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

    #[test]
    fn unpin_reset_drops_history_even_when_next_model_uses_same_wire_format() {
        let mut room = RoomState::new(crate::rooms::RoomSettings {
            thinking: Some(tellm_core::ThinkingLevel::High),
            ..crate::rooms::RoomSettings::default()
        });
        room.append_turn(
            WireFormat::Responses,
            vec![serde_json::json!({ "provider": "xai", "opaque": true })],
        );

        reset_room_model_context(&mut room);
        let reset = room.begin_turn(WireFormat::Responses).reset;

        assert!(room.history.is_empty());
        assert!(room.settings.thinking.is_none());
        assert!(matches!(reset, HistoryReset::WireFormatChanged { .. }));
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
        let access = AccessControl::new(AccessConfig::from_config(&config));
        let config = Arc::new(Mutex::new(config));
        let rooms = Arc::new(Mutex::new(RoomStates::default()));
        rooms.lock().await.get_or_default(42).settings.role = Some("preserve me".to_string());

        let release_save = Arc::new(std::sync::Barrier::new(2));
        let writer_release = Arc::clone(&release_save);
        let (save_started_tx, save_started_rx) = std_mpsc::channel();
        let (persistence, writer_thread) = persistence::spawn_with(
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
        let prompt = model_key_prompt_reply(&base, preset.api_key_secret);
        let skipped = model_key_skipped_reply(preset.key);
        let stored =
            model_key_stored_reply(preset.key, secrets::SecretDestination::CredentialsFile);

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
        let prompt = model_key_prompt_reply(&base, "mistral_api_key");
        let skipped = model_key_skipped_reply("mistral");
        let stored = model_key_stored_reply("mistral", secrets::SecretDestination::CredentialsFile);
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
    fn room_setup_uses_a_picker_only_when_model_switching_is_allowed() {
        let available = vec!["claude".to_string(), "openai".to_string()];
        let unpinned = build_room_setup_reply("claude", None, available.clone(), false);
        assert_eq!(unpinned.model_keys, Some(available));
        assert!(unpinned.text.contains("Current model: claude"));

        let pinned = build_room_setup_reply(
            "openai",
            Some("openai"),
            vec!["claude".to_string(), "openai".to_string()],
            false,
        );
        assert!(pinned.model_keys.is_none());
        assert!(pinned.text.contains("pinned to openai"));
    }

    #[test]
    fn update_log_line_contains_only_metadata() {
        assert_eq!(
            update_log_line(-100, UpdateKind::Message, UpdateRoute::Model),
            "update chat_id=-100 kind=message route=model"
        );
        assert_eq!(
            update_log_line(42, UpdateKind::EditedMessage, UpdateRoute::Ignored),
            "update chat_id=42 kind=edited_message route=ignored"
        );
    }

    #[test]
    fn get_updates_failures_are_rate_limited_and_recovery_resets_state() {
        let start = StdInstant::now();
        let mut state = GetUpdatesLogState::default();

        assert_eq!(
            state.record_failure("502 Bad Gateway", start),
            Some(GetUpdatesFailureReport {
                consecutive_failures: 1,
                elapsed: Duration::ZERO,
                suppressed: 0,
                first: true,
                error_changed: false,
            })
        );
        assert_eq!(
            state.record_failure("502 Bad Gateway", start + Duration::from_secs(2)),
            None
        );
        assert_eq!(
            state.record_failure("request send failed", start + Duration::from_secs(4)),
            Some(GetUpdatesFailureReport {
                consecutive_failures: 3,
                elapsed: Duration::from_secs(4),
                suppressed: 1,
                first: false,
                error_changed: true,
            })
        );
        assert_eq!(
            state.record_failure("request send failed", start + Duration::from_secs(34)),
            Some(GetUpdatesFailureReport {
                consecutive_failures: 4,
                elapsed: Duration::from_secs(34),
                suppressed: 0,
                first: false,
                error_changed: false,
            })
        );
        assert_eq!(
            state.record_success(start + Duration::from_secs(36)),
            Some(GetUpdatesRecovery {
                failures: 4,
                downtime: Duration::from_secs(36),
            })
        );
        assert_eq!(state.record_success(start + Duration::from_secs(37)), None);
    }

    #[test]
    fn log_duration_uses_milliseconds_until_a_full_second() {
        assert_eq!(format_log_duration(Duration::from_millis(999)), "999ms");
        assert_eq!(format_log_duration(Duration::from_secs(2)), "2s");
        assert_eq!(format_log_duration(Duration::from_millis(2_125)), "2.125s");
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
        assert!(hint.contains("registered owner"));
        assert!(!hint.contains("admin chat"));
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
        gemini.model_name = "gemini-3.6-flash".to_string();
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
