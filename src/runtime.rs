use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant as StdInstant, SystemTime, UNIX_EPOCH};

use tellm_config::{Config, WireFormat, secrets};
use tellm_core::{ChatRequest, ChatResponse, ContentPart};
use tellm_telegram::{IncomingMessage, Target, Telegram, TelegramError};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::{sleep, timeout};

use crate::access::{
    AccessConfig, AccessControl, AccessTime, ChatAccess, MAX_FAILED_PAIRINGS_PER_PROCESS,
    PairingAttempt,
};
use crate::commands::{self, CommandAction, CommandContext, CommandReject, KnownCommand, Route};
use crate::cost;
use crate::groups::{self, ChatKind};
use crate::memory::{self, ConversationMemory};
use crate::model_turn::{self, PreparedChatRequest};
use crate::ollama;
use crate::persistence::{self, Persistence};
use crate::rooms::{self, ChatMode, RoomState, RoomStates};
use crate::summary;

const LONG_POLL_TIMEOUT_S: u32 = 20;
const GET_UPDATES_RETRY_DELAY: Duration = Duration::from_secs(2);
const GET_UPDATES_FAILURE_LOG_INTERVAL: Duration = Duration::from_secs(30);
const CHAT_QUEUE_SIZE: usize = 32;
const CHAT_TASK_IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
/// How long the worker waits for an album's next item before treating the
/// album as complete. Telegram creates an album's messages together, so the
/// items reach the queue back to back; the wait only bounds a straggler.
const ALBUM_SETTLE: Duration = Duration::from_secs(1);
/// Telegram's album size limit (sendMediaGroup takes 2-10 items).
const MAX_ALBUM_ITEMS: usize = 10;
/// A text message at least this long, in UTF-16 code units as Telegram
/// counts text, may be a part of a long text that a Telegram app split into
/// several messages. The apps cut near the message length limit (checked
/// 2026-11-28 against their source): Telegram Desktop at the best break in
/// the second half of its limit, 4,096 or 8,192 with Premium (lib_ui
/// `TextUtilities::CutPart`, tdesktop `data_premium_limits.cpp`); Android
/// at most 300 characters before 4,096
/// (`ChatActivityEnterView.processSendingText`); iOS at the last line break
/// or full stop before 4,096 (`breakChatInputText`). So every part but the
/// last is longer than 2,048, unless iOS found no line break or full stop in
/// the second half. Nobody types this much and sends again within
/// [`SPLIT_TEXT_GAP_S`], and shorter messages never wait.
const SPLIT_TEXT_MIN_PART: usize = 2_000;
/// How long the worker waits for the next part of a split text, and the
/// largest gap between two parts' Telegram timestamps (whole seconds). The
/// app sends the parts back to back; the timestamps keep messages that only
/// queued up together while the room was busy from joining.
const SPLIT_TEXT_GAP_S: u64 = 2;
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
    /// The bot's own user id, from getMe, to tell replies to its messages.
    bot_user_id: Option<i64>,
    /// Whether the bot's privacy mode is on, from getMe: then groups where
    /// it isn't an admin send it only commands and replies to it.
    privacy_mode: Option<bool>,
    shutdown_tx: mpsc::Sender<ShutdownReason>,
    persistence: Persistence,
    workers: WorkerRegistry,
    queue_full_notices: Arc<StdMutex<BTreeSet<i64>>>,
}

impl RuntimeHandles {
    fn bot(&self) -> groups::Bot<'_> {
        groups::Bot {
            user_id: self.bot_user_id,
            username: self.bot_username.as_deref(),
        }
    }
}

struct ChatDispatcher {
    sender: mpsc::Sender<IncomingMessage>,
    handle: JoinHandle<()>,
    worker_id: u64,
    cancelled: Arc<AtomicBool>,
    /// Resolves when this worker's task ends, however it ends. A worker
    /// that replaces this one waits for it.
    done: Option<oneshot::Receiver<()>>,
    /// The worker this one replaced, which may still be finishing messages
    /// it had queued. Kept so shutdown aborts and joins it too.
    previous: Option<JoinHandle<()>>,
    /// Messages queued for the worker that it hasn't started handling.
    waiting: Arc<AtomicUsize>,
}

type WorkerRegistry = Arc<StdMutex<BTreeMap<i64, WorkerRegistration>>>;

struct WorkerRegistration {
    worker_id: u64,
    cancelled: Arc<AtomicBool>,
    abort_handle: tokio::task::AbortHandle,
    /// Messages queued for the worker that it hasn't started handling,
    /// including one read ahead while gathering an album or a split text.
    waiting: Arc<AtomicUsize>,
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

type TerminalSecretPrompts = Arc<StdMutex<TerminalPromptSlot>>;

/// The one terminal secret prompt that may be waiting, and whether the
/// terminal can still answer it.
#[derive(Default)]
struct TerminalPromptSlot {
    pending: Option<TerminalSecretPrompt>,
    /// Set when the stdin reader stops: at end of input (tellm started
    /// without a terminal, or Ctrl-D) or after `exit`. No prompt can be
    /// answered from then on.
    input_closed: bool,
}

#[derive(Clone)]
struct TerminalSecretPrompt {
    secret_name: String,
    line_tx: mpsc::UnboundedSender<String>,
    /// The room whose worker waits on this prompt; its later messages queue
    /// behind it until the prompt ends.
    chat_id: i64,
    /// Whether that room was already told its messages are waiting.
    wait_notice_sent: bool,
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
        let access = AccessControl::new(AccessConfig::from_config(&config));
        model_turn::warm_configured_provider_secrets(&config);
        print_startup_notice(access.startup_notice());

        let (shutdown_tx, shutdown_rx) = mpsc::channel(4);
        let terminal_prompts = Arc::default();

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

        if bot.can_read_all_group_messages == Some(false) {
            let config = self.config.lock().await;
            print_group_privacy_hints(&allowed_group_chat_ids(&AccessConfig::from_config(&config)));
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
            bot_user_id: Some(bot.id),
            privacy_mode: bot.can_read_all_group_messages.map(|reads_all| !reads_all),
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
                                    handle_membership_change(membership, &handles).await;
                                } else if let Some(message) = dispatchable_message(&mut update) {
                                    self.handle_update(message, &handles, &mut dispatchers).await;
                                } else if let Some(message) = update.edited_message {
                                    // Updates already queued before edited messages were removed
                                    // from allowed_updates can still arrive. Never turn an edit
                                    // into a second billed provider call.
                                    log_update_route(
                                        &message,
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

            // A worker that closed its queue but is still running is
            // finishing queued messages; keep it until it ends, so the next
            // message's worker can wait for it and shutdown can join it.
            dispatchers.retain(|chat_id, dispatcher| {
                if dispatcher
                    .previous
                    .as_ref()
                    .is_some_and(JoinHandle::is_finished)
                {
                    dispatcher.previous = None;
                }
                let keep = !dispatcher.handle.is_finished();
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
                    log_update_route(&message, UpdateKind::Message, UpdateRoute::Command);
                    let pairer = message.from.as_ref().map(|user| user.id);
                    drop(start_pair_attempt(chat_id, &code, pairer, handles).await);
                } else if send_hint {
                    log_update_route(&message, UpdateKind::Message, UpdateRoute::Ignored);
                    let telegram = self.telegram.clone();
                    tokio::spawn(async move {
                        let _ = telegram
                            .send_message(chat_id, &unknown_chat_hint(chat_id))
                            .await;
                    });
                } else {
                    log_update_route(&message, UpdateKind::Message, UpdateRoute::Ignored);
                }
            }
        }
    }
}

/// The bot's own membership changed in a chat. Being added by an owner
/// approves the chat; being added by anyone else arms a pairing code and
/// announces it on the console. Being removed ends the chat's access as
/// `/deny` does, so adding the bot back starts over. Any other change,
/// such as becoming an admin, changes nothing.
async fn handle_membership_change(
    membership: tellm_telegram::ChatMemberUpdated,
    handles: &RuntimeHandles,
) {
    let chat_id = membership.chat.id;
    let label = membership.chat.label();
    let was_in = membership.old_chat_member.is_present();
    let is_in = membership.new_chat_member.is_present();
    if was_in && !is_in {
        handle_removed(chat_id, &label, handles).await;
        return;
    }
    if was_in || !is_in {
        log::debug!(
            target: "tellm::access",
            "membership changed chat={label:?} status={}",
            membership.new_chat_member.status
        );
        return;
    }

    let adder = membership.from.as_ref().map(|user| user.id);
    let trusted_adder = match adder {
        Some(user_id) => {
            let config = handles.config.lock().await;
            config.telegram.owner_user_ids.contains(&user_id)
        }
        None => false,
    };
    if let Some(user_id) = adder.filter(|_| trusted_adder) {
        let persisted = update_config(handles, move |config| {
            let changed = allow_chat_in_config(config, chat_id);
            (changed, changed)
        })
        .await;
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
            let mut access = handles.access.lock().await;
            access.allow_chat(chat_id);
        }
        log::info!(
            target: "tellm::access",
            "chat auto-approved chat={label:?} owner_user_id={user_id} changed={changed}"
        );
        warn_privacy_mode(chat_id, handles);
        let setup = room_setup_reply(chat_id, false, handles).await;
        let handles = handles.clone();
        tokio::spawn(async move {
            let _ = send_room_setup(chat_id, setup, &handles).await;
        });
        return;
    }

    let pairing = {
        let mut access = handles.access.lock().await;
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
            warn_privacy_mode(chat_id, handles);
        }
        None => log::info!(
            target: "tellm::access",
            "added to already-allowed chat chat={label:?}"
        ),
    }
}

/// The bot left or was removed from a chat: it can't read or answer
/// there any more, and a chat it is added back to should start over
/// rather than come back approved or marked private.
async fn handle_removed(chat_id: i64, label: &str, handles: &RuntimeHandles) {
    if !chat_is_allowed(chat_id, handles).await {
        log::info!(target: "tellm::access", "removed from chat chat={label:?}");
        return;
    }
    match revoke_chat(chat_id, true, handles).await {
        Ok(_) => log::info!(
            target: "tellm::access",
            "removed from chat; access and room settings cleared chat={label:?}"
        ),
        Err(error) => log::error!(
            target: "tellm::access",
            "removed from chat; clearing it failed chat={label:?} error={error:?}"
        ),
    }
}

fn dispatchable_message(update: &mut tellm_telegram::Update) -> Option<IncomingMessage> {
    update.message.take()
}

/// Queue the update for its chat worker without ever blocking the poll
/// loop: awaiting a full queue would stall dispatch for every chat behind
/// one slow room, so a full queue drops the message with a busy notice.
/// Gathers an album into its first message so the model sees every photo
/// or file in one turn and replies once. Telegram delivers an album as one
/// message per item, all sharing a `media_group_id`; the items that follow
/// the first in the room's queue within `settle` of each other join it.
/// Returns the first queued message that isn't part of the album, if one was
/// read, so the caller handles it next.
async fn gather_album(
    first: IncomingMessage,
    receiver: &mut mpsc::Receiver<IncomingMessage>,
    settle: Duration,
) -> (IncomingMessage, Option<IncomingMessage>) {
    let Some(group) = first.media_group_id.clone() else {
        return (first, None);
    };
    let mut items = vec![first];
    let read_ahead = loop {
        if items.len() >= MAX_ALBUM_ITEMS {
            break None;
        }
        match timeout(settle, receiver.recv()).await {
            Ok(Some(next)) if next.media_group_id.as_deref() == Some(group.as_str()) => {
                items.push(next);
            }
            Ok(Some(next)) => break Some(next),
            Ok(None) | Err(_) => break None,
        }
    };
    (merge_album(items), read_ahead)
}

/// One message for a whole album: the first item, carrying every item's
/// caption (usually there is one) and the other items in `album`.
fn merge_album(mut items: Vec<IncomingMessage>) -> IncomingMessage {
    let mut rest = items.split_off(1);
    let Some(mut first) = items.pop() else {
        unreachable!("gather_album always passes the first item");
    };
    let captions: Vec<String> = std::iter::once(&first)
        .chain(&rest)
        .filter_map(model_turn::message_text)
        .map(str::trim)
        .filter(|caption| !caption.is_empty())
        .map(str::to_string)
        .collect();
    if !captions.is_empty() {
        first.text = None;
        first.caption = Some(captions.join("\n\n"));
    }
    for item in &mut rest {
        item.text = None;
        item.caption = None;
    }
    first.album = rest;
    first
}

/// Gathers a long text that a Telegram app split into several messages back
/// into its first part, so the model answers it once. The Bot API marks
/// nothing about a split, so a part is recognized by length, sender and
/// timestamp: a text message from the same sender, dated within
/// [`SPLIT_TEXT_GAP_S`] of the part before it, joins while every part so far
/// is at least [`SPLIT_TEXT_MIN_PART`] long. Returns the first queued
/// message that doesn't join, if one was read, so the caller handles it
/// next.
async fn gather_split_text(
    first: IncomingMessage,
    receiver: &mut mpsc::Receiver<IncomingMessage>,
    settle: Duration,
) -> (IncomingMessage, Option<IncomingMessage>) {
    let mut parts = vec![first];
    let read_ahead = loop {
        let Some(last) = parts.last().filter(|last| is_long_text(last)) else {
            break None;
        };
        match timeout(settle, receiver.recv()).await {
            Ok(Some(next)) if continues_split_text(last, &next) => parts.push(next),
            Ok(Some(next)) => break Some(next),
            Ok(None) | Err(_) => break None,
        }
    };
    (merge_split_text(parts), read_ahead)
}

fn is_long_text(message: &IncomingMessage) -> bool {
    message.media_group_id.is_none()
        && message
            .text
            .as_deref()
            .is_some_and(|text| text.encode_utf16().count() >= SPLIT_TEXT_MIN_PART)
}

/// Whether `next` is the part of a split text that follows `last`. A
/// message without a sender is never joined, since its parts can't be told
/// from another member's message.
fn continues_split_text(last: &IncomingMessage, next: &IncomingMessage) -> bool {
    let same_sender = matches!(
        (&last.from, &next.from),
        (Some(last), Some(next)) if last.id == next.id
    );
    let gap = next
        .date
        .checked_sub(last.date)
        .and_then(|gap| u64::try_from(gap).ok());
    same_sender
        && next.text.is_some()
        && next.media_group_id.is_none()
        && gap.is_some_and(|gap| gap <= SPLIT_TEXT_GAP_S)
}

/// One message for a split text: the first part, carrying every part's text
/// joined by line breaks. The apps cut at a line break, a space or after a
/// full stop and trim it from both parts, so a line break restores the most
/// common cut and keeps words apart at the others.
fn merge_split_text(parts: Vec<IncomingMessage>) -> IncomingMessage {
    let count = parts.len();
    let mut parts = parts.into_iter();
    let Some(mut first) = parts.next() else {
        unreachable!("gather_split_text always passes the first part");
    };
    if count > 1 {
        let mut text = first.text.take().unwrap_or_default();
        for part in parts {
            text.push('\n');
            text.push_str(part.text.as_deref().unwrap_or_default());
        }
        first.text = Some(text);
        first.split_parts = count;
    }
    first
}

fn send_to_chat_worker(
    chat_id: i64,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    dispatchers: &mut BTreeMap<i64, ChatDispatcher>,
) {
    let dispatcher = dispatchers
        .entry(chat_id)
        .or_insert_with(|| spawn_chat_worker(chat_id, handles.clone(), None));

    let message = match dispatcher.sender.try_send(message) {
        Ok(()) => {
            dispatcher.waiting.fetch_add(1, Ordering::AcqRel);
            if let Some(secret_name) =
                reserve_prompt_wait_notice(&handles.terminal_prompts, chat_id)
            {
                let telegram = handles.telegram.clone();
                tokio::spawn(async move {
                    let _ = telegram
                        .send_message(chat_id, &prompt_wait_notice(&secret_name))
                        .await;
                });
            }
            return;
        }
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

    // The old worker may still be finishing messages that reached its queue
    // before it closed; the new one starts after it, so order holds.
    let previous = dispatchers.remove(&chat_id).map(|mut old| {
        let done = old.done.take();
        (old.handle, done)
    });
    let dispatcher = spawn_chat_worker(chat_id, handles.clone(), previous);
    if dispatcher.sender.try_send(message).is_err() {
        log::error!(
            target: "tellm::dispatcher",
            "fresh worker rejected message chat_id={chat_id}"
        );
    }
    dispatchers.insert(chat_id, dispatcher);
}

/// Claims the one notice for a message that reaches the room whose worker is
/// waiting on the terminal key prompt. The message is kept and answered after
/// the prompt ends; without the notice the room looks unresponsive, then
/// answers everything sent meanwhile in a burst.
fn reserve_prompt_wait_notice(
    secret_prompts: &TerminalSecretPrompts,
    chat_id: i64,
) -> Option<String> {
    let mut slot = secret_prompts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let prompt = slot
        .pending
        .as_mut()
        .filter(|prompt| prompt.chat_id == chat_id && !prompt.wait_notice_sent)?;
    prompt.wait_notice_sent = true;
    Some(prompt.secret_name.clone())
}

fn prompt_wait_notice(secret_name: &str) -> String {
    format!(
        "This chat is waiting for {secret_name} in the tellm terminal. Messages sent here \
         meanwhile are answered once that prompt ends."
    )
}

fn reserve_queue_full_notice(pending: &Arc<StdMutex<BTreeSet<i64>>>, chat_id: i64) -> bool {
    pending
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(chat_id)
}

fn spawn_chat_worker(
    chat_id: i64,
    handles: RuntimeHandles,
    previous: Option<(JoinHandle<()>, Option<oneshot::Receiver<()>>)>,
) -> ChatDispatcher {
    let (sender, mut receiver) = mpsc::channel::<IncomingMessage>(CHAT_QUEUE_SIZE);
    let (done_tx, done_rx) = oneshot::channel::<()>();
    let (previous, previous_done) = match previous {
        Some((handle, done)) => (Some(handle), done),
        None => (None, None),
    };
    let worker_id = NEXT_CHAT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
    let waiting = Arc::new(AtomicUsize::new(0));
    let worker_waiting = Arc::clone(&waiting);
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let workers = Arc::clone(&handles.workers);
    let worker_guard_registry = Arc::clone(&workers);
    let handle = tokio::spawn(async move {
        // Dropped when this task ends, even by abort, which resolves `done`.
        let _done = done_tx;
        if let Some(previous_done) = previous_done {
            let _ = previous_done.await;
        }
        let _registry_guard = WorkerRegistryGuard {
            chat_id,
            worker_id,
            workers: worker_guard_registry,
        };
        // A message read while gathering an album that isn't part of it; it
        // is handled next, so the room's order holds.
        let mut read_ahead: Option<IncomingMessage> = None;
        loop {
            // A self-deny does not abort the task so it can send the command
            // confirmation. Observe its flag before waiting for another item
            // instead of retaining the denied room until the idle timeout.
            if worker_cancelled.load(Ordering::Acquire) || !chat_is_allowed(chat_id, &handles).await
            {
                break;
            }
            let received = match read_ahead.take() {
                Some(message) => Ok(Some(message)),
                None => timeout(CHAT_TASK_IDLE_TIMEOUT, receiver.recv()).await,
            };
            match received {
                Ok(Some(message)) => {
                    if worker_cancelled.load(Ordering::Acquire)
                        || !chat_is_allowed(chat_id, &handles).await
                    {
                        break;
                    }
                    let (message, next) = if message.media_group_id.is_some() {
                        gather_album(message, &mut receiver, ALBUM_SETTLE).await
                    } else {
                        let settle = Duration::from_secs(SPLIT_TEXT_GAP_S);
                        gather_split_text(message, &mut receiver, settle).await
                    };
                    read_ahead = next;
                    let parts = message_parts(&message);
                    let _ = worker_waiting.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                        Some(n.saturating_sub(parts))
                    });
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
                    // Close, then finish what is queued: the poll loop can
                    // send between the timeout and the close, and dropping
                    // the receiver would lose that message. From the close
                    // on, sends fail and the poll loop starts a new worker,
                    // which waits for this one; `recv` ends with `None`.
                    log::debug!(
                        target: "tellm::dispatcher",
                        "idle task reaped chat_id={chat_id}"
                    );
                    receiver.close();
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
                waiting: Arc::clone(&waiting),
            },
        );

    ChatDispatcher {
        sender,
        handle,
        worker_id,
        cancelled,
        done: Some(done_rx),
        previous,
        waiting,
    }
}

/// How many queued messages one handled message stands for: an album's
/// items, or the parts of a text a Telegram app split.
fn message_parts(message: &IncomingMessage) -> usize {
    (1 + message.album.len()).max(message.split_parts)
}

/// Whether the chat has messages queued behind the one being answered.
fn messages_waiting(handles: &RuntimeHandles, chat_id: i64) -> bool {
    handles
        .workers
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&chat_id)
        .is_some_and(|worker| worker.waiting.load(Ordering::Acquire) > 0)
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
        if let Some(previous) = dispatcher.previous {
            previous.abort();
            handles.push((chat_id, previous));
        }
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
                log_update_route(&message, UpdateKind::Message, UpdateRoute::Command);
                return handle_command(chat_id, action, handles, cancelled).await;
            }
            Route::Ignore => {
                log_update_route(&message, UpdateKind::Message, UpdateRoute::Ignored);
                return Ok(());
            }
            Route::UserMessage => {}
        }
    }

    let kind = {
        let config = handles.config.lock().await;
        groups::chat_kind(&config, chat_id)
    };
    if !model_turn::message_has_input(&message) {
        log_update_route(&message, UpdateKind::Message, UpdateRoute::Ignored);
        if kind == ChatKind::PrivateGroup {
            announce_newcomers(chat_id, &message, handles, cancelled).await;
        }
        return Ok(());
    }
    if kind == ChatKind::SharedGroup {
        if !groups::is_addressed(&message, handles.bot()) {
            log_update_route(&message, UpdateKind::Message, UpdateRoute::Ignored);
            return Ok(());
        }
        log_update_route(&message, UpdateKind::Message, UpdateRoute::Model);
        let message = groups::standalone(message, handles.bot());
        return run_model_turn(chat_id, message, handles, cancelled, TurnKind::Standalone).await;
    }
    log_update_route(&message, UpdateKind::Message, UpdateRoute::Model);
    handle_model_message(chat_id, message, handles, cancelled).await
}

/// Tells a group marked private that someone who isn't an owner joined, so
/// the owner can make it shared. Nothing changes on its own.
async fn announce_newcomers(
    chat_id: i64,
    message: &IncomingMessage,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
) {
    let owners = handles.config.lock().await.telegram.owner_user_ids.clone();
    let names = groups::newcomers(message, handles.bot(), &owners);
    if names.is_empty() {
        return;
    }
    log::info!(
        target: "tellm::access",
        "someone who isn't an owner joined a private group chat_id={chat_id} joined={}",
        names.len()
    );
    if worker_can_reply(chat_id, handles, cancelled).await {
        send_best_effort(handles, chat_id, &groups::join_notice(&names)).await;
    }
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
        // Telegram sends /start when someone first opens a direct chat with
        // the bot. In a group it is most likely another bot's.
        commands::ParsedRoute::Unknown { name: "start" } if chat_id > 0 => (KnownCommand::Help, ""),
        commands::ParsedRoute::Unknown { name } => {
            // In a shared group a command tellm doesn't know is most likely
            // another bot's, and a reply to each would only add noise.
            let shared = {
                let config = handles.config.lock().await;
                groups::chat_kind(&config, chat_id) == ChatKind::SharedGroup
            };
            if shared {
                return Ok(Route::Ignore);
            }
            return Ok(Route::Command(commands::unknown_command(name)));
        }
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
            chat_kind: groups::chat_kind(&config, chat_id),
        };
        commands::resolve(command, args, &context)
    };
    Ok(Route::Command(action))
}

async fn handle_command(
    chat_id: i64,
    action: CommandAction,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    match action {
        CommandAction::ResetHistory => {
            // Released before the send: every chat's worker needs this lock,
            // and a slow Telegram call must not stall them all.
            handles
                .rooms
                .lock()
                .await
                .get_or_default(chat_id)
                .reset_history();
            handles
                .telegram
                .send_message(chat_id, "Started a new chat.")
                .await
                .map_err(|error| error.to_string())
        }
        CommandAction::SummarizeChat => {
            summarize_into_new_chat(chat_id, handles, cancelled, summary::Trigger::Command)
                .await
                .map(|_| ())
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
        CommandAction::ModelUnchanged { model_key } => {
            send_command_reply(handles, chat_id, &model_unchanged_reply(&model_key)).await
        }
        CommandAction::SetModel { model_key } => {
            let capabilities = {
                let config = handles.config.lock().await;
                model_capabilities(&config, model_key.clone())
            };
            let mut settings_note = None;
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
                    room.settings.model_key = Some(model_key.clone());
                    reset_room_model_context(room);
                    settings_note = drop_unsupported_settings(room, &capabilities);
                })
                .await?;
            let reply = with_settings_note(model_set_reply(&model_key), settings_note);
            send_command_reply(handles, chat_id, &reply).await
        }
        CommandAction::PinModel { model_key } => {
            let pinned_key = model_key.clone();
            update_config(handles, move |config| {
                for model in config.models.values_mut() {
                    model.telegram_chat_ids.retain(|pinned| *pinned != chat_id);
                }
                if let Some(model) = config.models.get_mut(&pinned_key) {
                    model.telegram_chat_ids.push(chat_id);
                }
                (true, ())
            })
            .await
            .map_err(|error| format!("failed to persist model pin: {error}"))?;
            let capabilities = {
                let config = handles.config.lock().await;
                model_capabilities(&config, model_key.clone())
            };
            let mut settings_note = None;
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
                    reset_room_model_context(room);
                    settings_note = drop_unsupported_settings(room, &capabilities);
                })
                .await?;
            let reply = with_settings_note(
                format!(
                    "Room pinned to {model_key}. Chat history reset. /model unpin releases it."
                ),
                settings_note,
            );
            send_command_reply(handles, chat_id, &reply).await
        }
        CommandAction::UnpinModel => {
            let was_pinned = update_config(handles, move |config| {
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
                (was_pinned, was_pinned)
            })
            .await
            .map_err(|error| format!("failed to persist model unpin: {error}"))?;
            let reply = if was_pinned {
                // The room falls back to its stored selection or the default
                // model, which may not support what the pinned one did.
                let capabilities = {
                    let config = handles.config.lock().await;
                    let mut rooms = handles.rooms.lock().await;
                    room_capabilities(&config, rooms.get_or_default(chat_id), chat_id)
                };
                let mut settings_note = None;
                handles
                    .persistence
                    .mutate_room(&handles.rooms, chat_id, |room| {
                        reset_room_model_context(room);
                        settings_note = drop_unsupported_settings(room, &capabilities);
                    })
                    .await?;
                with_settings_note(
                    "Room unpinned. Chat history reset. /model KEY now switches models here."
                        .to_string(),
                    settings_note,
                )
            } else {
                "This room isn't pinned.".to_string()
            };
            send_command_reply(handles, chat_id, &reply).await
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

            // On disk, an entry the owner added by hand since startup wins
            // over the preset: the change only inserts a missing key.
            let already = update_config(handles, move |config| {
                if config.models.contains_key(preset.key) {
                    (false, true)
                } else {
                    config.models.insert(
                        preset.key.to_string(),
                        crate::wizard::model_config_from_preset(preset),
                    );
                    (true, false)
                }
            })
            .await
            .map_err(|error| format!("failed to persist added model: {error}"))?;

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
                    room.set_web_search(enabled);
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
        CommandAction::ShowCost => {
            let summary = {
                let config = handles.config.lock().await;
                let mut rooms = handles.rooms.lock().await;
                let room = rooms.get_or_default(chat_id);
                let model_key = selected_model_key(&config, room, chat_id);
                let has_price = config
                    .models
                    .get(&model_key)
                    .is_some_and(|model| model.price.is_some());
                let shared = groups::chat_kind(&config, chat_id) == ChatKind::SharedGroup;
                cost::summary(
                    &room.cost,
                    room.settings.shows_cost(shared),
                    &model_key,
                    has_price,
                    room.context_tokens,
                )
            };
            send_command_reply(handles, chat_id, &summary).await
        }
        CommandAction::SetCostLine { enabled } => {
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
                    room.settings.show_cost = Some(enabled);
                })
                .await?;
            let state = if enabled { "on" } else { "off" };
            send_command_reply(handles, chat_id, &format!("Cost line set to {state}.")).await
        }
        CommandAction::ShowMemory => {
            let enabled = memory_enabled_in(chat_id, handles).await;
            let loaded = memory::load().await;
            let path = match memory::path() {
                Ok(path) => path.display().to_string(),
                Err(error) => error,
            };
            send_command_reply(
                handles,
                chat_id,
                &memory::status_reply(&loaded, enabled, &path),
            )
            .await
        }
        CommandAction::SetMemory { enabled } => {
            handles
                .persistence
                .mutate_room(&handles.rooms, chat_id, |room| {
                    room.settings.memory = Some(enabled);
                })
                .await?;
            send_command_reply(handles, chat_id, &memory::toggle_reply(enabled)).await
        }
        CommandAction::Remember { fact } => {
            let reply = match memory::edit(move |memory| memory.remember(&fact)).await {
                Ok(outcome) => {
                    let enabled = memory_enabled_in(chat_id, handles).await;
                    memory::remember_reply(&outcome, enabled)
                }
                Err(error) => format!("Not saved: {error}"),
            };
            send_command_reply(handles, chat_id, &reply).await
        }
        CommandAction::Forget { number } => {
            let reply = match memory::edit(move |memory| memory.forget(number)).await {
                Ok(outcome) => memory::forget_reply(number, &outcome),
                Err(error) => format!("Memory not changed: {error}"),
            };
            send_command_reply(handles, chat_id, &reply).await
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
        CommandAction::ShowGroup { kind } => {
            send_command_reply(handles, chat_id, &groups::status_reply(kind, false)).await
        }
        CommandAction::SetGroupPrivate { private } => {
            // The room's conversation is kept either way: a shared group never
            // reads or writes it, so it continues if the group turns private.
            let changed = update_config(handles, move |config| {
                let listed = &mut config.telegram.private_group_ids;
                let changed = listed.contains(&chat_id) != private;
                if private && changed {
                    listed.push(chat_id);
                } else if changed {
                    listed.retain(|listed| *listed != chat_id);
                }
                (changed, changed)
            })
            .await
            .map_err(|error| format!("failed to persist group setting: {error}"))?;
            let kind = if private {
                ChatKind::PrivateGroup
            } else {
                ChatKind::SharedGroup
            };
            send_command_reply(handles, chat_id, &groups::status_reply(kind, changed)).await
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

/// Whether a model turn belongs to the room's conversation or stands alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnKind {
    /// A turn of the room's conversation (in chat or message mode): it
    /// reads and stores the room's history and memory, and the chat may be
    /// summarized after it.
    Conversation,
    /// A shared group's turn: no history, memory or summary goes out, and
    /// nothing of the room's conversation changes; only its cost is counted.
    Standalone,
}

async fn handle_model_message(
    chat_id: i64,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    run_model_turn(chat_id, message, handles, cancelled, TurnKind::Conversation).await
}

async fn run_model_turn(
    chat_id: i64,
    message: IncomingMessage,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
    kind: TurnKind,
) -> Result<(), String> {
    let message_id = message.message_id;
    let input = model_turn::content_parts_from_message(&handles.telegram, &message).await?;
    if input.is_empty() {
        return Ok(());
    }
    if !worker_can_reply(chat_id, handles, cancelled).await {
        return Ok(());
    }

    let prepared = match kind {
        TurnKind::Conversation => build_chat_request(chat_id, input, handles).await?,
        TurnKind::Standalone => build_standalone_request(chat_id, input, handles).await?,
    };
    for notice in [&prepared.reset_notice, &prepared.memory_notice]
        .into_iter()
        .flatten()
    {
        if worker_can_reply(chat_id, handles, cancelled).await {
            let _ = handles.telegram.send_message(chat_id, notice).await;
        }
    }

    let typing = AbortOnDrop(spawn_typing_indicator(handles.telegram.clone(), chat_id));
    let response = model_turn::dispatch_provider(&prepared.model_config, &prepared.request)
        .await
        .and_then(model_turn::require_reply_content);
    typing.abort();

    match response {
        Ok(mut response) => {
            if !worker_can_reply(chat_id, handles, cancelled).await {
                return Ok(());
            }
            let turn_items = std::mem::take(&mut response.turn_items);
            let reply_cost =
                cost::ReplyCost::new(response.usage, prepared.model_config.price.as_ref());
            // The cost joins the room's total only with the turn it paid for,
            // so a reply dropped by a reset never counts toward the new chat.
            let committed = {
                let mut rooms = handles.rooms.lock().await;
                rooms.get_mut(chat_id).and_then(|room| match kind {
                    TurnKind::Conversation => room
                        .append_turn_if_generation(
                            prepared.generation,
                            prepared.model_config.wire_format,
                            turn_items,
                        )
                        .then(|| {
                            room.cost.record(&reply_cost);
                            room.record_context(reply_cost.usage.as_ref());
                            room.memory = prepared.memory.clone();
                            room.settings.shows_cost(false).then(|| {
                                cost::cost_line(&reply_cost, &room.cost, room.context_tokens)
                            })
                        }),
                    // No conversation grows, so the cost line has no chat size;
                    // a standalone turn runs only in a shared group.
                    TurnKind::Standalone => (room.generation() == prepared.generation).then(|| {
                        room.cost.record(&reply_cost);
                        room.settings
                            .shows_cost(true)
                            .then(|| cost::cost_line(&reply_cost, &room.cost, None))
                    }),
                })
            };
            match committed {
                Some(cost_line) if worker_can_reply(chat_id, handles, cancelled).await => {
                    // In a shared group the answer shows whose question it
                    // answers; elsewhere only when newer messages wait, so
                    // it is clear which one it answers.
                    let quote = match kind {
                        TurnKind::Standalone => true,
                        TurnKind::Conversation => messages_waiting(handles, chat_id),
                    };
                    let target = if quote {
                        Target::reply(chat_id, message_id)
                    } else {
                        Target::from(chat_id)
                    };
                    let delivered =
                        model_turn::send_response(&handles.telegram, target, response, cost_line)
                            .await;
                    if kind == TurnKind::Standalone {
                        return delivered;
                    }
                    // Even an undelivered reply was committed, so the bound
                    // applies either way.
                    let bounded = keep_chat_within_budget(chat_id, handles, cancelled).await;
                    delivered.and(bounded)
                }
                _ => Ok(()),
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
                "call failed chat_id={chat_id} context_window={} error={:?}",
                error.context_window,
                error.message
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

/// Runs after each committed reply. A chat whose context reached its token
/// budget, or whose history passed the byte cap, is summarized into a new
/// one right away: the provider's cache still holds the conversation, and
/// this runs in the chat's worker, so the next message waits for it. When
/// summarizing fails (or a message-mode exchange alone is over the cap), the
/// oldest whole turns are dropped to the cap instead, and the chat is told.
async fn keep_chat_within_budget(
    chat_id: i64,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    let due = {
        let rooms = handles.rooms.lock().await;
        rooms.get(chat_id).and_then(RoomState::summary_due)
    };
    // Telegram failures here are reported, never allowed to skip a step:
    // the prune below is what keeps the next request under the cap.
    let mut result = Ok(());
    if let Some(trigger) = due {
        if !worker_can_reply(chat_id, handles, cancelled).await {
            return Ok(());
        }
        if let Some(notice) = trigger.notice() {
            send_best_effort(handles, chat_id, &notice).await;
        }
        match summarize_into_new_chat(chat_id, handles, cancelled, trigger).await {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) => result = Err(error),
        }
    }

    let dropped = {
        let mut rooms = handles.rooms.lock().await;
        rooms
            .get_mut(chat_id)
            .map_or(0, RoomState::prune_to_byte_cap)
    };
    if dropped > 0 {
        log::warn!(
            target: "tellm::history",
            "history over the byte cap; oldest turns dropped chat_id={chat_id} turns={dropped}"
        );
        if worker_can_reply(chat_id, handles, cancelled).await {
            send_best_effort(handles, chat_id, &summary::pruned_notice(dropped)).await;
        }
    }
    result
}

/// Sends a notice whose loss must not stop the work around it; a failure
/// is logged instead.
async fn send_best_effort(handles: &RuntimeHandles, chat_id: i64, text: &str) {
    if let Err(error) = send_command_reply(handles, chat_id, text).await {
        log::warn!(
            target: "tellm::telegram",
            "notice not delivered chat_id={chat_id} error={error:?}"
        );
    }
}

/// Asks the room's model to summarize the conversation and starts a new
/// conversation that carries the summary (`/new summary`, or a chat that
/// reached its budget). The request is built exactly like a turn in the room
/// (model, system prompt, history, tools and effort), so only its user
/// message differs from what the provider has cached. The summary is the old
/// conversation's last reply: its cost joins that conversation's total. A
/// failed request leaves the conversation as it was. Returns whether the
/// conversation now continues from a summary.
async fn summarize_into_new_chat(
    chat_id: i64,
    handles: &RuntimeHandles,
    cancelled: &AtomicBool,
    trigger: summary::Trigger,
) -> Result<bool, String> {
    let nothing_to_summarize = {
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id);
        match room.settings.mode {
            ChatMode::Message => Some(summary::NOTHING_TO_SUMMARIZE_IN_MESSAGE_MODE),
            ChatMode::Chat if room.history.is_empty() => Some(summary::NOTHING_TO_SUMMARIZE),
            ChatMode::Chat => None,
        }
    };
    if let Some(reply) = nothing_to_summarize {
        send_command_reply(handles, chat_id, reply).await?;
        return Ok(false);
    }

    let input = vec![ContentPart::Text {
        text: summary::SUMMARY_REQUEST.to_string(),
    }];
    let prepared = build_chat_request(chat_id, input, handles).await?;
    let turns = {
        let rooms = handles.rooms.lock().await;
        rooms
            .get(chat_id)
            .filter(|room| room.generation() == prepared.generation)
            .map(RoomState::turn_history)
    };
    let Some(turns) = turns.filter(|_| !prepared.request.history.is_empty()) else {
        // The room's model now uses another wire format than the history
        // was written in, so there is nothing this model can replay. Put the
        // history back; the next ordinary turn announces the reset.
        let mut rooms = handles.rooms.lock().await;
        if let Some(room) = rooms.get_mut(chat_id) {
            room.restore_failed_turn(prepared.generation, prepared.rollback);
        }
        drop(rooms);
        send_command_reply(handles, chat_id, summary::NOTHING_TO_SUMMARIZE).await?;
        return Ok(false);
    };

    let typing = AbortOnDrop(spawn_typing_indicator(handles.telegram.clone(), chat_id));
    let (response, left_out) = send_summary_request(prepared.request, &turns, |request| {
        let model_config = prepared.model_config.clone();
        async move { model_turn::dispatch_provider(&model_config, &request).await }
    })
    .await;
    typing.abort();

    let response = match response {
        Ok(response) => response,
        Err(error) => {
            log::warn!(
                target: "tellm::model",
                "summary failed chat_id={chat_id} error={:?}",
                error.message
            );
            let will_retry = {
                let mut rooms = handles.rooms.lock().await;
                rooms.get_mut(chat_id).and_then(|room| {
                    if !room.restore_failed_turn(prepared.generation, prepared.rollback) {
                        return None;
                    }
                    if trigger != summary::Trigger::Command {
                        room.failed_summaries = room.failed_summaries.saturating_add(1);
                    }
                    Some(room.failed_summaries < summary::MAX_AUTOMATIC_SUMMARY_FAILURES)
                })
            };
            if let Some(will_retry) = will_retry
                && worker_can_reply(chat_id, handles, cancelled).await
            {
                let reply = summary::failed_reply(trigger, &error.message, will_retry);
                send_command_reply(handles, chat_id, &reply).await?;
            }
            return Ok(false);
        }
    };
    if !worker_can_reply(chat_id, handles, cancelled).await {
        return Ok(false);
    }

    let summary = response.text.trim().to_string();
    let reply_cost = cost::ReplyCost::new(response.usage, prepared.model_config.price.as_ref());
    let committed = {
        let mut rooms = handles.rooms.lock().await;
        rooms.get_mut(chat_id).and_then(|room| {
            (room.generation() == prepared.generation).then(|| {
                room.cost.record(&reply_cost);
                room.record_context(reply_cost.usage.as_ref());
                let cost_line = room
                    .settings
                    .shows_cost(false)
                    .then(|| cost::cost_line(&reply_cost, &room.cost, room.context_tokens));
                room.continue_from_summary(summary.clone());
                cost_line
            })
        })
    };
    let Some(cost_line) = committed else {
        return Ok(false);
    };
    if worker_can_reply(chat_id, handles, cancelled).await {
        let mut reply = summary::carried_reply(&summary, left_out);
        if let Some(line) = cost_line {
            reply = model_turn::with_cost_line(&reply, &line);
        }
        send_command_reply(handles, chat_id, &reply).await?;
    }
    Ok(true)
}

/// Sends a summary request. When the provider says it is over the model's
/// context window, the request is sent again with the older half of the
/// turns left out, until it fits or a single turn was too much. Returns the
/// outcome and how many of the oldest turns the summary had to leave out.
async fn send_summary_request<F, Sent>(
    mut request: ChatRequest,
    turns: &rooms::TurnHistory,
    mut send: F,
) -> (Result<ChatResponse, model_turn::TurnError>, usize)
where
    F: FnMut(ChatRequest) -> Sent,
    Sent: std::future::Future<Output = Result<ChatResponse, model_turn::TurnError>>,
{
    let mut sent = turns.len();
    loop {
        let result = send(request.clone()).await.and_then(|response| {
            if response.text.trim().is_empty() {
                Err(summary::EMPTY_SUMMARY_ERROR.to_string().into())
            } else {
                Ok(response)
            }
        });
        match result {
            Err(error) if error.context_window => match summary::fewer_turns(sent) {
                Some(keep) => {
                    log::info!(
                        target: "tellm::model",
                        "summary request over the context window; retrying with fewer turns sent={sent} keep={keep}"
                    );
                    sent = keep;
                    request.history = turns.newest(keep);
                }
                None => return (Err(error), turns.len() - sent),
            },
            result => return (result, turns.len() - sent),
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
    // Read before taking the room locks. Commands and turns for this chat
    // share its queue, so the room's memory setting can't change meanwhile.
    let fresh_memory = if memory_enabled_in(chat_id, handles).await {
        memory::for_new_conversation(Some(memory::load().await))
    } else {
        ConversationMemory::default()
    };
    let config = handles.config.lock().await;
    let mut rooms = handles.rooms.lock().await;
    let room = rooms.get_or_default(chat_id);
    let model_key = selected_model_key(&config, room, chat_id);
    let model_config = config
        .models
        .get(&model_key)
        .cloned()
        .ok_or_else(|| format!("configured model \"{model_key}\" was not found"))?;
    Ok(model_turn::prepare_chat_request(
        room,
        model_config,
        input,
        fresh_memory,
    ))
}

/// A shared group's request: the room's model and settings, without its
/// history, memory or summary.
async fn build_standalone_request(
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
    Ok(model_turn::prepare_standalone_request(
        room,
        model_config,
        input,
    ))
}

async fn memory_enabled_in(chat_id: i64, handles: &RuntimeHandles) -> bool {
    let config = handles.config.lock().await;
    let rooms = handles.rooms.lock().await;
    let setting = rooms.get(chat_id).and_then(|room| room.settings.memory);
    memory::enabled_in(setting, chat_id, &config.telegram.owner_user_ids)
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
    let changed = update_config(handles, move |config| {
        let changed = allow_chat_in_config(config, target_chat_id);
        (changed, changed)
    })
    .await
    .map_err(|error| format!("failed to persist allowed chat {target_chat_id}: {error}"))?;

    {
        let mut access = handles.access.lock().await;
        access.allow_chat(target_chat_id);
    }

    warn_privacy_mode(target_chat_id, handles);

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
    // A command that denies its own chat is allowed to finish the
    // confirmation reply; its worker exits before dequeuing anything else.
    let result = revoke_chat(target_chat_id, target_chat_id != requester_chat_id, handles).await?;
    send_command_reply(
        handles,
        requester_chat_id,
        &format_deny_chat_reply(target_chat_id, &result),
    )
    .await
}

/// Ends a chat's access: config (allowed, private marking, model pins),
/// the live access check, its worker, and its room. `abort_worker` is false
/// only when the chat's own worker is running this, so it can still reply.
async fn revoke_chat(
    target_chat_id: i64,
    abort_worker: bool,
    handles: &RuntimeHandles,
) -> Result<DenyChatConfigResult, String> {
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
        // racing its final commit sees revocation synchronously.
        cancel_chat_worker(&handles.workers, target_chat_id, abort_worker);

        // The file gets the same change, applied to it as it is on disk.
        if result.changed()
            && let Err(error) = handles
                .persistence
                .update_config(move |config| {
                    deny_chat_in_config(config, target_chat_id);
                })
                .await
        {
            *config = before;
            if was_allowed {
                handles.access.lock().await.allow_chat(target_chat_id);
            }
            // Only a self-deny leaves the task alive. Restore its flag so
            // the normal worker error reply can report the failed save;
            // an aborted target worker and its queued work stay dropped.
            if !abort_worker {
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
    Ok(config_result)
}

async fn handle_pair_attempt(
    chat_id: i64,
    code: String,
    pairer_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<(), String> {
    let attempt = check_pair_code(chat_id, &code, handles).await;
    finish_pair_attempt(chat_id, attempt, pairer_user_id, handles).await
}

/// `/pair CODE` from a chat that is not allowed yet, run by the poll loop.
/// The code is checked here, in update order, so every attempt counts once
/// and in sequence. Saving and replying run on their own task: a slow
/// Telegram call or disk write must not hold up updates for every other
/// chat, and a stranger's chat can send `/pair` as often as it likes.
async fn start_pair_attempt(
    chat_id: i64,
    code: &str,
    pairer_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> tokio::task::JoinHandle<()> {
    let attempt = check_pair_code(chat_id, code, handles).await;
    let handles = handles.clone();
    tokio::spawn(async move {
        if let Err(error) = finish_pair_attempt(chat_id, attempt, pairer_user_id, &handles).await {
            log::warn!(
                target: "tellm::access",
                "pairing attempt failed chat_id={chat_id} error={error:?}"
            );
        }
    })
}

async fn check_pair_code(chat_id: i64, code: &str, handles: &RuntimeHandles) -> PairingAttempt {
    handles
        .access
        .lock()
        .await
        .attempt_pair(chat_id, code, now_access_time())
}

/// Saves a successful pairing and sends the reply for any attempt.
async fn finish_pair_attempt(
    chat_id: i64,
    attempt: PairingAttempt,
    pairer_user_id: Option<i64>,
    handles: &RuntimeHandles,
) -> Result<(), String> {
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
            warn_privacy_mode(chat_id, handles);
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
        PairingAttempt::Disabled { just_disabled } => {
            if just_disabled {
                log::error!(
                    target: "tellm::access",
                    "pairing disabled after {MAX_FAILED_PAIRINGS_PER_PROCESS} wrong codes across all chats; last chat_id={chat_id} action=\"restart tellm to pair a new chat; owners can still add the bot to groups\""
                );
            }
            send_command_reply(handles, chat_id, PAIRING_DISABLED_REPLY).await
        }
    }
}

const PAIRING_DISABLED_REPLY: &str = "Pairing is turned off after too many wrong codes. The bot owner can restart tellm to turn it back on.";

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
    model_capabilities(config, selected_model_key(config, room, chat_id))
}

fn model_capabilities(config: &Config, model_key: String) -> commands::RoomCapabilities {
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

/// Turns off the room settings its model can't use. A setting chosen under
/// the old model would otherwise make every message after a model switch
/// fail with a capability error. Returns the line that says so, if any.
fn drop_unsupported_settings(
    room: &mut RoomState,
    capabilities: &commands::RoomCapabilities,
) -> Option<String> {
    let web_search = room.settings.web_search && !capabilities.web_search;
    let image_generation = room.settings.image_generation && !capabilities.image_generation;
    if web_search {
        room.set_web_search(false);
    }
    if image_generation {
        room.settings.image_generation = false;
    }
    let (what, pronoun) = match (web_search, image_generation) {
        (true, true) => ("Web search and image generation are", "them"),
        (true, false) => ("Web search is", "it"),
        (false, true) => ("Image generation is", "it"),
        (false, false) => return None,
    };
    Some(format!(
        "{what} now off: {} doesn't support {pronoun}.",
        capabilities.model_key
    ))
}

fn with_settings_note(reply: String, note: Option<String>) -> String {
    match note {
        Some(note) => format!("{reply}\n{note}"),
        None => reply,
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
    update_config(handles, move |config| {
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
        (changed, became_owner)
    })
    .await
}

/// Changes the runtime's config and config.toml together. `change` runs on
/// the in-memory copy and, when it reports a change, again on the file as it
/// is on disk, so edits the owner made to the file since startup are kept. A
/// failed write restores the in-memory copy. Returns what `change` returned
/// for the in-memory copy.
async fn update_config<T>(
    handles: &RuntimeHandles,
    change: impl Fn(&mut Config) -> (bool, T) + Send + Sync + 'static,
) -> Result<T, String> {
    let change = Arc::new(change);
    let mut config = handles.config.lock().await;
    let before = config.clone();
    let (changed, value) = change(&mut config);
    if changed {
        let on_disk = Arc::clone(&change);
        let saved = handles
            .persistence
            .update_config(move |config| {
                on_disk(config);
            })
            .await;
        if let Err(error) = saved {
            *config = before;
            return Err(error);
        }
    }
    Ok(value)
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
        reserve_terminal_secret_prompt(&handles.terminal_prompts, chat_id, secret_name.clone())?;

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
        "{model_key} is already configured with no API key prompt. Select it here with /model {model_key}. If this endpoint needs a key, add api_key_secret = \"...\" under [models.{model_key}] in config.toml and restart tellm."
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
        "Could not complete the terminal prompt for {model_key}: {error}. Fallback: run tellm secret set {secret_name} in a separate terminal; the running bot uses the key from the next message without a restart."
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
    let (model_key, pinned, available, kind) = {
        let config = handles.config.lock().await;
        let mut rooms = handles.rooms.lock().await;
        let room = rooms.get_or_default(chat_id);
        (
            selected_model_key(&config, room, chat_id),
            pinned_model_key(&config, chat_id).map(str::to_string),
            config.models.keys().cloned().collect::<Vec<_>>(),
            groups::chat_kind(&config, chat_id),
        )
    };
    let mut setup = build_room_setup_reply(&model_key, pinned.as_deref(), available, became_owner);
    if kind == ChatKind::SharedGroup {
        setup.text.push_str("\n\n");
        setup.text.push_str(groups::NEW_GROUP_NOTE);
    }
    if chat_id < 0 && handles.privacy_mode == Some(true) {
        setup.text.push_str("\n\n");
        setup.text.push_str(PRIVACY_MODE_NOTE);
    }
    setup
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
        close_terminal_input(&secret_prompts);
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
    chat_id: i64,
    secret_name: String,
) -> Result<(mpsc::UnboundedReceiver<String>, TerminalPromptGuard), String> {
    let (line_tx, line_rx) = mpsc::unbounded_channel();
    {
        let mut slot = secret_prompts
            .lock()
            .map_err(|_| "terminal prompt lock poisoned".to_string())?;
        if slot.input_closed {
            return Err(TERMINAL_INPUT_CLOSED.to_string());
        }
        if slot.pending.is_some() {
            return Err(
                "another terminal prompt is already waiting; complete it first".to_string(),
            );
        }
        slot.pending = Some(TerminalSecretPrompt {
            secret_name: secret_name.clone(),
            line_tx,
            chat_id,
            wait_notice_sent: false,
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

const TERMINAL_INPUT_CLOSED: &str =
    "tellm's terminal input is closed, so no key can be entered there";

fn current_terminal_secret_prompt(
    secret_prompts: &TerminalSecretPrompts,
) -> Option<TerminalSecretPrompt> {
    secret_prompts.lock().ok()?.pending.clone()
}

/// Called when the stdin reader stops. Dropping a waiting prompt's sender
/// ends that prompt; the flag fails every later one at once instead of
/// leaving it to wait out its timeout for input that cannot come.
fn close_terminal_input(secret_prompts: &TerminalSecretPrompts) {
    let mut slot = secret_prompts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    slot.input_closed = true;
    slot.pending = None;
}

fn clear_terminal_secret_prompt(secret_prompts: &TerminalSecretPrompts, secret_name: &str) {
    let Ok(mut slot) = secret_prompts.lock() else {
        return;
    };
    if slot
        .pending
        .as_ref()
        .is_some_and(|prompt| prompt.secret_name == secret_name)
    {
        slot.pending = None;
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

/// Privacy mode is on: name the groups it limits, once at startup.
fn print_group_privacy_hints(chat_ids: &BTreeSet<i64>) {
    if chat_ids.is_empty() {
        return;
    }

    log::warn!(
        target: "tellm::telegram",
        "privacy mode is on groups={} action=\"{PRIVACY_MODE_ACTION}\"",
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

const PRIVACY_MODE_ACTION: &str = "unless the bot is an admin in a group, it gets only commands and replies to its messages there; turn privacy mode off with BotFather /setprivacy, then remove the bot from each group and add it again";

/// The group-chat line in the room setup message while privacy mode is on.
const PRIVACY_MODE_NOTE: &str = "My privacy mode is on, so unless I'm an admin here I only see \
commands and replies to my messages, not @mentions or anything else. To fix it, turn privacy \
mode off in BotFather (/setprivacy), then remove me from this group and add me again.";

/// Logs that a group the bot joined is limited by privacy mode, when getMe
/// said it is on.
fn warn_privacy_mode(chat_id: i64, handles: &RuntimeHandles) {
    if chat_id < 0 && handles.privacy_mode == Some(true) {
        log::warn!(target: "tellm::telegram", "{}", group_privacy_hint(chat_id));
    }
}

fn group_privacy_hint(chat_id: i64) -> String {
    format!("privacy mode is on chat_id={chat_id} action=\"{PRIVACY_MODE_ACTION}\"")
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

fn log_update_route(message: &IncomingMessage, kind: UpdateKind, route: UpdateRoute) {
    log::debug!(
        target: "tellm::telegram",
        "{}",
        update_log_line(
            message.chat.id,
            message.message_id,
            message.from.as_ref().map(|user| user.id),
            kind,
            &input_shape(message),
            route,
        )
    );
}

/// The message id tells a second message from the same one delivered twice,
/// the sender tells the user's own messages from another group member's, and
/// the input shape tells which message a model reply answered once the
/// message itself is deleted from the chat.
fn update_log_line(
    chat_id: i64,
    message_id: i64,
    sender_id: Option<i64>,
    kind: UpdateKind,
    input: &str,
    route: UpdateRoute,
) -> String {
    let sender = sender_id.map_or_else(|| "none".to_string(), |id| id.to_string());
    format!(
        "update chat_id={chat_id} message_id={message_id} from={sender} kind={} input={input} route={}",
        kind.as_str(),
        route.as_str()
    )
}

/// What a message carries, never its content: `text(8)`, `photo+caption(12)`,
/// `photos(7)` for an album, `text(5100,2 parts)` for a text a Telegram app
/// split, `document`, or `none`. Text starting with `/` is flagged, so a slash
/// command tellm doesn't know shows up as `route=model` with `slash`.
fn input_shape(message: &IncomingMessage) -> String {
    let items = || std::iter::once(message).chain(&message.album);
    let mut parts = Vec::new();
    for (count, one, many) in [
        (
            items().filter(|item| model_turn::has_photo(item)).count(),
            "photo",
            "photos",
        ),
        (
            items().filter(|item| item.document.is_some()).count(),
            "document",
            "documents",
        ),
    ] {
        match count {
            0 => {}
            1 => parts.push(one.to_string()),
            count => parts.push(format!("{many}({count})")),
        }
    }
    if let Some(text) = model_turn::message_text(message)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        let label = if message.text.is_some() {
            "text"
        } else {
            "caption"
        };
        let mut details = vec![text.chars().count().to_string()];
        if message.split_parts > 1 {
            details.push(format!("{} parts", message.split_parts));
        }
        if text.starts_with('/') {
            details.push("slash".to_string());
        }
        parts.push(format!("{label}({})", details.join(",")));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("+")
    }
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

/// The room's model: its pin, else its own selection while that model is
/// still configured, else the default. A selection whose model was removed
/// from or renamed in config.toml falls back to the default rather than
/// failing every message; it applies again if the model comes back.
fn selected_model_key(config: &Config, room: &RoomState, chat_id: i64) -> String {
    pinned_model_key(config, chat_id)
        .map(str::to_string)
        .or_else(|| {
            room.settings
                .model_key
                .clone()
                .filter(|key| config.models.contains_key(key))
        })
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
    private_removed: bool,
    removed_model_pins: Vec<String>,
}

impl DenyChatConfigResult {
    fn changed(&self) -> bool {
        self.allowed_removed || self.private_removed || !self.removed_model_pins.is_empty()
    }
}

fn deny_chat_in_config(config: &mut Config, chat_id: i64) -> DenyChatConfigResult {
    let before_allowed = config.telegram.allowed_chat_ids.len();
    config
        .telegram
        .allowed_chat_ids
        .retain(|allowed| *allowed != chat_id);
    let allowed_removed = config.telegram.allowed_chat_ids.len() != before_allowed;
    let before_private = config.telegram.private_group_ids.len();
    config
        .telegram
        .private_group_ids
        .retain(|private| *private != chat_id);
    let private_removed = config.telegram.private_group_ids.len() != before_private;

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
        private_removed,
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

fn model_unchanged_reply(model_key: &str) -> String {
    format!("Already using {model_key}. Nothing changed.")
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
        CommandReject::SharedGroupOwnerOnly => {
            "In a shared group only the bot's owner can use commands, except /help and /id. \
             Mention me or reply to me to ask something."
                .to_string()
        }
        CommandReject::StandaloneInSharedGroup => {
            "This group is shared, so every message stands alone: there is no conversation to \
             change or summarize. /group private makes it a private room."
                .to_string()
        }
        CommandReject::MemoryInSharedGroup => {
            "Memory stays off in a shared group, and I don't list or change your facts where \
             others can read them. Use the memory commands in a private group."
                .to_string()
        }
        CommandReject::UnknownGroupOption { value } => format!(
            "Unknown /group option \"{value}\". Use /group, /group private or /group shared."
        ),
        CommandReject::UnknownCommand { name } => {
            format!("Unknown command /{name}. Send /help for the list.")
        }
        CommandReject::UnknownNewOption { value } => format!(
            "Unknown /new option \"{value}\". Use /new to start over, or /new summary to carry a summary of this chat into the new one."
        ),
        CommandReject::MissingPairingCode => "Usage: /pair CODE.".to_string(),
        CommandReject::MissingMemoryFact => {
            "Usage: /remember TEXT, for example /remember I have two kids, aged 7 and 10."
                .to_string()
        }
        CommandReject::InvalidMemoryNumber { value } => match value {
            Some(value) => {
                format!("\"{value}\" isn't a fact number. Usage: /forget N; /memory lists them.")
            }
            None => "Usage: /forget N; /memory lists the numbers.".to_string(),
        },
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
        KnownCommand::Cost => "cost",
        KnownCommand::Memory => "memory",
        KnownCommand::Remember => "remember",
        KnownCommand::Forget => "forget",
        KnownCommand::Allow => "allow",
        KnownCommand::Deny => "deny",
        KnownCommand::Pair => "pair",
        KnownCommand::Ollama => "ollama",
        KnownCommand::Shutdown => "shutdown",
        KnownCommand::Group => "group",
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
- /new [summary] - start a new chat, or carry a summary of this one into it
- /id - show this Telegram chat id
- /mode chat|message - show or set conversation mode
- /model KEY - show or set model
- /model pin KEY | /model unpin - lock or release this room's model (owner)
- /model add [KEY] - list the provider catalog or add a preset (owner)
- /role TEXT|clear - show, set, or clear the system role
- /reasoning default|off|low|medium|high|max - show or set reasoning level
- /websearch on|off|status - toggle, set, or show web search
- /imagegen on|off|status - toggle, set, or show image generation
- /cost on|off - show this chat's cost, or show or hide the cost line under each reply
- /memory on|off - list your saved facts, or use them in this chat or not (owner)
- /remember TEXT - save a fact for new chats (owner)
- /forget N - remove saved fact N (owner)
- /allow CHAT_ID - allow a chat (owner)
- /deny CHAT_ID - deny a chat and clear its room state (owner)
- /pair CODE - pair a new chat
- /ollama unload - unload local Ollama models used by this session (owner)
- /group private|shared - show or set whether this group is a private room or shared (owner)
- /shutdown - stop tellm (owner)
- /help - show commands";

#[cfg(test)]
mod tests {
    use std::sync::mpsc as std_mpsc;

    use tellm_config::ModelConfig;
    use tellm_test_support::{MockHttpServer, MockResponse};

    use super::*;
    use crate::access::MAX_PAIRING_ATTEMPTS;
    use crate::rooms::HistoryReset;

    #[tokio::test]
    async fn config_writes_apply_their_change_to_the_file_as_it_is_on_disk() {
        let startup = Config {
            default_model: "openai".to_string(),
            models: BTreeMap::from([(
                "openai".to_string(),
                test_model(WireFormat::Responses, &[]),
            )]),
            telegram: tellm_config::TelegramConfig::default(),
        };
        // Since startup the owner added a model to config.toml by hand.
        let mut edited = startup.clone();
        edited.models.insert(
            "added-by-hand".to_string(),
            test_model(WireFormat::Compat, &[]),
        );
        let disk = Arc::new(StdMutex::new(edited));
        let writer_disk = Arc::clone(&disk);
        let (persistence, thread) = persistence::spawn_with(
            move |change: persistence::ConfigChange| {
                change(&mut writer_disk.lock().unwrap());
                Ok(())
            },
            |_settings| Ok(()),
        )
        .expect("spawn persistence writer");
        let config = Arc::new(Mutex::new(startup));
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);
        let handles = RuntimeHandles {
            telegram: Telegram::new("test-token"),
            config: Arc::clone(&config),
            rooms: Arc::new(Mutex::new(RoomStates::default())),
            access: Arc::new(Mutex::new(AccessControl::new(AccessConfig::default()))),
            terminal_prompts: Arc::default(),
            bot_username: None,
            bot_user_id: None,
            privacy_mode: None,
            shutdown_tx,
            persistence: persistence.clone(),
            workers: Arc::new(StdMutex::new(BTreeMap::new())),
            queue_full_notices: Arc::new(StdMutex::new(BTreeSet::new())),
        };

        assert!(persist_paired_chat(42, Some(7), &handles).await.unwrap());

        let on_disk = disk.lock().unwrap().clone();
        assert!(
            on_disk.models.contains_key("added-by-hand"),
            "the hand edit survives the runtime's write"
        );
        assert_eq!(on_disk.telegram.allowed_chat_ids, vec![42]);
        assert_eq!(on_disk.telegram.owner_user_ids, vec![7]);
        let in_memory = config.lock().await.clone();
        assert_eq!(in_memory.telegram.allowed_chat_ids, vec![42]);
        assert!(!in_memory.models.contains_key("added-by-hand"));

        // A change that reports nothing new writes nothing.
        let written_before = disk.lock().unwrap().clone();
        assert!(
            !update_config(&handles, |config| {
                let changed = allow_chat_in_config(config, 42);
                (changed, changed)
            })
            .await
            .unwrap()
        );
        assert_eq!(
            disk.lock().unwrap().telegram.allowed_chat_ids,
            written_before.telegram.allowed_chat_ids
        );

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
            |_change| Err("injected config failure".to_string()),
            |_settings| Ok(()),
        )
        .expect("spawn persistence writer");
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);
        let handles = RuntimeHandles {
            telegram: Telegram::new("test-token"),
            config: Arc::clone(&config),
            rooms: Arc::new(Mutex::new(RoomStates::default())),
            access: Arc::new(Mutex::new(access)),
            terminal_prompts: Arc::default(),
            bot_username: None,
            bot_user_id: None,
            privacy_mode: None,
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
                waiting: Arc::default(),
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
                private_group_ids: Vec::new(),
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
            move |_change| {
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
                waiting: Arc::default(),
            },
        );
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);
        let handles = RuntimeHandles {
            telegram: Telegram::new("test-token"),
            config: Arc::clone(&config),
            rooms: Arc::clone(&rooms),
            access: Arc::new(Mutex::new(access)),
            terminal_prompts: Arc::default(),
            bot_username: None,
            bot_user_id: None,
            privacy_mode: None,
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
            message_id: 1,
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
            media_group_id: None,
            album: Vec::new(),
            split_parts: 0,
            reply_to_message: None,
            new_chat_members: Vec::new(),
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

        let stale = RoomState::new(crate::rooms::RoomSettings {
            model_key: Some("renamed-since".to_string()),
            ..crate::rooms::RoomSettings::default()
        });
        assert_eq!(
            selected_model_key(&config, &stale, 7),
            "claude",
            "a selection whose model is gone falls back to the default"
        );
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
                private_group_ids: vec![-100],
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
                private_removed: true,
                removed_model_pins: vec!["gpt".to_string()],
            }
        );
        assert!(!config.telegram.allowed_chat_ids.contains(&-100));
        assert!(config.telegram.private_group_ids.is_empty());
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
                private_group_ids: Vec::new(),
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
    fn a_message_during_a_key_prompt_gets_one_wait_notice_in_that_chat() {
        let prompts: TerminalSecretPrompts = Arc::default();
        assert_eq!(
            reserve_prompt_wait_notice(&prompts, 7),
            None,
            "no prompt waiting"
        );

        let (_lines, guard) =
            reserve_terminal_secret_prompt(&prompts, 7, "gemini_api_key".to_string()).unwrap();
        assert_eq!(
            reserve_prompt_wait_notice(&prompts, 8),
            None,
            "other rooms are not held by the prompt"
        );
        assert_eq!(
            reserve_prompt_wait_notice(&prompts, 7).as_deref(),
            Some("gemini_api_key")
        );
        assert_eq!(
            reserve_prompt_wait_notice(&prompts, 7),
            None,
            "one notice per prompt"
        );

        drop(guard);
        assert_eq!(
            reserve_prompt_wait_notice(&prompts, 7),
            None,
            "prompt ended"
        );
        assert!(prompt_wait_notice("gemini_api_key").contains("gemini_api_key"));
    }

    #[test]
    fn terminal_prompts_fail_at_once_after_terminal_input_closes() {
        let prompts: TerminalSecretPrompts = Arc::default();
        let (mut waiting, _guard) =
            reserve_terminal_secret_prompt(&prompts, 7, "first_key".to_string()).unwrap();

        close_terminal_input(&prompts);

        assert!(
            matches!(
                waiting.try_recv(),
                Err(mpsc::error::TryRecvError::Disconnected)
            ),
            "a prompt that was waiting ends with the input"
        );
        let error = reserve_terminal_secret_prompt(&prompts, 7, "second_key".to_string())
            .err()
            .expect("no new prompt once input is closed");
        assert_eq!(error, TERMINAL_INPUT_CLOSED);
        let reply = model_key_prompt_failed_reply("claude", "anthropic_api_key", &error);
        assert!(
            reply.contains("tellm secret set anthropic_api_key"),
            "{reply}"
        );
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
            update_log_line(
                -100,
                17,
                Some(7),
                UpdateKind::Message,
                "text(8)",
                UpdateRoute::Model
            ),
            "update chat_id=-100 message_id=17 from=7 kind=message input=text(8) route=model"
        );
        assert_eq!(
            update_log_line(
                42,
                3,
                None,
                UpdateKind::EditedMessage,
                "none",
                UpdateRoute::Ignored
            ),
            "update chat_id=42 message_id=3 from=none kind=edited_message input=none route=ignored"
        );
    }

    fn album_item(message_id: i64, group: &str, caption: Option<&str>) -> IncomingMessage {
        let mut item = text_message("");
        item.message_id = message_id;
        item.text = None;
        item.caption = caption.map(str::to_string);
        item.photo = Some(vec![tellm_telegram::PhotoSize {
            file_id: format!("photo-{message_id}"),
            width: 1,
            height: 1,
            file_size: None,
        }]);
        item.media_group_id = Some(group.to_string());
        item
    }

    #[tokio::test]
    async fn an_album_is_one_message_and_a_message_after_it_waits_its_turn() {
        let (sender, mut receiver) = mpsc::channel(CHAT_QUEUE_SIZE);
        for id in 2..=7 {
            sender.send(album_item(id, "g", None)).await.unwrap();
        }
        sender.send(text_message("what do you see?")).await.unwrap();

        let (album, next) = gather_album(
            album_item(1, "g", Some("my trip")),
            &mut receiver,
            Duration::from_millis(50),
        )
        .await;

        assert_eq!(album.message_id, 1);
        assert_eq!(
            album
                .album
                .iter()
                .map(|item| item.message_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 5, 6, 7]
        );
        assert_eq!(album.caption.as_deref(), Some("my trip"));
        assert_eq!(input_shape(&album), "photos(7)+caption(7)");
        assert_eq!(
            next.and_then(|message| message.text).as_deref(),
            Some("what do you see?"),
            "the next message is handed back, not dropped or reordered"
        );
    }

    #[tokio::test]
    async fn album_gathering_stops_at_telegrams_limit_and_on_a_quiet_queue() {
        let (sender, mut receiver) = mpsc::channel(CHAT_QUEUE_SIZE);
        for id in 2..=11 {
            sender.send(album_item(id, "g", None)).await.unwrap();
        }
        let (album, next) = gather_album(
            album_item(1, "g", None),
            &mut receiver,
            Duration::from_millis(50),
        )
        .await;
        assert_eq!(album.album.len(), MAX_ALBUM_ITEMS - 1);
        assert!(next.is_none());
        assert_eq!(
            receiver.try_recv().unwrap().message_id,
            11,
            "an item past the limit stays queued"
        );

        let (album, next) = gather_album(
            album_item(20, "h", None),
            &mut receiver,
            Duration::from_millis(20),
        )
        .await;
        assert!(album.album.is_empty() && next.is_none());

        let (single, next) = timeout(
            Duration::from_secs(1),
            gather_album(text_message("hi"), &mut receiver, Duration::from_secs(60)),
        )
        .await
        .expect("a message outside an album is not held back");
        assert_eq!(single.text.as_deref(), Some("hi"));
        assert!(next.is_none());
    }

    #[test]
    fn an_albums_captions_all_reach_the_model() {
        let album = merge_album(vec![
            album_item(1, "g", Some("first")),
            album_item(2, "g", None),
            album_item(3, "g", Some(" third ")),
        ]);
        assert_eq!(album.caption.as_deref(), Some("first\n\nthird"));
        assert!(album.album.iter().all(|item| item.caption.is_none()));
    }

    /// A text message from user 7, as one part of what they sent.
    fn text_part(message_id: i64, date: i64, text: &str) -> IncomingMessage {
        let mut part = text_message(text);
        part.message_id = message_id;
        part.date = date;
        part.from = Some(tellm_telegram::User {
            id: 7,
            is_bot: false,
            first_name: "Ana".to_string(),
            username: None,
            can_read_all_group_messages: None,
        });
        part
    }

    #[tokio::test]
    async fn a_split_text_is_one_message_up_to_its_short_last_part() {
        let first = "a".repeat(SPLIT_TEXT_MIN_PART + 1_000);
        let second = "b".repeat(SPLIT_TEXT_MIN_PART);
        // A cut can leave a part starting with `/`; it is still text.
        let last = "/usr/bin ends it.";
        let (sender, mut receiver) = mpsc::channel(CHAT_QUEUE_SIZE);
        sender.send(text_part(2, 1000, &second)).await.unwrap();
        sender.send(text_part(3, 1001, last)).await.unwrap();
        sender.send(text_part(4, 1001, "thanks")).await.unwrap();

        let (text, next) = timeout(
            Duration::from_secs(1),
            gather_split_text(
                text_part(1, 1000, &first),
                &mut receiver,
                Duration::from_secs(60),
            ),
        )
        .await
        .expect("a short part ends the text without waiting");

        assert_eq!(text.message_id, 1);
        assert_eq!(
            text.text.as_deref(),
            Some(format!("{first}\n{second}\n{last}").as_str())
        );
        assert_eq!(text.split_parts, 3);
        assert_eq!(
            input_shape(&text),
            format!(
                "text({},3 parts)",
                first.len() + second.len() + last.len() + 2
            )
        );
        assert!(next.is_none());
        assert_eq!(
            receiver.try_recv().unwrap().message_id,
            4,
            "a message after the short last part stays queued"
        );
    }

    #[tokio::test]
    async fn only_a_long_text_waits_for_another_part() {
        let (sender, mut receiver) = mpsc::channel(CHAT_QUEUE_SIZE);
        sender.send(text_part(2, 1000, "and more")).await.unwrap();
        for short in ["hi".to_string(), "a".repeat(SPLIT_TEXT_MIN_PART - 1)] {
            let (text, next) = timeout(
                Duration::from_secs(1),
                gather_split_text(
                    text_part(1, 1000, &short),
                    &mut receiver,
                    Duration::from_secs(60),
                ),
            )
            .await
            .expect("a shorter message is not held back");
            assert_eq!(text.text, Some(short));
            assert_eq!(text.split_parts, 0);
            assert!(next.is_none(), "the queue is left alone");
        }

        // Telegram counts UTF-16 code units, two for each of these emoji.
        let emoji = "😀".repeat(SPLIT_TEXT_MIN_PART / 2);
        let (text, next) = gather_split_text(
            text_part(1, 1000, &emoji),
            &mut receiver,
            Duration::from_millis(50),
        )
        .await;
        assert_eq!(text.text, Some(format!("{emoji}\nand more")));
        assert_eq!(text.split_parts, 2);
        assert!(next.is_none());
    }

    #[tokio::test]
    async fn a_long_text_never_joins_another_sender_a_later_message_or_media() {
        let long = "a".repeat(SPLIT_TEXT_MIN_PART);
        let mut other_sender = text_part(2, 1000, "me too");
        other_sender.from.as_mut().unwrap().id = 8;
        // Queued while the room was busy: back to back in the queue, but sent
        // later than any part of a split.
        let later = text_part(2, 1003, "and another thing");
        let mut photo = album_item(2, "g", None);
        photo.media_group_id = None;
        photo.from = text_part(2, 1000, "").from;
        let mut anonymous = text_part(1, 1000, &long);
        anonymous.from = None;
        let mut anonymous_next = text_part(2, 1000, "who sent this?");
        anonymous_next.from = None;

        for (first, next) in [
            (text_part(1, 1000, &long), other_sender),
            (text_part(1, 1000, &long), later),
            (text_part(1, 1000, &long), photo),
            (anonymous, anonymous_next),
        ] {
            let (sender, mut receiver) = mpsc::channel(CHAT_QUEUE_SIZE);
            sender.send(next).await.unwrap();
            let (text, next) =
                gather_split_text(first, &mut receiver, Duration::from_millis(50)).await;
            assert_eq!(text.text.as_deref(), Some(long.as_str()));
            assert_eq!(text.split_parts, 0);
            assert_eq!(
                next.map(|message| message.message_id),
                Some(2),
                "the message is handed back to be handled next"
            );
        }

        let (_sender, mut receiver) = mpsc::channel(CHAT_QUEUE_SIZE);
        let (text, next) = gather_split_text(
            text_part(1, 1000, &long),
            &mut receiver,
            Duration::from_millis(20),
        )
        .await;
        assert_eq!(text.split_parts, 0, "a quiet queue ends the wait");
        assert!(next.is_none());
    }

    #[tokio::test]
    async fn the_worker_answers_a_split_text_once() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![completion("One answer.", 10, 5)]);
        let handles = mock_handles(&telegram, &provider);
        let first = "a".repeat(SPLIT_TEXT_MIN_PART);

        let dispatcher = spawn_chat_worker(42, handles.clone(), None);
        dispatcher
            .sender
            .try_send(text_part(1, 1000, &first))
            .unwrap();
        dispatcher
            .sender
            .try_send(text_part(2, 1000, "the rest"))
            .unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while sent_texts(&telegram).is_empty() {
            assert!(tokio::time::Instant::now() < deadline, "no reply sent");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let requests = provider.requests();
        assert_eq!(requests.len(), 1, "one turn for both parts");
        let messages = requests[0].json_body()["messages"].clone();
        assert_eq!(
            messages.as_array().unwrap().last().unwrap()["content"],
            format!("{first}\nthe rest")
        );
        assert_eq!(sent_texts(&telegram).len(), 1);
        tokio::time::timeout(
            Duration::from_secs(3),
            stop_chat_workers(BTreeMap::from([(42, dispatcher)]), &handles.workers),
        )
        .await
        .expect("the worker is joined");
    }

    const GROUP: i64 = -5;

    /// Handles where chat -5 is an allowed group, private or shared, user 7
    /// is the owner, and the bot is @tellm_bot with user id 99. The room
    /// shows no cost line, so the chat holds only the replies.
    async fn group_handles(
        telegram: &MockHttpServer,
        provider: &MockHttpServer,
        private: bool,
    ) -> RuntimeHandles {
        let mut handles = mock_handles(telegram, provider);
        let mut config = handles.config.lock().await.clone();
        config.telegram.allowed_chat_ids.push(GROUP);
        config.telegram.owner_user_ids = vec![7];
        if private {
            config.telegram.private_group_ids = vec![GROUP];
        }
        handles.access = Arc::new(Mutex::new(AccessControl::new(AccessConfig::from_config(
            &config,
        ))));
        handles.config = Arc::new(Mutex::new(config));
        handles.bot_username = Some("tellm_bot".to_string());
        handles.bot_user_id = Some(99);
        handles
            .rooms
            .lock()
            .await
            .get_or_default(GROUP)
            .settings
            .show_cost = Some(false);
        handles
    }

    /// A message just sent in the group by user `from`.
    fn group_message(from: i64, text: &str) -> IncomingMessage {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut message = text_part(1, i64::try_from(now).unwrap(), text);
        message.chat.id = GROUP;
        let sender = message.from.as_mut().unwrap();
        sender.id = from;
        sender.first_name = if from == 99 { "tellm" } else { "Ivo" }.to_string();
        message
    }

    #[tokio::test]
    async fn a_shared_group_answers_only_mentions_and_replies_each_on_its_own() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![
            completion("Option B.", 10, 5),
            completion("It is cheaper.", 10, 5),
        ]);
        let handles = group_handles(&telegram, &provider, false).await;
        {
            let mut rooms = handles.rooms.lock().await;
            let room = rooms.get_or_default(GROUP);
            room.settings.role = Some("Be brief.".to_string());
            room.summary = Some("- The owner moves in May.".to_string());
            room.append_turn(WireFormat::Compat, compat_turn("I move in May.", "Noted."));
        }
        let cancelled = AtomicBool::new(false);

        for text in ["anyone free tonight?", "/start"] {
            handle_allowed_message(GROUP, group_message(8, text), &handles, &cancelled)
                .await
                .unwrap();
        }
        assert!(
            provider.requests().is_empty() && telegram.requests().is_empty(),
            "a message not meant for the bot, or another bot's command, gets nothing"
        );

        handle_allowed_message(
            GROUP,
            group_message(8, "@tellm_bot which option?"),
            &handles,
            &cancelled,
        )
        .await
        .unwrap();
        let mut follow_up = group_message(8, "why?");
        follow_up.reply_to_message = Some(Box::new(group_message(99, "Option B.")));
        handle_allowed_message(GROUP, follow_up, &handles, &cancelled)
            .await
            .unwrap();

        // The role only: no history, memory or summary of the room's own
        // conversation.
        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].json_body()["messages"],
            serde_json::json!([
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "which option?" }
            ])
        );
        assert_eq!(
            requests[1].json_body()["messages"],
            serde_json::json!([
                { "role": "system", "content": "Be brief." },
                {
                    "role": "user",
                    "content": "Replying to your earlier message:\n> Option B.\n\nwhy?"
                }
            ])
        );
        assert_eq!(sent_texts(&telegram), vec!["Option B.", "It is cheaper."]);
        for request in sent_message_bodies(&telegram) {
            assert_eq!(
                request["reply_parameters"]["message_id"],
                serde_json::json!(1),
                "each answer quotes the question it answers"
            );
        }

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(GROUP).unwrap();
        assert_eq!(
            room.history,
            compat_turn("I move in May.", "Noted."),
            "the room's conversation is untouched"
        );
        assert_eq!(room.summary.as_deref(), Some("- The owner moves in May."));
        assert_eq!(room.context_tokens, None);
    }

    #[tokio::test]
    async fn a_private_group_answers_every_message_in_its_conversation() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![
            completion("In May.", 10, 5),
            completion("Sofia.", 10, 5),
        ]);
        let handles = group_handles(&telegram, &provider, true).await;
        handles
            .rooms
            .lock()
            .await
            .get_or_default(GROUP)
            .append_turn(WireFormat::Compat, compat_turn("I move soon.", "Noted."));

        handle_allowed_message(
            GROUP,
            group_message(8, "when again?"),
            &handles,
            &AtomicBool::new(false),
        )
        .await
        .unwrap();

        assert_eq!(
            provider.requests()[0].json_body()["messages"],
            serde_json::json!([
                { "role": "user", "content": "I move soon." },
                { "role": "assistant", "content": "Noted." },
                { "role": "user", "content": "when again?" }
            ])
        );
        assert_eq!(sent_texts(&telegram), vec!["In May."]);
        assert!(
            sent_message_bodies(&telegram)[0]
                .get("reply_parameters")
                .is_none(),
            "nothing else waits, so the answer needs no quote"
        );

        // Another message waits behind this one: the answer quotes its
        // question so it is clear which one it answers.
        let worker = tokio::spawn(std::future::pending::<()>());
        handles.workers.lock().unwrap().insert(
            GROUP,
            WorkerRegistration {
                worker_id: 1,
                cancelled: Arc::default(),
                abort_handle: worker.abort_handle(),
                waiting: Arc::new(AtomicUsize::new(1)),
            },
        );
        let mut question = group_message(8, "and where?");
        question.message_id = 5;
        handle_allowed_message(GROUP, question, &handles, &AtomicBool::new(false))
            .await
            .unwrap();
        worker.abort();
        assert_eq!(
            sent_message_bodies(&telegram)[1]["reply_parameters"]["message_id"],
            serde_json::json!(5)
        );
    }

    #[tokio::test]
    async fn in_a_shared_group_only_an_owner_changes_anything_and_group_marks_it() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let handles = group_handles(&telegram, &provider, false).await;
        let cancelled = AtomicBool::new(false);
        let send = |from, text| {
            let message = group_message(from, text);
            let handles = &handles;
            let cancelled = &cancelled;
            async move {
                handle_allowed_message(GROUP, message, handles, cancelled)
                    .await
                    .unwrap();
            }
        };

        send(8, "/new").await;
        send(7, "/group private").await;
        assert_eq!(
            handles.config.lock().await.telegram.private_group_ids,
            vec![GROUP]
        );
        send(7, "/group shared").await;
        assert!(
            handles
                .config
                .lock()
                .await
                .telegram
                .private_group_ids
                .is_empty()
        );
        send(7, "/group shared").await;

        assert_eq!(
            sent_texts(&telegram),
            vec![
                format_reject(CommandReject::SharedGroupOwnerOnly),
                groups::status_reply(ChatKind::PrivateGroup, true),
                groups::status_reply(ChatKind::SharedGroup, true),
                groups::status_reply(ChatKind::SharedGroup, false),
            ]
        );
    }

    #[tokio::test]
    async fn a_private_group_says_when_someone_who_isnt_an_owner_joins() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let joined = |members: &[(i64, &str)]| {
            let mut message = group_message(7, "");
            message.text = None;
            message.new_chat_members = members
                .iter()
                .map(|(id, name)| tellm_telegram::User {
                    id: *id,
                    is_bot: *id == 99,
                    first_name: (*name).to_string(),
                    username: None,
                    can_read_all_group_messages: None,
                })
                .collect();
            message
        };
        let cancelled = AtomicBool::new(false);

        let private = group_handles(&telegram, &provider, true).await;
        for members in [&[(99, "tellm"), (7, "Owner")][..], &[(8, "Ivo")][..]] {
            handle_allowed_message(GROUP, joined(members), &private, &cancelled)
                .await
                .unwrap();
        }
        let shared = group_handles(&telegram, &provider, false).await;
        handle_allowed_message(GROUP, joined(&[(9, "Mia")]), &shared, &cancelled)
            .await
            .unwrap();

        assert_eq!(sent_texts(&telegram), vec![groups::join_notice(&["Ivo"])]);
        assert!(provider.requests().is_empty());
    }

    #[tokio::test]
    async fn start_shows_help_in_a_direct_chat_and_model_keeps_the_same_model() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let handles = mock_handles(&telegram, &provider);
        handles
            .rooms
            .lock()
            .await
            .get_or_default(42)
            .append_turn(WireFormat::Compat, compat_turn("Hi.", "Hello."));
        let cancelled = AtomicBool::new(false);

        for text in ["/start", "/model local"] {
            handle_allowed_message(42, text_message(text), &handles, &cancelled)
                .await
                .unwrap();
        }

        let sent = sent_texts(&telegram);
        assert_eq!(sent.len(), 2, "{sent:?}");
        assert!(sent[0].contains("/new [summary]"), "{}", sent[0]);
        assert_eq!(sent[1], "Already using local. Nothing changed.");
        assert_eq!(
            handles.rooms.lock().await.get_or_default(42).history.len(),
            2,
            "the conversation goes on"
        );
    }

    fn membership(from: i64, old: &str, new: &str) -> tellm_telegram::ChatMemberUpdated {
        serde_json::from_value(serde_json::json!({
            "chat": { "id": GROUP, "title": "Family" },
            "from": { "id": from, "is_bot": false, "first_name": "Ana" },
            "old_chat_member": { "status": old },
            "new_chat_member": { "status": new },
        }))
        .unwrap()
    }

    async fn wait_for_sent(telegram: &MockHttpServer) -> Vec<String> {
        for _ in 0..200 {
            let sent = sent_requests(telegram);
            if !sent.is_empty() {
                return sent;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Vec::new()
    }

    /// Every message-sending request, by method name.
    fn sent_requests(telegram: &MockHttpServer) -> Vec<String> {
        telegram
            .requests()
            .iter()
            .filter(|request| request.path.contains("/send"))
            .map(|request| {
                request
                    .path
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect()
    }

    #[tokio::test]
    async fn the_bot_removed_from_a_group_forgets_it_and_a_promotion_changes_nothing() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let handles = group_handles(&telegram, &provider, true).await;
        handles
            .rooms
            .lock()
            .await
            .get_or_default(GROUP)
            .settings
            .role = Some("Be brief.".to_string());

        handle_membership_change(membership(7, "member", "administrator"), &handles).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            telegram.requests().is_empty(),
            "becoming an admin sends no second welcome"
        );
        assert!(chat_is_allowed(GROUP, &handles).await);

        handle_membership_change(membership(7, "administrator", "kicked"), &handles).await;
        {
            let config = handles.config.lock().await;
            assert!(!config.telegram.allowed_chat_ids.contains(&GROUP));
            assert!(config.telegram.private_group_ids.is_empty());
        }
        assert!(!chat_is_allowed(GROUP, &handles).await);
        assert_eq!(
            handles.rooms.lock().await.get_or_default(GROUP).settings,
            rooms::RoomSettings::default(),
            "the room starts over"
        );

        handle_membership_change(membership(7, "left", "member"), &handles).await;
        assert_eq!(wait_for_sent(&telegram).await, ["sendMessage"]);
        assert!(chat_is_allowed(GROUP, &handles).await);
        let welcome = telegram.requests()[0].json_body()["text"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        assert!(welcome.contains(groups::NEW_GROUP_NOTE), "{welcome}");
    }

    #[tokio::test]
    async fn a_new_group_is_told_it_starts_shared() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let shared = group_handles(&telegram, &provider, false).await;
        assert!(
            room_setup_reply(GROUP, false, &shared)
                .await
                .text
                .ends_with(groups::NEW_GROUP_NOTE)
        );
        let private = group_handles(&telegram, &provider, true).await;
        assert!(
            !room_setup_reply(GROUP, false, &private)
                .await
                .text
                .contains("shared")
        );
        assert!(
            !room_setup_reply(42, false, &private)
                .await
                .text
                .contains("shared"),
            "a direct chat isn't a group"
        );
    }

    #[test]
    fn input_shape_names_kind_and_length_never_content() {
        let secret = "my password is hunter2";
        let shape = input_shape(&text_message(secret));
        assert_eq!(shape, "text(22)");
        assert!(!shape.contains("hunter2"));

        assert_eq!(input_shape(&text_message("  Hi there \n")), "text(8)");
        assert_eq!(input_shape(&text_message("/start")), "text(6,slash)");
        assert_eq!(input_shape(&text_message("   ")), "none");

        let mut photo = text_message("");
        photo.text = None;
        photo.caption = Some("look at this".to_string());
        photo.photo = Some(vec![tellm_telegram::PhotoSize {
            file_id: "f".to_string(),
            width: 1,
            height: 1,
            file_size: None,
        }]);
        assert_eq!(input_shape(&photo), "photo+caption(12)");

        let mut document = text_message("");
        document.text = None;
        document.document = Some(tellm_telegram::Document {
            file_id: "d".to_string(),
            file_name: None,
            mime_type: None,
            file_size: None,
        });
        assert_eq!(input_shape(&document), "document");
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
        assert!(hint.contains("add it again"));
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
    fn switching_models_turns_off_settings_the_new_model_cannot_use() {
        let mut config = Config {
            default_model: "astra".to_string(),
            models: BTreeMap::new(),
            telegram: tellm_config::TelegramConfig::default(),
        };
        config
            .models
            .insert("astra".to_string(), test_model(WireFormat::Responses, &[]));
        config
            .models
            .insert("inkling".to_string(), test_model(WireFormat::Compat, &[]));
        let mut claude = test_model(WireFormat::Anthropic, &[]);
        claude.model_name = "claude-test".to_string();
        config.models.insert("claude".to_string(), claude);
        let both_on = || {
            RoomState::new(crate::rooms::RoomSettings {
                web_search: true,
                image_generation: true,
                ..crate::rooms::RoomSettings::default()
            })
        };

        let mut room = both_on();
        let note = drop_unsupported_settings(
            &mut room,
            &model_capabilities(&config, "inkling".to_string()),
        );
        assert_eq!(
            note.as_deref(),
            Some("Web search and image generation are now off: inkling doesn't support them.")
        );
        assert!(!room.settings.web_search && !room.settings.image_generation);

        let mut room = both_on();
        let note = drop_unsupported_settings(
            &mut room,
            &model_capabilities(&config, "claude".to_string()),
        );
        assert_eq!(
            note.as_deref(),
            Some("Image generation is now off: claude doesn't support it.")
        );
        assert!(room.settings.web_search && !room.settings.image_generation);

        let mut room = both_on();
        let note =
            drop_unsupported_settings(&mut room, &model_capabilities(&config, "astra".to_string()));
        assert_eq!(note, None, "a model that supports both keeps both");
        assert!(room.settings.web_search && room.settings.image_generation);

        assert_eq!(
            with_settings_note("Model set.".to_string(), note),
            "Model set."
        );
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

        for image_model in ["gemini-3.1-flash-image", "gemini-nano-banana-2.1"] {
            config.models.get_mut("gemini").unwrap().model_name = image_model.to_string();
            let capabilities = room_capabilities(&config, &room, 42);
            assert!(capabilities.web_search, "{image_model}");
            assert!(capabilities.image_generation, "{image_model}");
        }
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

    #[tokio::test]
    async fn a_replacement_worker_starts_after_the_worker_it_replaces() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let handles = mock_handles(&telegram, &provider);
        // Stands in for a reaped worker still finishing its queued messages.
        let (old_done_tx, old_done_rx) = oneshot::channel::<()>();
        let old = tokio::spawn(std::future::pending::<()>());

        let dispatcher = spawn_chat_worker(42, handles.clone(), Some((old, Some(old_done_rx))));
        dispatcher.sender.try_send(text_message("/id")).unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            sent_texts(&telegram).is_empty(),
            "nothing runs while the old worker may still be replying"
        );

        drop(old_done_tx);
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while sent_texts(&telegram).is_empty() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "queued message never ran"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(sent_texts(&telegram), vec![chat_id_reply(42)]);

        // Shutdown aborts and joins the replaced worker too.
        tokio::time::timeout(
            Duration::from_secs(3),
            stop_chat_workers(BTreeMap::from([(42, dispatcher)]), &handles.workers),
        )
        .await
        .expect("both workers are joined");
    }

    #[tokio::test]
    async fn new_chat_releases_the_rooms_lock_before_its_reply() {
        let telegram = MockHttpServer::start_stalled();
        let provider = MockHttpServer::start(Vec::new());
        let handles = Arc::new(mock_handles(&telegram, &provider));
        let command_handles = Arc::clone(&handles);
        let command = tokio::spawn(async move {
            handle_command(
                42,
                CommandAction::ResetHistory,
                &command_handles,
                &AtomicBool::new(false),
            )
            .await
        });
        // Wait until the reply is in flight to the stalled server.
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        while telegram.requests().is_empty() {
            assert!(tokio::time::Instant::now() < deadline, "reply never sent");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let rooms = tokio::time::timeout(Duration::from_millis(500), handles.rooms.lock())
            .await
            .expect("other chats can use the rooms while /new waits on Telegram");
        drop(rooms);
        command.abort();
    }

    #[tokio::test]
    async fn pair_attempts_from_unknown_chats_do_not_wait_for_their_reply() {
        let telegram = MockHttpServer::start_stalled();
        let provider = MockHttpServer::start(Vec::new());
        let handles = mock_handles(&telegram, &provider);
        handles.access.lock().await.arm_room(-7, now_access_time());

        let reply = tokio::time::timeout(
            Duration::from_millis(500),
            start_pair_attempt(-7, "not-the-code", Some(9), &handles),
        )
        .await
        .expect("the poll loop moves on while Telegram stalls");

        // The attempt was counted before the poll loop moved on.
        assert!(matches!(
            check_pair_code(-7, "not-the-code", &handles).await,
            PairingAttempt::Rejected {
                attempts_remaining
            } if attempts_remaining == MAX_PAIRING_ATTEMPTS - 2
        ));
        reply.abort();
    }

    /// Runtime handles for chat 42 with a keyless compat model, where both
    /// Telegram and the provider are local mock servers.
    fn mock_handles(telegram: &MockHttpServer, provider: &MockHttpServer) -> RuntimeHandles {
        let mut model = test_model(WireFormat::Compat, &[]);
        model.model_name = "local-model".to_string();
        model.base_url = Some(format!("{}/v1", provider.base_url()));
        model.api_key_secret = None;
        let config = Config {
            default_model: "local".to_string(),
            models: BTreeMap::from([("local".to_string(), model)]),
            telegram: tellm_config::TelegramConfig {
                allowed_chat_ids: vec![42],
                owner_user_ids: Vec::new(),
                private_group_ids: Vec::new(),
            },
        };
        let access = AccessControl::new(AccessConfig::from_config(&config));
        let (persistence, _writer) = persistence::spawn_with(|_change| Ok(()), |_settings| Ok(()))
            .expect("spawn persistence writer");
        let (shutdown_tx, _shutdown_rx) = mpsc::channel(1);
        RuntimeHandles {
            telegram: Telegram::with_base_urls(
                "test-token",
                telegram.base_url(),
                telegram.base_url(),
            ),
            config: Arc::new(Mutex::new(config)),
            rooms: Arc::new(Mutex::new(RoomStates::default())),
            access: Arc::new(Mutex::new(access)),
            terminal_prompts: Arc::default(),
            bot_username: None,
            bot_user_id: None,
            privacy_mode: None,
            shutdown_tx,
            persistence,
            workers: Arc::new(StdMutex::new(BTreeMap::new())),
            queue_full_notices: Arc::new(StdMutex::new(BTreeSet::new())),
        }
    }

    fn telegram_mock() -> MockHttpServer {
        MockHttpServer::start(
            (0..12)
                .map(|_| MockResponse::json(200, serde_json::json!({ "ok": true, "result": true })))
                .collect(),
        )
    }

    fn completion(text: &str, prompt_tokens: u64, completion_tokens: u64) -> MockResponse {
        MockResponse::json(
            200,
            serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": text },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            }),
        )
    }

    fn compat_turn(question: &str, answer: &str) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({ "role": "user", "content": question }),
            serde_json::json!({ "role": "assistant", "content": answer }),
        ]
    }

    /// The request body of every message tellm sent to the chat.
    fn sent_message_bodies(telegram: &MockHttpServer) -> Vec<serde_json::Value> {
        telegram
            .requests()
            .iter()
            .filter(|request| request.path.ends_with("/sendRichMessage"))
            .map(|request| request.json_body())
            .collect()
    }

    /// The text of every message tellm sent to the chat.
    fn sent_texts(telegram: &MockHttpServer) -> Vec<String> {
        telegram
            .requests()
            .iter()
            .filter(|request| request.path.ends_with("/sendRichMessage"))
            .map(|request| {
                request.json_body()["rich_message"]["markdown"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string()
            })
            .collect()
    }

    #[tokio::test]
    async fn new_summary_sends_an_ordinary_turn_and_carries_the_reply_into_a_new_chat() {
        let telegram = telegram_mock();
        let provider =
            MockHttpServer::start(vec![completion("- Moving to Sofia in May.", 1_200, 80)]);
        let handles = mock_handles(&telegram, &provider);
        {
            let mut rooms = handles.rooms.lock().await;
            let room = rooms.get_or_default(42);
            room.settings.role = Some("Be brief.".to_string());
            room.append_turn(WireFormat::Compat, compat_turn("I move in May.", "Noted."));
            room.cost.record(&cost::ReplyCost::new(
                Some(tellm_core::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                }),
                None,
            ));
        }

        summarize_into_new_chat(
            42,
            &handles,
            &AtomicBool::new(false),
            summary::Trigger::Command,
        )
        .await
        .unwrap();

        // Same system prompt and history as an ordinary turn; only the user
        // message is the summary instruction.
        let body = provider.requests()[0].json_body();
        assert_eq!(body["model"], "local-model");
        assert_eq!(
            body["messages"],
            serde_json::json!([
                { "role": "system", "content": "Be brief." },
                { "role": "user", "content": "I move in May." },
                { "role": "assistant", "content": "Noted." },
                { "role": "user", "content": summary::SUMMARY_REQUEST }
            ])
        );

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(42).unwrap();
        assert!(room.history.is_empty());
        assert_eq!(room.summary.as_deref(), Some("- Moving to Sofia in May."));
        assert_eq!(room.cost, cost::ChatCost::default());
        drop(rooms);

        let sent = sent_texts(&telegram);
        assert_eq!(
            sent,
            vec![
                "Summary carried into a new chat:\n\n- Moving to Sofia in May.\n\n\
                 Tokens 1.2k in · 80 out · no price set · 1.3k of 150k tokens"
                    .to_string()
            ]
        );
    }

    #[tokio::test]
    async fn a_failed_summary_keeps_the_conversation() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![MockResponse::json(
            500,
            serde_json::json!({ "error": { "message": "boom" } }),
        )]);
        let handles = mock_handles(&telegram, &provider);
        let turn = compat_turn("I move in May.", "Noted.");
        handles
            .rooms
            .lock()
            .await
            .get_or_default(42)
            .append_turn(WireFormat::Compat, turn.clone());

        summarize_into_new_chat(
            42,
            &handles,
            &AtomicBool::new(false),
            summary::Trigger::Command,
        )
        .await
        .unwrap();

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(42).unwrap();
        assert_eq!(room.history, turn);
        assert_eq!(room.summary, None);
        drop(rooms);
        let sent = sent_texts(&telegram);
        assert_eq!(sent.len(), 1, "{sent:?}");
        assert!(sent[0].contains("continues unchanged"), "{sent:?}");
        assert!(sent[0].contains("boom"), "{sent:?}");
    }

    #[tokio::test]
    async fn new_summary_has_nothing_to_do_without_a_conversation_or_in_message_mode() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(Vec::new());
        let handles = mock_handles(&telegram, &provider);

        summarize_into_new_chat(
            42,
            &handles,
            &AtomicBool::new(false),
            summary::Trigger::Command,
        )
        .await
        .unwrap();
        {
            let mut rooms = handles.rooms.lock().await;
            let room = rooms.get_or_default(42);
            room.settings.mode = ChatMode::Message;
            room.append_turn(WireFormat::Compat, compat_turn("One-off?", "Yes."));
        }
        summarize_into_new_chat(
            42,
            &handles,
            &AtomicBool::new(false),
            summary::Trigger::Command,
        )
        .await
        .unwrap();

        assert!(provider.requests().is_empty());
        assert_eq!(
            sent_texts(&telegram),
            vec![
                summary::NOTHING_TO_SUMMARIZE.to_string(),
                summary::NOTHING_TO_SUMMARIZE_IN_MESSAGE_MODE.to_string()
            ]
        );
    }

    fn text_message(text: &str) -> IncomingMessage {
        IncomingMessage {
            message_id: 1,
            chat: tellm_telegram::Chat {
                id: 42,
                title: None,
            },
            from: None,
            date: 1000,
            text: Some(text.to_string()),
            caption: None,
            photo: None,
            document: None,
            media_group_id: None,
            album: Vec::new(),
            split_parts: 0,
            reply_to_message: None,
            new_chat_members: Vec::new(),
        }
    }

    fn summary_request_for(turns: &rooms::TurnHistory) -> ChatRequest {
        ChatRequest {
            model: "model".to_string(),
            system: None,
            history: turns.newest(turns.len()),
            input: Vec::new(),
            thinking: tellm_core::ThinkingLevel::Low,
            web_search: false,
            image_generation: false,
        }
    }

    fn five_turns() -> rooms::TurnHistory {
        let mut room = RoomState::new(crate::rooms::RoomSettings::default());
        for turn in 0..5 {
            room.append_turn(
                WireFormat::Compat,
                compat_turn(&format!("q{turn}"), &format!("a{turn}")),
            );
        }
        room.turn_history()
    }

    fn too_long() -> model_turn::TurnError {
        model_turn::TurnError {
            message: "too long".to_string(),
            context_window: true,
        }
    }

    fn summary_reply(text: &str) -> ChatResponse {
        ChatResponse {
            text: text.to_string(),
            images: Vec::new(),
            turn_items: Vec::new(),
            usage: None,
        }
    }

    #[tokio::test]
    async fn a_summary_over_the_context_window_is_retried_with_the_older_half_left_out() {
        let turns = five_turns();
        let mut sent = Vec::new();

        let (result, left_out) =
            send_summary_request(summary_request_for(&turns), &turns, |request| {
                sent.push(request.history.clone());
                let fits = request.history.len() <= 4;
                async move {
                    if fits {
                        Ok(summary_reply("Notes."))
                    } else {
                        Err(too_long())
                    }
                }
            })
            .await;

        assert_eq!(result.unwrap().text, "Notes.");
        assert_eq!(left_out, 3);
        assert_eq!(
            sent.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![10, 6, 4]
        );
        assert_eq!(sent[2][0]["content"], "q3", "the newest turns are kept");
    }

    #[tokio::test]
    async fn a_summary_retry_stops_at_one_turn_and_other_errors_are_not_retried() {
        let turns = five_turns();
        let mut sent = Vec::new();
        let (result, left_out) =
            send_summary_request(summary_request_for(&turns), &turns, |request| {
                sent.push(request.history.len());
                async { Err(too_long()) }
            })
            .await;
        assert!(result.unwrap_err().context_window);
        assert_eq!(left_out, 4);
        assert_eq!(sent, vec![10, 6, 4, 2]);

        let mut attempts = 0;
        let (result, left_out) =
            send_summary_request(summary_request_for(&turns), &turns, |_request| {
                attempts += 1;
                async { Err(model_turn::TurnError::from("boom".to_string())) }
            })
            .await;
        assert_eq!(result.unwrap_err().message, "boom");
        assert_eq!((attempts, left_out), (1, 0));

        let (result, _) =
            send_summary_request(summary_request_for(&turns), &turns, |_request| async {
                Ok(summary_reply(" \n"))
            })
            .await;
        assert_eq!(result.unwrap_err().message, summary::EMPTY_SUMMARY_ERROR);
    }

    #[tokio::test]
    async fn a_reply_that_reaches_the_token_budget_is_followed_by_a_summary() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![
            completion("Sure.", 149_000, 2_000),
            completion("- Notes.", 151_500, 300),
        ]);
        let handles = mock_handles(&telegram, &provider);
        handles
            .rooms
            .lock()
            .await
            .get_or_default(42)
            .append_turn(WireFormat::Compat, compat_turn("I move in May.", "Noted."));

        handle_model_message(
            42,
            text_message("And then?"),
            &handles,
            &AtomicBool::new(false),
        )
        .await
        .unwrap();

        let requests = provider.requests();
        assert_eq!(requests.len(), 2);
        let messages = requests[1].json_body()["messages"].clone();
        assert_eq!(messages.as_array().unwrap().len(), 5);
        assert_eq!(messages[3]["content"], "Sure.");
        assert_eq!(messages[4]["content"], summary::SUMMARY_REQUEST);

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(42).unwrap();
        assert!(room.history.is_empty());
        assert_eq!(room.summary.as_deref(), Some("- Notes."));
        assert_eq!(room.context_tokens, None);
        drop(rooms);

        assert_eq!(
            sent_texts(&telegram),
            vec![
                "Sure.\n\nTokens 149k in · 2k out · no price set · 151k of 150k tokens".to_string(),
                summary::Trigger::TokenBudget.notice().unwrap(),
                "Summary carried into a new chat:\n\n- Notes.\n\n\
                 Tokens 151.5k in · 300 out · no price set · 151.8k of 150k tokens"
                    .to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn automatic_summaries_stop_after_failing_twice() {
        let telegram = telegram_mock();
        let failure =
            || MockResponse::json(500, serde_json::json!({ "error": { "message": "boom" } }));
        let provider = MockHttpServer::start(vec![
            completion("Sure.", 149_000, 2_000),
            failure(),
            completion("Again.", 152_000, 2_000),
            failure(),
        ]);
        let handles = mock_handles(&telegram, &provider);

        for text in ["And then?", "And now?"] {
            handle_model_message(42, text_message(text), &handles, &AtomicBool::new(false))
                .await
                .unwrap();
        }

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(42).unwrap();
        assert_eq!(room.failed_summaries, 2);
        assert_eq!(room.context_tokens, Some(154_000));
        assert_eq!(room.summary_due(), None, "no third attempt on its own");
        drop(rooms);
        let sent = sent_texts(&telegram);
        assert!(sent.last().unwrap().contains("won't try again"), "{sent:?}");

        handles
            .rooms
            .lock()
            .await
            .get_mut(42)
            .unwrap()
            .reset_history();
        assert_eq!(
            handles.rooms.lock().await.get(42).unwrap().failed_summaries,
            0
        );
    }

    #[tokio::test]
    async fn a_failed_automatic_summary_keeps_the_chat_and_tries_again_later() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![
            completion("Sure.", 149_000, 2_000),
            MockResponse::json(500, serde_json::json!({ "error": { "message": "boom" } })),
        ]);
        let handles = mock_handles(&telegram, &provider);

        handle_model_message(
            42,
            text_message("And then?"),
            &handles,
            &AtomicBool::new(false),
        )
        .await
        .unwrap();

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(42).unwrap();
        assert_eq!(room.history.len(), 2);
        assert_eq!(room.summary, None);
        assert_eq!(room.context_tokens, Some(151_000));
        assert_eq!(room.summary_due(), Some(summary::Trigger::TokenBudget));
        drop(rooms);

        let sent = sent_texts(&telegram);
        assert_eq!(sent.len(), 3, "{sent:?}");
        assert!(
            sent[2].contains("try again after the next reply"),
            "{sent:?}"
        );
        assert!(sent[2].contains("boom"), "{sent:?}");
    }

    #[tokio::test]
    async fn history_over_the_byte_cap_is_pruned_and_announced_when_summarizing_fails() {
        let telegram = telegram_mock();
        let provider = MockHttpServer::start(vec![
            completion("Sure.", 1_000, 10),
            MockResponse::json(500, serde_json::json!({ "error": { "message": "boom" } })),
        ]);
        let handles = mock_handles(&telegram, &provider);
        handles.rooms.lock().await.get_or_default(42).append_turn(
            WireFormat::Compat,
            compat_turn(&"x".repeat(rooms::MAX_HISTORY_BYTES), "Long."),
        );

        handle_model_message(
            42,
            text_message("And then?"),
            &handles,
            &AtomicBool::new(false),
        )
        .await
        .unwrap();

        let rooms = handles.rooms.lock().await;
        let room = rooms.get(42).unwrap();
        assert_eq!(
            room.history,
            compat_turn("And then?", "Sure."),
            "only the turn that fits is kept"
        );
        drop(rooms);

        let sent = sent_texts(&telegram);
        assert_eq!(sent.len(), 4, "{sent:?}");
        assert_eq!(sent[1], summary::Trigger::HistoryBytes.notice().unwrap());
        assert!(sent[2].contains("Couldn't summarize"), "{sent:?}");
        assert_eq!(sent[3], summary::pruned_notice(1));
    }

    #[tokio::test]
    async fn history_over_the_byte_cap_is_pruned_even_when_telegram_fails() {
        let telegram = MockHttpServer::start(
            (0..12)
                .map(|_| {
                    MockResponse::json(
                        500,
                        serde_json::json!({ "ok": false, "description": "boom" }),
                    )
                })
                .collect(),
        );
        let provider = MockHttpServer::start(vec![
            completion("Sure.", 1_000, 10),
            MockResponse::json(500, serde_json::json!({ "error": { "message": "boom" } })),
        ]);
        let handles = mock_handles(&telegram, &provider);
        handles.rooms.lock().await.get_or_default(42).append_turn(
            WireFormat::Compat,
            compat_turn(&"x".repeat(rooms::MAX_HISTORY_BYTES), "Long."),
        );

        let result = handle_model_message(
            42,
            text_message("And then?"),
            &handles,
            &AtomicBool::new(false),
        )
        .await;

        assert!(result.is_err(), "the undelivered reply is still reported");
        let rooms = handles.rooms.lock().await;
        assert_eq!(
            rooms.get(42).unwrap().history,
            compat_turn("And then?", "Sure."),
            "the prune ran although no notice could be sent"
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
            max_effort: None,
            off_effort: None,
            max_output_tokens: None,
            price: None,
        }
    }

    #[test]
    fn help_text_is_markdown_bullets_without_raw_angle_placeholders() {
        assert!(HELP_TEXT.lines().all(|line| line.starts_with("- /")));
        assert!(HELP_TEXT.contains("- /new [summary] - start a new chat"));
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
