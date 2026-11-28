use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tellm_core::{ThinkingLevel, Usage};

use tellm_config::{ConfigError, WireFormat};

use crate::cost::ChatCost;
use crate::summary;

const ROOMS_FILE: &str = "rooms.toml";
/// A memory safety net for the opaque history, not its usual bound. History
/// is append-only within a conversation, and a chat is summarized into a new
/// one when its context reaches [`summary::CONTEXT_SUMMARY_TOKENS`] or its
/// history passes this size. Only when summarizing fails does
/// [`RoomState::prune_to_byte_cap`] drop the oldest whole turns.
pub const MAX_HISTORY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChatMode {
    #[default]
    Chat,
    Message,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoomSettings {
    #[serde(default)]
    pub model_key: Option<String>,
    #[serde(default)]
    pub mode: ChatMode,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingLevel>,
    #[serde(default)]
    pub web_search: bool,
    #[serde(default)]
    pub image_generation: bool,
    /// Whether each reply ends with its cost line. `None` follows the
    /// default (see [`RoomSettings::shows_cost`]), so the file only records
    /// rooms that chose.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_cost: Option<bool>,
    /// Whether new conversations get the owner's memory. `None` follows the
    /// default: on in an owner's direct chat, off elsewhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<bool>,
}

impl Default for RoomSettings {
    fn default() -> Self {
        Self {
            model_key: None,
            mode: ChatMode::Chat,
            role: None,
            thinking: None,
            web_search: false,
            image_generation: false,
            show_cost: None,
            memory: None,
        }
    }
}

impl RoomSettings {
    /// Whether replies end with their cost line: the room's choice, or by
    /// default on, except in a shared group, where everyone would read what
    /// each answer costs the owner.
    pub fn shows_cost(&self, shared_group: bool) -> bool {
        self.show_cost.unwrap_or(!shared_group)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomState {
    pub settings: RoomSettings,
    pub wire_format: Option<WireFormat>,
    pub history: Vec<Value>,
    /// Running cost of the current conversation; in memory only, like the
    /// conversation itself.
    pub cost: ChatCost,
    /// The memory block the current conversation started with. It stays
    /// until the conversation ends, so the system prompt never changes under
    /// replayed provider state and prompt caching keeps working.
    pub memory: Option<String>,
    /// The summary of an earlier conversation that the current one carries
    /// (`/new summary`). It is set when that conversation ends and stays
    /// until this one does, so it keeps the system prompt byte-stable the
    /// same way `memory` does.
    pub summary: Option<String>,
    /// The conversation's size in tokens as of its latest committed reply:
    /// everything that reply's request sent, plus what it wrote, as the
    /// provider reported it. `None` when the provider reported no usage, in
    /// message mode, and before the first reply.
    pub context_tokens: Option<u64>,
    /// Automatic summaries of the current conversation that failed. At
    /// [`summary::MAX_AUTOMATIC_SUMMARY_FAILURES`] the conversation is no
    /// longer summarized on its own, since each attempt sends all of it
    /// again; `/new summary` still works.
    pub failed_summaries: u8,
    history_turns: VecDeque<HistoryTurn>,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HistoryTurn {
    item_count: usize,
    approx_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HistoryReset {
    None,
    WireFormatChanged {
        previous: Option<WireFormat>,
        new: WireFormat,
    },
}

pub struct TurnStart {
    pub reset: HistoryReset,
    pub rollback: Option<FailedTurnRollback>,
}

pub struct FailedTurnRollback {
    wire_format: Option<WireFormat>,
    history: Vec<Value>,
    history_turns: VecDeque<HistoryTurn>,
}

impl RoomState {
    pub fn new(settings: RoomSettings) -> Self {
        Self {
            settings,
            wire_format: None,
            history: Vec::new(),
            cost: ChatCost::default(),
            memory: None,
            summary: None,
            context_tokens: None,
            failed_summaries: 0,
            history_turns: VecDeque::new(),
            generation: 0,
        }
    }

    /// Changes whenever pending provider work must no longer affect this room.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn begin_turn(&mut self, wire_format: WireFormat) -> TurnStart {
        if self.wire_format != Some(wire_format) {
            let previous = self.wire_format;
            let rollback = FailedTurnRollback {
                wire_format: previous,
                history: std::mem::take(&mut self.history),
                history_turns: std::mem::take(&mut self.history_turns),
            };
            self.wire_format = Some(wire_format);
            self.generation = self.generation.wrapping_add(1);
            return TurnStart {
                reset: HistoryReset::WireFormatChanged {
                    previous,
                    new: wire_format,
                },
                rollback: Some(rollback),
            };
        }

        TurnStart {
            reset: HistoryReset::None,
            rollback: None,
        }
    }

    pub fn append_turn(&mut self, wire_format: WireFormat, turn_items: Vec<Value>) {
        if self.wire_format != Some(wire_format) {
            self.wire_format = Some(wire_format);
            self.invalidate_history();
        }

        match self.settings.mode {
            ChatMode::Chat => self.push_turn(turn_items),
            // Message mode stays stateless at request time, but keeps the
            // latest exchange so `/mode chat` can continue from it.
            ChatMode::Message => {
                self.clear_history_storage();
                self.push_turn(turn_items);
            }
        }
    }

    /// Starts a new conversation: the history, its running cost and size,
    /// and the memory and summary it started with.
    pub fn reset_history(&mut self) {
        self.invalidate_history();
        self.cost = ChatCost::default();
        self.memory = None;
        self.summary = None;
        self.context_tokens = None;
        self.failed_summaries = 0;
    }

    /// Records the conversation's size after a committed reply, from the
    /// usage the provider reported for it. A message-mode request carries no
    /// conversation, so it has no size to record.
    pub fn record_context(&mut self, usage: Option<&Usage>) {
        self.context_tokens = match self.settings.mode {
            ChatMode::Chat => usage.map(Usage::context_size),
            ChatMode::Message => None,
        };
    }

    /// Whether the conversation has grown enough to be summarized into a new
    /// one, and why: its context reached the token budget, or its history
    /// passed the byte cap. Only a chat-mode conversation is summarized, and
    /// only until its automatic summaries have failed too often.
    pub fn summary_due(&self) -> Option<summary::Trigger> {
        if self.settings.mode != ChatMode::Chat
            || self.history.is_empty()
            || self.failed_summaries >= summary::MAX_AUTOMATIC_SUMMARY_FAILURES
        {
            return None;
        }
        if self
            .context_tokens
            .is_some_and(|tokens| tokens >= summary::CONTEXT_SUMMARY_TOKENS)
        {
            Some(summary::Trigger::TokenBudget)
        } else if self.history_bytes() > MAX_HISTORY_BYTES {
            Some(summary::Trigger::HistoryBytes)
        } else {
            None
        }
    }

    /// The last-resort bound, for when summarizing failed or isn't possible:
    /// drops the oldest whole turns until the history fits in
    /// [`MAX_HISTORY_BYTES`] and returns how many turns are gone, so the
    /// caller can say so in the chat. The kept turns then follow a different
    /// history, so their wire-format crate decides what is still replayable.
    pub fn prune_to_byte_cap(&mut self) -> usize {
        let turns_before = self.history_turns.len();
        let mut pruned = false;
        while self.history_bytes() > MAX_HISTORY_BYTES {
            let Some(oldest) = self.history_turns.pop_front() else {
                break;
            };
            self.history.drain(..oldest.item_count);
            pruned = true;
        }
        if pruned {
            self.history_prefix_changed();
        }
        turns_before - self.history_turns.len()
    }

    /// A copy of the conversation's turns, for a request that may have to
    /// leave the oldest ones out.
    pub fn turn_history(&self) -> TurnHistory {
        let mut items = self.history.iter();
        TurnHistory {
            wire_format: self.wire_format,
            turns: self
                .history_turns
                .iter()
                .map(|turn| items.by_ref().take(turn.item_count).cloned().collect())
                .collect(),
        }
    }

    /// Starts a new conversation that carries `summary` of the one ending
    /// now. Like [`Self::reset_history`], it drops the history, its cost and
    /// its memory.
    pub fn continue_from_summary(&mut self, summary: String) {
        self.reset_history();
        self.summary = Some(summary);
    }

    /// Apply a `/websearch` change. The tool list precedes the whole
    /// conversation on the wire, so flipping it is a history edit for
    /// provider state that is bound to what came before it; the conversation
    /// itself is kept. Commands and model turns share the per-chat queue, so
    /// no request built from the old history is in flight here.
    pub fn set_web_search(&mut self, enabled: bool) {
        if self.settings.web_search == enabled {
            return;
        }
        self.settings.web_search = enabled;
        self.history_prefix_changed();
    }

    /// Commit a provider turn only if no reset or model transition happened
    /// while the request was in flight.
    pub fn append_turn_if_generation(
        &mut self,
        expected_generation: u64,
        wire_format: WireFormat,
        turn_items: Vec<Value>,
    ) -> bool {
        if self.generation != expected_generation {
            return false;
        }
        self.append_turn(wire_format, turn_items);
        true
    }

    /// Restore a speculative wire-format transition made by `begin_turn`
    /// without overwriting a newer reset or any settings changed while the
    /// provider call was in flight. Same-format failures have no rollback
    /// token and leave the room, including its generation, untouched.
    pub fn restore_failed_turn(
        &mut self,
        expected_generation: u64,
        rollback: Option<FailedTurnRollback>,
    ) -> bool {
        if self.generation != expected_generation {
            return false;
        }
        let Some(rollback) = rollback else {
            return true;
        };
        self.wire_format = rollback.wire_format;
        self.history = rollback.history;
        self.history_turns = rollback.history_turns;
        self.generation = self.generation.wrapping_add(1);
        true
    }

    fn invalidate_history(&mut self) {
        self.clear_history_storage();
        self.generation = self.generation.wrapping_add(1);
    }

    fn clear_history_storage(&mut self) {
        self.history.clear();
        self.history_turns.clear();
    }

    /// History is append-only: a conversation that grows too large is
    /// summarized into a new one instead of losing its beginning, which would
    /// change the prefix every provider caches.
    fn push_turn(&mut self, turn_items: Vec<Value>) {
        self.store_turn(turn_items);
    }

    fn store_turn(&mut self, turn_items: Vec<Value>) {
        let turn = HistoryTurn {
            item_count: turn_items.len(),
            approx_bytes: turn_items.iter().fold(0_usize, |total, value| {
                total.saturating_add(approx_json_bytes(value))
            }),
        };
        self.history.extend(turn_items);
        self.history_turns.push_back(turn);
    }

    /// tellm changed what precedes the kept turns: it pruned earlier turns or
    /// changed the tool list. Opaque items are interpreted only by the
    /// wire-format crate that produced them, so that crate decides what is
    /// still replayable. A turn it returns empty is dropped whole.
    fn history_prefix_changed(&mut self) {
        let Some(replayable) = replayable_after_prefix_change(self.wire_format) else {
            return;
        };

        let mut items = std::mem::take(&mut self.history).into_iter();
        for turn in std::mem::take(&mut self.history_turns) {
            let turn_items = replayable(items.by_ref().take(turn.item_count).collect());
            if !turn_items.is_empty() {
                self.store_turn(turn_items);
            }
        }
    }

    pub fn history_bytes(&self) -> usize {
        self.history_turns.iter().fold(0_usize, |total, turn| {
            total.saturating_add(turn.approx_bytes)
        })
    }
}

/// How a wire format's crate rewrites a kept turn once what precedes it
/// changed, or `None` when its turns replay unchanged. Anthropic removes
/// thinking blocks; the other formats keep their turns as they are.
fn replayable_after_prefix_change(
    wire_format: Option<WireFormat>,
) -> Option<fn(Vec<Value>) -> Vec<Value>> {
    match wire_format {
        Some(WireFormat::Anthropic) => Some(tellm_anthropic::without_thinking_blocks),
        Some(WireFormat::Responses | WireFormat::Compat | WireFormat::Gemini) | None => None,
    }
}

/// A copy of a conversation's turns, each a complete provider-native
/// request/response exchange.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnHistory {
    wire_format: Option<WireFormat>,
    turns: Vec<Vec<Value>>,
}

impl TurnHistory {
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// The newest `keep` turns, as a request sends them. All of them is the
    /// stored history unchanged. Leaving older turns out changes what
    /// precedes the kept ones, so each kept turn then goes through its
    /// wire-format crate the way pruning does, and a turn that comes back
    /// empty is left out too.
    pub fn newest(&self, keep: usize) -> Vec<Value> {
        let skip = self.turns.len().saturating_sub(keep);
        let kept = self.turns[skip..].iter().cloned();
        match replayable_after_prefix_change(self.wire_format).filter(|_| skip > 0) {
            Some(replayable) => kept.flat_map(replayable).collect(),
            None => kept.flatten().collect(),
        }
    }
}

fn approx_json_bytes(value: &Value) -> usize {
    match value {
        Value::Null => 4,
        Value::Bool(value) => usize::from(*value) + 4,
        Value::Number(value) => value.to_string().len(),
        Value::String(value) => value.len().saturating_add(2),
        Value::Array(values) => values.iter().fold(2_usize, |total, value| {
            total.saturating_add(approx_json_bytes(value).saturating_add(1))
        }),
        Value::Object(values) => values.iter().fold(2_usize, |total, (key, value)| {
            total
                .saturating_add(key.len().saturating_add(3))
                .saturating_add(approx_json_bytes(value).saturating_add(1))
        }),
    }
}

#[derive(Debug, Clone, Default)]
pub struct RoomStates {
    rooms: BTreeMap<i64, RoomState>,
}

impl RoomStates {
    pub fn from_settings(settings: BTreeMap<i64, RoomSettings>) -> Self {
        Self {
            rooms: settings
                .into_iter()
                .map(|(chat_id, settings)| (chat_id, RoomState::new(settings)))
                .collect(),
        }
    }

    pub fn get_or_default(&mut self, chat_id: i64) -> &mut RoomState {
        self.rooms
            .entry(chat_id)
            .or_insert_with(|| RoomState::new(RoomSettings::default()))
    }

    pub fn get(&self, chat_id: i64) -> Option<&RoomState> {
        self.rooms.get(&chat_id)
    }

    pub fn get_mut(&mut self, chat_id: i64) -> Option<&mut RoomState> {
        self.rooms.get_mut(&chat_id)
    }

    pub fn settings(&self) -> BTreeMap<i64, RoomSettings> {
        self.rooms
            .iter()
            .map(|(chat_id, state)| (*chat_id, state.settings.clone()))
            .collect()
    }

    pub fn reset_all_history(&mut self) {
        for room in self.rooms.values_mut() {
            room.reset_history();
        }
    }

    pub fn remove(&mut self, chat_id: i64) -> Option<RoomState> {
        self.rooms.remove(&chat_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct RoomsToml {
    #[serde(default)]
    rooms: BTreeMap<String, RoomSettings>,
}

pub fn rooms_path() -> Result<PathBuf, ConfigError> {
    Ok(config_dir()?.join(ROOMS_FILE))
}

fn config_dir() -> Result<PathBuf, ConfigError> {
    #[cfg(test)]
    if let Some(path) = test_config_dir_override() {
        return Ok(path);
    }

    tellm_config::config_dir()
}

#[cfg(test)]
fn test_config_dir_override() -> Option<PathBuf> {
    TEST_CONFIG_DIR_OVERRIDE
        .get_or_init(Default::default)
        .lock()
        .expect("test rooms config dir override lock poisoned")
        .clone()
}

#[cfg(test)]
fn set_test_config_dir_override(path: Option<PathBuf>) {
    *TEST_CONFIG_DIR_OVERRIDE
        .get_or_init(Default::default)
        .lock()
        .expect("test rooms config dir override lock poisoned") = path;
}

#[cfg(test)]
static TEST_CONFIG_DIR_OVERRIDE: std::sync::OnceLock<std::sync::Mutex<Option<PathBuf>>> =
    std::sync::OnceLock::new();

pub fn load_settings() -> Result<BTreeMap<i64, RoomSettings>, ConfigError> {
    let path = rooms_path()?;
    if !path.exists() {
        return Ok(BTreeMap::new());
    }

    let text = fs::read_to_string(path)?;
    let file: RoomsToml = toml::from_str(&text)?;
    let mut rooms = BTreeMap::new();
    let mut problems = Vec::new();
    for (chat_id, settings) in file.rooms {
        match chat_id.parse::<i64>() {
            Ok(id) => {
                rooms.insert(id, settings);
            }
            Err(_) => problems.push(format!("rooms.toml contains invalid chat_id \"{chat_id}\"")),
        }
    }

    if problems.is_empty() {
        Ok(rooms)
    } else {
        Err(ConfigError::Invalid(problems))
    }
}

pub fn save_settings(settings: &BTreeMap<i64, RoomSettings>) -> Result<(), ConfigError> {
    let path = rooms_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = RoomsToml {
        rooms: settings
            .iter()
            .map(|(chat_id, settings)| (chat_id.to_string(), settings.clone()))
            .collect(),
    };
    tellm_config::write_atomic(&path, &toml::to_string_pretty(&file)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn room_settings_roundtrip_through_rooms_toml() {
        with_temp_config_dir(|dir| {
            let mut settings = BTreeMap::new();
            settings.insert(
                42,
                RoomSettings {
                    model_key: Some("claude".to_string()),
                    mode: ChatMode::Chat,
                    role: Some("Be precise.".to_string()),
                    thinking: Some(ThinkingLevel::High),
                    web_search: true,
                    image_generation: true,
                    show_cost: None,
                    memory: Some(true),
                },
            );
            settings.insert(
                -7,
                RoomSettings {
                    model_key: Some("ollama".to_string()),
                    mode: ChatMode::Message,
                    role: None,
                    thinking: Some(ThinkingLevel::Off),
                    web_search: false,
                    image_generation: false,
                    show_cost: Some(false),
                    memory: None,
                },
            );

            save_settings(&settings).unwrap();
            let loaded = load_settings().unwrap();

            assert_eq!(loaded, settings);
            let text = std::fs::read_to_string(dir.join("rooms.toml")).unwrap();
            assert!(text.contains("[rooms.42]"));
            assert!(text.contains("[rooms.-7]"));
            // Only the room that hides its cost line records the setting.
            assert_eq!(text.matches("show_cost").count(), 1, "{text}");
            assert!(text.contains("show_cost = false"), "{text}");
        });
    }

    #[test]
    fn missing_rooms_file_loads_empty_settings() {
        with_temp_config_dir(|_dir| {
            assert!(load_settings().unwrap().is_empty());
        });
    }

    #[test]
    fn invalid_room_chat_id_is_reported() {
        with_temp_config_dir(|dir| {
            std::fs::write(
                dir.join("rooms.toml"),
                "[rooms.not-a-chat]\nmodel_key = \"claude\"\n",
            )
            .unwrap();

            let error = load_settings().unwrap_err();

            assert!(
                matches!(error, ConfigError::Invalid(problems) if problems[0].contains("not-a-chat"))
            );
        });
    }

    #[test]
    fn chat_mode_keeps_opaque_history_for_same_wire_format() {
        let mut room = RoomState::new(RoomSettings::default());

        assert_eq!(
            room.begin_turn(WireFormat::Anthropic).reset,
            HistoryReset::WireFormatChanged {
                previous: None,
                new: WireFormat::Anthropic,
            }
        );
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({ "role": "user", "content": "hi" })],
        );

        assert_eq!(
            room.begin_turn(WireFormat::Anthropic).reset,
            HistoryReset::None
        );
        assert_eq!(room.history.len(), 1);
    }

    #[test]
    fn wire_format_switch_resets_opaque_history() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({ "role": "assistant", "content": [] })],
        );

        let reset = room.begin_turn(WireFormat::Responses).reset;

        assert_eq!(
            reset,
            HistoryReset::WireFormatChanged {
                previous: Some(WireFormat::Anthropic),
                new: WireFormat::Responses,
            }
        );
        assert!(room.history.is_empty());
        assert_eq!(room.wire_format, Some(WireFormat::Responses));
    }

    #[test]
    fn message_mode_retains_latest_turn_without_chaining() {
        let mut room = RoomState::new(RoomSettings {
            mode: ChatMode::Message,
            ..RoomSettings::default()
        });

        assert!(matches!(
            room.begin_turn(WireFormat::Compat).reset,
            HistoryReset::WireFormatChanged { .. }
        ));
        room.append_turn(
            WireFormat::Compat,
            vec![serde_json::json!({ "role": "assistant", "content": "one-off" })],
        );

        assert_eq!(
            room.history,
            vec![serde_json::json!({ "role": "assistant", "content": "one-off" })]
        );
        assert_eq!(
            room.begin_turn(WireFormat::Compat).reset,
            HistoryReset::None
        );
        room.append_turn(
            WireFormat::Compat,
            vec![serde_json::json!({ "role": "assistant", "content": "latest" })],
        );
        assert_eq!(
            room.history,
            vec![serde_json::json!({ "role": "assistant", "content": "latest" })]
        );
    }

    #[test]
    fn history_is_append_only_however_many_turns_it_holds() {
        let mut room = RoomState::new(RoomSettings::default());
        for turn in 0..100 {
            room.append_turn(
                WireFormat::Responses,
                vec![
                    serde_json::json!({ "turn": turn, "role": "user" }),
                    serde_json::json!({ "turn": turn, "role": "assistant" }),
                ],
            );
        }

        assert_eq!(room.history.len(), 200);
        assert_eq!(room.history[0]["turn"], 0);
        assert_eq!(room.history.last().unwrap()["turn"], 99);
        assert_eq!(room.prune_to_byte_cap(), 0);
    }

    #[test]
    fn the_byte_cap_backstop_never_keeps_half_a_provider_turn() {
        let mut room = RoomState::new(RoomSettings::default());
        let payload = "x".repeat(MAX_HISTORY_BYTES / 2);
        room.append_turn(
            WireFormat::Anthropic,
            vec![
                serde_json::json!({ "turn": 1, "content": payload }),
                serde_json::json!({ "turn": 1, "marker": "complete" }),
            ],
        );
        room.append_turn(
            WireFormat::Anthropic,
            vec![
                serde_json::json!({ "turn": 2, "content": "y".repeat(MAX_HISTORY_BYTES / 2) }),
                serde_json::json!({ "turn": 2, "marker": "complete" }),
            ],
        );
        // Appending never drops anything; the runtime summarizes instead.
        assert_eq!(room.history.len(), 4);
        assert!(room.history_bytes() > MAX_HISTORY_BYTES);

        assert_eq!(room.prune_to_byte_cap(), 1);

        assert_eq!(room.history.len(), 2);
        assert!(room.history.iter().all(|item| item["turn"] == 2));
        assert!(room.history_bytes() <= MAX_HISTORY_BYTES);

        // A single turn over the cap goes too: the bound is on memory.
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({ "turn": 3, "content": "z".repeat(MAX_HISTORY_BYTES) })],
        );
        assert_eq!(room.prune_to_byte_cap(), 2);
        assert!(room.history.is_empty());
    }

    #[test]
    fn anthropic_thinking_is_replayed_verbatim_until_the_backstop_prunes() {
        let mut room = RoomState::new(RoomSettings::default());
        let mut big_first_turn = anthropic_turn(0);
        with_text(&mut big_first_turn, "z".repeat(MAX_HISTORY_BYTES / 2));
        room.append_turn(WireFormat::Anthropic, big_first_turn);
        for turn in 1..40 {
            room.append_turn(WireFormat::Anthropic, anthropic_turn(turn));
        }
        assert_eq!(thinking_blocks(&room.history), 40);
        assert_eq!(room.history[2..4], anthropic_turn(1)[..]);

        let mut big_last_turn = anthropic_turn(40);
        with_text(&mut big_last_turn, "z".repeat(MAX_HISTORY_BYTES / 2));
        room.append_turn(WireFormat::Anthropic, big_last_turn);
        assert_eq!(thinking_blocks(&room.history), 41);

        assert_eq!(room.prune_to_byte_cap(), 1);

        // The kept turns now follow a different history than the one their
        // thinking was produced after, so none of it may be replayed.
        assert_eq!(room.history.len(), 80);
        assert_eq!(room.history[0]["content"][0]["text"], "question 1");
        assert_eq!(thinking_blocks(&room.history), 0);
        assert_eq!(
            room.history[1]["content"],
            serde_json::json!([{ "type": "text", "text": "answer 1" }])
        );
        assert_eq!(
            room.history_bytes(),
            room.history.iter().map(approx_json_bytes).sum::<usize>()
        );
    }

    #[test]
    fn a_summary_request_leaving_turns_out_strips_anthropic_thinking_from_the_rest() {
        let mut room = RoomState::new(RoomSettings::default());
        for turn in 0..3 {
            room.append_turn(WireFormat::Anthropic, anthropic_turn(turn));
        }
        let turns = room.turn_history();
        assert_eq!(turns.len(), 3);

        // All turns: the stored history, verbatim.
        assert_eq!(turns.newest(3), room.history);
        assert_eq!(turns.newest(10), room.history);

        let newest = turns.newest(2);
        assert_eq!(newest.len(), 4);
        assert_eq!(newest[0]["content"][0]["text"], "question 1");
        assert_eq!(thinking_blocks(&newest), 0);
        // The room itself is untouched.
        assert_eq!(thinking_blocks(&room.history), 3);
    }

    #[test]
    fn other_wire_formats_leave_turns_out_without_rewriting_the_rest() {
        let mut room = RoomState::new(RoomSettings::default());
        let reasoning = |turn: usize| {
            vec![
                serde_json::json!({ "role": "user", "content": format!("q{turn}") }),
                serde_json::json!({ "type": "reasoning", "encrypted_content": format!("r{turn}") }),
            ]
        };
        for turn in 0..3 {
            room.append_turn(WireFormat::Responses, reasoning(turn));
        }

        assert_eq!(room.turn_history().newest(1), reasoning(2));
    }

    #[test]
    fn the_context_size_is_the_last_replys_input_and_output() {
        let mut room = RoomState::new(RoomSettings::default());
        let usage = Usage {
            input_tokens: 1_000,
            cached_input_tokens: 120_000,
            cache_write_tokens: 4_000,
            output_tokens: 2_500,
            reasoning_tokens: 2_000,
            ..Usage::default()
        };

        room.record_context(Some(&usage));
        assert_eq!(room.context_tokens, Some(127_500));

        room.record_context(None);
        assert_eq!(
            room.context_tokens, None,
            "unknown usage means unknown size"
        );

        room.record_context(Some(&usage));
        room.reset_history();
        assert_eq!(room.context_tokens, None);

        room.settings.mode = ChatMode::Message;
        room.record_context(Some(&usage));
        assert_eq!(room.context_tokens, None);
    }

    #[test]
    fn a_chat_is_due_for_a_summary_at_the_token_budget_or_the_byte_cap() {
        let mut room = RoomState::new(RoomSettings::default());
        room.context_tokens = Some(summary::CONTEXT_SUMMARY_TOKENS);
        assert_eq!(room.summary_due(), None, "nothing to summarize yet");

        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);
        room.context_tokens = Some(summary::CONTEXT_SUMMARY_TOKENS - 1);
        assert_eq!(room.summary_due(), None);
        room.context_tokens = None;
        assert_eq!(room.summary_due(), None);
        room.context_tokens = Some(summary::CONTEXT_SUMMARY_TOKENS);
        assert_eq!(room.summary_due(), Some(summary::Trigger::TokenBudget));

        room.context_tokens = None;
        room.append_turn(
            WireFormat::Responses,
            vec![serde_json::json!({ "content": "x".repeat(MAX_HISTORY_BYTES) })],
        );
        assert_eq!(room.summary_due(), Some(summary::Trigger::HistoryBytes));

        room.settings.mode = ChatMode::Message;
        assert_eq!(room.summary_due(), None, "message mode is never summarized");
    }

    #[test]
    fn web_search_toggle_drops_anthropic_thinking_and_keeps_the_conversation() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(WireFormat::Anthropic, anthropic_turn(0));
        room.append_turn(WireFormat::Anthropic, anthropic_turn(1));
        let generation = room.generation();
        let bytes_with_thinking = room.history_bytes();

        room.set_web_search(true);

        assert!(room.settings.web_search);
        assert_eq!(room.history.len(), 4);
        assert_eq!(thinking_blocks(&room.history), 0);
        assert_eq!(room.history[0]["content"][0]["text"], "question 0");
        assert_eq!(room.history[3]["content"][0]["text"], "answer 1");
        assert!(room.history_bytes() < bytes_with_thinking);
        assert_eq!(room.generation(), generation);
    }

    #[test]
    fn unchanged_web_search_setting_keeps_anthropic_history_verbatim() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(WireFormat::Anthropic, anthropic_turn(0));

        room.set_web_search(false);

        assert_eq!(room.history, anthropic_turn(0));
    }

    #[test]
    fn web_search_toggle_keeps_other_wire_formats_verbatim() {
        let mut room = RoomState::new(RoomSettings::default());
        let turn = vec![
            serde_json::json!({ "type": "reasoning", "encrypted_content": "opaque" }),
            serde_json::json!({
                "role": "assistant",
                "content": [{ "type": "thinking", "thinking": "not Anthropic's" }]
            }),
        ];
        room.append_turn(WireFormat::Responses, turn.clone());

        room.set_web_search(true);

        assert_eq!(room.history, turn);
    }

    #[test]
    fn thinking_only_turn_is_dropped_whole_and_pruning_stays_turn_aligned() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(WireFormat::Anthropic, anthropic_turn(0));
        room.append_turn(
            WireFormat::Anthropic,
            vec![
                serde_json::json!({
                    "role": "user",
                    "content": [{ "type": "text", "text": "cut off" }]
                }),
                serde_json::json!({
                    "role": "assistant",
                    "content": [{
                        "type": "thinking",
                        "thinking": "ran out of tokens",
                        "signature": "sig-cut-off"
                    }]
                }),
            ],
        );
        room.append_turn(WireFormat::Anthropic, anthropic_turn(2));

        room.set_web_search(true);

        assert_eq!(room.history.len(), 4);
        assert_eq!(room.history[2]["content"][0]["text"], "question 2");

        // Two turns are left. A turn that fills the byte cap together with
        // the newer of them makes the backstop drop exactly the older one.
        let kept_bytes = room.history[2..]
            .iter()
            .map(approx_json_bytes)
            .sum::<usize>();
        let mut big_turn = anthropic_turn(3);
        with_text(&mut big_turn, String::new());
        let overhead = big_turn.iter().map(approx_json_bytes).sum::<usize>();
        big_turn[0]["content"][1]["text"] =
            serde_json::json!("z".repeat(MAX_HISTORY_BYTES - kept_bytes - overhead));
        room.append_turn(WireFormat::Anthropic, big_turn);
        assert_eq!(
            room.history_bytes(),
            MAX_HISTORY_BYTES
                + room.history[..2]
                    .iter()
                    .map(approx_json_bytes)
                    .sum::<usize>()
        );

        assert_eq!(room.prune_to_byte_cap(), 1);

        assert_eq!(room.history.len(), 4);
        assert_eq!(room.history[0]["role"], "user");
        assert_eq!(room.history[0]["content"][0]["text"], "question 2");
        assert_eq!(room.history[2]["content"][0]["text"], "question 3");
        assert_eq!(thinking_blocks(&room.history), 0);
        assert_eq!(
            room.history_bytes(),
            room.history.iter().map(approx_json_bytes).sum::<usize>()
        );
    }

    #[test]
    fn reset_generation_rejects_stale_commit_and_rollback() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(
            WireFormat::Responses,
            vec![serde_json::json!({ "turn": "before" })],
        );
        let start = room.begin_turn(WireFormat::Responses);
        assert_eq!(start.reset, HistoryReset::None);
        let request_generation = room.generation();

        room.reset_history();

        assert!(!room.append_turn_if_generation(
            request_generation,
            WireFormat::Responses,
            vec![serde_json::json!({ "turn": "stale response" })],
        ));
        assert!(!room.restore_failed_turn(request_generation, start.rollback));
        assert!(room.history.is_empty());
    }

    #[test]
    fn failed_wire_transition_can_restore_prior_history_once() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({ "turn": "anthropic" })],
        );
        let start = room.begin_turn(WireFormat::Responses);
        assert!(matches!(
            start.reset,
            HistoryReset::WireFormatChanged { .. }
        ));
        let request_generation = room.generation();
        room.settings.web_search = true;

        assert!(room.restore_failed_turn(request_generation, start.rollback));
        assert_eq!(room.wire_format, Some(WireFormat::Anthropic));
        assert_eq!(room.history[0]["turn"], "anthropic");
        assert!(room.settings.web_search);
        assert_ne!(room.generation(), request_generation);
    }

    #[test]
    fn failed_same_wire_turn_leaves_midflight_settings_and_room_state_untouched() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(
            WireFormat::Responses,
            vec![serde_json::json!({ "turn": "before" })],
        );
        let start = room.begin_turn(WireFormat::Responses);
        assert_eq!(start.reset, HistoryReset::None);
        assert!(start.rollback.is_none());
        let request_generation = room.generation();

        room.settings.role = Some("Changed while the provider was running".to_string());
        room.settings.web_search = true;
        let expected = room.clone();

        assert!(room.restore_failed_turn(request_generation, start.rollback));
        assert_eq!(room, expected);
    }

    #[test]
    fn room_states_extract_persistable_settings_without_history() {
        let mut settings = BTreeMap::new();
        settings.insert(
            1,
            RoomSettings {
                model_key: Some("claude".to_string()),
                ..RoomSettings::default()
            },
        );
        let mut states = RoomStates::from_settings(settings.clone());
        states.get_or_default(1).append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({"opaque": true})],
        );
        states.get_or_default(2).settings.web_search = true;

        let persisted = states.settings();

        assert_eq!(persisted[&1], settings[&1]);
        assert!(persisted[&2].web_search);
    }

    #[test]
    fn room_states_reset_all_history_keeps_settings() {
        let mut states = RoomStates::default();
        let room = states.get_or_default(1);
        room.settings.role = Some("terse".to_string());
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({"role": "assistant"})],
        );

        states.reset_all_history();

        let room = states.get_or_default(1);
        assert!(room.history.is_empty());
        assert_eq!(room.settings.role.as_deref(), Some("terse"));
    }

    #[test]
    fn reset_history_starts_the_chat_cost_over_and_appending_keeps_it() {
        let mut room = RoomState::new(RoomSettings::default());
        room.cost.record(&crate::cost::ReplyCost::new(
            Some(tellm_core::Usage {
                input_tokens: 10,
                output_tokens: 5,
                reported_cost_usd: Some(0.01),
                ..Default::default()
            }),
            None,
        ));
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({"role": "assistant"})],
        );
        assert_eq!(room.cost.replies, 1);

        room.reset_history();

        assert_eq!(room.cost, crate::cost::ChatCost::default());
    }

    #[test]
    fn rooms_toml_without_show_cost_follows_the_default_cost_line() {
        let file: RoomsToml = toml::from_str("[rooms.5]\nweb_search = true\n").unwrap();
        assert_eq!(file.rooms["5"].show_cost, None);
        assert!(file.rooms["5"].shows_cost(false), "on in a private room");
        assert!(!file.rooms["5"].shows_cost(true), "off in a shared group");
        assert!(file.rooms["5"].web_search);
        assert_eq!(file.rooms["5"].memory, None);
        let file: RoomsToml = toml::from_str("[rooms.5]\nshow_cost = false\n").unwrap();
        assert!(!file.rooms["5"].shows_cost(false));
        let file: RoomsToml = toml::from_str("[rooms.5]\nshow_cost = true\n").unwrap();
        assert!(file.rooms["5"].shows_cost(true));
    }

    #[test]
    fn rooms_toml_records_memory_only_when_a_room_chose_it() {
        let mut settings = RoomSettings::default();
        let file = |settings: &RoomSettings| {
            toml::to_string(&RoomsToml {
                rooms: BTreeMap::from([("5".to_string(), settings.clone())]),
            })
            .unwrap()
        };

        assert!(!file(&settings).contains("memory"));
        settings.memory = Some(false);
        assert!(file(&settings).contains("memory = false"));
    }

    #[test]
    fn reset_history_forgets_the_conversation_memory() {
        let mut room = RoomState::new(RoomSettings::default());
        room.memory = Some("Facts".to_string());
        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);

        room.reset_history();

        assert_eq!(room.memory, None);
        assert!(room.history.is_empty());
    }

    #[test]
    fn a_summary_starts_a_new_conversation_and_lasts_until_the_next_one() {
        let mut room = RoomState::new(RoomSettings::default());
        room.memory = Some("Facts".to_string());
        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 1})]);
        room.cost.record(&crate::cost::ReplyCost::new(
            Some(tellm_core::Usage {
                input_tokens: 10,
                output_tokens: 5,
                reported_cost_usd: Some(0.01),
                ..Default::default()
            }),
            None,
        ));
        let generation = room.generation();

        room.continue_from_summary("Notes.".to_string());

        assert_eq!(room.summary.as_deref(), Some("Notes."));
        assert!(room.history.is_empty());
        assert_eq!(room.memory, None);
        assert_eq!(room.cost, crate::cost::ChatCost::default());
        assert_ne!(room.generation(), generation);

        room.append_turn(WireFormat::Responses, vec![serde_json::json!({"turn": 2})]);
        assert_eq!(room.summary.as_deref(), Some("Notes."));

        room.reset_history();
        assert_eq!(room.summary, None);
    }

    #[test]
    fn room_states_remove_chat_drops_persisted_settings() {
        let mut states = RoomStates::from_settings(BTreeMap::from([(
            1,
            RoomSettings {
                model_key: Some("claude".to_string()),
                ..RoomSettings::default()
            },
        )]));

        states.remove(1);

        assert!(!states.settings().contains_key(&1));
    }

    #[test]
    fn stale_provider_commit_does_not_recreate_removed_room() {
        let mut states = RoomStates::default();
        let room = states.get_or_default(1);
        room.begin_turn(WireFormat::Responses);
        let generation = room.generation();
        states.remove(1);

        let committed = states.get_mut(1).is_some_and(|room| {
            room.append_turn_if_generation(
                generation,
                WireFormat::Responses,
                vec![serde_json::json!({ "stale": true })],
            )
        });

        assert!(!committed);
        assert!(states.get(1).is_none());
    }

    fn anthropic_turn(turn: usize) -> Vec<Value> {
        vec![
            serde_json::json!({
                "role": "user",
                "content": [{ "type": "text", "text": format!("question {turn}") }]
            }),
            serde_json::json!({
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": format!("reasoning {turn}"),
                        "signature": format!("sig-{turn}")
                    },
                    { "type": "text", "text": format!("answer {turn}") }
                ]
            }),
        ]
    }

    /// Adds a text block to a turn's user message.
    fn with_text(turn: &mut [Value], text: String) {
        turn[0]["content"]
            .as_array_mut()
            .expect("user content blocks")
            .push(serde_json::json!({ "type": "text", "text": text }));
    }

    fn thinking_blocks(history: &[Value]) -> usize {
        history
            .iter()
            .filter_map(|item| item["content"].as_array())
            .flatten()
            .filter(|block| block["type"] == "thinking")
            .count()
    }

    fn with_temp_config_dir(test: impl FnOnce(&std::path::Path)) {
        let _guard = TEST_LOCK
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let temp = TempConfigDir::new();

        test(&temp.path);
    }

    struct TempConfigDir {
        path: std::path::PathBuf,
    }

    impl TempConfigDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "tellm-rooms-test-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            set_test_config_dir_override(Some(path.clone()));
            Self { path }
        }
    }

    impl Drop for TempConfigDir {
        fn drop(&mut self) {
            set_test_config_dir_override(None);
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    static TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
}
