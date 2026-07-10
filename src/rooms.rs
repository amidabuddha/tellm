use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tellm_core::ThinkingLevel;

use tellm_config::{ConfigError, WireFormat};

const ROOMS_FILE: &str = "rooms.toml";
/// Keep enough context for long conversations without allowing opaque provider
/// state to grow without bound. Pruning always removes complete provider turns.
pub const MAX_HISTORY_TURNS: usize = 32;
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomState {
    pub settings: RoomSettings,
    pub wire_format: Option<WireFormat>,
    pub history: Vec<Value>,
    history_turns: VecDeque<HistoryTurn>,
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

impl RoomState {
    pub fn new(settings: RoomSettings) -> Self {
        Self {
            settings,
            wire_format: None,
            history: Vec::new(),
            history_turns: VecDeque::new(),
        }
    }

    pub fn begin_turn(&mut self, wire_format: WireFormat) -> HistoryReset {
        if self.wire_format != Some(wire_format) {
            let previous = self.wire_format;
            self.wire_format = Some(wire_format);
            self.clear_history_storage();
            return HistoryReset::WireFormatChanged {
                previous,
                new: wire_format,
            };
        }

        HistoryReset::None
    }

    pub fn append_turn(&mut self, wire_format: WireFormat, turn_items: Vec<Value>) {
        if self.wire_format != Some(wire_format) {
            self.wire_format = Some(wire_format);
            self.clear_history_storage();
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

    pub fn reset_history(&mut self) {
        self.clear_history_storage();
    }

    fn clear_history_storage(&mut self) {
        self.history.clear();
        self.history_turns.clear();
    }

    fn push_turn(&mut self, turn_items: Vec<Value>) {
        let turn = HistoryTurn {
            item_count: turn_items.len(),
            approx_bytes: turn_items.iter().fold(0_usize, |total, value| {
                total.saturating_add(approx_json_bytes(value))
            }),
        };
        self.history.extend(turn_items);
        self.history_turns.push_back(turn);

        while self.history_turns.len() > MAX_HISTORY_TURNS
            || self.history_bytes() > MAX_HISTORY_BYTES
        {
            let Some(oldest) = self.history_turns.pop_front() else {
                break;
            };
            self.history.drain(..oldest.item_count);
        }
    }

    fn history_bytes(&self) -> usize {
        self.history_turns.iter().fold(0_usize, |total, turn| {
            total.saturating_add(turn.approx_bytes)
        })
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

    pub fn insert(&mut self, chat_id: i64, room: RoomState) {
        self.rooms.insert(chat_id, room);
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
                },
            );

            save_settings(&settings).unwrap();
            let loaded = load_settings().unwrap();

            assert_eq!(loaded, settings);
            let text = std::fs::read_to_string(dir.join("rooms.toml")).unwrap();
            assert!(text.contains("[rooms.42]"));
            assert!(text.contains("[rooms.-7]"));
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
            room.begin_turn(WireFormat::Anthropic),
            HistoryReset::WireFormatChanged {
                previous: None,
                new: WireFormat::Anthropic,
            }
        );
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({ "role": "user", "content": "hi" })],
        );

        assert_eq!(room.begin_turn(WireFormat::Anthropic), HistoryReset::None);
        assert_eq!(room.history.len(), 1);
    }

    #[test]
    fn wire_format_switch_resets_opaque_history() {
        let mut room = RoomState::new(RoomSettings::default());
        room.append_turn(
            WireFormat::Anthropic,
            vec![serde_json::json!({ "role": "assistant", "content": [] })],
        );

        let reset = room.begin_turn(WireFormat::Responses);

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
            room.begin_turn(WireFormat::Compat),
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
        assert_eq!(room.begin_turn(WireFormat::Compat), HistoryReset::None);
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
    fn history_limit_prunes_complete_provider_turns() {
        let mut room = RoomState::new(RoomSettings::default());
        for turn in 0..=MAX_HISTORY_TURNS {
            room.append_turn(
                WireFormat::Responses,
                vec![
                    serde_json::json!({ "turn": turn, "role": "user" }),
                    serde_json::json!({ "turn": turn, "role": "assistant" }),
                ],
            );
        }

        assert_eq!(room.history.len(), MAX_HISTORY_TURNS * 2);
        assert_eq!(room.history[0]["turn"], 1);
        assert_eq!(room.history[1]["turn"], 1);
        assert_eq!(room.history.last().unwrap()["turn"], MAX_HISTORY_TURNS);
    }

    #[test]
    fn history_byte_limit_never_keeps_half_a_provider_turn() {
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

        assert_eq!(room.history.len(), 2);
        assert!(room.history.iter().all(|item| item["turn"] == 2));
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
