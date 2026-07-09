//! Configuration and secret storage.
//!
//! Non-secrets live in a human-editable TOML file at
//! `~/.config/tellm/config.toml` (XDG on Linux, `~/Library/Application
//! Support` is deliberately NOT used — self-hosters expect dotfile-style
//! config they can diff and back up; `dirs::config_dir()` gives us the
//! platform-appropriate location).
//!
//! Secrets (bot token, API keys) never enter the TOML. Target design:
//! OS keychain via `keyring-core` with direct platform-store registration
//! (Apple native keychain, Windows native, or zbus Secret Service; checked
//! 2026-07-05 against keyring 4.1.3's broken `v1` wrapper), falling back to a
//! `0600` credentials file for headless hosts, with env-var overrides.
//! Encrypting a file with a key stored on the same disk is theater — we
//! don't do it.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use serde::{Deserialize, Serialize};
use tellm_core::ThinkingLevel;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("config parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("config serialize error: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("no config directory available on this platform")]
    NoConfigDir,
    #[error("invalid config:\n{}", .0.join("\n"))]
    Invalid(Vec<String>),
}

/// Which wire format a configured model speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireFormat {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI Responses API (OpenAI, xAI, Meta Model API).
    Responses,
    /// OpenAI chat-completions-compatible (Ollama, DeepSeek, OpenRouter, ...).
    Compat,
    /// Google Interactions API (Gemini).
    Gemini,
}

/// One configured model. Capability routing is EXPLICIT — never inferred
/// from the spelling of a user-chosen key (a lesson from console-chat-gpt).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub wire_format: WireFormat,
    pub model_name: String,
    #[serde(default)]
    pub base_url: Option<String>,
    /// Name of the secret holding this model's provider API key. The secret
    /// value itself never enters `config.toml`.
    #[serde(default)]
    pub api_key_secret: Option<String>,
    /// Telegram chats pinned to this model ("room pinning").
    #[serde(default)]
    pub telegram_chat_ids: Vec<i64>,
    #[serde(default)]
    pub thinking: ThinkingLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TelegramConfig {
    #[serde(default)]
    pub allowed_chat_ids: Vec<i64>,
    /// Telegram USER ids proven to own this bot (recorded when they complete
    /// a code pairing — console access is the ownership proof). Chats these
    /// users add the bot to are auto-approved with the model picker, no code
    /// needed.
    #[serde(default)]
    pub owner_user_ids: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub default_model: String,
    pub models: BTreeMap<String, ModelConfig>,
    #[serde(default)]
    pub telegram: TelegramConfig,
}

impl Config {
    /// Semantic validation beyond deserialization. Called at startup and by
    /// the wizard before writing; all problems are reported at once.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let mut problems = Vec::new();

        if !self.models.contains_key(&self.default_model) {
            problems.push(format!(
                "default_model \"{}\" is not defined under [models]",
                self.default_model
            ));
        }

        let mut pinned: std::collections::BTreeMap<i64, &str> = Default::default();
        for (key, model) in &self.models {
            if model.wire_format == WireFormat::Compat && model.base_url.is_none() {
                problems.push(format!(
                    "model \"{key}\": wire_format \"compat\" requires base_url (there is no default compat endpoint)"
                ));
            }
            for chat_id in &model.telegram_chat_ids {
                if let Some(previous) = pinned.insert(*chat_id, key) {
                    problems.push(format!(
                        "chat_id {chat_id} is pinned to both \"{previous}\" and \"{key}\" — a room can be pinned to at most one model"
                    ));
                }
            }
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(ConfigError::Invalid(problems))
        }
    }
}

/// Load and semantically validate in one step — the runtime entry point.
pub fn load_validated() -> Result<Config, ConfigError> {
    let config = load()?;
    config.validate()?;
    Ok(config)
}

pub fn config_dir() -> Result<PathBuf, ConfigError> {
    #[cfg(test)]
    if let Some(path) = test_config_dir_override() {
        return Ok(path);
    }

    Ok(dirs::config_dir()
        .ok_or(ConfigError::NoConfigDir)?
        .join("tellm"))
}

#[cfg(test)]
fn test_config_dir_override() -> Option<PathBuf> {
    TEST_CONFIG_DIR_OVERRIDE
        .get_or_init(Default::default)
        .lock()
        .expect("test config dir override lock poisoned")
        .clone()
}

#[cfg(test)]
fn set_test_config_dir_override(path: Option<PathBuf>) {
    *TEST_CONFIG_DIR_OVERRIDE
        .get_or_init(Default::default)
        .lock()
        .expect("test config dir override lock poisoned") = path;
}

#[cfg(test)]
static TEST_CONFIG_DIR_OVERRIDE: std::sync::OnceLock<std::sync::Mutex<Option<PathBuf>>> =
    std::sync::OnceLock::new();

pub fn config_path() -> Result<PathBuf, ConfigError> {
    Ok(config_dir()?.join("config.toml"))
}

pub fn load() -> Result<Config, ConfigError> {
    let text = std::fs::read_to_string(config_path()?)?;
    Ok(toml::from_str(&text)?)
}

pub fn save(config: &Config) -> Result<(), ConfigError> {
    let dir = config_dir()?;
    std::fs::create_dir_all(&dir)?;
    write_atomic(&config_path()?, &toml::to_string_pretty(config)?)?;
    Ok(())
}

/// Write via a same-directory temp file + rename so a crash mid-write can
/// never truncate the previous version. These files are rewritten on every
/// /allow, /model, and toggle — and credentials.toml may hold every API key.
pub fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    write_atomic_with_mode(path, contents, None)
}

fn write_atomic_with_mode(path: &Path, contents: &str, mode: Option<u32>) -> std::io::Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
        })?;
    let tmp = path.with_file_name(format!(".{file_name}.tmp"));
    let _ = std::fs::remove_file(&tmp);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    if let Some(mode) = mode {
        options.mode(mode);
    }
    #[cfg(not(unix))]
    let _ = mode;
    let mut file = options.open(&tmp)?;
    file.write_all(contents.as_bytes())?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Secret storage facade: env-var override, OS keychain, then `0600` file.
pub mod secrets {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    #[cfg(feature = "keychain")]
    use std::sync::OnceLock;

    use super::*;

    const KEYCHAIN_SERVICE: &str = "tellm";
    const CREDENTIALS_FILE: &str = "credentials.toml";

    /// Well-known secret names.
    pub const TELEGRAM_BOT_TOKEN: &str = "telegram_bot_token";
    pub const ANTHROPIC_API_KEY: &str = "anthropic_api_key";
    pub const OPENAI_API_KEY: &str = "openai_api_key";
    pub const XAI_API_KEY: &str = "xai_api_key";
    pub const META_MODEL_API_KEY: &str = "meta_model_api_key";
    pub const GEMINI_API_KEY: &str = "gemini_api_key";

    #[derive(Debug, thiserror::Error)]
    pub enum SecretError {
        #[error("io error: {0}")]
        Io(#[from] std::io::Error),
        #[error("credentials parse error: {0}")]
        Parse(#[from] toml::de::Error),
        #[error("credentials serialize error: {0}")]
        Serialize(#[from] toml::ser::Error),
        #[error("no config directory available on this platform")]
        NoConfigDir,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SecretDestination {
        OsKeychain,
        CredentialsFile,
    }

    impl SecretDestination {
        pub fn status_message(self) -> &'static str {
            match self {
                Self::OsKeychain => "stored in OS keychain",
                Self::CredentialsFile => "stored in credentials.toml (0600)",
            }
        }

        pub fn location_label(self) -> &'static str {
            match self {
                Self::OsKeychain => "the OS keychain",
                Self::CredentialsFile => "credentials.toml (0600)",
            }
        }
    }

    pub fn get(name: &str) -> Option<String> {
        std::env::var(env_var_name(name))
            .ok()
            .or_else(|| keychain_get(name))
            .or_else(|| file_get(name))
    }

    pub fn set(name: &str, value: &str) -> Result<SecretDestination, SecretError> {
        if keychain_set(name, value) {
            return Ok(SecretDestination::OsKeychain);
        }
        file_set(name, value)?;
        Ok(SecretDestination::CredentialsFile)
    }

    pub fn set_nonempty(name: &str, value: &str) -> Result<Option<SecretDestination>, SecretError> {
        let value = value.trim();
        if value.is_empty() {
            return Ok(None);
        }
        set(name, value).map(Some)
    }

    fn env_var_name(name: &str) -> String {
        let suffix = name
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() {
                    ch.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect::<String>();
        format!("TELLM_{suffix}")
    }

    #[cfg(feature = "keychain")]
    fn keychain_get(name: &str) -> Option<String> {
        keychain_entry(name)
            .ok()
            .and_then(|entry| entry.get_password().ok())
    }

    #[cfg(feature = "keychain")]
    fn keychain_entry(name: &str) -> Result<keyring_core::Entry, String> {
        if keychain_disabled_for_tests() {
            return Err("keychain disabled for tests".to_string());
        }
        ensure_keychain_store()?;
        keyring_core::Entry::new(KEYCHAIN_SERVICE, name).map_err(|error| error.to_string())
    }

    #[cfg(feature = "keychain")]
    fn ensure_keychain_store() -> Result<(), String> {
        KEYCHAIN_INIT
            .get_or_init(register_keychain_store)
            .as_ref()
            .map(|_| ())
            .map_err(Clone::clone)
    }

    #[cfg(feature = "keychain")]
    static KEYCHAIN_INIT: OnceLock<Result<(), String>> = OnceLock::new();

    #[cfg(feature = "keychain")]
    fn register_keychain_store() -> Result<(), String> {
        let store = platform_keychain_store()?;
        keyring_core::set_default_store(store);
        Ok(())
    }

    #[cfg(all(feature = "keychain", target_os = "macos"))]
    fn platform_keychain_store() -> Result<std::sync::Arc<keyring_core::CredentialStore>, String> {
        let store: std::sync::Arc<keyring_core::CredentialStore> =
            apple_native_keyring_store::keychain::Store::new()
                .map_err(|error| error.to_string())?;
        Ok(store)
    }

    #[cfg(all(feature = "keychain", target_os = "windows"))]
    fn platform_keychain_store() -> Result<std::sync::Arc<keyring_core::CredentialStore>, String> {
        let store: std::sync::Arc<keyring_core::CredentialStore> =
            windows_native_keyring_store::Store::new().map_err(|error| error.to_string())?;
        Ok(store)
    }

    #[cfg(all(
        feature = "keychain",
        unix,
        not(any(target_os = "macos", target_os = "ios", target_os = "android"))
    ))]
    fn platform_keychain_store() -> Result<std::sync::Arc<keyring_core::CredentialStore>, String> {
        let store: std::sync::Arc<keyring_core::CredentialStore> =
            zbus_secret_service_keyring_store::Store::new().map_err(|error| error.to_string())?;
        Ok(store)
    }

    #[cfg(all(
        feature = "keychain",
        not(any(
            target_os = "macos",
            target_os = "windows",
            all(
                unix,
                not(any(target_os = "ios", target_os = "android", target_os = "macos"))
            )
        ))
    ))]
    fn platform_keychain_store() -> Result<std::sync::Arc<keyring_core::CredentialStore>, String> {
        Err("no keychain backend is compiled for this platform".to_string())
    }

    #[cfg(not(feature = "keychain"))]
    fn keychain_get(_name: &str) -> Option<String> {
        None
    }

    #[cfg(feature = "keychain")]
    fn keychain_set(name: &str, value: &str) -> bool {
        keychain_entry(name)
            .and_then(|entry| entry.set_password(value).map_err(|error| error.to_string()))
            .ok()
            .is_some()
    }

    #[cfg(not(feature = "keychain"))]
    fn keychain_set(_name: &str, _value: &str) -> bool {
        false
    }

    fn file_get(name: &str) -> Option<String> {
        read_file_secrets()
            .ok()
            .and_then(|secrets| secrets.get(name).cloned())
    }

    fn file_set(name: &str, value: &str) -> Result<(), SecretError> {
        let mut secrets = read_file_secrets()?;
        secrets.insert(name.to_string(), value.to_string());
        write_file_secrets(&secrets)
    }

    fn read_file_secrets() -> Result<BTreeMap<String, String>, SecretError> {
        let path = credentials_path()?;
        if !path.exists() {
            return Ok(BTreeMap::new());
        }
        ensure_credentials_permissions(&path)?;
        let text = fs::read_to_string(path)?;
        Ok(toml::from_str(&text)?)
    }

    fn write_file_secrets(secrets: &BTreeMap<String, String>) -> Result<(), SecretError> {
        let path = credentials_path()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let text = toml::to_string_pretty(secrets)?;
        super::write_atomic_with_mode(&path, &text, Some(0o600))?;
        ensure_credentials_permissions(&path)?;
        Ok(())
    }

    fn credentials_path() -> Result<PathBuf, SecretError> {
        Ok(super::config_dir()
            .map_err(|_| SecretError::NoConfigDir)?
            .join(CREDENTIALS_FILE))
    }

    #[cfg(unix)]
    fn ensure_credentials_permissions(path: &PathBuf) -> Result<(), SecretError> {
        let metadata = fs::metadata(path)?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode != 0o600 {
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(())
    }

    #[cfg(not(unix))]
    fn ensure_credentials_permissions(_path: &PathBuf) -> Result<(), SecretError> {
        Ok(())
    }

    fn keychain_disabled_for_tests() -> bool {
        #[cfg(test)]
        {
            *TEST_KEYCHAIN_DISABLED
                .get_or_init(Default::default)
                .lock()
                .expect("test keychain disable lock poisoned")
        }
        #[cfg(not(test))]
        {
            false
        }
    }

    #[cfg(test)]
    fn set_keychain_disabled_for_tests(disabled: bool) {
        *TEST_KEYCHAIN_DISABLED
            .get_or_init(Default::default)
            .lock()
            .expect("test keychain disable lock poisoned") = disabled;
    }

    #[cfg(test)]
    static TEST_KEYCHAIN_DISABLED: std::sync::OnceLock<std::sync::Mutex<bool>> =
        std::sync::OnceLock::new();

    #[cfg(test)]
    pub(super) fn disable_keychain_for_tests(disabled: bool) {
        set_keychain_disabled_for_tests(disabled);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(wire_format: WireFormat, base_url: Option<&str>, chat_ids: &[i64]) -> ModelConfig {
        ModelConfig {
            wire_format,
            model_name: "m".into(),
            base_url: base_url.map(Into::into),
            api_key_secret: None,
            telegram_chat_ids: chat_ids.to_vec(),
            thinking: ThinkingLevel::default(),
        }
    }

    fn config(models: &[(&str, ModelConfig)], default_model: &str) -> Config {
        Config {
            default_model: default_model.into(),
            models: models
                .iter()
                .map(|(k, m)| (k.to_string(), m.clone()))
                .collect(),
            telegram: TelegramConfig::default(),
        }
    }

    fn problems(config: &Config) -> Vec<String> {
        match config.validate() {
            Ok(()) => Vec::new(),
            Err(ConfigError::Invalid(problems)) => problems,
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn valid_config_passes() {
        let c = config(
            &[
                ("claude", model(WireFormat::Anthropic, None, &[1])),
                (
                    "ollama",
                    model(WireFormat::Compat, Some("http://localhost:11434/v1"), &[2]),
                ),
            ],
            "claude",
        );
        assert!(c.validate().is_ok());
    }

    #[test]
    fn missing_default_model_is_reported() {
        let c = config(
            &[("claude", model(WireFormat::Anthropic, None, &[]))],
            "gone",
        );
        let p = problems(&c);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("default_model"), "{p:?}");
    }

    #[test]
    fn compat_without_base_url_is_reported() {
        let c = config(&[("x", model(WireFormat::Compat, None, &[]))], "x");
        let p = problems(&c);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("base_url"), "{p:?}");
    }

    #[test]
    fn duplicate_room_pin_is_reported() {
        let c = config(
            &[
                ("a", model(WireFormat::Anthropic, None, &[42])),
                ("b", model(WireFormat::Responses, None, &[42])),
            ],
            "a",
        );
        let p = problems(&c);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("42"), "{p:?}");
    }

    #[test]
    fn all_problems_reported_at_once() {
        let c = config(
            &[
                ("a", model(WireFormat::Compat, None, &[7])),
                ("b", model(WireFormat::Anthropic, None, &[7])),
            ],
            "missing",
        );
        assert_eq!(problems(&c).len(), 3);
    }

    #[test]
    fn config_roundtrips_through_toml() {
        let c = config(
            &[("claude", model(WireFormat::Anthropic, None, &[1, 2]))],
            "claude",
        );
        let text = toml::to_string_pretty(&c).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert!(back.validate().is_ok());
        assert_eq!(back.default_model, "claude");
        assert_eq!(back.models["claude"].telegram_chat_ids, vec![1, 2]);
    }

    #[test]
    fn secrets_file_roundtrip_uses_credentials_toml() {
        with_temp_config_dir(|dir| {
            let destination = secrets::set("anthropic_api_key", "sk-ant-test").unwrap();

            assert_eq!(destination, secrets::SecretDestination::CredentialsFile);
            assert_eq!(
                secrets::get("anthropic_api_key"),
                Some("sk-ant-test".to_string())
            );
            let text = std::fs::read_to_string(dir.join("credentials.toml")).unwrap();
            assert!(text.contains("anthropic_api_key"));
            assert!(text.contains("sk-ant-test"));
        });
    }

    #[test]
    fn set_nonempty_trims_and_skips_empty_values() {
        with_temp_config_dir(|_dir| {
            assert_eq!(
                secrets::set_nonempty("openai_api_key", "   ").unwrap(),
                None
            );
            assert_eq!(
                secrets::set_nonempty("openai_api_key", " sk-openai ").unwrap(),
                Some(secrets::SecretDestination::CredentialsFile)
            );
            assert_eq!(
                secrets::get("openai_api_key"),
                Some("sk-openai".to_string())
            );
        });
    }

    #[test]
    fn secret_destination_messages_are_operator_facing() {
        assert_eq!(
            secrets::SecretDestination::OsKeychain.status_message(),
            "stored in OS keychain"
        );
        assert_eq!(
            secrets::SecretDestination::CredentialsFile.status_message(),
            "stored in credentials.toml (0600)"
        );
        assert_eq!(
            secrets::SecretDestination::OsKeychain.location_label(),
            "the OS keychain"
        );
        assert_eq!(
            secrets::SecretDestination::CredentialsFile.location_label(),
            "credentials.toml (0600)"
        );
    }

    #[test]
    fn secrets_file_preserves_existing_entries() {
        with_temp_config_dir(|_dir| {
            secrets::set("telegram_bot_token", "bot-token").unwrap();
            secrets::set("openai_api_key", "sk-openai").unwrap();

            assert_eq!(
                secrets::get("telegram_bot_token"),
                Some("bot-token".to_string())
            );
            assert_eq!(
                secrets::get("openai_api_key"),
                Some("sk-openai".to_string())
            );
        });
    }

    #[test]
    fn env_secret_overrides_file_secret() {
        with_temp_config_dir(|_dir| {
            secrets::set("test_secret", "file-value").unwrap();
            // Rust 2024 marks process environment mutation unsafe because it is
            // global; this test holds TEST_LOCK for the whole mutation window.
            unsafe {
                std::env::set_var("TELLM_TEST_SECRET", "env-value");
            }
            assert_eq!(secrets::get("test_secret"), Some("env-value".to_string()));
            unsafe {
                std::env::remove_var("TELLM_TEST_SECRET");
            }
        });
    }

    #[test]
    fn missing_secret_returns_none() {
        with_temp_config_dir(|_dir| {
            assert_eq!(secrets::get("missing"), None);
        });
    }

    #[cfg(unix)]
    #[test]
    fn credentials_file_permissions_are_0600() {
        use std::os::unix::fs::PermissionsExt;

        with_temp_config_dir(|dir| {
            secrets::set("openai_api_key", "sk-openai").unwrap();

            let mode = std::fs::metadata(dir.join("credentials.toml"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        });
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
                "tellm-config-test-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            set_test_config_dir_override(Some(path.clone()));
            secrets::disable_keychain_for_tests(true);
            Self { path }
        }
    }

    impl Drop for TempConfigDir {
        fn drop(&mut self) {
            secrets::disable_keychain_for_tests(false);
            set_test_config_dir_override(None);
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    static TEST_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
}
