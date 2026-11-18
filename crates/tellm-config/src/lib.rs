//! Configuration and secret storage.
//!
//! Non-secrets live in a human-editable `tellm/config.toml` below the
//! platform configuration directory returned by `dirs::config_dir()` (for
//! example, XDG config on Linux and `~/Library/Application Support` on
//! macOS).
//!
//! Secrets (bot token, API keys) never enter `config.toml`. Target design:
//! OS keychain via `keyring-core` with direct platform-store registration
//! (Apple native keychain, Windows native, or zbus Secret Service; checked
//! 2026-07-05 against keyring 4.1.3's broken `v1` wrapper), falling back to a
//! `0600` credentials file for headless hosts, with env-var overrides.
//! Encrypting a file with a key stored on the same disk is theater — we
//! don't do it.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

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
    /// Explicitly permit cleartext HTTP for a keyless, non-loopback compat
    /// endpoint (for example, Ollama on a trusted LAN). Credential-bearing
    /// endpoints still require HTTPS even when this is enabled.
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_insecure_http: bool,
    /// Name of the secret holding this model's provider API key. The secret
    /// value itself never enters `config.toml`.
    #[serde(default)]
    pub api_key_secret: Option<String>,
    /// Telegram chats pinned to this model ("room pinning").
    #[serde(default)]
    pub telegram_chat_ids: Vec<i64>,
    #[serde(default)]
    pub thinking: ThinkingLevel,
    /// Provider value sent when the thinking level is `max`, such as `"max"`
    /// or `"xhigh"`. The tiers above `high` differ from model to model, so
    /// this is explicit instead of inferred from `model_name`. When unset,
    /// the wire format's default top tier is sent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_effort: Option<String>,
    /// Provider value sent when the thinking level is `off`, such as
    /// `"none"`, `"minimal"` or `"low"`. Whether reasoning can be disabled,
    /// and what the lowest tier is called, differs from model to model. When
    /// unset, no reasoning setting is sent and the provider's own default
    /// applies, which on some models is their highest tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub off_effort: Option<String>,
    /// Upper limit on one reply's output tokens, sent as Anthropic
    /// `max_tokens`; thinking counts toward it. Only the `anthropic` wire
    /// format takes it, because the other formats leave output length to the
    /// provider. When unset, the Anthropic client's default applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// The model's list prices, used to estimate what each reply cost from
    /// the token usage the provider reports. Prices change and are the
    /// user's to keep current; tellm never fetches them. When unset, the
    /// cost line shows the provider's own reported cost if it sends one
    /// (xAI, OpenRouter), and token counts otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub price: Option<ModelPrice>,
}

/// US-dollar list prices for one model. Token prices are per million tokens;
/// `web_search` is per 1,000 searches. Only `input` and `output` are
/// required: cache reads and writes fall back to the input price, which can
/// overstate a cached reply but never understates one, and searches fall back
/// to free.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPrice {
    /// Input tokens that were neither read from nor written to a cache.
    pub input: f64,
    /// Input tokens read from the provider's prompt cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input: Option<f64>,
    /// Input tokens written to the provider's prompt cache, where the
    /// provider bills writes (Anthropic, OpenAI GPT-5.6 and later).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
    /// Output tokens, reasoning included.
    pub output: f64,
    /// Provider-native web searches, per 1,000.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search: Option<f64>,
}

impl ModelPrice {
    /// Estimated US-dollar cost of one reply's usage at these prices.
    pub fn cost_usd(&self, usage: &tellm_core::Usage) -> f64 {
        let per_token =
            |price_per_million: f64, tokens: u64| price_per_million * tokens as f64 / 1_000_000.0;
        per_token(self.input, usage.input_tokens)
            + per_token(
                self.cached_input.unwrap_or(self.input),
                usage.cached_input_tokens,
            )
            + per_token(
                self.cache_write.unwrap_or(self.input),
                usage.cache_write_tokens,
            )
            + per_token(self.output, usage.output_tokens)
            + self.web_search.unwrap_or(0.0) * usage.web_searches as f64 / 1_000.0
    }

    fn fields(&self) -> [(&'static str, Option<f64>); 5] {
        [
            ("input", Some(self.input)),
            ("cached_input", self.cached_input),
            ("cache_write", self.cache_write),
            ("output", Some(self.output)),
            ("web_search", self.web_search),
        ]
    }
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
            if key.trim().is_empty() {
                problems.push("model key must not be empty".to_string());
            } else if key.chars().any(char::is_whitespace) {
                problems.push(format!(
                    "model \"{key}\": key must not contain whitespace because /model commands use one token"
                ));
            }
            if model.model_name.trim().is_empty() {
                problems.push(format!("model \"{key}\": model_name must not be empty"));
            }
            for (setting, value, examples) in [
                ("max_effort", &model.max_effort, "\"max\" or \"xhigh\""),
                ("off_effort", &model.off_effort, "\"none\" or \"low\""),
            ] {
                if let Some(value) = value
                    && (value.is_empty() || value.chars().any(char::is_whitespace))
                {
                    problems.push(format!(
                        "model \"{key}\": {setting} must be a single provider effort value such as {examples}"
                    ));
                }
            }
            if let Some(price) = &model.price {
                for (field, value) in price.fields() {
                    if let Some(value) = value
                        && !(value.is_finite() && value >= 0.0)
                    {
                        problems.push(format!(
                            "model \"{key}\": price.{field} must be a non-negative number of US dollars"
                        ));
                    }
                }
            }
            if let Some(max_output_tokens) = model.max_output_tokens {
                if model.wire_format != WireFormat::Anthropic {
                    problems.push(format!(
                        "model \"{key}\": max_output_tokens is only used by the anthropic wire format"
                    ));
                } else if max_output_tokens == 0 {
                    problems.push(format!(
                        "model \"{key}\": max_output_tokens must be greater than zero"
                    ));
                }
            }

            if model.wire_format == WireFormat::Compat && model.base_url.is_none() {
                problems.push(format!(
                    "model \"{key}\": wire_format \"compat\" requires base_url (there is no default compat endpoint)"
                ));
            }

            let nonempty_base_url = match model.base_url.as_deref() {
                Some(base_url) if base_url.trim().is_empty() => {
                    problems.push(format!("model \"{key}\": base_url must not be empty"));
                    None
                }
                base_url => base_url,
            };
            let parsed_base_url =
                nonempty_base_url.and_then(|base_url| match url::Url::parse(base_url) {
                    Ok(url) => Some(url),
                    Err(error) => {
                        problems.push(format!(
                            "model \"{key}\": base_url is not a valid absolute URL: {error}"
                        ));
                        None
                    }
                });
            let has_secret = match model.api_key_secret.as_deref() {
                Some(secret_name) if secret_name.trim().is_empty() => {
                    problems.push(format!("model \"{key}\": api_key_secret must not be empty"));
                    false
                }
                Some(secret_name) if secrets::is_reserved_name(secret_name) => {
                    problems.push(format!(
                        "model \"{key}\": api_key_secret uses a reserved internal prefix"
                    ));
                    false
                }
                Some(_) => true,
                None => false,
            };

            if model.wire_format != WireFormat::Compat && !has_secret {
                // An explicitly empty name already has the more actionable
                // diagnostic above; avoid reporting the same defect twice.
                if model.api_key_secret.is_none() {
                    problems.push(format!(
                        "model \"{key}\": wire_format \"{:?}\" requires api_key_secret",
                        model.wire_format
                    ));
                }
            }
            if let Some(base_url) = parsed_base_url.as_ref()
                && (!base_url.username().is_empty() || base_url.password().is_some())
            {
                problems.push(format!(
                    "model \"{key}\": base_url must not contain embedded credentials"
                ));
            }
            if let Some(base_url) = parsed_base_url.as_ref()
                && base_url.scheme() != "https"
            {
                if has_secret {
                    problems.push(format!(
                        "model \"{key}\": credential-bearing base_url must use https://"
                    ));
                } else if model.wire_format != WireFormat::Compat
                    || base_url.scheme() != "http"
                    || (!is_loopback_http_url(base_url) && !model.allow_insecure_http)
                {
                    problems.push(format!(
                        "model \"{key}\": base_url must use https:// unless it is a keyless loopback compat endpoint or allow_insecure_http = true explicitly permits keyless compat HTTP"
                    ));
                }
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

fn is_false(value: &bool) -> bool {
    !*value
}

fn is_loopback_http_url(value: &url::Url) -> bool {
    value.scheme() == "http"
        && match value.host() {
            Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
            Some(url::Host::Ipv4(address)) => address.is_loopback(),
            Some(url::Host::Ipv6(address)) => address.is_loopback(),
            None => false,
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
    let (tmp, mut file) = create_unique_temp(path, mode)?;
    let result = (|| {
        file.write_all(contents.as_bytes())?;
        file.sync_all()?;
        drop(file);
        publish_atomic_file(path, &tmp)
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn publish_atomic_file(path: &Path, tmp: &Path) -> std::io::Result<()> {
    publish_atomic_file_with_sync(path, tmp, sync_parent_directory)
}

fn publish_atomic_file_with_sync(
    path: &Path,
    tmp: &Path,
    sync_parent: impl FnOnce(&Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    std::fs::rename(tmp, path)?;
    // Once rename succeeds, the new file is the visible committed state.
    // Report success to callers even if the platform/filesystem cannot fsync
    // the directory; treating that durability warning as a failed commit
    // would make callers roll memory back behind the already-published file.
    if let Err(error) = sync_parent(path) {
        log::warn!(
            target: "tellm::config",
            "atomic file published but parent directory sync failed error=\"{error}\""
        );
    }
    Ok(())
}

fn create_unique_temp(path: &Path, mode: Option<u32>) -> std::io::Result<(PathBuf, File)> {
    let file_name = path.file_name().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "path has no file name")
    })?;

    for _ in 0..8 {
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random)
            .map_err(|error| std::io::Error::other(format!("random temp-file name: {error}")))?;
        let mut tmp_name = OsString::from(".");
        tmp_name.push(file_name);
        tmp_name.push(format!(
            ".tmp-{}-{:032x}",
            std::process::id(),
            u128::from_ne_bytes(random)
        ));
        let tmp = path.with_file_name(tmp_name);

        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        if let Some(mode) = mode {
            options.mode(mode);
        }
        #[cfg(not(unix))]
        let _ = mode;

        match options.open(&tmp) {
            Ok(file) => return Ok((tmp, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "could not allocate a unique atomic-write temp file",
    ))
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    File::open(parent)?.sync_all()
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> std::io::Result<()> {
    // Opening directories as files is not portable on Windows. Atomic rename
    // still protects the file contents there; Unix additionally attempts to
    // persist the directory entry above.
    Ok(())
}

/// Secret storage facade: env-var override, then the explicitly selected
/// persisted destination (legacy entries remain keychain-first).
pub mod secrets {
    use std::collections::BTreeMap;
    use std::fs;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};

    #[cfg(feature = "keychain")]
    const KEYCHAIN_SERVICE: &str = "tellm";
    const CREDENTIALS_FILE: &str = "credentials.toml";
    const FILE_PREFERENCE_PREFIX: &str = "__tellm_prefer_credentials_file__:";

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
        #[error(
            "cannot store secret \"{name}\" while environment variable {variable} overrides every persisted value; unset it first"
        )]
        EnvironmentOverride { name: String, variable: String },
        #[error("secret name \"{0}\" uses a reserved internal prefix")]
        ReservedName(String),
        #[error(
            "stored secret \"{name}\" in the OS keychain but failed to remove its stale credentials.toml fallback: {source}"
        )]
        FallbackCleanup {
            name: String,
            #[source]
            source: Box<SecretError>,
        },
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
        if is_reserved_name(name) {
            return None;
        }
        if let Some(value) = std::env::var_os(env_var_name(name)) {
            // Presence is authoritative even when the platform value is not
            // UTF-8. Fail closed instead of silently using an older persisted
            // credential the operator expected this variable to override.
            return value.into_string().ok();
        }

        // Read the file before the keychain because its per-secret preference
        // marker must outrank a stale keychain value. Keeping this read fresh
        // also lets a running daemon observe `tellm secret set` rotations from
        // a separate invocation without a restart.
        let file_secrets = read_file_secrets().ok();
        persisted_secret(name, file_secrets.as_ref(), || keychain_get(name))
    }

    fn persisted_secret(
        name: &str,
        file_secrets: Option<&BTreeMap<String, String>>,
        keychain_get: impl FnOnce() -> Option<String>,
    ) -> Option<String> {
        let file_value = file_secrets.and_then(|secrets| secrets.get(name).cloned());
        let prefer_file =
            file_secrets.is_some_and(|secrets| secrets.contains_key(&file_preference_key(name)));
        if prefer_file {
            return file_value;
        }
        keychain_get().or(file_value)
    }

    pub fn set(name: &str, value: &str) -> Result<SecretDestination, SecretError> {
        if is_reserved_name(name) {
            return Err(SecretError::ReservedName(name.to_string()));
        }
        let variable = env_var_name(name);
        if std::env::var_os(&variable).is_some() {
            return Err(SecretError::EnvironmentOverride {
                name: name.to_string(),
                variable,
            });
        }
        set_with_backends(
            name,
            value,
            keychain_set,
            keychain_read,
            file_set,
            file_remove,
        )
    }

    fn set_with_backends(
        name: &str,
        value: &str,
        keychain_set: impl FnOnce(&str, &str) -> bool,
        keychain_get: impl FnOnce(&str) -> Result<Option<String>, String>,
        file_set: impl FnOnce(&str, &str) -> Result<(), SecretError>,
        file_remove: impl FnOnce(&str) -> Result<(), SecretError>,
    ) -> Result<SecretDestination, SecretError> {
        if keychain_set(name, value) {
            return finish_keychain_write(name, file_remove);
        }

        match keychain_get(name) {
            // A failed idempotent write is harmless: the requested value is
            // already at the higher-priority destination.
            Ok(Some(stored)) if stored == value => finish_keychain_write(name, file_remove),
            // A file fallback records its own precedence marker atomically.
            // It therefore remains authoritative if an older keychain value
            // becomes readable again after a transient backend failure.
            Ok(Some(_)) | Ok(None) | Err(_) => {
                file_set(name, value)?;
                Ok(SecretDestination::CredentialsFile)
            }
        }
    }

    fn finish_keychain_write(
        name: &str,
        file_remove: impl FnOnce(&str) -> Result<(), SecretError>,
    ) -> Result<SecretDestination, SecretError> {
        file_remove(name).map_err(|source| SecretError::FallbackCleanup {
            name: name.to_string(),
            source: Box::new(source),
        })?;
        Ok(SecretDestination::OsKeychain)
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
    fn keychain_read(name: &str) -> Result<Option<String>, String> {
        if keychain_disabled_for_tests() {
            return Ok(None);
        }
        let entry = keychain_entry(name)?;
        match entry.get_password() {
            Ok(value) => Ok(Some(value)),
            Err(keyring_core::Error::NoEntry) => Ok(None),
            Err(error) => Err(error.to_string()),
        }
    }

    #[cfg(feature = "keychain")]
    fn keychain_get(name: &str) -> Option<String> {
        keychain_read(name).ok().flatten()
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
    fn keychain_read(_name: &str) -> Result<Option<String>, String> {
        Ok(None)
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

    fn file_set(name: &str, value: &str) -> Result<(), SecretError> {
        let _guard = credentials_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut secrets = read_file_secrets()?;
        secrets.insert(name.to_string(), value.to_string());
        secrets.insert(file_preference_key(name), "true".to_string());
        write_file_secrets(&secrets)
    }

    fn file_remove(name: &str) -> Result<(), SecretError> {
        let _guard = credentials_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut secrets = read_file_secrets()?;
        let removed_value = secrets.remove(name).is_some();
        let removed_preference = secrets.remove(&file_preference_key(name)).is_some();
        if removed_value || removed_preference {
            // Replacing the credentials file atomically also makes removal of
            // this entry atomic with respect to concurrent readers.
            write_file_secrets(&secrets)?;
        }
        Ok(())
    }

    fn credentials_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn file_preference_key(name: &str) -> String {
        format!("{FILE_PREFERENCE_PREFIX}{name}")
    }

    pub fn is_reserved_name(name: &str) -> bool {
        name.starts_with(FILE_PREFERENCE_PREFIX)
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

    #[cfg(feature = "keychain")]
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

    #[cfg(test)]
    mod tests {
        use std::cell::Cell;

        use super::*;

        #[test]
        fn failed_keychain_rotation_selects_the_durable_file_fallback() {
            let wrote_file = Cell::new(false);
            let removed_file = Cell::new(false);

            let destination = set_with_backends(
                "openai_api_key",
                "new-value",
                |_, _| false,
                |_| Ok(Some("old-value".to_string())),
                |_, _| {
                    wrote_file.set(true);
                    Ok(())
                },
                |_| {
                    removed_file.set(true);
                    Ok(())
                },
            )
            .unwrap();

            assert_eq!(destination, SecretDestination::CredentialsFile);
            assert!(wrote_file.get());
            assert!(!removed_file.get());
        }

        #[test]
        fn transient_keychain_read_failure_selects_the_file_fallback() {
            let wrote_file = Cell::new(false);

            let destination = set_with_backends(
                "openai_api_key",
                "new-value",
                |_, _| false,
                |_| Err("keychain locked".to_string()),
                |_, _| {
                    wrote_file.set(true);
                    Ok(())
                },
                |_| panic!("file fallback must not be removed"),
            )
            .unwrap();

            assert_eq!(destination, SecretDestination::CredentialsFile);
            assert!(wrote_file.get());
        }

        #[test]
        fn preferred_file_value_shadows_a_recovered_stale_keychain() {
            let name = "openai_api_key";
            let file = BTreeMap::from([
                (name.to_string(), "new-file-value".to_string()),
                (file_preference_key(name), "true".to_string()),
            ]);
            let read_keychain = Cell::new(false);

            let value = persisted_secret(name, Some(&file), || {
                read_keychain.set(true);
                Some("old-keychain-value".to_string())
            });

            assert_eq!(value.as_deref(), Some("new-file-value"));
            assert!(!read_keychain.get());
        }

        #[test]
        fn failed_idempotent_keychain_write_stays_keychain_first() {
            let wrote_file = Cell::new(false);
            let removed_file = Cell::new(false);

            let destination = set_with_backends(
                "openai_api_key",
                "current-value",
                |_, _| false,
                |_| Ok(Some("current-value".to_string())),
                |_, _| {
                    wrote_file.set(true);
                    Ok(())
                },
                |_| {
                    removed_file.set(true);
                    Ok(())
                },
            )
            .unwrap();

            assert_eq!(destination, SecretDestination::OsKeychain);
            assert!(!wrote_file.get());
            assert!(removed_file.get());
        }

        #[test]
        fn successful_keychain_write_removes_stale_file_fallback() {
            let removed_file = Cell::new(false);

            let destination = set_with_backends(
                "openai_api_key",
                "new-value",
                |_, _| true,
                |_| panic!("successful keychain writes do not need a readback"),
                |_, _| panic!("successful keychain writes must not fall back"),
                |_| {
                    removed_file.set(true);
                    Ok(())
                },
            )
            .unwrap();

            assert_eq!(destination, SecretDestination::OsKeychain);
            assert!(removed_file.get());
        }

        #[test]
        fn keychain_fallback_cleanup_failure_is_reported_clearly() {
            let error = set_with_backends(
                "openai_api_key",
                "new-value",
                |_, _| true,
                |_| panic!("successful keychain writes do not need a readback"),
                |_, _| panic!("successful keychain writes must not fall back"),
                |_| Err(SecretError::NoConfigDir),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                SecretError::FallbackCleanup { name, source }
                    if name == "openai_api_key" && matches!(*source, SecretError::NoConfigDir)
            ));
        }
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
            allow_insecure_http: false,
            api_key_secret: (wire_format != WireFormat::Compat).then(|| "test_api_key".to_string()),
            telegram_chat_ids: chat_ids.to_vec(),
            thinking: ThinkingLevel::default(),
            max_effort: None,
            off_effort: None,
            max_output_tokens: None,
            price: None,
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
    fn empty_model_fields_and_missing_secrets_are_aggregated() {
        let mut missing = model(WireFormat::Anthropic, Some("   "), &[]);
        missing.model_name = "  ".to_string();
        missing.api_key_secret = None;
        let mut empty = model(WireFormat::Compat, Some("https://provider.example/v1"), &[]);
        empty.api_key_secret = Some("\t".to_string());
        let c = config(&[("missing", missing), ("empty", empty)], "missing");

        let p = problems(&c);

        assert_eq!(p.len(), 4, "{p:?}");
        assert!(
            p.iter().any(|problem| problem.contains("model_name")),
            "{p:?}"
        );
        assert!(
            p.iter().any(|problem| problem.contains("base_url must")),
            "{p:?}"
        );
        assert!(
            p.iter()
                .any(|problem| problem.contains("requires api_key_secret")),
            "{p:?}"
        );
        assert!(
            p.iter()
                .any(|problem| problem.contains("api_key_secret must")),
            "{p:?}"
        );
    }

    #[test]
    fn model_keys_must_roundtrip_through_single_token_commands() {
        let c = config(
            &[
                ("", model(WireFormat::Anthropic, None, &[])),
                ("two words", model(WireFormat::Anthropic, None, &[])),
            ],
            "",
        );

        let p = problems(&c);

        assert_eq!(p.len(), 2, "{p:?}");
        assert!(
            p.iter()
                .any(|problem| problem.contains("must not be empty"))
        );
        assert!(p.iter().any(|problem| problem.contains("whitespace")));
    }

    #[test]
    fn internal_secret_marker_prefix_is_reserved() {
        let reserved = "__tellm_prefer_credentials_file__:openai_api_key";
        let mut configured = model(WireFormat::Responses, None, &[]);
        configured.api_key_secret = Some(reserved.to_string());
        let c = config(&[("openai", configured)], "openai");

        let p = problems(&c);

        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("reserved internal prefix"), "{p:?}");
        with_temp_config_dir(|_| {
            assert!(matches!(
                secrets::set(reserved, "value").unwrap_err(),
                secrets::SecretError::ReservedName(name) if name == reserved
            ));
            assert_eq!(secrets::get(reserved), None);
        });
    }

    #[test]
    fn credentialed_custom_url_requires_https_but_keyless_loopback_is_allowed() {
        let credentialed = model(
            WireFormat::Responses,
            Some("http://provider.example/v1"),
            &[],
        );
        let keyless_loopback = model(WireFormat::Compat, Some("http://127.0.0.1:11434/v1"), &[]);
        let c = config(
            &[("credentialed", credentialed), ("ollama", keyless_loopback)],
            "credentialed",
        );

        let p = problems(&c);

        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("https://"), "{p:?}");
    }

    #[test]
    fn keyless_remote_http_endpoint_requires_explicit_opt_in() {
        let mut remote = model(
            WireFormat::Compat,
            Some("http://192.168.1.20:11434/v1"),
            &[],
        );
        let c = config(&[("remote", remote.clone())], "remote");

        let p = problems(&c);

        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("allow_insecure_http = true"), "{p:?}");

        remote.allow_insecure_http = true;
        assert!(config(&[("remote", remote)], "remote").validate().is_ok());
    }

    #[test]
    fn insecure_http_opt_in_never_permits_credentials() {
        let mut credentialed = model(
            WireFormat::Compat,
            Some("http://192.168.1.20:11434/v1"),
            &[],
        );
        credentialed.api_key_secret = Some("ollama_api_key".to_string());
        credentialed.allow_insecure_http = true;

        let p = problems(&config(&[("remote", credentialed)], "remote"));

        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("credential-bearing"), "{p:?}");
    }

    #[test]
    fn insecure_http_opt_in_does_not_permit_other_schemes() {
        let mut remote = model(WireFormat::Compat, Some("ftp://192.168.1.20:11434/v1"), &[]);
        remote.allow_insecure_http = true;

        let p = problems(&config(&[("remote", remote)], "remote"));

        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("must use https://"), "{p:?}");
    }

    #[test]
    fn credentialed_https_custom_url_is_valid() {
        let c = config(
            &[(
                "proxy",
                model(
                    WireFormat::Responses,
                    Some("HTTPS://provider.example/v1"),
                    &[],
                ),
            )],
            "proxy",
        );

        assert!(c.validate().is_ok());
    }

    #[test]
    fn malformed_urls_and_embedded_credentials_are_rejected() {
        let c = config(
            &[
                (
                    "malformed",
                    model(WireFormat::Responses, Some("https://"), &[]),
                ),
                (
                    "embedded",
                    model(
                        WireFormat::Responses,
                        Some("https://user:password@provider.example/v1"),
                        &[],
                    ),
                ),
            ],
            "malformed",
        );

        let p = problems(&c);

        assert_eq!(p.len(), 2, "{p:?}");
        assert!(
            p.iter()
                .any(|problem| problem.contains("valid absolute URL"))
        );
        assert!(
            p.iter()
                .any(|problem| problem.contains("embedded credentials"))
        );
    }

    #[test]
    fn config_roundtrips_through_toml() {
        let c = config(
            &[("claude", model(WireFormat::Anthropic, None, &[1, 2]))],
            "claude",
        );
        let text = toml::to_string_pretty(&c).unwrap();
        assert!(!text.contains("allow_insecure_http"));
        assert!(!text.contains("max_effort"), "{text}");
        assert!(!text.contains("off_effort"), "{text}");
        assert!(!text.contains("max_output_tokens"), "{text}");
        let back: Config = toml::from_str(&text).unwrap();
        assert!(back.validate().is_ok());
        assert_eq!(back.default_model, "claude");
        assert_eq!(back.models["claude"].telegram_chat_ids, vec![1, 2]);
        assert!(!back.models["claude"].allow_insecure_http);
        assert_eq!(back.models["claude"].max_effort, None);
        assert_eq!(back.models["claude"].off_effort, None);
    }

    #[test]
    fn max_effort_roundtrips_through_toml_and_is_optional() {
        let mut gpt = model(WireFormat::Responses, None, &[]);
        gpt.max_effort = Some("max".to_string());
        let c = config(&[("gpt", gpt)], "gpt");

        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("max_effort = \"max\""), "{text}");
        let back: Config = toml::from_str(&text).unwrap();

        assert_eq!(back.models["gpt"].max_effort.as_deref(), Some("max"));
        assert!(back.validate().is_ok());

        // A config written before the setting existed still loads.
        let legacy: Config = toml::from_str(
            "default_model = \"gpt\"\n\n[models.gpt]\nwire_format = \"responses\"\nmodel_name = \"m\"\napi_key_secret = \"k\"\n",
        )
        .unwrap();
        assert_eq!(legacy.models["gpt"].max_effort, None);
        assert!(legacy.validate().is_ok());
    }

    #[test]
    fn max_effort_must_be_one_token() {
        for bad in ["", " ", "x high", "max\n"] {
            let mut gpt = model(WireFormat::Responses, None, &[]);
            gpt.max_effort = Some(bad.to_string());
            let c = config(&[("gpt", gpt)], "gpt");

            let p = problems(&c);

            assert_eq!(p.len(), 1, "{bad:?}: {p:?}");
            assert!(p[0].contains("max_effort"), "{bad:?}: {p:?}");
        }
    }

    #[test]
    fn off_effort_roundtrips_through_toml_and_is_optional() {
        let mut meta = model(WireFormat::Responses, None, &[]);
        meta.off_effort = Some("minimal".to_string());
        let c = config(&[("meta", meta)], "meta");

        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("off_effort = \"minimal\""), "{text}");
        let back: Config = toml::from_str(&text).unwrap();

        assert_eq!(back.models["meta"].off_effort.as_deref(), Some("minimal"));
        assert!(back.validate().is_ok());

        // A config written before the setting existed still loads.
        let legacy: Config = toml::from_str(
            "default_model = \"meta\"\n\n[models.meta]\nwire_format = \"responses\"\nmodel_name = \"m\"\napi_key_secret = \"k\"\n",
        )
        .unwrap();
        assert_eq!(legacy.models["meta"].off_effort, None);
        assert!(legacy.validate().is_ok());
    }

    #[test]
    fn max_output_tokens_roundtrips_through_toml_and_is_optional() {
        let mut claude = model(WireFormat::Anthropic, None, &[]);
        claude.max_output_tokens = Some(21_000);
        let c = config(&[("claude", claude)], "claude");

        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("max_output_tokens = 21000"), "{text}");
        let back: Config = toml::from_str(&text).unwrap();

        assert_eq!(back.models["claude"].max_output_tokens, Some(21_000));
        assert!(back.validate().is_ok());

        // A config written before the setting existed still loads.
        let legacy: Config = toml::from_str(
            "default_model = \"claude\"\n\n[models.claude]\nwire_format = \"anthropic\"\nmodel_name = \"m\"\napi_key_secret = \"k\"\n",
        )
        .unwrap();
        assert_eq!(legacy.models["claude"].max_output_tokens, None);
        assert!(legacy.validate().is_ok());
    }

    #[test]
    fn price_roundtrips_through_toml_as_a_model_subtable_and_is_optional() {
        let mut gpt = model(WireFormat::Responses, None, &[]);
        gpt.price = Some(ModelPrice {
            input: 8.0,
            cached_input: Some(0.4),
            cache_write: Some(10.0),
            output: 40.0,
            web_search: Some(10.0),
        });
        let c = config(&[("gpt", gpt)], "gpt");

        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("[models.gpt.price]"), "{text}");
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.models["gpt"].price, c.models["gpt"].price);
        assert!(back.validate().is_ok());

        let minimal: Config = toml::from_str(
            "default_model = \"gpt\"\n\n[models.gpt]\nwire_format = \"responses\"\nmodel_name = \"m\"\napi_key_secret = \"k\"\n\n[models.gpt.price]\ninput = 2\noutput = 10\n",
        )
        .unwrap();
        let price = minimal.models["gpt"].price.unwrap();
        assert_eq!((price.input, price.output), (2.0, 10.0));
        assert_eq!(price.cached_input, None);
        assert!(minimal.validate().is_ok());

        let misspelled = toml::from_str::<Config>(
            "default_model = \"gpt\"\n\n[models.gpt]\nwire_format = \"responses\"\nmodel_name = \"m\"\napi_key_secret = \"k\"\n\n[models.gpt.price]\ninput = 2\noutput = 10\ncached = 1\n",
        );
        assert!(
            misspelled.is_err(),
            "unknown price fields must not be ignored"
        );
    }

    #[test]
    fn price_values_must_be_finite_and_non_negative() {
        let mut gpt = model(WireFormat::Responses, None, &[]);
        gpt.price = Some(ModelPrice {
            input: -1.0,
            cached_input: Some(f64::NAN),
            cache_write: None,
            output: f64::INFINITY,
            web_search: Some(0.0),
        });
        let error = config(&[("gpt", gpt)], "gpt")
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("price.input"), "{error}");
        assert!(error.contains("price.cached_input"), "{error}");
        assert!(error.contains("price.output"), "{error}");
        assert!(!error.contains("price.web_search"), "{error}");
    }

    #[test]
    fn price_cost_applies_each_rate_and_falls_back_to_input_for_cache_tokens() {
        let usage = tellm_core::Usage {
            input_tokens: 1_000_000,
            cached_input_tokens: 2_000_000,
            cache_write_tokens: 1_000_000,
            output_tokens: 500_000,
            reasoning_tokens: 400_000,
            web_searches: 3,
            reported_cost_usd: None,
        };
        let full = ModelPrice {
            input: 8.0,
            cached_input: Some(0.4),
            cache_write: Some(10.0),
            output: 40.0,
            web_search: Some(10.0),
        };
        // 8 + 0.8 + 10 + 20 + 0.03
        assert!((full.cost_usd(&usage) - 38.83).abs() < 1e-9);

        let minimal = ModelPrice {
            input: 2.0,
            cached_input: None,
            cache_write: None,
            output: 10.0,
            web_search: None,
        };
        // 2 + 4 + 2 + 5, searches free
        assert!((minimal.cost_usd(&usage) - 13.0).abs() < 1e-9);
    }

    #[test]
    fn max_output_tokens_must_be_positive_and_on_an_anthropic_model() {
        let mut claude = model(WireFormat::Anthropic, None, &[]);
        claude.max_output_tokens = Some(0);
        let p = problems(&config(&[("claude", claude)], "claude"));
        assert_eq!(p.len(), 1, "{p:?}");
        assert!(p[0].contains("greater than zero"), "{p:?}");

        // The other wire formats send no output limit, so the setting would
        // be silently ignored there.
        for wire_format in [WireFormat::Responses, WireFormat::Gemini] {
            let mut other = model(wire_format, None, &[]);
            other.max_output_tokens = Some(21_000);
            let p = problems(&config(&[("other", other)], "other"));
            assert_eq!(p.len(), 1, "{wire_format:?}: {p:?}");
            assert!(p[0].contains("anthropic wire format"), "{p:?}");
        }
    }

    #[test]
    fn off_effort_must_be_one_token() {
        for bad in ["", " ", "very low", "none\n"] {
            let mut gpt = model(WireFormat::Responses, None, &[]);
            gpt.off_effort = Some(bad.to_string());
            let c = config(&[("gpt", gpt)], "gpt");

            let p = problems(&c);

            assert_eq!(p.len(), 1, "{bad:?}: {p:?}");
            assert!(p[0].contains("off_effort"), "{bad:?}: {p:?}");
        }
    }

    #[test]
    fn insecure_http_opt_in_roundtrips_through_toml() {
        let mut remote = model(WireFormat::Compat, Some("http://ollama.lan:11434/v1"), &[]);
        remote.allow_insecure_http = true;
        let c = config(&[("ollama", remote)], "ollama");

        let text = toml::to_string_pretty(&c).unwrap();
        assert!(text.contains("allow_insecure_http = true"), "{text}");
        let back: Config = toml::from_str(&text).unwrap();

        assert!(back.models["ollama"].allow_insecure_http);
        assert!(back.validate().is_ok());
    }

    #[test]
    fn concurrent_atomic_writers_never_collide_or_publish_partial_toml() {
        const WRITERS: usize = 12;

        with_temp_config_dir(|dir| {
            let path = dir.join("concurrent.toml");
            let candidates = (0..WRITERS)
                .map(|writer| {
                    format!(
                        "writer = {writer}\npayload = \"{}\"\n",
                        char::from(b'a' + writer as u8)
                            .to_string()
                            .repeat(128 * 1024)
                    )
                })
                .collect::<Vec<_>>();
            let barrier = std::sync::Arc::new(std::sync::Barrier::new(WRITERS));
            let handles = candidates
                .iter()
                .cloned()
                .map(|candidate| {
                    let path = path.clone();
                    let barrier = std::sync::Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        write_atomic(&path, &candidate)
                    })
                })
                .collect::<Vec<_>>();

            for handle in handles {
                handle.join().unwrap().unwrap();
            }

            let published = std::fs::read_to_string(&path).unwrap();
            let parsed: toml::Value = toml::from_str(&published).unwrap();
            assert!(parsed.get("writer").is_some());
            assert!(candidates.iter().any(|candidate| candidate == &published));
            assert_no_atomic_temps(dir, "concurrent.toml");
        });
    }

    #[test]
    fn failed_atomic_rename_cleans_up_its_unique_temp_file() {
        with_temp_config_dir(|dir| {
            let target = dir.join("destination.toml");
            std::fs::create_dir(&target).unwrap();

            assert!(write_atomic(&target, "value = 1\n").is_err());
            assert_no_atomic_temps(dir, "destination.toml");
        });
    }

    #[test]
    fn post_rename_directory_sync_failure_is_still_a_committed_write() {
        with_temp_config_dir(|dir| {
            let target = dir.join("committed.toml");
            let tmp = dir.join(".committed.toml.test-tmp");
            std::fs::write(&tmp, "value = 2\n").unwrap();

            publish_atomic_file_with_sync(&target, &tmp, |_| {
                Err(std::io::Error::other("injected directory sync failure"))
            })
            .expect("rename already committed the new file");

            assert_eq!(std::fs::read_to_string(target).unwrap(), "value = 2\n");
            assert!(!tmp.exists());
        });
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
            let error = secrets::set("test_secret", "rotated-value").unwrap_err();
            assert!(matches!(
                error,
                secrets::SecretError::EnvironmentOverride { name, variable }
                    if name == "test_secret" && variable == "TELLM_TEST_SECRET"
            ));
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

    fn assert_no_atomic_temps(dir: &std::path::Path, file_name: &str) {
        let prefix = format!(".{file_name}.tmp-");
        let leftovers = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .filter(|name| name.to_string_lossy().starts_with(&prefix))
            .collect::<Vec<_>>();
        assert!(leftovers.is_empty(), "leftover temp files: {leftovers:?}");
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
