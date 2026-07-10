use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use tellm_config::Config;

pub const PAIRING_CODE_TTL: Duration = Duration::from_secs(10 * 60);
pub const LOCKOUT_WINDOW: Duration = Duration::from_secs(5 * 60);
pub const LOCKOUT_DURATION: Duration = Duration::from_secs(5 * 60);
pub const MAX_PAIRING_ATTEMPTS: u8 = 5;
pub const SHUTDOWN_STALE_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AccessTime {
    seconds: u64,
}

impl AccessTime {
    pub fn from_unix_seconds(seconds: u64) -> Self {
        Self { seconds }
    }

    pub fn as_unix_seconds(self) -> u64 {
        self.seconds
    }

    pub fn saturating_add(self, duration: Duration) -> Self {
        Self {
            seconds: self.seconds.saturating_add(duration.as_secs()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AccessConfig {
    pub allowed_chat_ids: BTreeSet<i64>,
    pub owner_user_ids: BTreeSet<i64>,
    pub pinned_chat_ids: BTreeSet<i64>,
}

impl AccessConfig {
    pub fn from_config(config: &Config) -> Self {
        let pinned_chat_ids = config
            .models
            .values()
            .flat_map(|model| model.telegram_chat_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        Self {
            allowed_chat_ids: config.telegram.allowed_chat_ids.iter().copied().collect(),
            owner_user_ids: config.telegram.owner_user_ids.iter().copied().collect(),
            pinned_chat_ids,
        }
    }

    fn effective_allowed(&self) -> BTreeSet<i64> {
        self.allowed_chat_ids
            .union(&self.pinned_chat_ids)
            .copied()
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartupNotice {
    /// No chats are allowed yet; per-room pairing codes are issued on first
    /// contact (message or being added to a group).
    PairingMode,
    Restricted {
        allowed_chat_count: usize,
    },
}

/// The pairing code armed for one room. Pairing is per-room and re-armable;
/// approving one chat never disables pairing for others.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomPairing {
    pub code: String,
    /// True when this call issued or rotated the code — the runtime prints
    /// it to the console only then, to avoid console spam per message.
    pub newly_issued: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatAccess {
    Allowed,
    Unknown { send_hint: bool },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingAttempt {
    Paired,
    AlreadyAllowed,
    Rejected { attempts_remaining: u8 },
    LockedOut { until: AccessTime },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShutdownAccess {
    Allowed,
    NotAdmin,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PairingState {
    code: String,
    expires_at: AccessTime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AttemptState {
    window_started_at: AccessTime,
    count: u8,
    locked_until: Option<AccessTime>,
}

pub struct AccessControl {
    allowed_chat_ids: BTreeSet<i64>,
    owner_user_ids: BTreeSet<i64>,
    pending: BTreeMap<i64, PairingState>,
    attempts: BTreeMap<i64, AttemptState>,
    hinted_unknown_chats: BTreeSet<i64>,
    code_generator: Box<dyn FnMut() -> String + Send>,
}

impl AccessControl {
    pub fn new(config: AccessConfig, _now: AccessTime) -> Self {
        Self::new_with_generator(config, secure_pairing_code)
    }

    pub fn new_with_generator(
        config: AccessConfig,
        code_generator: impl FnMut() -> String + Send + 'static,
    ) -> Self {
        Self {
            allowed_chat_ids: config.effective_allowed(),
            owner_user_ids: config.owner_user_ids,
            pending: BTreeMap::new(),
            attempts: BTreeMap::new(),
            hinted_unknown_chats: BTreeSet::new(),
            code_generator: Box::new(code_generator),
        }
    }

    pub fn startup_notice(&self) -> StartupNotice {
        if self.allowed_chat_ids.is_empty() {
            StartupNotice::PairingMode
        } else {
            StartupNotice::Restricted {
                allowed_chat_count: self.allowed_chat_ids.len(),
            }
        }
    }

    /// Arm (or refresh) the pairing code for an unknown room. Returns None
    /// for already-allowed chats. Codes rotate on expiry; `newly_issued`
    /// tells the runtime when to print the code to the console.
    pub fn arm_room(&mut self, chat_id: i64, now: AccessTime) -> Option<RoomPairing> {
        if self.allowed_chat_ids.contains(&chat_id) {
            return None;
        }
        let (code, newly_issued) = self.ensure_room_code(chat_id, now);
        Some(RoomPairing { code, newly_issued })
    }

    pub fn check_chat(&mut self, chat_id: i64) -> ChatAccess {
        if self.allowed_chat_ids.contains(&chat_id) {
            return ChatAccess::Allowed;
        }

        ChatAccess::Unknown {
            send_hint: self.hinted_unknown_chats.insert(chat_id),
        }
    }

    /// Non-mutating access check for queued/in-flight work. Unlike
    /// `check_chat`, this must not consume the one-time unknown-chat hint.
    pub fn is_chat_allowed(&self, chat_id: i64) -> bool {
        self.allowed_chat_ids.contains(&chat_id)
    }

    pub fn allow_chat(&mut self, chat_id: i64) {
        self.allowed_chat_ids.insert(chat_id);
        self.hinted_unknown_chats.remove(&chat_id);
    }

    pub fn deny_chat(&mut self, chat_id: i64) {
        self.allowed_chat_ids.remove(&chat_id);
        self.hinted_unknown_chats.remove(&chat_id);
        self.attempts.remove(&chat_id);
        self.pending.remove(&chat_id);
    }

    /// Register an owner USER at runtime (code pairing proves console
    /// access). Owners are people, not chats: privileged commands work for
    /// them from any chat, and /deny can never strand the bot.
    pub fn add_owner(&mut self, user_id: i64) {
        self.owner_user_ids.insert(user_id);
    }

    #[cfg(test)]
    pub fn is_owner(&self, user_id: i64) -> bool {
        self.owner_user_ids.contains(&user_id)
    }

    pub fn attempt_pair(
        &mut self,
        chat_id: i64,
        submitted_code: &str,
        now: AccessTime,
    ) -> PairingAttempt {
        if self.allowed_chat_ids.contains(&chat_id) {
            return PairingAttempt::AlreadyAllowed;
        }

        if let Some(until) = self.locked_until(chat_id, now) {
            return PairingAttempt::LockedOut { until };
        }

        let (expected, _) = self.ensure_room_code(chat_id, now);
        if constant_time_eq(expected.as_bytes(), submitted_code.as_bytes()) {
            self.allowed_chat_ids.insert(chat_id);
            self.attempts.remove(&chat_id);
            self.hinted_unknown_chats.remove(&chat_id);
            self.pending.remove(&chat_id);
            return PairingAttempt::Paired;
        }

        self.record_failed_attempt(chat_id, now)
    }

    /// Gate for privileged commands (/allow, /deny, /shutdown, /model
    /// pin|unpin|add): the sender must be a registered owner user, and the
    /// message must be fresh so replayed updates cannot act.
    pub fn check_privileged(
        &self,
        sender_user_id: Option<i64>,
        message_unix_seconds: u64,
        now: AccessTime,
    ) -> ShutdownAccess {
        if !sender_user_id.is_some_and(|user_id| self.owner_user_ids.contains(&user_id)) {
            return ShutdownAccess::NotAdmin;
        }
        if now.as_unix_seconds().saturating_sub(message_unix_seconds)
            > SHUTDOWN_STALE_AFTER.as_secs()
        {
            return ShutdownAccess::Stale;
        }
        ShutdownAccess::Allowed
    }

    /// The currently armed code for a room, if any (does not rotate).
    pub fn room_code(&self, chat_id: i64) -> Option<&str> {
        self.pending
            .get(&chat_id)
            .map(|pairing| pairing.code.as_str())
    }

    /// Get the room's current code, issuing or rotating it when missing or
    /// expired. Returns (code, newly_issued).
    fn ensure_room_code(&mut self, chat_id: i64, now: AccessTime) -> (String, bool) {
        let needs_issue = self
            .pending
            .get(&chat_id)
            .is_none_or(|pairing| now >= pairing.expires_at);
        if needs_issue {
            let code = normalize_pairing_code((self.code_generator)());
            self.pending.insert(
                chat_id,
                PairingState {
                    code,
                    expires_at: now.saturating_add(PAIRING_CODE_TTL),
                },
            );
        }
        let code = self
            .pending
            .get(&chat_id)
            .expect("code just ensured")
            .code
            .clone();
        (code, needs_issue)
    }

    fn locked_until(&mut self, chat_id: i64, now: AccessTime) -> Option<AccessTime> {
        let attempt = self.attempts.get_mut(&chat_id)?;
        match attempt.locked_until {
            Some(until) if now < until => Some(until),
            Some(_) => {
                self.attempts.remove(&chat_id);
                None
            }
            None => None,
        }
    }

    fn record_failed_attempt(&mut self, chat_id: i64, now: AccessTime) -> PairingAttempt {
        let attempt = self.attempts.entry(chat_id).or_insert(AttemptState {
            window_started_at: now,
            count: 0,
            locked_until: None,
        });
        if now
            .as_unix_seconds()
            .saturating_sub(attempt.window_started_at.as_unix_seconds())
            >= LOCKOUT_WINDOW.as_secs()
        {
            attempt.window_started_at = now;
            attempt.count = 0;
            attempt.locked_until = None;
        }

        attempt.count = attempt.count.saturating_add(1);
        if attempt.count >= MAX_PAIRING_ATTEMPTS {
            let until = now.saturating_add(LOCKOUT_DURATION);
            attempt.locked_until = Some(until);
            return PairingAttempt::LockedOut { until };
        }

        PairingAttempt::Rejected {
            attempts_remaining: MAX_PAIRING_ATTEMPTS - attempt.count,
        }
    }
}

fn normalize_pairing_code(code: String) -> String {
    let digits = code
        .chars()
        .filter(char::is_ascii_digit)
        .take(6)
        .collect::<String>();
    if digits.len() == 6 {
        digits
    } else {
        format!(
            "{:06}",
            digits.parse::<u32>().unwrap_or_default() % 1_000_000
        )
    }
}

fn secure_pairing_code() -> String {
    let mut bytes = [0_u8; 4];
    let max = u32::MAX - (u32::MAX % 1_000_000);
    loop {
        getrandom::fill(&mut bytes).expect("secure random pairing code");
        let value = u32::from_le_bytes(bytes);
        if value < max {
            return format!("{:06}", value % 1_000_000);
        }
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left_byte = left.get(index).copied().unwrap_or_default();
        let right_byte = right.get(index).copied().unwrap_or_default();
        diff |= usize::from(left_byte ^ right_byte);
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tellm_config::{Config, ModelConfig, TelegramConfig, WireFormat};
    use tellm_core::ThinkingLevel;

    fn model(wire_format: WireFormat, base_url: Option<&str>, chat_ids: &[i64]) -> ModelConfig {
        ModelConfig {
            wire_format,
            model_name: "m".into(),
            base_url: base_url.map(Into::into),
            allow_insecure_http: false,
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

    #[test]
    fn empty_access_config_enters_pairing_mode_and_arms_per_room() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456", "654321"]);

        assert_eq!(access.startup_notice(), StartupNotice::PairingMode);

        let first = access.arm_room(7, time(100)).unwrap();
        assert_eq!(first.code, "123456");
        assert!(first.newly_issued);

        // Re-arming before expiry keeps the code and doesn't reprint.
        let again = access.arm_room(7, time(200)).unwrap();
        assert_eq!(again.code, "123456");
        assert!(!again.newly_issued);

        // A different room gets its own code.
        let other = access.arm_room(8, time(200)).unwrap();
        assert_eq!(other.code, "654321");
        assert!(other.newly_issued);
        assert_eq!(access.room_code(7), Some("123456"));
        assert_eq!(access.room_code(8), Some("654321"));
    }

    #[test]
    fn pairing_stays_available_when_allowlist_is_nonempty() {
        let mut config = AccessConfig::default();
        config.allowed_chat_ids.insert(10);
        let mut access = access_with_codes(config, &["123456"]);

        assert_eq!(
            access.startup_notice(),
            StartupNotice::Restricted {
                allowed_chat_count: 1,
            }
        );
        assert_eq!(access.check_chat(10), ChatAccess::Allowed);
        assert_eq!(
            access.check_chat(99),
            ChatAccess::Unknown { send_hint: true }
        );
        assert_eq!(
            access.check_chat(99),
            ChatAccess::Unknown { send_hint: false }
        );
        // Approving new rooms never disables — the unknown chat pairs.
        let pairing = access.arm_room(99, time(1)).unwrap();
        assert_eq!(
            access.attempt_pair(99, &pairing.code, time(2)),
            PairingAttempt::Paired
        );
        assert_eq!(access.room_code(99), None, "pending cleared on success");
        // Arming an allowed room is a no-op.
        assert!(access.arm_room(99, time(3)).is_none());
    }

    #[test]
    fn non_mutating_access_check_does_not_consume_unknown_chat_hint() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456"]);

        assert!(!access.is_chat_allowed(99));
        assert_eq!(
            access.check_chat(99),
            ChatAccess::Unknown { send_hint: true }
        );
    }

    #[test]
    fn pinned_chats_count_as_allowed() {
        let c = config(
            &[("claude", model(WireFormat::Anthropic, None, &[42]))],
            "claude",
        );
        let config = AccessConfig::from_config(&c);
        let mut access = access_with_codes(config, &["123456"]);

        assert_eq!(
            access.startup_notice(),
            StartupNotice::Restricted {
                allowed_chat_count: 1,
            }
        );
        assert_eq!(access.check_chat(42), ChatAccess::Allowed);
    }

    #[test]
    fn allow_and_deny_chat_take_effect_immediately() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456"]);

        assert!(matches!(
            access.check_chat(42),
            ChatAccess::Unknown { send_hint: true }
        ));
        access.allow_chat(42);
        assert_eq!(access.check_chat(42), ChatAccess::Allowed);
        access.deny_chat(42);
        assert!(matches!(
            access.check_chat(42),
            ChatAccess::Unknown { send_hint: true }
        ));
    }

    #[test]
    fn correct_pairing_code_allows_chat_and_clears_hint() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456"]);

        assert_eq!(
            access.check_chat(77),
            ChatAccess::Unknown { send_hint: true }
        );
        assert_eq!(
            access.attempt_pair(77, "123456", time(1)),
            PairingAttempt::Paired
        );
        assert_eq!(access.check_chat(77), ChatAccess::Allowed);
        assert_eq!(
            access.attempt_pair(77, "123456", time(2)),
            PairingAttempt::AlreadyAllowed
        );
    }

    #[test]
    fn wrong_pairing_attempts_lock_per_chat_then_recover() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456", "777777"]);

        for attempt in 1..5 {
            assert_eq!(
                access.attempt_pair(77, "000000", time(attempt)),
                PairingAttempt::Rejected {
                    attempts_remaining: 5 - attempt as u8,
                }
            );
        }
        assert_eq!(
            access.attempt_pair(77, "000000", time(5)),
            PairingAttempt::LockedOut { until: time(305) }
        );
        assert_eq!(
            access.attempt_pair(77, "123456", time(304)),
            PairingAttempt::LockedOut { until: time(305) }
        );

        // Lockouts are per chat: another room pairs with its own code while
        // 77 is locked out.
        let other = access.arm_room(88, time(304)).unwrap();
        assert_eq!(other.code, "777777");
        assert_eq!(
            access.attempt_pair(88, "777777", time(304)),
            PairingAttempt::Paired
        );

        assert_eq!(
            access.attempt_pair(77, "123456", time(305)),
            PairingAttempt::Paired
        );
    }

    #[test]
    fn room_pairing_code_rotates_after_ten_minutes() {
        let mut access = access_with_codes(AccessConfig::default(), &["111111", "222222"]);

        let armed = access.arm_room(88, time(100)).unwrap();
        assert_eq!(armed.code, "111111");

        // Attempt after expiry: the code rotates first, so the stale code is
        // rejected and the new one is available for the console to reprint.
        assert_eq!(
            access.attempt_pair(88, "111111", time(700)),
            PairingAttempt::Rejected {
                attempts_remaining: 4,
            }
        );
        assert_eq!(access.room_code(88), Some("222222"));
        assert_eq!(
            access.attempt_pair(88, "222222", time(701)),
            PairingAttempt::Paired
        );
    }

    #[test]
    fn pairing_compare_rejects_partial_code() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456"]);

        assert_eq!(
            access.attempt_pair(77, "12345", time(1)),
            PairingAttempt::Rejected {
                attempts_remaining: 4,
            }
        );
    }

    #[test]
    fn privileged_commands_require_owner_user_and_fresh_timestamp() {
        let mut config = AccessConfig::default();
        config.owner_user_ids.insert(1);
        let access = access_with_codes(config, &["123456"]);

        assert_eq!(
            access.check_privileged(Some(2), 100, time(100)),
            ShutdownAccess::NotAdmin
        );
        assert_eq!(
            access.check_privileged(None, 100, time(100)),
            ShutdownAccess::NotAdmin,
            "anonymous senders (no from) are never privileged"
        );
        assert_eq!(
            access.check_privileged(Some(1), 39, time(100)),
            ShutdownAccess::Stale
        );
        assert_eq!(
            access.check_privileged(Some(1), 40, time(100)),
            ShutdownAccess::Allowed
        );
    }

    #[test]
    fn runtime_owner_registration_applies_to_live_set() {
        let mut access = access_with_codes(AccessConfig::default(), &["123456"]);

        // P1 lesson: promotion must take effect without a restart.
        assert_eq!(
            access.check_privileged(Some(7), 100, time(100)),
            ShutdownAccess::NotAdmin
        );
        access.add_owner(7);
        assert!(access.is_owner(7));
        assert_eq!(
            access.check_privileged(Some(7), 100, time(100)),
            ShutdownAccess::Allowed
        );

        // Owners are people, not chats: denying a chat does not revoke
        // ownership (and can therefore never strand the bot — P2 lesson).
        access.deny_chat(7);
        assert_eq!(
            access.check_privileged(Some(7), 100, time(100)),
            ShutdownAccess::Allowed
        );
    }

    fn access_with_codes(config: AccessConfig, codes: &[&str]) -> AccessControl {
        let mut codes = std::collections::VecDeque::from(
            codes
                .iter()
                .map(|code| code.to_string())
                .collect::<Vec<_>>(),
        );
        AccessControl::new_with_generator(config, move || {
            codes.pop_front().unwrap_or_else(|| "999999".to_string())
        })
    }

    fn time(seconds: u64) -> AccessTime {
        AccessTime::from_unix_seconds(seconds)
    }
}
