//! The owner's curated memory: a short list of facts that tellm adds to the
//! system prompt when a conversation starts, so they don't have to be
//! repeated in every new chat.
//!
//! The file is plain text the owner can also edit by hand: one fact per line,
//! an optional `- ` or `* ` bullet, and lines starting with `#` ignored. tellm
//! edits it line by line, so comments and layout survive `/remember` and
//! `/forget`.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use tokio::task::spawn_blocking;

pub const MEMORY_FILE: &str = "memory.md";
/// Characters of facts, bullets and comments not counted: roughly 500 tokens
/// added to every request in a room that uses memory.
pub const MAX_MEMORY_CHARS: usize = 2_000;

const NEW_FILE_HEADER: &str =
    "# tellm memory: one fact per line. Lines starting with # are ignored.";
const PROMPT_HEADER: &str = "Facts and preferences the user saved for you to remember:";
const APPLIES_LATER: &str = "It takes effect in your next chat (/new starts one).";

/// Serializes read-modify-write edits from different chats.
static EDITS: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Memory {
    /// The file's lines as written.
    lines: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Remembered {
    Saved { number: usize, used: usize },
    AlreadySaved { number: usize },
    Full { used: usize, adding: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Forgotten {
    Removed { fact: String },
    NoSuchFact { count: usize },
}

/// What a conversation starting now gets: the memory block for its system
/// prompt, and a notice when memory is on but can't be used.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ConversationMemory {
    pub prompt: Option<String>,
    pub notice: Option<String>,
}

impl Memory {
    pub fn parse(text: &str) -> Self {
        Self {
            lines: text.lines().map(str::to_string).collect(),
        }
    }

    pub fn render(&self) -> String {
        let mut text = self.lines.join("\n");
        if !text.is_empty() {
            text.push('\n');
        }
        text
    }

    pub fn facts(&self) -> Vec<&str> {
        self.lines
            .iter()
            .filter_map(|line| fact_text(line))
            .collect()
    }

    /// Characters of facts, the measure the size limit applies to.
    pub fn used_chars(&self) -> usize {
        self.facts().iter().map(|fact| fact.chars().count()).sum()
    }

    pub fn over_limit(&self) -> bool {
        self.used_chars() > MAX_MEMORY_CHARS
    }

    pub fn remember(&mut self, fact: &str) -> Remembered {
        let fact = normalize_fact(fact);
        let facts = self.facts();
        if let Some(index) = facts
            .iter()
            .position(|saved| saved.to_lowercase() == fact.to_lowercase())
        {
            return Remembered::AlreadySaved { number: index + 1 };
        }
        let used = self.used_chars();
        let adding = fact.chars().count();
        if used + adding > MAX_MEMORY_CHARS {
            return Remembered::Full { used, adding };
        }
        let number = facts.len() + 1;
        if self.lines.is_empty() {
            self.lines.push(NEW_FILE_HEADER.to_string());
        }
        self.lines.push(format!("- {fact}"));
        Remembered::Saved {
            number,
            used: used + adding,
        }
    }

    /// Removes fact `number`, counting from 1 as `/memory` lists them.
    pub fn forget(&mut self, number: usize) -> Forgotten {
        let line_index = self
            .lines
            .iter()
            .enumerate()
            .filter(|(_, line)| fact_text(line).is_some())
            .nth(number.wrapping_sub(1))
            .map(|(index, _)| index);
        match line_index {
            Some(index) => {
                let line = self.lines.remove(index);
                Forgotten::Removed {
                    fact: fact_text(&line).unwrap_or_default().to_string(),
                }
            }
            None => Forgotten::NoSuchFact {
                count: self.facts().len(),
            },
        }
    }

    /// The system-prompt block, or `None` when there are no facts.
    pub fn prompt_block(&self) -> Option<String> {
        let facts = self.facts();
        if facts.is_empty() {
            return None;
        }
        let mut block = PROMPT_HEADER.to_string();
        for fact in facts {
            block.push_str("\n- ");
            block.push_str(fact);
        }
        Some(block)
    }
}

fn fact_text(line: &str) -> Option<&str> {
    let line = line.trim();
    if line.starts_with('#') {
        return None;
    }
    let fact = match line {
        "-" | "*" => "",
        _ => line
            .strip_prefix("- ")
            .or_else(|| line.strip_prefix("* "))
            .unwrap_or(line)
            .trim(),
    };
    (!fact.is_empty()).then_some(fact)
}

/// A fact is one line: whitespace, line breaks included, collapses to single
/// spaces.
pub fn normalize_fact(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Memory is on by default only in an owner's direct chat with the bot, so
/// the owner's facts don't reach groups or other people's chats unless the
/// owner turns it on there. In the Bot API a private chat's id is the user's
/// id ("MTProto user IDs are equal to bot API user dialog IDs"; checked
/// 2026-11-18 against core.telegram.org/api/bots/ids).
pub fn enabled_in(setting: Option<bool>, chat_id: i64, owner_user_ids: &[i64]) -> bool {
    setting.unwrap_or_else(|| owner_user_ids.contains(&chat_id))
}

/// `loaded` is `None` when memory is off in the room.
pub fn for_new_conversation(loaded: Option<Result<Memory, String>>) -> ConversationMemory {
    match loaded {
        None => ConversationMemory::default(),
        Some(Ok(memory)) if memory.over_limit() => ConversationMemory {
            prompt: None,
            notice: Some(format!(
                "This chat runs without memory: {MEMORY_FILE} holds {} characters of facts, \
                 over the {} limit. Remove facts with /forget N or shorten the file.",
                with_commas(memory.used_chars()),
                with_commas(MAX_MEMORY_CHARS),
            )),
        },
        Some(Ok(memory)) => ConversationMemory {
            prompt: memory.prompt_block(),
            notice: None,
        },
        Some(Err(error)) => ConversationMemory {
            prompt: None,
            notice: Some(format!("This chat runs without memory: {error}")),
        },
    }
}

pub fn path() -> Result<PathBuf, String> {
    tellm_config::config_dir()
        .map(|dir| dir.join(MEMORY_FILE))
        .map_err(|error| error.to_string())
}

/// A missing file is an empty memory. A file that exists is tightened to
/// owner-only before it is used.
pub fn load_from(path: &Path) -> Result<Memory, String> {
    match fs::read_to_string(path) {
        Ok(text) => {
            tellm_config::restrict_to_owner(path)
                .map_err(|error| format!("couldn't make {} private: {error}", path.display()))?;
            Ok(Memory::parse(&text))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Memory::default()),
        Err(error) => Err(format!("couldn't read {}: {error}", path.display())),
    }
}

pub fn save_to(path: &Path, memory: &Memory) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("couldn't create {}: {error}", parent.display()))?;
    }
    tellm_config::write_atomic_private(path, &memory.render())
        .map_err(|error| format!("couldn't save {}: {error}", path.display()))
}

/// Load, edit and save under one lock; the file is written only if the edit
/// changed it.
pub fn edit_at<T>(path: &Path, edit: impl FnOnce(&mut Memory) -> T) -> Result<T, String> {
    let _edits = EDITS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let mut memory = load_from(path)?;
    let before = memory.clone();
    let result = edit(&mut memory);
    if memory != before {
        save_to(path, &memory)?;
    }
    Ok(result)
}

pub async fn load() -> Result<Memory, String> {
    spawn_blocking(|| load_from(&path()?))
        .await
        .map_err(|error| format!("memory read task failed: {error}"))?
}

pub async fn edit<T: Send + 'static>(
    edit: impl FnOnce(&mut Memory) -> T + Send + 'static,
) -> Result<T, String> {
    spawn_blocking(move || edit_at(&path()?, edit))
        .await
        .map_err(|error| format!("memory edit task failed: {error}"))?
}

pub fn status_reply(memory: &Result<Memory, String>, enabled_here: bool, path: &str) -> String {
    let memory = match memory {
        Ok(memory) => memory,
        Err(error) => return format!("Memory can't be read: {error}"),
    };
    let state = if enabled_here {
        "Memory is on in this chat"
    } else {
        "Memory is off in this chat (/memory on uses it here)"
    };
    let facts = memory.facts();
    if facts.is_empty() {
        return format!("{state}. It's empty: add a fact with /remember TEXT.\nFile: {path}");
    }

    let count = match facts.len() {
        1 => "1 fact".to_string(),
        count => format!("{count} facts"),
    };
    let mut reply = format!(
        "{state}: {count}, {} of {} characters.",
        with_commas(memory.used_chars()),
        with_commas(MAX_MEMORY_CHARS),
    );
    if memory.over_limit() {
        reply.push_str(" That's over the limit, so new chats don't use it.");
    }
    for (index, fact) in facts.iter().enumerate() {
        reply.push_str(&format!("\n{}. {fact}", index + 1));
    }
    reply.push_str(&format!(
        "\n/forget N removes one. Changes take effect in your next chat (/new).\nFile: {path}"
    ));
    reply
}

pub fn remember_reply(outcome: &Remembered, enabled_here: bool) -> String {
    match outcome {
        Remembered::Saved { number, used } => {
            let mut reply = format!(
                "Remembered as #{number} ({} of {} characters). {APPLIES_LATER}",
                with_commas(*used),
                with_commas(MAX_MEMORY_CHARS),
            );
            if !enabled_here {
                reply.push_str(" Memory is off in this chat; /memory on uses it here.");
            }
            reply
        }
        Remembered::AlreadySaved { number } => format!("Already remembered as #{number}."),
        Remembered::Full { used, adding } => format!(
            "Not saved: memory would hold {} of {} characters. Remove a fact with /forget N first.",
            with_commas(used + adding),
            with_commas(MAX_MEMORY_CHARS),
        ),
    }
}

pub fn forget_reply(number: usize, outcome: &Forgotten) -> String {
    match outcome {
        Forgotten::Removed { fact } => format!("Forgot #{number}: {fact}\n{APPLIES_LATER}"),
        Forgotten::NoSuchFact { count: 0 } => "Memory is empty.".to_string(),
        Forgotten::NoSuchFact { count: 1 } => {
            format!("There's no fact #{number}; memory has 1. /memory lists it.")
        }
        Forgotten::NoSuchFact { count } => {
            format!("There's no fact #{number}; memory has {count}. /memory lists them.")
        }
    }
}

pub fn toggle_reply(enabled: bool) -> String {
    let state = if enabled { "on" } else { "off" };
    format!("Memory set to {state} for this chat. {APPLIES_LATER}")
}

fn with_commas(value: usize) -> String {
    let digits = value.to_string();
    let mut text = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            text.push(',');
        }
        text.push(digit);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_skip_comments_blank_lines_and_bullets() {
        let memory = Memory::parse(
            "# About me\n\n- I am 41.\n* Two kids, 7 and 10.\nLive in Sofia.\n-\n  # indented note\n",
        );

        assert_eq!(
            memory.facts(),
            vec!["I am 41.", "Two kids, 7 and 10.", "Live in Sofia."]
        );
        assert_eq!(memory.used_chars(), 8 + 19 + 14);
    }

    #[test]
    fn remember_appends_a_bullet_and_keeps_the_owners_layout() {
        let mut memory = Memory::parse("# Family\n- Two kids\n\n# Work\n");

        let outcome = memory.remember("  Works   as an\narchitect ");

        assert_eq!(
            outcome,
            Remembered::Saved {
                number: 2,
                used: 9 + 20,
            }
        );
        assert_eq!(
            memory.render(),
            "# Family\n- Two kids\n\n# Work\n- Works as an architect\n"
        );
    }

    #[test]
    fn remember_starts_a_new_file_with_a_format_note() {
        let mut memory = Memory::default();

        memory.remember("I am 41.");

        assert_eq!(memory.render(), format!("{NEW_FILE_HEADER}\n- I am 41.\n"));
        assert_eq!(memory.facts(), vec!["I am 41."]);
    }

    #[test]
    fn a_fact_starting_with_a_hash_survives_a_reload() {
        let mut memory = Memory::default();
        memory.remember("#1 priority is family");

        let reloaded = Memory::parse(&memory.render());

        assert_eq!(reloaded.facts(), vec!["#1 priority is family"]);
    }

    #[test]
    fn remember_reports_duplicates_without_changing_the_file() {
        let mut memory = Memory::parse("- I am 41.\n");

        assert_eq!(
            memory.remember("i am 41."),
            Remembered::AlreadySaved { number: 1 }
        );
        assert_eq!(memory.render(), "- I am 41.\n");
    }

    #[test]
    fn remember_refuses_to_go_over_the_limit() {
        let mut memory = Memory::parse(&format!("- {}\n", "a".repeat(MAX_MEMORY_CHARS - 5)));
        let before = memory.clone();

        assert_eq!(
            memory.remember("123456"),
            Remembered::Full {
                used: MAX_MEMORY_CHARS - 5,
                adding: 6,
            }
        );
        assert_eq!(memory, before);
        assert!(matches!(
            memory.remember("12345"),
            Remembered::Saved { used, .. } if used == MAX_MEMORY_CHARS
        ));
    }

    #[test]
    fn forget_removes_the_numbered_fact_and_keeps_comments() {
        let mut memory = Memory::parse("# Family\n- Two kids\n# Home\n- Sofia\n- Has a dog\n");

        assert_eq!(
            memory.forget(2),
            Forgotten::Removed {
                fact: "Sofia".to_string(),
            }
        );
        assert_eq!(
            memory.render(),
            "# Family\n- Two kids\n# Home\n- Has a dog\n"
        );
        assert_eq!(memory.forget(3), Forgotten::NoSuchFact { count: 2 });
        assert_eq!(memory.forget(0), Forgotten::NoSuchFact { count: 2 });
    }

    #[test]
    fn prompt_block_lists_the_facts_under_a_header() {
        let memory = Memory::parse("# me\nI am 41.\n- Two kids\n");

        assert_eq!(
            memory.prompt_block().as_deref(),
            Some(
                "Facts and preferences the user saved for you to remember:\n- I am 41.\n- Two kids"
            )
        );
        assert_eq!(Memory::parse("# only a comment\n").prompt_block(), None);
    }

    #[test]
    fn memory_defaults_on_only_in_an_owners_direct_chat() {
        let owners = [7];

        assert!(enabled_in(None, 7, &owners));
        assert!(!enabled_in(None, -100_420, &owners));
        assert!(!enabled_in(None, 8, &owners));
        assert!(enabled_in(Some(true), -100_420, &owners));
        assert!(!enabled_in(Some(false), 7, &owners));
    }

    #[test]
    fn new_conversations_skip_unusable_memory_with_a_notice() {
        assert_eq!(for_new_conversation(None), ConversationMemory::default());

        let usable = for_new_conversation(Some(Ok(Memory::parse("- I am 41.\n"))));
        assert!(
            usable
                .prompt
                .is_some_and(|prompt| prompt.ends_with("- I am 41."))
        );
        assert_eq!(usable.notice, None);

        let empty = for_new_conversation(Some(Ok(Memory::default())));
        assert_eq!(empty, ConversationMemory::default());

        let long = Memory::parse(&format!("- {}\n", "a".repeat(MAX_MEMORY_CHARS + 1)));
        let over = for_new_conversation(Some(Ok(long)));
        assert_eq!(over.prompt, None);
        assert!(
            over.notice
                .is_some_and(|notice| notice.contains("2,001 characters"))
        );

        let unreadable = for_new_conversation(Some(Err("couldn't read memory.md".to_string())));
        assert_eq!(unreadable.prompt, None);
        assert_eq!(
            unreadable.notice.as_deref(),
            Some("This chat runs without memory: couldn't read memory.md")
        );
    }

    #[test]
    fn edits_save_owner_only_files_and_skip_unchanged_ones() {
        let dir = std::env::temp_dir().join(format!(
            "tellm-memory-test-{}-{:x}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        let path = dir.join(MEMORY_FILE);

        assert_eq!(load_from(&path), Ok(Memory::default()));
        assert_eq!(
            edit_at(&path, |memory| memory.forget(1)),
            Ok(Forgotten::NoSuchFact { count: 0 })
        );
        assert!(!path.exists(), "an unchanged memory is not written");

        let saved = edit_at(&path, |memory| memory.remember("I am 41."));
        assert!(matches!(saved, Ok(Remembered::Saved { number: 1, .. })));
        assert_eq!(
            load_from(&path).map(|memory| memory.facts().join("|")),
            Ok("I am 41.".to_string())
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn status_lists_numbered_facts_with_usage_and_the_file() {
        let memory = Ok(Memory::parse("- I am 41.\n- Two kids\n"));

        assert_eq!(
            status_reply(&memory, true, "/cfg/memory.md"),
            "Memory is on in this chat: 2 facts, 16 of 2,000 characters.\n\
             1. I am 41.\n2. Two kids\n\
             /forget N removes one. Changes take effect in your next chat (/new).\n\
             File: /cfg/memory.md"
        );
        assert!(
            status_reply(&Ok(Memory::default()), false, "/cfg/memory.md")
                .starts_with("Memory is off in this chat (/memory on uses it here). It's empty")
        );
        assert_eq!(
            status_reply(&Err("denied".to_string()), true, "/cfg/memory.md"),
            "Memory can't be read: denied"
        );
    }

    #[test]
    fn replies_say_when_changes_take_effect() {
        assert_eq!(
            remember_reply(
                &Remembered::Saved {
                    number: 3,
                    used: 1_234
                },
                true
            ),
            "Remembered as #3 (1,234 of 2,000 characters). \
             It takes effect in your next chat (/new starts one)."
        );
        assert!(
            remember_reply(&Remembered::Saved { number: 1, used: 5 }, false)
                .ends_with("Memory is off in this chat; /memory on uses it here.")
        );
        assert_eq!(
            remember_reply(
                &Remembered::Full {
                    used: 1_990,
                    adding: 20
                },
                true
            ),
            "Not saved: memory would hold 2,010 of 2,000 characters. \
             Remove a fact with /forget N first."
        );
        assert_eq!(
            forget_reply(
                2,
                &Forgotten::Removed {
                    fact: "Sofia".to_string(),
                }
            ),
            "Forgot #2: Sofia\nIt takes effect in your next chat (/new starts one)."
        );
        assert_eq!(
            forget_reply(9, &Forgotten::NoSuchFact { count: 4 }),
            "There's no fact #9; memory has 4. /memory lists them."
        );
        assert_eq!(
            toggle_reply(false),
            "Memory set to off for this chat. It takes effect in your next chat (/new starts one)."
        );
    }

    #[test]
    fn counts_get_thousands_separators() {
        assert_eq!(with_commas(0), "0");
        assert_eq!(with_commas(999), "999");
        assert_eq!(with_commas(2_000), "2,000");
        assert_eq!(with_commas(1_234_567), "1,234,567");
    }
}
