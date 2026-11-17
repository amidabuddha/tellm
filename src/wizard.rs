use std::collections::BTreeMap;
use std::io::{BufRead, Write};

use tellm_config::{Config, ModelConfig, TelegramConfig, WireFormat, config_path, secrets};
use tellm_core::ThinkingLevel;
use tellm_telegram::Telegram;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProviderPreset {
    pub(crate) key: &'static str,
    pub(crate) label: &'static str,
    pub(crate) wire_format: WireFormat,
    pub(crate) model_name: &'static str,
    pub(crate) base_url: Option<&'static str>,
    pub(crate) api_key_secret: &'static str,
    /// Written to the model's `max_effort` config: the provider value for
    /// `thinking = "max"`. `None` leaves the wire format's default tier.
    pub(crate) max_effort: Option<&'static str>,
    /// Written to the model's `off_effort` config: the provider value for
    /// `thinking = "off"`. `None` sends no reasoning setting, so the
    /// provider's default applies.
    pub(crate) off_effort: Option<&'static str>,
}

/// The built-in provider catalog, shared with /model add.
pub(crate) fn provider_presets() -> &'static [ProviderPreset] {
    PROVIDER_PRESETS
}

pub(crate) fn preset_by_key(key: &str) -> Option<&'static ProviderPreset> {
    PROVIDER_PRESETS
        .iter()
        .find(|preset| preset.key.eq_ignore_ascii_case(key))
}

pub(crate) fn model_config_from_preset(preset: &ProviderPreset) -> ModelConfig {
    ModelConfig {
        wire_format: preset.wire_format,
        model_name: preset.model_name.to_string(),
        base_url: preset.base_url.map(str::to_string),
        allow_insecure_http: false,
        api_key_secret: Some(preset.api_key_secret.to_string()),
        telegram_chat_ids: Vec::new(),
        thinking: ThinkingLevel::default(),
        max_effort: preset.max_effort.map(str::to_string),
        off_effort: preset.off_effort.map(str::to_string),
        max_output_tokens: None,
    }
}

// Each preset is its lab's most capable model that an ordinary API key can
// call; that is what tellm is for. All checked 2026-11-16:
// - platform.claude.com model docs: Claude Fable 5.2 uses ID claude-fable-5-2
//   and is Anthropic's most capable generally available model (Claude Mythos
//   5.2 is limited to verified organizations). Thinking is always adaptive,
//   and effort accepts low/medium/high/xhigh/max.
// - developers.openai.com API model docs: GPT-6 Astra uses ID gpt-6-astra, is
//   OpenAI's most capable model, supports Responses with web search and
//   image generation, and accepts reasoning effort low/medium/high/xhigh/max.
// - docs.x.ai model docs: Grok 4.7 uses ID grok-4.7 and is the flagship; it
//   accepts reasoning effort low/medium/high/xhigh and never max.
// - dev.meta.ai Model API docs: Muse Wonder uses ID muse-wonder-1.0, is
//   Meta's most capable model, is in preview on Model API, and accepts
//   reasoning effort from minimal to max (rechecked 2026-11-17: "none" is
//   an HTTP 400, and the default when no effort is sent is max).
// - ai.google.dev model docs: Gemini 4 Argon uses ID gemini-4-argon and is
//   Google's flagship; thinking level accepts low/medium/high and rejects
//   minimal.
//
// None of the five can turn reasoning off, and without an effort each one
// reasons at its own default, which is the top tier on Muse Wonder. So each
// preset writes its model's lowest tier as `off_effort`, and `off` is the
// cheapest setting instead of the provider's default.
const PROVIDER_PRESETS: &[ProviderPreset] = &[
    ProviderPreset {
        key: "anthropic",
        label: "Anthropic Claude Fable 5.2",
        wire_format: WireFormat::Anthropic,
        model_name: "claude-fable-5-2",
        base_url: None,
        api_key_secret: secrets::ANTHROPIC_API_KEY,
        max_effort: None,
        off_effort: Some("low"),
    },
    ProviderPreset {
        key: "openai",
        label: "OpenAI GPT-6 Astra",
        wire_format: WireFormat::Responses,
        model_name: "gpt-6-astra",
        base_url: None,
        api_key_secret: secrets::OPENAI_API_KEY,
        max_effort: Some("max"),
        off_effort: Some("low"),
    },
    ProviderPreset {
        key: "xai",
        label: "xAI Grok 4.7",
        wire_format: WireFormat::Responses,
        model_name: "grok-4.7",
        base_url: Some(tellm_openai::XAI_BASE_URL),
        api_key_secret: secrets::XAI_API_KEY,
        max_effort: None,
        off_effort: Some("low"),
    },
    ProviderPreset {
        key: "meta",
        label: "Meta Muse Wonder 1.0",
        wire_format: WireFormat::Responses,
        model_name: "muse-wonder-1.0",
        base_url: Some(tellm_openai::META_MODEL_API_BASE_URL),
        api_key_secret: secrets::META_MODEL_API_KEY,
        max_effort: Some("max"),
        off_effort: Some("minimal"),
    },
    ProviderPreset {
        key: "gemini",
        label: "Google Gemini 4 Argon",
        wire_format: WireFormat::Gemini,
        model_name: "gemini-4-argon",
        base_url: None,
        api_key_secret: secrets::GEMINI_API_KEY,
        max_effort: None,
        off_effort: Some("low"),
    },
];

pub async fn run_first_run(
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    writeln!(output, "tellm first-run setup")?;
    writeln!(
        output,
        "Create a Telegram bot with BotFather, then paste its token here."
    )?;
    let telegram_token = prompt_required(input, output, "Telegram bot token: ")?;

    let bot = Telegram::new(telegram_token.clone()).get_me().await?;
    if !bot.is_bot {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Telegram token did not resolve to a bot account",
        )
        .into());
    }
    match bot.username {
        Some(username) => writeln!(output, "Validated Telegram bot @{username}.")?,
        None => writeln!(output, "Validated Telegram bot {}.", bot.first_name)?,
    }

    writeln!(output)?;
    writeln!(output, "Choose one provider to configure now:")?;
    for (idx, preset) in PROVIDER_PRESETS.iter().enumerate() {
        writeln!(
            output,
            "  {}. {} ({})",
            idx + 1,
            preset.label,
            preset.model_name
        )?;
    }
    let preset = prompt_provider(input, output)?;
    let api_key = prompt_required(input, output, &format!("{} API key: ", preset.label))?;

    let config = config_for_preset(preset);
    let telegram_destination = secrets::set_nonempty(secrets::TELEGRAM_BOT_TOKEN, &telegram_token)?
        .expect("prompt_required returned a non-empty Telegram token");
    let provider_destination = secrets::set_nonempty(preset.api_key_secret, &api_key)?
        .expect("prompt_required returned a non-empty provider key");
    // A config file marks setup complete, so publish it only after every
    // referenced secret has been stored successfully.
    tellm_config::save(&config)?;

    writeln!(output)?;
    writeln!(output, "Config saved to {}.", config_path()?.display())?;
    writeln!(
        output,
        "Telegram bot token {}.",
        telegram_destination.status_message()
    )?;
    writeln!(
        output,
        "{} API key {}.",
        preset.label,
        provider_destination.status_message()
    )?;
    writeln!(
        output,
        "Pairing mode is enabled until a Telegram chat claims the bot with /pair CODE."
    )?;
    writeln!(
        output,
        "When the runtime prints a pairing code, send that command to your bot."
    )?;
    output.flush()?;
    Ok(())
}

fn prompt_provider(
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> std::io::Result<&'static ProviderPreset> {
    loop {
        let raw = prompt_optional(input, output, "Provider [1]: ")?;
        if let Some(preset) = provider_from_choice(&raw) {
            return Ok(preset);
        }
        writeln!(
            output,
            "Enter a number from 1 to {}.",
            PROVIDER_PRESETS.len()
        )?;
    }
}

fn prompt_required(
    input: &mut impl BufRead,
    output: &mut impl Write,
    prompt: &str,
) -> std::io::Result<String> {
    loop {
        let value = prompt_optional(input, output, prompt)?;
        if !value.is_empty() {
            return Ok(value);
        }
        writeln!(output, "This value is required.")?;
    }
}

fn prompt_optional(
    input: &mut impl BufRead,
    output: &mut impl Write,
    prompt: &str,
) -> std::io::Result<String> {
    write!(output, "{prompt}")?;
    output.flush()?;
    let mut line = String::new();
    if input.read_line(&mut line)? == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "first-run input ended before setup was complete",
        ));
    }
    Ok(line.trim().to_string())
}

fn provider_from_choice(choice: &str) -> Option<&'static ProviderPreset> {
    let choice = choice.trim();
    if choice.is_empty() {
        return PROVIDER_PRESETS.first();
    }
    let index = choice.parse::<usize>().ok()?.checked_sub(1)?;
    PROVIDER_PRESETS.get(index)
}

fn config_for_preset(preset: &ProviderPreset) -> Config {
    let mut models = BTreeMap::new();
    models.insert(preset.key.to_string(), model_config_from_preset(preset));

    Config {
        default_model: preset.key.to_string(),
        models,
        telegram: TelegramConfig::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_provider_choice_defaults_to_first_preset() {
        assert_eq!(
            provider_from_choice("").map(|preset| preset.key),
            Some("anthropic")
        );
    }

    #[test]
    fn provider_choice_uses_one_based_numbers() {
        assert_eq!(
            provider_from_choice("2").map(|preset| preset.key),
            Some("openai")
        );
        assert!(provider_from_choice("0").is_none());
        assert!(provider_from_choice("99").is_none());
        assert!(provider_from_choice("openai").is_none());
    }

    #[test]
    fn preset_config_references_secret_name_without_secret_value() {
        let preset = provider_from_choice("3").unwrap();
        let config = config_for_preset(preset);
        let model = &config.models["xai"];

        assert_eq!(config.default_model, "xai");
        assert_eq!(model.wire_format, WireFormat::Responses);
        assert_eq!(model.model_name, "grok-4.7");
        assert_eq!(model.base_url.as_deref(), Some(tellm_openai::XAI_BASE_URL));
        assert_eq!(model.api_key_secret.as_deref(), Some(secrets::XAI_API_KEY));
        assert!(model.telegram_chat_ids.is_empty());
    }

    #[test]
    fn preset_max_effort_is_written_to_the_model_config() {
        let openai = model_config_from_preset(preset_by_key("openai").unwrap());
        let meta = model_config_from_preset(preset_by_key("meta").unwrap());
        let xai = model_config_from_preset(preset_by_key("xai").unwrap());

        // Astra and Wonder list "max"; Grok stops at the Responses default.
        assert_eq!(openai.max_effort.as_deref(), Some("max"));
        assert_eq!(meta.max_effort.as_deref(), Some("max"));
        assert_eq!(xai.max_effort, None);
    }

    #[test]
    fn presets_write_each_models_lowest_tier_as_off_effort() {
        // None of the preset models can turn reasoning off; Muse Wonder is
        // the one with a tier below "low".
        for (key, lowest) in [
            ("anthropic", "low"),
            ("openai", "low"),
            ("xai", "low"),
            ("meta", "minimal"),
            ("gemini", "low"),
        ] {
            let model = model_config_from_preset(preset_by_key(key).unwrap());

            assert_eq!(model.off_effort.as_deref(), Some(lowest), "{key}");
        }
    }

    #[test]
    fn presets_name_each_labs_most_capable_model() {
        for (key, label, model_name, wire_format) in [
            (
                "anthropic",
                "Anthropic Claude Fable 5.2",
                "claude-fable-5-2",
                WireFormat::Anthropic,
            ),
            (
                "openai",
                "OpenAI GPT-6 Astra",
                "gpt-6-astra",
                WireFormat::Responses,
            ),
            ("xai", "xAI Grok 4.7", "grok-4.7", WireFormat::Responses),
            (
                "meta",
                "Meta Muse Wonder 1.0",
                "muse-wonder-1.0",
                WireFormat::Responses,
            ),
            (
                "gemini",
                "Google Gemini 4 Argon",
                "gemini-4-argon",
                WireFormat::Gemini,
            ),
        ] {
            let preset = preset_by_key(key).unwrap();

            assert_eq!(preset.label, label);
            assert_eq!(preset.model_name, model_name);
            assert_eq!(preset.wire_format, wire_format);
        }
        assert_eq!(provider_presets().len(), 5);
    }

    #[test]
    fn required_prompt_returns_unexpected_eof_after_one_prompt() {
        let mut input = std::io::Cursor::new(Vec::<u8>::new());
        let mut output = Vec::new();

        let error = prompt_required(&mut input, &mut output, "Telegram bot token: ").unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(
            error.to_string(),
            "first-run input ended before setup was complete"
        );
        assert_eq!(output, b"Telegram bot token: ");
    }
}
