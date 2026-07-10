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
    }
}

// Checked 2026-07-04 against platform.claude.com model docs,
// developers.openai.com API model docs, and docs.x.ai model docs.
// Checked 2026-07-09 against dev.meta.ai Model API docs for Muse Spark.
// Checked 2026-07-09 against docs.x.ai model docs for Grok 4.5.
// Checked 2026-07-09 against developers.openai.com latest-model guide for
// GPT-5.6 Sol (Responses API; reasoning.effort low/medium/high/xhigh/max).
const PROVIDER_PRESETS: &[ProviderPreset] = &[
    ProviderPreset {
        key: "anthropic",
        label: "Anthropic Claude Fable 5",
        wire_format: WireFormat::Anthropic,
        model_name: "claude-fable-5",
        base_url: None,
        api_key_secret: secrets::ANTHROPIC_API_KEY,
    },
    ProviderPreset {
        key: "openai",
        label: "OpenAI GPT-5.6 Sol",
        wire_format: WireFormat::Responses,
        model_name: "gpt-5.6-sol",
        base_url: None,
        api_key_secret: secrets::OPENAI_API_KEY,
    },
    ProviderPreset {
        key: "xai",
        label: "xAI Grok 4.5",
        wire_format: WireFormat::Responses,
        model_name: "grok-4.5",
        base_url: Some(tellm_openai::XAI_BASE_URL),
        api_key_secret: secrets::XAI_API_KEY,
    },
    ProviderPreset {
        key: "meta",
        label: "Meta Muse Spark 1.1",
        wire_format: WireFormat::Responses,
        model_name: "muse-spark-1.1",
        base_url: Some(tellm_openai::META_MODEL_API_BASE_URL),
        api_key_secret: secrets::META_MODEL_API_KEY,
    },
    ProviderPreset {
        key: "gemini",
        label: "Google Gemini 3.5 Flash",
        wire_format: WireFormat::Gemini,
        model_name: "gemini-3.5-flash",
        base_url: None,
        api_key_secret: secrets::GEMINI_API_KEY,
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
    tellm_config::save(&config)?;
    let telegram_destination = secrets::set_nonempty(secrets::TELEGRAM_BOT_TOKEN, &telegram_token)?
        .expect("prompt_required returned a non-empty Telegram token");
    let provider_destination = secrets::set_nonempty(preset.api_key_secret, &api_key)?
        .expect("prompt_required returned a non-empty provider key");

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
    input.read_line(&mut line)?;
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
        assert_eq!(model.model_name, "grok-4.5");
        assert_eq!(model.base_url.as_deref(), Some(tellm_openai::XAI_BASE_URL));
        assert_eq!(model.api_key_secret.as_deref(), Some(secrets::XAI_API_KEY));
        assert!(model.telegram_chat_ids.is_empty());
    }
}
