# tellm

**Your own multi-model AI assistant in Telegram. Bring your API keys.**

One small binary. No Docker, no Python, no config scavenger hunt. Chat with
Claude, GPT, Grok, Gemini, or your local Ollama models from any Telegram
client — usage billed directly by your API providers, not by a subscription.

> **Status: pre-alpha.** The design is in
> [DESIGN.md](DESIGN.md); the battle-tested reference implementation lives in
> [console-chat-gpt](https://github.com/amidabuddha/console-chat-gpt).

## Why

- **Pay per use, not per month.** Occasional access to frontier models costs
  cents via their APIs. tellm is the mobile-friendly front end for that.
- **Per-room model pinning.** One Telegram chat is your Opus room, another is
  your Grok room, a third runs your local Ollama model. Switch by switching
  chats.
- **Secure by default.** An unpaired bot answers to nobody: claim it with a
  one-time code printed in its terminal (`/pair 123456`), rotated every 10
  minutes, rate-limited per chat. Secrets live in your OS keychain, not in a
  dotfile.
- **Not an agent.** No shell access, no skills registry, no browser control,
  no MCP. The entire attack surface is HTTPS calls to Telegram and your model
  providers. If you want an agent, run OpenClaw — in a VM. If you want to
  *talk to models* from your pocket without handing anything the keys to your
  machine, that's tellm.

## Install From Source

```sh
git clone https://github.com/amidabuddha/tellm.git
cd tellm
cargo build --release
./target/release/tellm
```

The first run wizard asks for a Telegram bot token and one provider API key,
then prints a pairing code. Message your bot `/pair 123456` on Telegram. Done
— under two minutes, zero file editing.

## Supported APIs (direct, no middleware)

| Wire format | Providers |
|---|---|
| Anthropic Messages | Claude (with prompt caching + adaptive thinking) |
| OpenAI Responses | OpenAI, xAI (web search, image generation) |
| Chat completions | Ollama, DeepSeek, OpenRouter, any compatible endpoint |
| Google Interactions | Gemini |

## `config.toml` Example

`config.toml` contains routing and room policy only. Secret values stay in the
OS keychain or `credentials.toml`; `api_key_secret` is just the lookup name.

```toml
default_model = "claude"

[telegram]
allowed_chat_ids = []
owner_user_ids = []
max_concurrent_updates = 8

[models.claude]
wire_format = "anthropic"
model_name = "claude-fable-5"
api_key_secret = "anthropic_api_key"
telegram_chat_ids = []
thinking = "max"

[models.gpt]
wire_format = "responses"
model_name = "gpt-5.5"
api_key_secret = "openai_api_key"
telegram_chat_ids = []
thinking = "high"

[models.grok]
wire_format = "responses"
model_name = "grok-4.3"
base_url = "https://api.x.ai/v1"
api_key_secret = "xai_api_key"
telegram_chat_ids = []
thinking = "high"

[models.gemini]
wire_format = "gemini"
model_name = "gemini-3.5-flash"
api_key_secret = "gemini_api_key"
telegram_chat_ids = []
thinking = "high"

# Any OpenAI chat-completions-compatible paid endpoint, such as Mistral,
# DeepSeek, OpenRouter, or a proxy. base_url is required for compat models.
[models.remote_compat]
wire_format = "compat"
model_name = "provider-model-name"
base_url = "https://provider.example/v1"
api_key_secret = "remote_compat_api_key"
telegram_chat_ids = []
thinking = "medium"

# Local Ollama is also compat, but it is normally keyless. Omit
# api_key_secret entirely so tellm does not warm or prompt for a key.
[models.ollama]
wire_format = "compat"
model_name = "llama3.3:70b"
base_url = "http://localhost:11434/v1"
telegram_chat_ids = []
thinking = "off"
```

The four wiring shapes are:

- Built-in provider default endpoint: omit `base_url`, set `api_key_secret`.
- Provider variant on the same wire format: set `base_url` and `api_key_secret`
  (xAI through `responses` is the built-in example).
- Remote chat-completions-compatible endpoint: `wire_format = "compat"` with
  both `base_url` and `api_key_secret`.
- Local/keyless compatible endpoint: `wire_format = "compat"` with `base_url`
  and no `api_key_secret`, usually Ollama.

`telegram_chat_ids` is only for locked room pins. Normal `/model KEY`
selection is stored in `rooms.toml`, so these arrays usually stay empty.

## Setting Up Model Rooms

For one group per model, disable Telegram privacy mode via BotFather
(`/setprivacy`) once, then add the bot to each group from your owner account.
Owner-added rooms are approved automatically and get the model picker.

Use `/model KEY` for a room-local model selection. That persists in
`rooms.toml` and intentionally leaves `telegram_chat_ids` empty. Use `/model pin
KEY` only when you want a locked model room; that writes the chat id into the
chosen model's `telegram_chat_ids` in `config.toml`, and the room will always
use that model until `/model unpin`.

Use `/model add` to list built-in provider presets. `/model add KEY` writes a
built-in preset into `config.toml`; if its API key is missing, tellm asks for it
in the terminal so the key never goes through Telegram. For a custom model
already present in `config.toml`, `/model add KEY` uses that model's
`api_key_secret` and opens the same terminal prompt.

For local Ollama models using `base_url = "http://localhost:11434/v1"` (or
`127.0.0.1` / `[::1]` on port `11434`), tellm checks the local port before a
request. If Ollama is not running, it starts `ollama serve` and waits briefly
before dispatching the message. If tellm started that Ollama process, it stops
it again during normal tellm shutdown, first asking Ollama to unload the local
models tellm invoked so model runner processes do not stay resident in memory.
Ollama processes that were already running are left alone. Owners can also send
`/ollama unload` to unload local Ollama models used by the current tellm
session without stopping tellm or `ollama serve`.

At startup, tellm also reads every unique `api_key_secret` referenced by
configured models. This surfaces OS keychain permission prompts before the
first model call in the current session; missing keys are reported but do not
stop local/keyless models from running.

## Troubleshooting

### Group chats

If tellm works in direct messages but ignores plain text in a group, disable
privacy mode via BotFather (`/setprivacy`) and re-add the bot to the group.
The runtime logs one stderr breadcrumb per received update with chat id, update
kind, and route (`command`, `model`, or `ignored`), but never message content.

## License

MIT OR Apache-2.0, at your option.
