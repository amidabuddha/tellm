# tellm

**Your own multi-model AI assistant in Telegram. Bring your API keys.**

One small binary. No Docker, no Python, no config scavenger hunt. Chat with
Claude, GPT, Grok, Gemini, or your local Ollama models from any Telegram
client — usage billed directly by your API providers, not by a subscription.

> **Status: alpha.** Runs as the author's daily driver across every supported
> provider, but it hasn't been road-tested by anyone else yet — expect rough
> edges and the occasional breaking change. It grew out of
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
  no MCP. The whole attack surface is HTTPS calls to Telegram and your model
  providers — nothing on your machine to inject into or steal, beyond keys
  that live in your OS keychain.

## Why not OpenClaw?

OpenClaw and similar agent frameworks put an LLM in your messenger *and* give
it hands: shell, filesystem, a skills registry, browser control. That power
comes with a matching attack surface: exposed control planes, prompt-injection
paths to data exfiltration, and malicious entries in extension registries.

tellm is the deliberate opposite. It has no hands. It relays messages between
Telegram and your model providers and does nothing else: no shell to hijack,
no skill to poison, no control UI to leak a token. If you want an autonomous
agent, run one — in a VM. If you just want to *talk to models* from your phone
without handing anything the keys to your machine, that's tellm.

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
| OpenAI Responses | OpenAI, xAI, Meta Model API (web search; OpenAI image generation) |
| Chat completions | Ollama, DeepSeek, OpenRouter, any compatible endpoint |
| Google Interactions | Gemini (including image models) |

## Commands

Send these to the bot in any allowed chat. Owner-only commands are marked.

| Command | Does |
|---|---|
| `/help` | List commands |
| `/new` | Reset this chat's conversation |
| `/id` | Show this chat's Telegram id |
| `/mode chat\|message` | Multi-turn context, or one question per message |
| `/model [KEY]` | Show or switch this room's model |
| `/model add [KEY]` | List the provider catalog, or add a preset (owner) |
| `/model pin KEY` · `/model unpin` | Lock or release this room's model (owner) |
| `/role TEXT\|clear` | Set or clear the system prompt |
| `/reasoning default\|off\|low\|medium\|high\|max` | Thinking effort |
| `/websearch on\|off\|status` | Provider-native web search |
| `/imagegen on\|off\|status` | Image generation (capable models only) |
| `/pair CODE` | Claim an unpaired chat with the code from the terminal |
| `/allow CHAT_ID` · `/deny CHAT_ID` | Approve or revoke a chat (owner) |
| `/ollama unload` | Unload this session's local Ollama models (owner) |
| `/shutdown` | Stop tellm (owner) |

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

[models.meta]
wire_format = "responses"
model_name = "muse-spark-1.1"
base_url = "https://api.meta.ai/v1"
api_key_secret = "meta_model_api_key"
telegram_chat_ids = []
thinking = "max"

[models.gemini]
wire_format = "gemini"
model_name = "gemini-3.5-flash"
api_key_secret = "gemini_api_key"
telegram_chat_ids = []
thinking = "high"

# Gemini image generation requires a Gemini image model id, for example:
# model_name = "gemini-3.1-flash-image"

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
  (xAI and Meta Model API through `responses` are built-in examples).
- Remote chat-completions-compatible endpoint: `wire_format = "compat"` with
  both `base_url` and `api_key_secret`.
- Local/keyless compatible endpoint: `wire_format = "compat"` with `base_url`
  and no `api_key_secret`, usually Ollama.

`telegram_chat_ids` is only for locked room pins. Normal `/model KEY`
selection is stored in `rooms.toml`, so these arrays usually stay empty.
`thinking` in `config.toml` is the model default. A room only writes
`thinking` to `rooms.toml` after `/reasoning LEVEL`; `/reasoning default`
clears that room override and returns to the selected model's default.

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
`http://127.0.0.1:11434/v1` / `http://[::1]:11434/v1`), tellm checks the local
port before a request. If Ollama is not running, it starts `ollama serve` and
waits briefly before dispatching the message. If tellm started that Ollama
process, it stops it again during tellm shutdown, including terminal
`exit`/`quit`, Telegram `/shutdown`, SIGINT, SIGTERM, and ordinary panic
unwinds. Before stopping the child, tellm asks Ollama to unload the local models
that actually completed a request in this tellm session so model runner
processes do not stay resident in memory, then sends SIGTERM before falling
back to SIGKILL. Already-evicted models are forgotten instead of retried
forever. SIGKILL and power loss cannot run cleanup. Ollama processes that were
already running, including possible leftovers from a prior hard kill, are left
alone. Owners can also send
`/ollama unload` to unload local Ollama models used by the current tellm
session without stopping tellm or `ollama serve`.

That lifecycle management is intentionally narrow: it is inferred only for the
default local HTTP endpoint above. Non-default ports, `https://` proxies,
`0.0.0.0`/LAN hosts, and remote compat servers are treated as ordinary compat
endpoints and are neither auto-started nor unloaded.

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
