# tellm

**Your own multi-model AI assistant in Telegram. Bring your API keys.**

One small binary that lets you chat with Claude, GPT, Grok, Muse, Gemini or
local Ollama models from any Telegram client. Your API providers bill you
directly for what you use.

> **Status: alpha.** It is the author's daily driver, but nobody else has
> road-tested it yet: expect rough edges and the occasional breaking change.
> It grew out of [console-chat-gpt](https://github.com/amidabuddha/console-chat-gpt).

## Why

- **Pay per use.** Frontier models cost cents per question through their
  APIs, and every reply shows what it cost.
- **One room per model.** Each Telegram chat can use its own model, so you
  switch models by switching chats.
- **Private by default.** An unpaired bot answers nobody until you claim it
  with a one-time code from its terminal. API keys stay in your OS keychain
  (or an owner-only file on headless machines) and never pass through
  Telegram.
- **Memory you control.** Save facts about yourself once and every new chat
  knows them.

## Why not OpenClaw?

Agent frameworks give the model hands: shell, files, browser, skill
registries. That brings their attack surface with them. tellm has no hands. It
only relays messages between Telegram and your model providers, so there is
no shell to hijack and no skill to poison.

## Install

Download the archive for your platform from
[GitHub Releases](https://github.com/amidabuddha/tellm/releases) (macOS Apple
Silicon and Intel, Linux arm64 and x86_64, Windows x86_64), check it against
`sha256.sum` and put `tellm` on your `PATH`. On macOS or Linux:

```sh
mkdir -p "$HOME/.local/bin"
install -m 0755 tellm "$HOME/.local/bin/.tellm.new"
mv -f "$HOME/.local/bin/.tellm.new" "$HOME/.local/bin/tellm"
tellm
```

Replace an installed binary this way rather than copying over it: macOS can
kill a binary that was rewritten in place.

On macOS, the release binaries are not signed with an Apple developer
certificate or notarized by Apple. If you downloaded the archive with a web
browser, macOS may refuse to start tellm, saying it can't verify the
developer. Clear the download mark once:

```sh
xattr -d com.apple.quarantine "$HOME/.local/bin/tellm"
```

or allow it under System Settings → Privacy & Security → Open Anyway. For
the same reason, after each upgrade macOS asks once for your login password
for each key tellm stored in the keychain; choose Always Allow and it stays
quiet until the next upgrade.

On first run, a wizard asks for a Telegram bot token and one provider API key,
then prints a pairing code. Send `/pair CODE` to your bot and you're done.

`tellm --version` (or `-v`) prints the installed version.

### Build from source

You need Git and Rust 1.88 or newer.

```sh
git clone https://github.com/amidabuddha/tellm.git
cd tellm
cargo build --locked --release   # binary: target/release/tellm
```

On macOS, each new build makes the keychain ask again for every stored key.
To keep an Always Allow across rebuilds, sign each build with your own Apple
Development certificate, which any Apple Account can create free in Xcode →
Settings → Accounts → Manage Certificates → + → Apple Development:

```sh
codesign --force --sign "Apple Development" --identifier tellm target/release/tellm
```

If `codesign` finds no valid identity although Xcode lists the certificate,
add Apple's
[WWDR G3 intermediate certificate](https://www.apple.com/certificateauthority/AppleWWDRCAG3.cer)
to your login keychain.

The checks CI runs before every merge:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

## Supported APIs

All direct, with no middleware.

| Wire format | Providers |
|---|---|
| Anthropic Messages | Claude (prompt caching, adaptive thinking) |
| OpenAI Responses | OpenAI, xAI, Meta Model API (web search; OpenAI image generation) |
| Chat completions | Ollama, DeepSeek, OpenRouter, any compatible endpoint |
| Google Interactions | Gemini (including image models) |

## Commands

| Command | Does |
|---|---|
| `/help` | List commands |
| `/new` · `/new summary` | Start a new conversation, or carry a summary of this one into it |
| `/id` | Show this chat's Telegram id |
| `/mode chat\|message` | Multi-turn conversation, or one question per message |
| `/model [KEY]` | Show or switch this chat's model |
| `/model add [KEY]` | List the built-in providers, or add one (owner) |
| `/model pin KEY` · `/model unpin` | Lock or release this chat's model (owner) |
| `/role TEXT\|clear` | Set or clear the system prompt |
| `/reasoning default\|off\|low\|medium\|high\|max` | Thinking effort |
| `/websearch on\|off\|status` | Provider-native web search |
| `/imagegen on\|off\|status` | Image generation (capable models only) |
| `/cost` · `/cost on\|off` | This chat's running cost, or show/hide the cost line |
| `/memory` · `/memory on\|off` | List saved facts, or use them in this chat or not (owner) |
| `/remember TEXT` · `/forget N` | Save a fact, or remove fact N (owner) |
| `/pair CODE` | Claim an unpaired chat with the code from the terminal |
| `/allow CHAT_ID` · `/deny CHAT_ID` | Approve or revoke a chat (owner) |
| `/ollama unload` | Unload local Ollama models this session used (owner) |
| `/group` · `/group private\|shared` | Show, or set (owner), whether this group is your private room or shared |
| `/shutdown` | Stop tellm (owner) |

## Configuration

`config.toml`, `rooms.toml`, `credentials.toml` and `memory.md` live in your OS
configuration folder under `tellm/`. `config.toml` never holds secrets:
`api_key_secret` only names the key, and the key itself is stored in the
keychain or `credentials.toml`. A `TELLM_<SECRET_NAME>` environment variable
overrides both.

`/model add` writes a ready-made entry for each lab's most capable model.
A fuller example:

```toml
default_model = "gpt"

[telegram]
allowed_chat_ids = []
owner_user_ids = []
private_group_ids = []

[models.gpt]
wire_format = "responses"
model_name = "gpt-6.2-astra"
api_key_secret = "openai_api_key"
telegram_chat_ids = []
thinking = "high"
max_effort = "max"
off_effort = "low"

[models.claude]
wire_format = "anthropic"
model_name = "claude-fable-5-2"
api_key_secret = "anthropic_api_key"
thinking = "max"
off_effort = "low"
max_output_tokens = 64000

[models.grok]
wire_format = "responses"
model_name = "grok-4.7"
base_url = "https://api.x.ai/v1"
api_key_secret = "xai_api_key"

[models.meta]
wire_format = "responses"
model_name = "muse-wonder-1.0"
base_url = "https://api.meta.ai/v1"
api_key_secret = "meta_model_api_key"
max_effort = "max"
off_effort = "minimal"

[models.gemini]
wire_format = "gemini"
model_name = "gemini-4-carbon"   # image generation needs an image model id
api_key_secret = "gemini_api_key"

[models.remote_compat]           # any chat-completions endpoint
wire_format = "compat"
model_name = "provider-model-name"
base_url = "https://provider.example/v1"
api_key_secret = "remote_compat_api_key"

[models.ollama]                  # local and keyless: no api_key_secret
wire_format = "compat"
model_name = "llama3.3:70b"
base_url = "http://localhost:11434/v1"
thinking = "off"
```

- **`base_url`**: leave it out for a provider's own endpoint. Set it for other
  endpoints on the same wire format, such as xAI or Meta. It is required for
  `compat`. Endpoints that receive a key must use HTTPS. Keyless HTTP works on
  loopback, and a trusted LAN endpoint needs `allow_insecure_http = true`.
- **`thinking`**: the model's default effort. `/reasoning` overrides it per
  chat.
- **`max_effort` / `off_effort`**: the provider values sent for `max` and
  `off`, since they differ per model. Without `max_effort`, tellm sends `max`
  on `anthropic`, `xhigh` on `responses` and `high` elsewhere. Without
  `off_effort`, tellm sends no setting and the model uses its own default.
- **`max_output_tokens`** (anthropic only): the reply limit, thinking
  included. The default is 16000. tellm doesn't stream, so a higher limit
  means a longer wait.
- **`telegram_chat_ids`**: chats locked to this model with `/model pin`. A plain
  `/model KEY` is saved in `rooms.toml` instead.

### Reply costs

Each reply ends with a line such as
`Cost ≈ $0.031 (2 searches) · this chat ≈ $0.42 · 48k of 150k tokens`. The
last part is the chat's size: at 150k tokens tellm summarizes it into a fresh
chat. `/cost` shows the same totals in more detail, including how much of the
input came from the provider's cache.
xAI and OpenRouter report what they charged. For other providers tellm
multiplies the reported tokens by the model's price table; the built-in Claude,
OpenAI, Meta and Gemini entries include one. Add one for any other model:

```toml
[models.gpt.price]   # US dollars
input = 8.0          # per million input tokens
cached_input = 0.40  # per million cached input tokens
cache_write = 10.0   # per million tokens written to the cache
output = 40.0        # per million output tokens, reasoning included
web_search = 10.0    # per 1,000 web searches
```

Only `input` and `output` are required. Without a price, the line shows token
counts. tellm never fetches prices, so keep the table current. A `≥` means some
replies had no known cost. "This chat" restarts with `/new`, `/model`, `/role`
and `/mode message`. Failed replies aren't counted. Your provider's billing
page is the final word.

## Memory

Save facts once and every new conversation starts with them:

```text
/remember I'm 41 and live in Sofia.
/remember Two kids, aged 7 and 10.
/memory
/forget 2
```

- The facts live in `memory.md` next to `config.toml`, readable only by your
  user account. You can edit it by hand: one fact per line, `#` for comments.
- Memory is on in your direct chat with the bot. In groups and other people's
  chats it stays off until you send `/memory on` there.
- Changes take effect in your next chat (`/new`).
- The limit is 4,096 characters (one Telegram message), about 1,000 tokens per
  message. Most emoji count as two characters, as in Telegram. The facts go to
  the chat's provider with every message, so save only what you'd tell it.

## Model rooms

To give each model its own group, turn off privacy mode once in BotFather
(`/setprivacy`), then add the bot to each group from your owner account. Groups
you add are approved automatically and show a model picker.

- `/model KEY` switches the chat's model.
- `/model pin KEY` locks it until `/model unpin`.
- `/model add KEY` adds a built-in provider. If its key is missing, tellm asks
  for it in the terminal, never in Telegram.

**Shared groups:** every group starts shared, because tellm can't tell who
else reads it. There it answers only messages that @mention it or reply to
it, each on its own: no conversation history, no memory, and nothing is
stored. A reply includes the message it answers, so follow-ups still make
sense. Only owners can use commands there, except `/help` and `/id`. If a
group is just you, send `/group private` to make it a normal room; it then
answers every message as one conversation. If someone who isn't an owner
joins a private group, tellm says so in the group, and `/group shared` makes
it shared again.

**Ollama:** for `localhost:11434`, tellm starts `ollama serve` when needed.
On shutdown it unloads the models this session used and stops the server again
if it started it. Images are sent only to models whose `ollama show` lists
`vision`. Other ports and hosts are treated as ordinary endpoints.

**Limits:** when a chat reaches 150k tokens (or 4 MiB of history), tellm
summarizes it right after that reply and continues in a fresh chat that
carries the summary. `/new summary` does the same at any time. Attachments can
be up to 20 MiB. Edited messages are ignored, so an edit never triggers a
second paid request.

## Troubleshooting

- **The bot ignores plain text in a group:** turn off privacy mode in
  BotFather (`/setprivacy`) and add the bot to the group again.
- **Keychain prompts at startup:** tellm reads each configured key once at
  startup, so the prompts come before your first message rather than in the
  middle of a chat. On macOS they return after each upgrade or rebuild,
  because the new binary has a new signature; see
  [Install](#install) and [Build from source](#build-from-source).
- **macOS says it can't verify the developer:** the release binaries aren't
  notarized; clear the download mark as shown under [Install](#install).
- **Logs:** each update is logged with its chat id and route, never with the
  message content.

## License

MIT OR Apache-2.0, at your option.
