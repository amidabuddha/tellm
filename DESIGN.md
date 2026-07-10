# tellm — Design

Standalone, minimal, self-hosted Telegram gateway to frontier LLMs.
Successor to the Telegram runtime of [console-chat-gpt](https://github.com/amidabuddha/console-chat-gpt),
which remains the reference implementation and the author's fallback during the port.

## Positioning

> Binary-first personal Telegram LLM gateway with first-run pairing, direct
> provider APIs, per-room model pinning, and no Docker/config ceremony.

- Usage is billed directly by your API providers. tellm never touches billing.
- **Not an agent.** No shell, no skills registry, no browser control, no MCP.
  The entire capability surface is calls to Telegram and model providers.
  Provider traffic is HTTPS by default; cleartext is limited to keyless compat
  endpoints on loopback or a per-model `allow_insecure_http = true` opt-in.
  This is deliberate (see "Why not OpenClaw?" in the README).

## Architecture

```
                 ┌────────────────┐
 Telegram  <───> │ tellm (binary) │  runtime: polling loop, per-chat ordered
                 │  ├ tellm-telegram   dispatch, command router, pairing,
                 │  ├ tellm-config     first-run wizard, session state
                 │  └ tellm-core  │  unified ChatRequest/ChatResponse + Provider trait
                 └───────┬────────┘
        ┌────────────┬───┴────────┬──────────────┐
  tellm-anthropic tellm-openai tellm-compat  tellm-gemini
   Messages API   Responses API chat completions Interactions API
   Anthropic      OpenAI/xAI/Meta Ollama/DeepSeek Gemini
                                OpenRouter/...
```

**No middleware.** Each wire-format crate talks to the provider's latest
native API directly via `reqwest` + `serde`. No SDK crates, no unichat
equivalent, no router layer. Dispatch is a plain enum match in the binary.

**One crate per concern** (workspace):

| Crate | Concern |
|---|---|
| `tellm-core` | Unified types (`ChatRequest`, `ChatResponse`, `ThinkingLevel`, `ContentPart`), opaque-history contract, `Provider` trait |
| `tellm-telegram` | Bot API client: long polling, rich→HTML→plain delivery chain, chunking, markdown→HTML |
| `tellm-config` | TOML config (non-secrets), keychain/0600 secret store, explicit capability routing |
| `tellm-anthropic` | Anthropic Messages API |
| `tellm-openai` | OpenAI Responses API (OpenAI, xAI, Meta via base_url) |
| `tellm-compat` | OpenAI chat-completions dialect (Ollama, DeepSeek, OpenRouter, any compatible) |
| `tellm-gemini` | Google Interactions API (Gemini) |
| root binary | Runtime loop, command router, sessions, pairing, wizard |

Inside the root binary, runtime-owned concerns are split into
`src/{runtime,wizard,rooms,access,commands}.rs`; `tellm-config` deliberately
keeps only nonsecret config, secret storage, and validation.

## Unified parameter set

The core model is deliberately minimal — only what is common across providers:

- `model`, `system`, `history` (opaque, see below) + `input` (text / image /
  document parts for the new user turn)
- `thinking: off | low | medium | high | max`
- `web_search: bool`, `image_generation: bool`
- `max_tokens`

### Opaque provider state

Provider conversations carry state that must be echoed back **verbatim** on
later turns: Anthropic web-search results include `encrypted_content`
(missing or modified ⇒ 400), and OpenAI Responses reasoning/function-call
items must be replayed in stateless usage (the Python reference already does
this via `response_parser`). A fully unified history type would destroy that
state, so:

- Room history is stored as **provider-native JSON items**, opaque to the
  runtime, tagged with the wire format that produced them.
- The unified types are the construction API for new input and the
  extraction API for display — never the storage format.
- Each `ChatResponse` returns `turn_items` (the full exchange in the
  provider's own history shape); the runtime appends them verbatim.
- Switching a room's wire format resets the opaque history (announced to the
  user).
- History stays memory-bounded: retain at most 32 complete provider-native
  turns and approximately 4 MiB per room, pruning oldest whole turns so a
  request/response pair is never split.

Thinking-level translation (verified against live provider docs 2026-07-05;
xAI Grok 4.5 refreshed 2026-07-09):

| Level | Anthropic Messages | OpenAI Responses | xAI Responses | Chat completions | Gemini |
|---|---|---|---|---|---|
| off | omit `thinking` | omit `reasoning` | omit | omit `reasoning_effort` | omit `thinking_level` |
| low/medium/high | adaptive + `output_config.effort` | `reasoning: {effort}` | same | `reasoning_effort` | `thinking_level` |
| max | `effort: "max"` (valid on ALL adaptive models) | `effort: "xhigh"` (model-dependent) | clamps to `"high"` (grok-4.5 has no xhigh) | clamps to `"high"` ("max" not in the dialect) | clamps to `"high"` |

Validity caveats (2026-07-05):
- **"off" means provider default everywhere, not "no thinking"**: Sonnet 5 /
  Fable 5 think anyway, gpt-5.6-sol reasons at its own default, grok-4.5 defaults to high,
  Gemini thinks dynamically. User-facing text must say "default", not "off".
- OpenAI effort is model-dependent: the GPT-5.6 family (sol/terra/luna) adds a
  `max` tier above `xhigh`, so `/reasoning max` sends `max` there and `xhigh` on
  older OpenAI and Meta Muse Spark; Grok 4.5 has no tier above `high` and cannot
  disable reasoning, so its `Max` clamps to `high` and `Off` omits the field. An
  unsupported value errors explicitly and the user can lower the room's level.
- Gemini `medium` is invalid on gemini-3-pro-preview (low/high only), so that
  model should be configured with `thinking = "low"` or `thinking = "high"`;
  gemini-3.1-pro and 3.5-flash accept medium.
- Anthropic default max_tokens is 16000: thinking tokens count toward
  max_tokens, and 4096 risked mid-thought truncation at high/max effort.
- **Capability toggles are gated at the toggle**: `/imagegen on` and
  `/websearch on` refuse in rooms whose wire format or model id can never
  honor them (image generation: Anthropic/compat/xAI/Meta/non-image Gemini models;
  web search: compat), naming the model and endpoint. Support that still varies
  per model *inside* a capable format (for example, which OpenAI models can use
  the image tool) stays a request-time error, with the error reply suggesting
  the off command.

Notes:
- **API surface details are pinned at implementation time against live
  provider docs, with the check date recorded in a source comment** — never
  from anyone's cached knowledge. Current pins (checked
  2026-07-04): Anthropic web search is `web_search_20260318` with
  `allowed_callers: ["direct"]` — from `_20260209` onward the default routes
  search through server-side code execution (dynamic filtering), which tellm
  doesn't want for plain chat.
- Anthropic `budget_tokens` is **dead** on current models (400) — never send it.
- Anthropic prompt caching: `cache_control: {type: ephemeral}` on the system
  block; keep the system prompt byte-stable (no timestamps — see the
  silent-invalidator list in Anthropic's caching docs).
- Handle Anthropic `stop_reason: refusal` before reading content.
- Handle Anthropic `stop_reason: pause_turn` inside the provider crate by
  continuing the request with the paused assistant message appended; return all
  assistant messages from the loop in `turn_items` so opaque history remains
  replayable.
- OpenAI/Meta/xAI Responses are used statelessly (`store: false`): prior
  provider-native input/output items are replayed in `input`, and every
  response `output` item is returned in `turn_items` verbatim. When reasoning
  is requested, `include: ["reasoning.encrypted_content"]` is set so encrypted
  reasoning can be replayed on later stateless turns. OpenAI system prompts use
  top-level `instructions`; xAI currently rejects `instructions`, so xAI
  system prompts are sent as compatible input message items instead (checked
  2026-07-04; still shown as input messages in 2026-07-09 Grok 4.5 docs).
- OpenAI Responses web search uses the `web_search` tool. xAI Responses uses
  `web_search` plus `x_search` for tellm's search toggle (checked
  2026-07-09 against docs.x.ai for Grok 4.5). Meta Model API Responses uses
  the same `web_search` tool,
  supports image understanding through `input_image` and `input_file`, and
  does not document image generation on this surface (checked 2026-07-09
  against dev.meta.ai). OpenAI image generation uses `image_generation`, and
  `image_generation_call.result` base64 payloads become `GeneratedImage`; xAI
  and Meta image generation are unsupported in the Responses crate.
- Chat-completions compat uses the broad OpenAI/Ollama dialect: `messages`,
  `stream: false`, `max_tokens`, `reasoning_effort`, and image input via
  `image_url` data URLs (checked 2026-07-04). Web search, image generation,
  and documents are reported as unsupported before any provider call.
- Gemini Interactions uses `POST /v1beta/interactions`, `x-goog-api-key`,
  `store:false`, `stream:false`, `system_instruction`, and stateless `input`
  as an array of Interactions `Step` objects (checked 2026-07-05 against
  ai.google.dev). Raw response `steps` are replayed through `turn_items`
  verbatim, including `thought` and Google Search call/result signatures.
  Web search is `tools: [{type:"google_search",
  search_types:["web_search"]}]`; image generation is allowed only for Gemini
  image model ids (`*-image*`) and sends `response_format: {"type":"image"}`;
  generated image content becomes `GeneratedImage`. The image-model gate is a
  fail-closed naming heuristic, not a capability-discovery API; update it if
  Google's public model ids stop using `-image`.
- **File upload requires no extraction.** PDFs/images pass through natively:
  Anthropic `document`/`image` blocks, OpenAI `input_file`/`input_image`,
  Gemini `document`/`image` content blocks.
  `.txt` is read as text. Providers that can't take a part get a clear
  "unsupported" message, not silent degradation.
- **No streaming anywhere.** Telegram delivers complete messages; providers
  are called non-streaming. This deletes the largest complexity class of the
  parent app.
- HTTP clients use explicit timeouts. Provider calls have a 10s connect timeout
  and long total timeout for multi-minute turns; production instances of each
  wire-format client share a process-wide `reqwest` connection pool. Telegram
  calls use bounded request/upload/download timeouts and long-poll timeout plus
  a grace window.

## Runtime model (ported from console-chat-gpt)

- Long polling (`getUpdates`, modest timeout so terminal controls stay responsive).
- **Strictly ordered execution per chat via one tokio mpsc task per chat**
  (decided 2026-07-04; idle tasks are reaped). The channel gives ordering and
  backpressure for free and is the tokio-native equivalent of the parent's
  chained-futures design. The queue is bounded (32 per chat); when a room
  backs up behind a slow turn, further messages are dropped with a busy
  notice instead of blocking the poll loop for every other chat. At most one
  busy-notice send may be outstanding per room.
- In-memory conversations; **room settings persist** across restarts
  (model pin, mode, role, optional thinking override, web-search toggle,
  image-generation toggle) in `config_dir()/rooms.toml` — runtime state
  deliberately separate from the user-owned config.toml. If a room has no
  `thinking` entry, it uses the selected model's `config.toml` thinking value;
  `/reasoning default` clears the room override. Conversations don't persist.
- Modes: `chat` (multi-turn) and `message` (stateless per message). Message
  mode requests never send prior history, but the runtime keeps the latest
  exchange in memory so switching back with `/mode chat` can continue from the
  last message-mode reply.
- Runtime command mutations are serialized through the same per-chat queue as
  model turns. `/model` and `/role` reset that room's in-memory history so
  provider-native opaque state is not replayed under a different model or
  system prompt. Terminal `reset` clears all in-memory histories while keeping
  room settings. Monotonic room generations prevent an older in-flight result
  or rollback from repopulating reset or revoked state.
- Telegram downloads accept images, PDFs, and text documents up to 20 MiB.
  Declared oversize files are rejected before download; unknown-length bodies
  are read incrementally and stopped at the same limit.
- Edited-message updates are deliberately ignored so correcting a Telegram
  message cannot silently trigger a second billed model call.
- Delivery: `sendRichMessage` → HTML `sendMessage` → plain text, with
  chunking at 32000/3900 chars. Fallback triggers ported from the Python
  implementation's error-marker list. Generated images preserve their provider
  MIME type when uploaded. Telegram transport and API errors strip
  token-bearing request URLs and redact any echoed bot token before the error can
  reach logs or operator-facing output.
- Shutdown (terminal `exit`/`quit`, Telegram `/shutdown`, SIGINT/SIGTERM)
  stops polling, marks every room worker cancelled, aborts and joins them, and
  only then performs Ollama unload/child cleanup. Draining provider turns could
  block exit for minutes, while unloading before their futures are gone can
  immediately reload a model. Closing stdin merely disables terminal controls;
  it does not stop a daemonized runtime. Only the tellm-started Ollama child
  gets automatic graceful cleanup (see § Configuration).

### Telegram commands (v1)

`/new`, `/id`, `/mode`, `/model` (incl. `pin`/`unpin`/`add`, owner-gated), `/role`,
`/reasoning`, `/websearch`, `/imagegen`, `/allow`, `/deny`, `/pair`,
`/ollama unload` (owner-gated), `/shutdown` (owner-gated), `/help`.
Dropped as waste:
`/webfetch` (fold into websearch decision later), assistants, AI-managed mode,
MCP tooling.
Known commands accept Telegram's group form (`/cmd@BotName`); commands
addressed to a different bot are ignored, and unknown slash commands pass
through as normal model input rather than producing "command not found" noise.
Argument semantics: `/id` replies with the current Telegram chat id,
`/mode [chat|message]`, `/model [key]`, `/role [text]`
with `clear|off|none|reset` clearing the role, `/reasoning
[default|off|low|medium|high|max]`, `/websearch [on|off|status]` (no argument
toggles), `/imagegen [on|off|status]` (no argument toggles),
`/ollama unload` to unload local Ollama models invoked by this tellm process,
and `/allow CHAT_ID` / `/deny CHAT_ID` from any owner, and `/pair CODE`.
If a chat id appears in a model's `telegram_chat_ids`, that model is the
locked room model: `/model` shows the pin, `/model KEY` refuses, `/help`
notes the pin, and any stored `rooms.toml` model selection is ignored for that
chat. Plain `/model KEY` is intentionally lighter-weight: it writes the room's
selection to `rooms.toml` and leaves `telegram_chat_ids` empty. Use `/model pin
KEY` only when a Telegram room should always be locked to that model across
restart and local room-setting changes.

## Access control (ported + hardened)

- Default-deny with **per-room, re-armable pairing**: any contact from
  an unknown chat — a message, or the bot being added to a group (heard via
  `my_chat_member`) — arms a 6-digit code for THAT room, printed to the
  terminal only (never sent over Telegram) and announced with the group
  title. `/pair CODE` typed in the room approves it. Constant-time compare,
  10-minute per-room rotation, per-chat 5-attempt / 5-minute lockout.
  Approving one room never disables pairing for others. Unknown chats get
  one hint containing the chat id, the /pair instruction, and the /allow
  alternative.
- **Owner users**: completing a code pairing records the pairing
  user's Telegram id in `telegram.owner_user_ids`. When a recorded owner
  adds the bot to a chat (`my_chat_member.from`), the room is auto-approved
  and receives the model picker immediately — one pairing code per bot
  lifetime, not per room. Unknown adders still get the per-room code gate.
- **Privilege belongs to users, not chats.** There is no
  `admin_chat_ids`. `owner_user_ids` is the single privilege concept:
  recorded on code pairing (console access = ownership proof), checked
  against `message.from` on every privileged command (`/allow`, `/deny`,
  `/shutdown`, `/ollama unload`, `/model pin|unpin|add`), valid from any chat
  the owner is in. This closes the group-admin hole (group members inheriting
  chat-based privilege) and makes admin-stranding via `/deny` structurally
  impossible. Caveat: Telegram anonymous-admin mode hides `from` — owners must
  post non-anonymously (or use a private chat) for privileged commands.
- After approval the room replies with its current model and the /model /
  /model pin next steps; on group approval the console prints the privacy-
  mode hint.
- Admin chats can approve or revoke rooms at runtime: `/allow CHAT_ID` adds the
  chat to `allowed_chat_ids`, persists config, and takes effect immediately;
  `/deny CHAT_ID` removes `allowed_chat_ids` and model pins for the chat,
  cancels and drops that room's queued/in-flight work, clears its room state,
  persists config/rooms, and takes effect before any stale result can be sent
  or committed. Live revocation and worker cancellation happen before waiting
  for config persistence. If that config write fails, config and access are
  restored, but already-aborted work stays dropped; after config commits, a
  room-cleanup write failure never re-allows or recreates the denied room.
- Model-room mappings count as allowed chats. They no longer suppress
  pairing (pairing is per-room now); the startup notice states how many
  chats are allowed and that new rooms pair on first contact.
- `/shutdown` requires a registered owner sender; messages older than
  60 seconds are rejected as stale so old updates cannot shut down a fresh
  process.
- Runtime stderr breadcrumbs log one line per received message update with
  chat id, update kind, and route (`command` / `model` / `ignored`) only; never
  message content. Negative allowed or pinned chat ids print a group privacy
  hint at startup because Telegram privacy mode can filter plain group text
  before tellm receives it.

## Configuration

- **API keys never travel through Telegram chat** (they would transit
  Telegram's servers and persist in history on both ends). Console-side
  entry: `tellm secret set NAME` or `/model add KEY` from an owner chat, which
  asks for the preset's secret in the tellm terminal using the same visible
  prompt style as first-run setup. If `KEY` is an already-configured custom
  model, `/model add KEY` instead prompts for that model's `api_key_secret`.
  The terminal-control reader has one prompt slot: while a prompt is active,
  terminal lines go to that prompt, `reset`/`exit`/`quit` are rejected as API
  keys rather than executed, and the prompt expires after five minutes so it
  cannot block later prompts indefinitely. Both paths store through the
  secret facade and are picked up per request without restart. On startup,
  tellm reads each unique configured provider `api_key_secret` once so OS
  keychain permission prompts happen before the first model call in that
  process; missing provider keys are logged but remain non-fatal until a model
  that requires them is used. The missing-key provider error names the console
  fallback.
- Non-secrets: human-editable TOML at `dirs::config_dir()/tellm/config.toml`.
  **Capability routing is explicit** (`wire_format` field per model) — never
  inferred from the spelling of user-chosen keys (parent-app lesson).
  Provider credentials are referenced by secret name (`api_key_secret`),
  never stored inline. Keyless endpoints, especially local Ollama, omit
  `api_key_secret`; `/model add KEY` must not open a key prompt for those
  configured models, and their compat requests omit `Authorization` entirely.
  Custom URLs are parsed as absolute URLs and reject embedded credentials.
  Credential-bearing URLs always require HTTPS. Keyless compat URLs may use
  HTTP on an explicit loopback host by default; a non-loopback HTTP endpoint
  requires `allow_insecure_http = true` in that model's table, making the
  cleartext LAN trust decision visible and local to that endpoint. The opt-in
  never permits credentials over HTTP or permits another URL scheme. Model
  keys are nonempty, whitespace-free command tokens.
- Local Ollama convenience: for compat models whose `base_url` points at the
  default local HTTP Ollama endpoint (`http://localhost:11434`,
  `http://127.0.0.1:11434`, or `http://[::1]:11434`, with or without `/v1`),
  the runtime checks the TCP port before dispatch. If it is down, tellm starts
  `ollama serve` once and waits briefly for readiness. A tellm-started Ollama
  child is stopped during tellm shutdown, including terminal `exit`/`quit`,
  Telegram `/shutdown`, SIGINT, SIGTERM, and ordinary panic unwinds, after
  tellm asks Ollama to unload every local model whose request began in the
  current tellm process via `keep_alive: 0` (checked 2026-07-05 against
  docs.ollama.com/api/generate). The child shutdown path sends SIGTERM, waits
  briefly, then falls back to SIGKILL. A 404 / not-found unload response means
  the model is already gone and is removed from tellm's in-memory tracking.
  SIGKILL and power loss cannot run cleanup. An Ollama server that was already
  running, including a possible orphan from an earlier hard kill, is left alone
  rather than adopted and killed. Other compat endpoints, including `https://`
  proxies, LAN binds, remote hosts, and local proxies on different ports, are
  never auto-started or unloaded. Owners can also send `/ollama unload` to
  unload local Ollama models invoked by the current tellm process without
  stopping tellm or `ollama serve`.
- `config.toml`, `rooms.toml`, and `credentials.toml` use collision-safe unique
  same-directory temp files, file sync, and rename, so a crash or concurrent
  writer cannot expose a partial file. Unix parent-directory sync is attempted
  after commit; failure is a durability warning, not a false failed-commit
  signal. One ordered persistence thread owns runtime config/room writes, so an
  aborted worker cannot detach an older write that lands after `/deny`. Failed
  mutations roll settings back without resurrecting invalidated history. The
  files remain separate transactions: a reported I/O failure between config
  and room persistence can leave stale room settings on disk, but never restores
  access or conversation history.
- **Semantic validation at startup** (`Config::validate()`): default model
  exists; model keys are command-safe; model names and secret names are
  nonempty and secret names cannot use tellm's internal marker prefix;
  non-compat models name a secret; compat models have a valid absolute URL;
  credential-bearing endpoints use HTTPS; non-loopback keyless compat HTTP has
  the explicit per-model opt-in described above; URLs contain no embedded
  credentials; and each chat is pinned to at most one model. All problems are
  reported at once; the bot refuses invalid config.
- Secrets (bot token, API keys): OS keychain via `keyring-core` plus direct
  platform-store registration (Apple native keychain, Windows native, zbus
  Secret Service; checked 2026-07-05 after keyring 4.1.3's `v1` wrapper failed
  to register a default store), fallback `0600` file for headless hosts,
  env-var override (`TELLM_<SECRET_NAME>`). Secret writes report the actual
  destination. A successful keychain write removes the stale file entry. A new
  file fallback records an internal preference marker atomically with the value,
  so a recovered stale keychain cannot shadow it; legacy unmarked files remain
  keychain-first. An active environment override must be unset before rotation,
  since it remains the effective value. Credentials-file read/modify/write is
  serialized within one process; multiple tellm processes must not share and
  mutate the same config directory concurrently.
  No encrypted-file theater.
- First run: when `config.toml` is absent, the interactive wizard asks for a
  Telegram bot token, validates it with `getMe`, asks for one provider choice
  from the built-in list (checked 2026-07-09: Claude Fable 5, GPT-5.6 Sol,
  Grok 4.5, Muse Spark 1.1, Gemini 3.5 Flash), stores Telegram/provider
  secrets via the secret facade, writes nonsecret config, and explains the
  `/pair CODE` claim step.
  Target: install-to-chatting under two minutes, zero file editing.

## Porting method

**Golden tests, not transcription.** The pure functions (`chunk_text`,
`markdown_to_html`, fallback-trigger classification) get test vectors
generated from the Python reference implementation
(`../consoleChatGPT/console_gpt/telegram_bot.py`) before the Rust versions
are written. Network fallback chains get a small mocked-Telegram harness.

## Distribution

- `cargo-dist` → GitHub Releases with static binaries (macOS arm64/x86_64,
  Linux x86_64/arm64, Windows). rustls (no OpenSSL linkage).
- Brew tap once there's traction. A sample Dockerfile may exist but is never
  the headline (binary-first, not anti-Docker).

## Non-goals (v1)

- No MCP, no tools beyond provider-native web search / image generation.
- No streaming. No conversation persistence. No multi-messenger.
- No OpenAI Assistants, no AI-managed model routing.
- No PDF text extraction (native provider passthrough instead).

## Open questions

1. Anthropic web_fetch: include with web search toggle or leave out.
2. Rich-message availability detection: probe once at startup vs per-send
   fallback (parent does per-send; tellm currently implements per-send).

Resolved: per-chat ordered dispatch = one tokio mpsc task per chat
(2026-07-04, see § Runtime).
Resolved: Gemini Interactions API mapping and native client implementation
(2026-07-04, see § Unified parameter set).
