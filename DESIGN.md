# tellm — Design

Standalone, minimal, self-hosted Telegram gateway to frontier LLMs.
Successor to the Telegram runtime of [console-chat-gpt](https://github.com/amidabuddha/console-chat-gpt),
which remains the reference implementation and the author's fallback during the port.

## Positioning

> Binary-first personal Telegram LLM gateway with first-run pairing, direct
> provider APIs, per-room model pinning, and no Docker/config ceremony.

- Usage is billed directly by your API providers. tellm never touches billing:
  it only reads the token usage each reply reports and shows what the reply
  cost (see § Usage and cost).
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
| `tellm-core` | Unified types (`ChatRequest`, `ChatResponse`, `Usage`, `ThinkingLevel`, `ContentPart`), opaque-history contract, `Provider` trait |
| `tellm-telegram` | Bot API client: long polling, rich→HTML→plain delivery chain, chunking, markdown→HTML |
| `tellm-config` | TOML config (non-secrets), keychain/0600 secret store, explicit capability routing |
| `tellm-anthropic` | Anthropic Messages API |
| `tellm-openai` | OpenAI Responses API (OpenAI, xAI, Meta via base_url) |
| `tellm-compat` | OpenAI chat-completions dialect (Ollama, DeepSeek, OpenRouter, any compatible) |
| `tellm-gemini` | Google Interactions API (Gemini) |
| `tellm-test-support` | Dev-only shared mock HTTP support for provider and Telegram integration tests |
| root binary | Runtime loop, command router, sessions, pairing, wizard, local Ollama lifecycle |

Inside the root binary, runtime-owned concerns are split into
`src/{runtime,model_turn,persistence,wizard,rooms,access,commands,ollama,cost,memory,summary}.rs`.
The private `model_turn` module owns Telegram input/attachment normalization,
provider-neutral request construction, provider dispatch and secret lookup,
and model-response delivery. `runtime` retains access checks, cancellation,
and generation-guarded commit/rollback orchestration. The private `persistence`
module owns the ordered blocking writer plus serialized room setting snapshots
and their rollback semantics. The private `ollama` module owns recognition and
lifecycle management for the narrowly supported local endpoint; `tellm-config`
deliberately keeps only nonsecret config, secret storage, and validation. The
private `cost` module turns a reply's usage into its cost line and keeps each
room's running total. The private `memory` module owns the owner's memory file
and the memory block it adds to the system prompt (see § Memory). The private
`summary` module owns the summary request and the summary block a new
conversation carries (see § Summaries).

## Unified parameter set

The core model is deliberately minimal — only what is common across providers:

- `model`, `system`, `history` (opaque, see below) + `input` (text / image /
  document parts for the new user turn)
- `thinking: off | low | medium | high | max`
- `web_search: bool`, `image_generation: bool`

Each `ChatResponse` also carries the reply's `usage`, normalized as described
in § Usage and cost.

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
- History is append-only within a conversation. Dropping its oldest turns
  would change the prefix every provider caches, so a conversation that
  grows too large is summarized into a new one instead (§ Summaries). About
  4 MiB of history per room is a memory safety net: passing it also triggers
  the summary. Only when summarizing fails, or a message-mode exchange alone
  is over it, does tellm prune the oldest whole turns down to the cap, so a
  request/response pair is never split, and it says in the chat how many
  turns it dropped. The notices are best-effort: a Telegram failure is
  logged and reported, but never skips the summary or the prune.
- Verbatim replay holds only while history is append-only. When the runtime
  itself changes what precedes the kept turns (the last-resort pruning
  above, a summary request that leaves the oldest turns out, or a
  `/websearch` toggle, which changes the tool list), it hands each kept turn
  back to the wire-format crate that produced it and sends or stores what
  that crate says is still replayable; a turn returned empty is dropped
  whole. The runtime still never inspects the items. Anthropic removes
  thinking blocks (see Notes); the other wire formats keep their turns
  unchanged.

Thinking-level translation (verified against live provider docs 2026-07-05;
Gemini 3.6 Flash refreshed 2026-07-26;
OpenAI availability and xAI Grok 4.5 refreshed 2026-07-10; Anthropic Claude
Opus 5 refreshed 2026-07-26; xAI and Meta top tiers and the preset models
refreshed 2026-11-16):

| Level | Anthropic Messages | OpenAI Responses | xAI Responses | Chat completions | Gemini |
|---|---|---|---|---|---|
| off | `off_effort` as `output_config.effort`, else no effort; never `thinking` | `off_effort`, else omit `reasoning` | same | `off_effort`, else omit `reasoning_effort` | `off_effort`, else omit `thinking_level` |
| low/medium/high | adaptive + `output_config.effort` | `reasoning: {effort}` | same | `reasoning_effort` | `thinking_level` |
| max | `max_effort`, else `"max"` (valid on ALL adaptive models) | `max_effort`, else `"xhigh"` | same | `max_effort`, else `"high"` ("max" not in the dialect) | `max_effort`, else `"high"` |

Validity caveats (OpenAI refreshed 2026-07-10; Anthropic refreshed
2026-07-26; xAI, Meta, and the preset models' defaults refreshed 2026-11-16):
- **"off" is configuration, like the top tier.** `/reasoning off` sends the
  model's optional `off_effort` value from `config.toml`. When that is unset
  tellm sends no reasoning setting and the provider default applies, which is
  not "no thinking": Claude Fable 5.2 defaults to `high`; gpt-6.2-astra reasons
  at its own default and rejects `none`; grok-4.7 defaults to `high`;
  muse-wonder-1.0 defaults to `max`; Gemini 4 Carbon defaults to `medium`. None
  of the preset models can turn thinking off, so each preset writes its
  model's lowest tier (`minimal` for muse-wonder-1.0, checked 2026-11-17
  against dev.meta.ai; `low` for the others) and "off" is their cheapest
  setting. A model that can disable reasoning takes `off_effort = "none"`. On
  the Anthropic wire format the value is sent as `output_config.effort`
  without a `thinking` field, since effort works with or without thinking.
  User-facing text must not promise "no thinking".
- The tier above `high` differs from model to model, so it is configuration
  and not code: `/reasoning max` sends the model's optional `max_effort` value
  from `config.toml`, and the wire format's default when that is unset. The
  defaults are `"max"` for Anthropic, `"xhigh"` for Responses (accepted by
  GPT-5.2 and later, every Muse model, and every current Grok, where 4.5
  treats it as `high`; an older OpenAI model that stops at `high` needs
  `max_effort = "high"`), and `"high"` for chat completions and Gemini. Models
  that list a higher tier opt in with `max_effort = "max"`: OpenAI's GPT-5.6,
  GPT-6, GPT-6.1, and GPT-6.2 families, Meta's muse-wonder-1.0, muse-spark-1.3, and
  muse-ember-1.0, and DeepSeek over chat completions. No Grok model lists
  `max`. Gemini 4 Neon and Gemini 4 Hydrogen list `xhigh` above `high` and opt
  in with `max_effort = "xhigh"`; Gemini 4 Carbon and Argon stop at `high`
  (checked 2026-11-27 against the model pages on aistudio.google.com/docs,
  where ai.google.dev now redirects). Built-in presets write the value their
  model needs. tellm checks only
  that the value is one token; an unsupported value errors explicitly at the
  provider and the user can lower the room's level.
- Gemini thinking levels are model-dependent: gemini-4-carbon,
  gemini-4-argon, gemini-3.8-flash, and gemini-3.6-flash accept
  low/medium/high. Carbon, Argon, and 3.8 Flash reject `minimal`, so `low` is
  the lowest `off_effort` they take (checked 2026-11-16; Carbon checked
  2026-11-17 against Google's Gemini 4 Carbon developer guide).
  gemini-4-neon accepts low/medium/high/xhigh with `high` the default and
  answers `minimal` with HTTP 400, so it takes `off_effort = "low"` and
  `max_effort = "xhigh"` (checked 2026-11-27 against Google Cloud's Gemini 4
  Neon developer's guide).
- Anthropic sends the model's optional `max_output_tokens` as `max_tokens`,
  and 16000 when it is unset: thinking tokens count toward `max_tokens`, and
  4096 risked mid-thought truncation at high/max effort. The setting exists
  only for the Anthropic wire format, and validation rejects it elsewhere,
  because the other formats send no output limit. The Claude Fable 5.2 preset
  writes 64000 so that a reply at `max` effort has room to finish: Anthropic
  says to set a large `max_tokens` at `high` and above on that model and gives
  no number for it, 64k is its starting point for Claude Opus 5 at `xhigh` or
  `max` (checked 2026-11-17 against platform.claude.com, effort), and the
  model's output maximum is 128k (checked 2026-11-16).
- A larger Anthropic limit is paid for in waiting, because requests are not
  streamed and the whole reply arrives in one HTTP response. Anthropic's SDKs
  refuse a non-streamed request above 21,333 `max_tokens`, the amount they
  expect to fit in their 10-minute timeout, and its guidance is to avoid a
  large `max_tokens` without streaming. tellm keeps the no-streaming boundary
  and follows the guidance for direct integrations instead: the Anthropic
  request timeout grows with the limit at that ratio and never drops below 10
  minutes (30 minutes at 64000), and the client sets a 60-second TCP
  keep-alive so that networks which drop idle connections do not drop a long
  request (checked 2026-11-16 against platform.claude.com, thinking and API
  errors). A request that still times out is an ordinary provider error.
- **Capability toggles are gated at the toggle**: `/imagegen on` and
  `/websearch on` refuse in rooms whose wire format or model id can never
  honor them (image generation: Anthropic/compat/xAI/Meta/non-image Gemini models;
  web search: compat), naming the model and endpoint. Support that still varies
  per model *inside* a capable format (for example, which OpenAI models can use
  the image tool) stays a request-time error, with the error reply suggesting
  the off command. Changing a room's model (`/model KEY`, `/model pin`,
  `/model unpin`) turns off the toggles the new model can never honor, by the
  same rule, and the reply says which, so a setting chosen under the old
  model can't make every following message fail.

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
- Handle Anthropic `stop_reason: max_tokens` (checked 2026-11-16 against
  platform.claude.com, stop reasons and the Claude Fable 5.2 migration
  guide): thinking counts toward the limit, so a reply can stop mid-answer or
  before any text. A cut-off answer is delivered with a closing note that says
  so, while the stored turn keeps the model's own content. A reply cut off
  before any text is a provider error, and the error reply suggests
  `/reasoning low` or a higher `max_output_tokens`.
- The same rule holds on the other wire formats (checked 2026-11-27
  against developers.openai.com, reasoning guide and API reference). A
  Responses reply with `status: incomplete` and `incomplete_details.reason:
  max_output_tokens` (which also covers a full context window) gets the
  cut-off note, another reason gets a general one, and `content_filter` is
  a refusal. A chat-completions reply with `finish_reason: length` gets the
  cut-off note. Either one without text is a provider error whose reply
  suggests `/reasoning low` or `/new summary`; `max_output_tokens` is named
  only for Anthropic, the one wire format that takes it.
- Handle Anthropic `stop_reason: model_context_window_exceeded` (checked
  2026-11-27 against platform.claude.com, handling stop reasons): the reply
  stopped because the chat filled the context window. A cut-off answer gets
  a closing note pointing at `/new summary`; one without text is the same
  context-window error as an oversized prompt, so the error reply offers
  `/new summary` too.
- A chat-completions 200 is not always an answer (checked 2026-11-27
  against openrouter.ai, API errors): a body with an `error` object and no
  `choices`, or with an `error` on `choices[0]` and `finish_reason: error`,
  is a provider error carrying that message, and a body with no choice at
  all is an error too, never an empty turn.
- Every client looks at the HTTP status before it parses the body. A
  success must be JSON; an error body that is not JSON (an HTML or
  plain-text page from a proxy or gateway in front of the API) becomes an
  API error with that status and a one-line excerpt, the page title for
  HTML, instead of a JSON decoding error that hides the status.
- A request over the model's context window is told apart from other API
  errors only where the provider documents an error for it (checked
  2026-11-19): Anthropic's HTTP 400 `invalid_request_error` "prompt is too
  long" (platform.claude.com, context windows); OpenAI's `error.code`
  `context_length_exceeded` (developers.openai.com: the Responses reference
  says only that such a request fails with a 400, and OpenAI's Agents API
  error table defines this code as "Input exceeds the model's context
  window"); and OpenRouter's `error.metadata.error_type`
  `context_length_exceeded` on chat completions (openrouter.ai, errors).
  xAI, Meta, Gemini, DeepSeek and the rest of the chat-completions dialect
  document none, so theirs stay ordinary API errors. A recognized one
  becomes `ProviderError::ContextWindowExceeded`: the chat's error reply
  suggests `/new summary`, and a summary request is retried with fewer
  turns (§ Summaries).
- A reply with neither text nor an image is a failed turn in every wire
  format: nothing is stored and the chat gets an error. Stored, it would be
  replayed as a question without a visible answer, and a reasoning-only turn
  can break the room, because Meta answers a reasoning item that sits directly
  before a user message with HTTP 400 (checked 2026-11-16 against
  dev.meta.ai).
- Anthropic thinking blocks are bound to everything sent before them:
  `system`, `tools`, and every earlier message (checked 2026-11-16 against
  platform.claude.com, preserved thinking / prefix checks). Replaying one
  after any of that changed is a 400 on Claude Fable 5.2 and can be one on the
  other models with preserved thinking. `system` cannot change under a kept
  history (`/role` resets it), so the edits tellm makes are the last-resort
  pruning, a summary request that leaves older turns out, and the
  web-search tool toggle. After any of them, `tellm-anthropic` removes every
  `thinking` and `redacted_thinking` block from the kept turns — all of them,
  because a block may not be left out from between two that stay — and the
  model continues without that earlier reasoning. An assistant message that
  held only thinking is removed, and a turn left with no assistant message is
  dropped whole.
- OpenAI/Meta/xAI Responses are used statelessly (`store: false`): prior
  provider-native input/output items are replayed in `input`, and every
  response `output` item is returned in `turn_items` verbatim. When reasoning
  is requested, `include: ["reasoning.encrypted_content"]` is set so encrypted
  reasoning can be replayed on later stateless turns. xAI and Meta get the
  flag at every level, `off` included: their models reason anyway, both return
  encrypted reasoning only on request (grok-4.7 always does), and Meta answers
  a replayed reasoning item without it with HTTP 400 (checked 2026-11-16
  against docs.x.ai and dev.meta.ai). OpenAI returns it by default and still
  accepts the flag.
- Every Responses provider gets the system prompt as the first item of
  `input`, never as top-level `instructions` (checked 2026-11-19). Models
  starting with GPT-6.2 append `instructions` after `input` and bill them at
  the uncached rate, and OpenAI says to put reusable instructions in a
  `developer` message as the first input item (developers.openai.com, latest
  model and keep the prefix unchanged). The role is `developer` for OpenAI
  and Meta, whose Responses docs accept it and recommend it for behavior
  instructions (dev.meta.ai), and `system` for xAI, whose docs put the
  system prompt first with that role and never mention `developer`
  (docs.x.ai). The role follows the provider detection in the next note,
  not a per-model list.
- A Responses request gets xAI or Meta handling when its base URL names the
  provider (`api.x.ai`, `api.meta.ai`) or, behind a proxy, when its model id
  starts with `grok-` or `muse-`; every Meta Model API id starts with `muse-`
  (checked 2026-11-17 against dev.meta.ai).
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
  `stream: false`, `reasoning_effort`, and image input via `image_url` data
  URLs, while leaving output length to the endpoint's default rather than
  sending `max_tokens` (checked 2026-07-04). Web search, image generation, and
  documents are reported as unsupported before any provider call.
- Gemini Interactions uses `POST /v1beta/interactions`, `x-goog-api-key`,
  `store:false`, `stream:false`, `system_instruction`, and stateless `input`
  as an array of Interactions `Step` objects (checked 2026-07-26 against
  ai.google.dev for Gemini 3.6 Flash; rechecked 2026-11-16 for Gemini 4
  Argon). Gemini 4 Carbon is a drop-in model id on the same API (checked
  2026-11-17 against Google's Gemini 4 Carbon developer guide), and so is
  Gemini 4 Neon, whose Interactions thinking levels are lowercase (checked
  2026-11-27 against Google Cloud's Gemini 4 Neon developer's guide).
  Requests omit deprecated sampling parameters and end in a `user_input`
  step, never a prefilled model turn. Long decode past 64K output tokens on
  Gemini 4 Carbon, Neon, and Argon needs background execution or a
  continuation token; tellm uses
  neither, so a reply that stops early comes back with a non-completed status
  and is reported as unsupported follow-up handling.
  Raw response `steps` are replayed through `turn_items`
  verbatim, including `thought` and Google Search call/result signatures.
  Web search is `tools: [{type:"google_search",
  search_types:["web_search"]}]`; image generation is allowed only for Gemini
  image model ids and sends `response_format: {"type":"image"}`;
  generated image content becomes `GeneratedImage`. The image-model gate is a
  fail-closed naming heuristic, not a capability-discovery API: an id counts
  when it contains `-image` (gemini-3.1-flash-image, gemini-3-pro-image) or
  `nano-banana` (gemini-nano-banana-2.1, the current image model; checked
  2026-11-16). Update it when Google's public image model ids change shape
  again.
- **File upload requires no extraction.** PDFs/images pass through natively:
  Anthropic `document`/`image` blocks, OpenAI `input_file`/`input_image`,
  Gemini `document`/`image` content blocks.
  `.txt` is read as text. Providers that can't take a part get a clear
  "unsupported" message, not silent degradation; for a PDF on a
  chat-completions model, the reply suggests sending a screenshot as a photo
  or switching to a model that reads PDFs.
  Telegram photos always arrive as JPEG, but an image sent as a file keeps
  whatever type the sender's client gave it. Anthropic and OpenAI take JPEG,
  PNG, GIF and WebP (checked 2026-11-27, platform.claude.com vision and
  developers.openai.com images and vision); Gemini also takes HEIC, HEIF,
  BMP and TIFF; xAI and Meta check their own. An image of another type is
  refused before the request is sent, and the reply suggests sending it as
  a photo.
- **No streaming anywhere.** Telegram delivers complete messages; providers
  are called non-streaming. This deletes the largest complexity class of the
  parent app.
- HTTP clients use explicit timeouts. Provider calls have a 10s connect timeout
  and long total timeout for multi-minute turns; production instances of each
  wire-format client share a process-wide `reqwest` connection pool. Telegram
  calls use bounded request/upload/download timeouts and long-poll timeout plus
  a grace window.

## Usage and cost

Every reply ends with what it cost, so a user can decide whether to continue.
tellm reads only what the provider already returns with the reply; it never
calls a billing API and never fetches prices.

- `ChatResponse.usage` is a provider-neutral `Usage` with disjoint buckets:
  ordinary input (neither read from nor written to a cache), cache reads,
  cache writes, output (reasoning included, since every provider bills
  reasoning as output), the reasoning part of that output for display,
  provider-native web searches, and the cost the provider reports, if any.
  Disjoint buckets matter because the shapes differ: OpenAI-style APIs count
  cache reads and writes inside their input total, while Anthropic reports
  them beside it. Each wire-format crate converts its own shape; a reply made
  of several requests (Anthropic `pause_turn`) sums them for cost, and
  records the last request's input plus output as the conversation's size,
  since each request resent everything before it. `usage` is `None`
  when the provider sent no usage block, and missing fields count as zero.
- Field mappings (checked 2026-11-18):
  - Anthropic Messages: `input_tokens` is input outside the cache;
    `cache_read_input_tokens` and `cache_creation_input_tokens` are reported
    beside it; `server_tool_use.web_search_requests` counts billed searches
    (platform.claude.com, prompt caching and pricing).
  - OpenAI Responses: `input_tokens` includes
    `input_tokens_details.cached_tokens` and
    `input_tokens_details.cache_write_tokens`, so ordinary input is the
    remainder; `output_tokens` includes `output_tokens_details.reasoning_tokens`
    (developers.openai.com, prompt caching). Web search is billed per call and
    counted as `web_search_call` output items (developers.openai.com pricing).
    Meta Model API reports the same OpenAI-shaped fields and no cost.
  - xAI Responses adds `usage.cost_in_usd_ticks`, the billed amount after
    caching discounts and including server-side tool fees, at 10^10 ticks per
    US dollar (docs.x.ai, cost tracking).
  - Chat completions: `prompt_tokens` includes
    `prompt_tokens_details.cached_tokens` and, where the endpoint bills
    writes, `prompt_tokens_details.cache_write_tokens`; `completion_tokens`
    includes `completion_tokens_details.reasoning_tokens` (openrouter.ai usage
    accounting). DeepSeek reports cache reads as `prompt_cache_hit_tokens`
    (api-docs.deepseek.com, context caching). OpenRouter's `usage.cost` is
    the amount charged in credits, which are US dollars; it is read only from
    an openrouter.ai URL, because the field has no shared meaning in the
    dialect.
  - Gemini Interactions: `total_input_tokens` is the prompt and
    `total_cached_tokens` the cached part of it; `total_thought_tokens` is
    added to `total_output_tokens`, since output is billed including thinking;
    `grounding_tool_count[].count` becomes web searches, which Google bills
    per query (Google's Interactions API reference and the js-genai
    Interactions `Usage` type; ai.google.dev pricing).
    `total_tool_use_tokens` is not priced: for Google Search it holds the
    grounding results, which Google's pricing does not charge as input
    (cloud.google.com Vertex AI pricing).
- Cost of one reply: a cost the provider reports wins, because it is what was
  billed. Otherwise the model's optional `price` table in `config.toml` gives
  an estimate: each bucket times its per-million-token price, plus searches
  times the per-1,000 search price. Only `input` and `output` are required; a
  missing cache price falls back to `input`, which can overstate a cached
  reply but never understates it, and a missing search price counts searches
  as free. Prices are the user's to keep current. Without a reported cost or a
  price, the reply shows its token counts instead.
- Delivery: the cost line follows the reply text after a blank line, or comes
  as its own message after an image-only reply. Only the delivered copy
  carries it; the stored turn keeps the model's own content, so the model
  never sees its bill. An estimate is marked `≈`; a chat total that includes
  replies with no known cost is marked `≥`.
- The line ends with the conversation's size against its token budget, in
  the same token format, for example `· 48k of 150k tokens` (§ Summaries).
  It is left out when the size is unknown: the provider reported no usage,
  or the room is in message mode. `/cost` shows the size too.
- `/cost` also shows the conversation's cache hit rate: cache reads as a
  share of all input tokens (ordinary, cache reads and cache writes), summed
  over its replies. It appears once a reply has reported nonzero input.
- Each room keeps the running total of its current conversation in memory,
  like the conversation itself. A reply's cost joins the total only when its
  turn is committed, so a reply dropped by a reset never counts toward the
  new chat. A summary reply (§ Summaries) counts toward the conversation it
  summarizes, and the conversation that carries it starts at zero. The total starts over with the conversation: `/new`, `/model`,
  `/role`, `/mode message`, and the terminal `reset`. A failed reply is not
  counted even though the provider may have billed it, so the provider's own
  billing stays the source of truth.

## Memory

- A short list of facts the owner curates (age, family, where they live,
  standing preferences) that tellm adds to the system prompt, so they don't
  have to be repeated in every new conversation. It holds facts, not
  conversations; conversations still don't persist.
- File: `config_dir()/memory.md`, next to `config.toml`. One fact per line,
  with an optional `- ` or `* ` bullet; lines starting with `#` are comments.
  tellm edits it line by line, so the owner's comments and layout survive.
  It is written atomically with mode `0600` and tightened to `0600` whenever
  it is read, since an editor can save it as `0644`; a file that can't be made
  private is not used. A missing file is an empty memory. Edits from
  different chats are serialized.
- Only the owner writes it, with `/remember` and `/forget` or by hand. Models
  get no tool to write it (see Non-goals), so it never collects a model's
  guesses.
- Size limit: 4,096 characters of facts (bullets and comments don't count),
  counted as Telegram counts text (UTF-16 code units), so the facts fit one
  Telegram message (checked 2026-11-28 against
  core.telegram.org/bots/api#sendmessage): about 1,000 tokens per request.
  `/remember` refuses to go over it. A
  hand-edited file over the limit, or one that can't be read, is not used: the
  conversation starts without memory and the chat gets a notice saying why.
- Where it applies: the room setting `memory` in `rooms.toml`. Unset, it is on
  only in an owner's direct chat with the bot, recognized by its chat id being
  an owner's user id (Bot API user dialog ids equal user ids; checked
  2026-11-18 against core.telegram.org/api/bots/ids). It is off in groups and
  in other people's chats unless the owner runs `/memory on` there, so the
  owner's facts reach no one else by default.
- System prompt: the room's role, a blank line, then a header line and the
  facts as a bullet list. A carried summary follows the memory block
  (§ Summaries).
- A conversation keeps the memory it started with. The runtime reads the file
  whenever it builds a request in a room with memory on, but uses it only for
  a request that sends no history: the first turn of a conversation and every
  message-mode request. Later turns reuse the block the room stored when an
  earlier turn committed. The system prompt therefore stays byte-stable for
  prompt caching, provider state is never replayed under a changed system
  prompt, and editing memory never resets a conversation the way `/role`
  does. Changes take effect in the next conversation. The stored block is
  cleared with the conversation (`/new`, `/model`, `/role`, `/mode message`,
  terminal `reset`).
- Privacy: in a room with memory on, every request carries the facts to that
  room's provider. tellm does not log them.

## Summaries

- `/new summary`, or tellm on its own when a chat reaches its budget, asks
  the room's current model to summarize the conversation, sends the summary
  to the chat, and starts a new conversation that carries it. Like `/new`,
  `/new summary` is not owner-gated.
- The summary request is an ordinary turn of the conversation: the same
  model, system prompt, full history, tools, web-search and image settings,
  and effort as any turn in the room, built by the same request path. Only
  the user message differs: a fixed instruction in `summary.rs` asking for
  compact notes (facts, decisions and why, open questions, next steps) in
  the conversation's language, without web search. The request prefix is
  therefore the one the provider has cached.
- Its reply is the old conversation's last reply: its cost joins that
  conversation's total and gets the usual cost line, but its turn is not
  stored. The conversation then resets, and the room keeps the summary text,
  in memory only like the conversation.
- The new conversation's system prompt is the role, the memory block, and
  then "Summary of the earlier part of this conversation:" with the summary,
  each separated by a blank line. The summary is fixed from the reset until
  the conversation ends, so the system prompt stays byte-stable like the
  memory block. `/new`, `/model`, `/role`, `/mode message` and the terminal
  `reset` clear it with the conversation.
- A chat without history, or in message mode, has nothing to summarize and
  is told so. A failed summary request, or a reply without text, leaves the
  conversation unchanged, and the chat says so.
- Budget: after each committed chat-mode reply the room records the
  conversation's size, that reply's input tokens in every bucket (ordinary,
  cache reads, cache writes) plus its output tokens, as the provider
  reported them; for a reply that took several requests, the last one's.
  Unknown usage records no size, and message mode records
  none. When the size reaches `CONTEXT_SUMMARY_TOKENS` (150,000), or the
  history passes the 4 MiB cap, tellm sends a short notice and runs the
  summary right after delivering that reply, while the provider's cache
  still holds the conversation. It runs in the chat's worker, so the next
  message waits for it. If it fails, the history stays, and the check runs
  again after the next reply. Each attempt resends the whole conversation,
  so after `MAX_AUTOMATIC_SUMMARY_FAILURES` (2) failures in one
  conversation tellm stops summarizing it on its own and says so; `/new
  summary` still tries on request, and a new conversation starts the count
  again. A history over the byte cap is still pruned.
- The budget is one constant for every provider, with no per-model or
  config setting (checked 2026-11-19). Anthropic's server-side compaction
  starts at 150,000 input tokens by default, and OpenAI's compaction
  examples use `compact_threshold: 200000`. The lowest long-context price
  step among the presets is Grok 4.7's: a prompt of 200K tokens or more
  bills the whole request at the higher rate. GPT-6.2 Astra's step is above
  272K input tokens, and Claude Fable 5.2 has none. Gemini 4 Carbon and Muse
  Wonder have windows of about 1M tokens, Grok 4.7 500K.
- When the provider says a summary request is over the model's context
  window (see Notes in § Unified parameter set), the request is sent again
  with the older half of the turns left out, until it fits or a single turn
  was too much. The kept turns go through their wire-format crate as after
  any history edit (§ Opaque provider state), and the summary message says
  how many of the oldest turns it leaves out.

## Runtime model (ported from console-chat-gpt)

- Long polling (`getUpdates`, modest timeout so terminal controls stay responsive).
- **Strictly ordered execution per chat via one tokio mpsc task per chat**
  (decided 2026-07-04; idle tasks are reaped). An idle task closes its
  queue and finishes whatever reached it before the close; the next
  message starts a new task that waits for the old one to end, so no
  message is lost and two tasks never serve one chat at once. Shutdown
  joins both. The channel gives ordering and
  backpressure for free and is the tokio-native equivalent of the parent's
  chained-futures design. The queue is bounded (32 per chat); when a room
  backs up behind a slow turn, further messages are dropped with a busy
  notice instead of blocking the poll loop for every other chat. At most one
  busy-notice send may be outstanding per room.
- Chats that are not allowed have no worker. The poll loop checks their
  `/pair` code itself, so attempts count in update order, and hands the
  save and the reply to a separate task; their hints are sent the same way.
  Nothing an unknown chat sends makes the poll loop wait on Telegram or disk.
- After each committed reply, the chat's worker checks the conversation's
  budget (§ Summaries) before it takes the next message, and summarizes the
  chat into a new one when it is due.
- In-memory conversations; **room settings persist** across restarts
  (model pin, mode, role, optional thinking override, web-search toggle,
  image-generation toggle, cost-line toggle, memory toggle) in `config_dir()/rooms.toml` — runtime state
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
  system prompt; `/model unpin` does the same even if the effective wire format
  does not change. `/websearch` keeps the conversation: it changes the tool
  list, not the instructions, so it only triggers the opaque-state repair
  described in § Opaque provider state. Terminal `reset` clears all in-memory
  histories while keeping room settings. Monotonic room generations prevent an
  older in-flight result or rollback from repopulating reset or revoked state.
- The `model_turn` boundary converts Telegram text, photos, and documents into
  core content parts. Downloads accept images, PDFs, and text documents up to
  20 MiB. Declared oversize files are rejected before download; unknown-length
  bodies are read incrementally and stopped at the same limit.
- Albums are one turn. Telegram delivers photos or files sent together as one
  message per item sharing a `media_group_id` (2-10 items, checked 2026-11-27
  against core.telegram.org/bots/api#sendmediagroup). The room's worker
  gathers the items that follow the first in its queue, each within 1 s of
  the last and up to 10, into one turn: every caption, then every item in
  order. A message that arrives right after the album is answered after it.
- A long text that a Telegram app split is one turn. The apps send a text
  over the message length limit as several messages, and the Bot API marks
  nothing about it (checked 2026-11-28 against the apps' source: Telegram
  Desktop cuts at the best break in the second half of its limit, 4,096 or
  8,192 with Premium; Android at most 300 characters before 4,096; iOS at
  the last line break or full stop before 4,096). So a text message of at
  least 2,000 UTF-16 code units, the unit Telegram counts, may be a part:
  the worker waits up to 2 s for the next, and a text message from the same
  sender dated within 2 s of the part before it joins, after a line break,
  while every part so far is that long. Shorter messages never wait. A part
  starting with `/` joins as text, and a long command such as `/role` is
  joined the same way. A part that arrives later, or an iOS first part
  shorter than 2,000, is still answered on its own. The log line shows the
  joined text as, for example, `input=text(5100,2 parts)`.
- Edited-message updates are deliberately ignored so correcting a Telegram
  message cannot silently trigger a second billed model call.
- Delivery: `sendRichMessage` → HTML `sendMessage` → plain text, with
  chunking at 32000/3900 chars. Fallback triggers ported from the Python
  implementation's error-marker list. Model selection uses a one-time Telegram
  reply keyboard with a text-command fallback. Generated images preserve their
  provider MIME type when uploaded. Telegram transport and API errors strip
  token-bearing request URLs and redact any echoed bot token before the error can
  reach logs or operator-facing output.
- Shutdown (terminal `exit`/`quit`, Telegram `/shutdown`, SIGINT/SIGTERM)
  stops polling, marks every room worker cancelled, aborts and joins them, and
  only then performs Ollama unload/child cleanup. Draining provider turns could
  block exit for minutes, while unloading before their futures are gone can
  immediately reload a model. Closing stdin merely disables terminal controls;
  it does not stop a daemonized runtime. It also ends any terminal key
  prompt, and from then on `/model add` replies at once that no key can be
  entered there and to run `tellm secret set NAME` instead. Only the tellm-started Ollama child
  gets automatic graceful cleanup (see § Configuration).

### Telegram commands (v1)

`/new` (incl. `summary`), `/id`, `/mode`, `/model` (incl. `pin`/`unpin`/`add`, owner-gated), `/role`,
`/reasoning`, `/websearch`, `/imagegen`, `/cost`, `/memory`, `/remember`,
`/forget` (the three memory commands owner-gated), `/allow`, `/deny`, `/pair`,
`/ollama unload` (owner-gated), `/shutdown` (owner-gated), `/help`.
Dropped as waste:
`/webfetch` (fold into websearch decision later), assistants, AI-managed mode,
MCP tooling.
Known commands accept Telegram's group form (`/cmd@BotName`); commands
addressed to a different bot are ignored. An unknown command (a name of
Telegram's command shape: 1–32 letters, digits or underscores, such as
`/start` or a typo) gets "Unknown command /NAME. Send /help for the list."
instead of a billed model reply that answers the command as if it were a
question. Text that only starts with `/`, such as a path, stays model input.
Argument semantics: `/new [summary]` starts a new conversation, and with
`summary` carries a summary of this one into it (§ Summaries); any other
argument is refused, so a mistyped `summary` never ends a conversation. `/id`
replies with the current Telegram chat id,
`/mode [chat|message]`, `/model [key]`, `/role [text]`
with `clear|off|none|reset` clearing the role, `/reasoning
[default|off|low|medium|high|max]`, `/websearch [on|off|status]` (no argument
toggles), `/imagegen [on|off|status]` (no argument toggles), `/cost
[on|off|status]` (no argument or `status` shows the conversation's running
cost, token breakdown, cache hit rate and size; `on`/`off` shows or hides
the cost line under each reply, on by default), `/memory [on|off|status]` (no
argument or `status` lists the saved facts with numbers, the characters used,
whether memory is on in this room, and the file's path; `on`/`off` sets the
room's memory setting),
`/remember TEXT` (whitespace and line breaks collapse to one line; a fact
already saved, ignoring case, is not added twice), `/forget N` (`#N` also
accepted),
`/ollama unload` to unload local Ollama models invoked by this tellm process,
and `/allow CHAT_ID` / `/deny CHAT_ID` from any owner, and `/pair CODE`.
If a chat id appears in a model's `telegram_chat_ids`, that model is the
locked room model: `/model` shows the pin, `/model KEY` refuses, `/help`
notes the pin, and any stored `rooms.toml` model selection is ignored for that
chat. Plain `/model KEY` is intentionally lighter-weight: it writes the room's
selection to `rooms.toml` and leaves `telegram_chat_ids` empty. Use `/model pin
KEY` only when a Telegram room should always be locked to that model across
restart and local room-setting changes. A stored selection whose model is no longer in
config.toml (removed or renamed) is ignored the same way: the room uses
`default_model`, and `/model` shows it as the default, until a model with
that key is configured again.

## Access control (ported + hardened)

- Default-deny with **per-room, re-armable pairing**: any contact from
  an unknown chat — a message, or the bot being added to a group (heard via
  `my_chat_member`) — arms a 6-digit code for THAT room, printed to the
  terminal only (never sent over Telegram) and announced with the group
  title. `/pair CODE` typed in the room approves it. Constant-time compare,
  10-minute per-room rotation, per-chat 5-attempt / 5-minute lockout.
  Because every chat has its own code and lockout, an attacker who opens
  many chats would otherwise get guesses in proportion to them, and a match
  makes the guesser an owner. So wrong codes also spend one process-wide
  budget of 20: once it is spent, pairing is off in every chat — even the
  right code is refused and no new codes are armed — until tellm restarts.
  The console logs this once. Owners still add the bot to groups without a
  code, so only pairing a new owner or a stranger's chat waits for the
  restart. That bounds an attacker to 20 guesses (one in 50,000) per
  process. Approving one room never disables pairing for others. Unknown chats get
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
  chat-based privilege) and makes owner-stranding via `/deny` structurally
  impossible. Caveat: Telegram anonymous-admin mode hides `from` — owners must
  post non-anonymously (or use a private chat) for privileged commands.
- After approval the room replies with its current model and the /model /
  /model pin next steps plus a one-time model reply keyboard; on group approval
  the console prints the privacy-mode hint.
- Registered owners can approve or revoke rooms from any chat they are in:
  `/allow CHAT_ID` adds the
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
- Runtime stderr diagnostics use an RFC 3339 local timestamp with milliseconds
  and UTC offset, followed by a level and component. Received-message
  breadcrumbs are `DEBUG` records containing chat id, message id, sender user
  id (`none` when Telegram names no sender), update kind, and route
  (`command` / `model` / `ignored`) only; never message content. The message
  id tells a second message from one delivered twice, and the sender tells
  the owner's messages from another group member's. Negative
  allowed or pinned chat ids produce one aggregated group-privacy warning plus
  a debug id list at startup because Telegram privacy mode can filter plain
  group text before tellm receives it. Repeated identical `getUpdates`
  failures are reported on the first failure and at most every 30 seconds;
  changed failures are reported immediately, followed by an `INFO` recovery
  record. Interactive prompts and explicit subcommand output are not log
  records and remain unprefixed.

## Configuration

- **API keys never travel through Telegram chat** (they would transit
  Telegram's servers and persist in history on both ends). Console-side
  entry: `tellm secret set NAME` or `/model add KEY` from a registered owner, which
  asks for the preset's secret in the tellm terminal using the same visible
  prompt style as first-run setup. If `KEY` is an already-configured custom
  model, `/model add KEY` instead prompts for that model's `api_key_secret`.
  The terminal-control reader has one prompt slot: while a prompt is active,
  terminal lines go to that prompt, `reset`/`exit`/`quit` are rejected as API
  keys rather than executed, and the prompt expires after five minutes so it
  cannot block later prompts indefinitely. The room that ran `/model add`
  waits on its prompt: messages sent there meanwhile are kept and answered
  in order once the prompt ends, and the first of them gets a single notice
  saying so, since the burst of replies would otherwise look like duplicates.
  Both paths store through the
  secret facade and are picked up per request without restart. On startup,
  tellm reads each unique configured provider `api_key_secret` once so OS
  keychain permission prompts happen before the first model call in that
  process; missing provider keys are logged but remain non-fatal until a model
  that requires them is used. The missing-key provider error names the console
  fallback.
- Non-secrets: human-editable TOML at `dirs::config_dir()/tellm/config.toml`.
  tellm reads it once at startup, so hand edits take effect at the next
  start. The runtime writes it only for `/allow`, `/deny`, pairing, an
  owner's group add, and `/model pin|unpin|add`. Each of those re-reads the
  file and applies just its own change to what is on disk, so edits made
  while tellm runs are kept; a file that no longer parses is left alone and
  the command fails. Comments are not kept, because the file is written back
  from its parsed values. Every file tellm writes (config.toml, rooms.toml,
  memory, credentials) is replaced atomically through a temp file and a
  rename beside it; a path that is a symlink is followed to its target, so
  a linked file stays linked.
  **Capability routing is explicit** (`wire_format` field per model) — never
  inferred from the spelling of user-chosen keys (parent-app lesson). The two
  ends of the reasoning scale follow the same rule: the optional per-model
  `max_effort` and `off_effort` name the provider values for
  `thinking = "max"` and `thinking = "off"` instead of tellm inferring them
  from `model_name`.
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
  `ollama serve` once with `OLLAMA_HOST=127.0.0.1:11434` and waits briefly
  for readiness. Ollama binds there by default (checked 2026-11-27 against
  docs.ollama.com FAQ), so tellm starts it only for `localhost` and
  `127.0.0.1`; a down `[::1]` endpoint is an error that says so. While a
  child tellm started is still running, tellm never starts another: an
  unreachable endpoint with a live child is an error naming its pid, and a
  child that has exited is reaped and may be started again. Before dispatching a new
  image input, tellm calls `POST /api/show` and requires the model's reported
  `capabilities` to contain `vision`; a missing capability, malformed metadata,
  or metadata API failure rejects the turn before inference (checked 2026-08-02
  against docs.ollama.com/api-reference/show-model-details). This dynamic check
  applies only to the recognized local Ollama endpoint; other compat endpoints
  retain their provider-defined image behavior. A tellm-started Ollama child is
  stopped during tellm shutdown, including terminal `exit`/`quit`, Telegram
  `/shutdown`, SIGINT, SIGTERM, and ordinary panic unwinds, after tellm asks
  Ollama to unload every local model whose request began in the current tellm
  process via `keep_alive: 0` (checked 2026-07-05 against
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
- `config.toml`, `rooms.toml`, `credentials.toml`, and `memory.md` use collision-safe unique
  same-directory temp files, file sync, and rename, so a crash or concurrent
  writer cannot expose a partial file. Unix parent-directory sync is attempted
  after commit; failure is a durability warning, not a false failed-commit
  signal. The root binary's `persistence` module owns one ordered thread for
  runtime config/room writes, so an aborted worker cannot detach an older write
  that lands after `/deny`. It also serializes room setting snapshots: failed
  ordinary mutations roll settings back without resurrecting invalidated
  history, while failed denied-room cleanup never restores the in-memory room.
  The files remain separate transactions: a reported I/O failure between
  config and room persistence can leave stale room settings on disk, but never
  restores access or conversation history.
- **Semantic validation at startup** (`Config::validate()`): default model
  exists; model keys are command-safe; model names and secret names are
  nonempty and secret names cannot use tellm's internal marker prefix;
  `max_effort` and `off_effort`, when present, are each one whitespace-free
  token; `max_output_tokens`, when present, is positive and belongs to an
  Anthropic model; every `price` value, when present, is a finite,
  non-negative number of US dollars, and unknown `price` fields are a parse
  error rather than silently ignored;
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
  keychain-first. Because the file can outrank the keychain, a
  credentials.toml that exists but cannot be read or parsed makes every
  stored secret count as missing (logged as an error) rather than letting a
  possibly stale keychain value through. A parse error names only the byte
  position, never the line, which may hold a key. An active environment override must be unset before rotation,
  since it remains the effective value. Credentials-file read/modify/write is
  serialized within one process; multiple tellm processes must not share and
  mutate the same config directory concurrently.
  No encrypted-file theater.
- First run: when `config.toml` is absent, the interactive wizard asks for a
  Telegram bot token, validates it with `getMe`, asks for one provider choice
  from the built-in list, which is each lab's most capable model that an
  ordinary API key can call (rechecked 2026-11-27: Claude Fable 5.2, GPT-6.2
  Astra, Grok 4.7, Muse Wonder 1.0 in preview, Gemini 4 Carbon), each with the
  list prices that could be checked against the provider's pricing page
  (Claude Fable 5.2 checked 2026-11-18, GPT-6.2 Astra 2026-11-19, Muse Wonder
  1.0 and Gemini 4 Carbon 2026-11-27; Grok needs none because xAI reports its
  billed cost). Gemini 4 Carbon gets the price Google charges from 2026-12-01,
  twice its launch price, and no search price, because Google's first 5,000
  searches a month are free and a missing search price counts searches as
  free. A secret whose `TELLM_*` variable is set is taken from it without a
  prompt and is not stored, since that variable overrides any stored value;
  a set but empty or non-UTF-8 variable stops the wizard with its name. The
  wizard validates the config it is about to write before it stores
  anything, then stores the other Telegram/provider
  secrets via the secret facade, writes nonsecret config last, and explains the
  `/pair CODE` claim step. EOF before completion exits once with an error rather
  than repeatedly prompting on a closed input stream.
  Target: install-to-chatting under two minutes, zero file editing.

## Porting method

**Golden tests, not transcription.** The pure functions (`chunk_text`,
`markdown_to_html`, fallback-trigger classification) get test vectors
generated from the Python reference implementation
(`../consoleChatGPT/console_gpt/telegram_bot.py`) before the Rust versions
are written. Network fallback chains get a small mocked-Telegram harness.

## Distribution

- `cargo-dist` → GitHub Releases with standalone archives (macOS arm64/x86_64,
  Linux GNU x86_64/arm64, Windows x86_64) plus SHA-256 checksums. rustls keeps
  provider transport free of OpenSSL linkage; the GNU Linux builds retain their
  normal glibc runtime dependency and are not described as fully static.
- The macOS binaries carry no Developer ID signature and are not notarized:
  the Apple Silicon one has only the linker's ad-hoc signature and the Intel
  one is unsigned. macOS therefore blocks a browser-downloaded copy until the
  user clears its quarantine mark or allows it in System Settings, and the
  keychain asks once per stored secret after each upgrade, because a binary
  without a Team ID is identified by its cdhash. Signing and notarizing
  would need a paid Apple Developer Program membership and the certificate
  in the release workflow's secrets. Local builds can avoid the keychain
  prompts with a free Apple Development certificate (README § Build from
  source).
- Brew tap once there's traction. A sample Dockerfile may exist but is never
  the headline (binary-first, not anti-Docker).

## Non-goals (v1)

- No MCP, no tools beyond provider-native web search / image generation.
- No streaming. No conversation persistence (the curated memory in § Memory
  holds facts, not conversations, and a carried summary lives in memory
  only, like the conversation). No multi-messenger.
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
