# Repository Instructions

These rules apply to every contributor — human, Claude, Codex, or otherwise.

## Source of truth

- **DESIGN.md is authoritative.** Any change that alters architecture,
  scope, security behavior, or a provider mapping MUST update DESIGN.md in
  the same commit. If code and DESIGN.md disagree, that is a bug in the
  commit that introduced the disagreement.
- The Python reference implementation lives in
  `../consoleChatGPT/console_gpt/telegram_bot.py`. Behavior ports are
  specified by golden test vectors generated from it, not by reading intent
  into the prose.

## Work packaging

- Tickets live in **TICKETS.md**. One ticket per commit series; keep commits
  scoped to one behavioral change with an imperative subject line and a body
  that explains why.
- Golden tests precede ports: `chunk_text` and `markdown_to_html` must be
  implemented against `crates/tellm-telegram/tests/golden/*.json` (remove the
  `#[ignore]` attributes; do not edit the vectors to make tests pass —
  regenerate them from the Python reference if you believe they are wrong,
  and say so in the commit body).

## API surface rule

- **Never write provider API version strings, parameter names, or defaults
  from memory or cached knowledge** — including your own training data and
  any bundled reference. Check the provider's live documentation at
  implementation time and record the check date in a source comment, e.g.
  `// checked 2026-07-04 against platform.claude.com`.
- Known pins as of 2026-07-04: Anthropic web search is
  `web_search_20260318` and MUST set `allowed_callers: ["direct"]` (from
  `_20260209` onward the default routes search through server-side code
  execution, which tellm does not want). Anthropic `budget_tokens` is
  rejected on current models — adaptive thinking + `output_config.effort`
  only.

## Verification (required before every commit)

```sh
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

(`--workspace` matters: from the root, plain `cargo test` runs only the
binary crate and silently skips every library's tests.)

CI (`.github/workflows/ci.yml`) runs this same gate on Linux, macOS, and
Windows for every push and pull request — the keychain stores are
platform-gated, so only CI compiles all three.

## Hard boundaries (do not cross without a DESIGN.md change)

- No streaming. No MCP. No agent tooling beyond provider-native web search /
  image generation. No SDK crates for providers — raw `reqwest` + `serde`.
- Secrets never enter `config.toml` or logs. No encrypted-file secret
  storage (keychain or 0600 plain file only).
- The unified core stays minimal; provider-specific state rides in opaque
  `turn_items` (see DESIGN.md § Opaque provider state), never as new core
  variants per provider feature.
