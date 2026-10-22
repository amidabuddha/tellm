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

- The maintainer's backlog (TICKETS.md) is deliberately local and untracked —
  a fresh clone does not have it. Contributors propose and pick up work
  through GitHub issues instead. One ticket/issue per commit series; keep
  commits scoped to one behavioral change with an imperative subject line and
  a body that explains why.
- Golden tests precede ports: `chunk_text` and `markdown_to_html` must be
  implemented against `crates/tellm-telegram/tests/golden/*.json` (remove the
  `#[ignore]` attributes; do not edit the vectors to make tests pass —
  regenerate them from the Python reference if you believe they are wrong,
  and say so in the commit body).

## Git workflow

- For routine commit-and-push requests, use `git` directly for staging,
  committing, and pushing. Do not invoke `gh` or a GitHub publishing workflow
  unless a GitHub-specific operation is actually required.
- Do not create pull requests for this repository unless the maintainer
  explicitly requests one. A request to commit and push means exactly those
  operations and stops after the push succeeds.
- Commit locally as work progresses. Push only when the maintainer asks;
  when a milestone is done, ask once whether to push now or not yet.
- `.github/workflows/release.yml` publishes a release when a version tag is
  pushed. Bump the version and tag only for an explicitly requested release;
  installing the local binary needs neither.

## Closure: install the binary and clean up

When a change is complete, close it out in this order:

1. Run the Verification gate below and commit locally.
2. Build the release binary and replace the installed one atomically:

   ```sh
   cargo build --release
   install -m 755 target/release/tellm ~/.local/bin/.tellm.new
   mv -f ~/.local/bin/.tellm.new ~/.local/bin/tellm
   ```

   `~/.local/bin` comes first on the maintainer's `PATH`. Never overwrite the
   installed file in place (for example `cp` onto it): macOS caches code
   signatures and can kill the rewritten binary.
3. Verify: `cmp target/release/tellm ~/.local/bin/tellm` succeeds and
   `command -v tellm` prints `~/.local/bin/tellm`.
4. Remove the build output with `cargo clean`. Keep `Cargo.lock`, source,
   `TICKETS.md` and configuration; secrets live in the keychain or the
   user's config directory, never in this repository.

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
