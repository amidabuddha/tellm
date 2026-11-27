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

- Propose and pick up work through GitHub issues. One issue per commit series; keep
  commits scoped to one behavioral change with an imperative subject line and
  a body that explains why.
- Golden tests precede ports: `chunk_text` and `markdown_to_html` must be
  implemented against `crates/tellm-telegram/tests/golden/*.json` (remove the
  `#[ignore]` attributes; do not edit the vectors to make tests pass —
  regenerate them from the Python reference if you believe they are wrong,
  and say so in the commit body).

## Documentation

- Docs describe the current state only. Do not add dated implementation logs, acceptance or validation evidence, release reports or completed-task lists; put evidence in commit messages.

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
2. Build the release binary, sign it, and replace the installed one
   atomically:

   ```sh
   cargo build --release
   codesign --force --sign "tellm local" --identifier tellm target/release/tellm
   install -m 755 target/release/tellm ~/.local/bin/.tellm.new
   mv -f ~/.local/bin/.tellm.new ~/.local/bin/tellm
   ```

   `~/.local/bin` comes first on the maintainer's `PATH`. Never overwrite the
   installed file in place (for example `cp` onto it): macOS caches code
   signatures and can kill the rewritten binary.

   `tellm local` is a self-signed code-signing identity in the maintainer's
   login keychain. The linker's ad-hoc signature changes with every build, so
   macOS would treat each build as a new app and ask for the login password
   once for every stored secret. Signed with the same identity and
   identifier, every build has the same designated requirement
   (`identifier tellm and certificate leaf = H"…"`), so a keychain
   "Always Allow" grant carries over to the next build. If the identity is
   missing, create a Code Signing certificate named `tellm local` with
   Keychain Access's Certificate Assistant; the first start after that asks
   once more per secret.
3. Verify: `cmp target/release/tellm ~/.local/bin/tellm` succeeds,
   `codesign -d -r- ~/.local/bin/tellm` shows `identifier tellm and
   certificate leaf`, and `command -v tellm` prints `~/.local/bin/tellm`.
4. Remove the build output with `cargo clean`. Keep `Cargo.lock`, source
   and configuration; secrets live in the keychain or the
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
