#!/usr/bin/env python3
"""Golden test vector generator for tellm-telegram pure functions.

The functions and predicates below are VERBATIM copies or direct extractions
from the reference implementation
in console-chat-gpt (console_gpt/telegram_bot.py, commit a9e560d):
`_chunk_text`, `_telegram_markdown_to_html`, and the rich/HTML delivery fallback
classifiers. Do not "improve" them here — their exact behavior, quirks included,
is the specification.

Run from this directory:  python3 generate_vectors.py
Outputs: chunk_text.json, markdown_to_html.json, fallback_classification.json

Note for the Rust implementer: Python string indices are Unicode code
points, not bytes. The emoji/cyrillic cases exist to fail byte-indexed
implementations by construction.
"""

import html
import json
import re
from typing import List

# --- verbatim from console_gpt/telegram_bot.py -----------------------------


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            split = text.rfind("\n", start, end)
            if split > start:
                end = split + 1
        chunks.append(text[start:end])
        start = end
    return chunks


def _telegram_markdown_to_html(text: str) -> str:
    """Convert a subset of markdown-like model output to Telegram-safe HTML."""
    if not text:
        return ""

    fenced_blocks: List[str] = []

    def _capture_fenced_block(match: "re.Match[str]") -> str:
        block = match.group(1) or ""
        escaped = html.escape(block.strip("\n"))
        fenced_blocks.append(f"<pre><code>{escaped}</code></pre>")
        return f"@@TG_CODEBLOCK_{len(fenced_blocks) - 1}@@"

    without_fenced = re.sub(r"```(?:[^\n`]+)?\n([\s\S]*?)```", _capture_fenced_block, text)
    escaped_text = html.escape(without_fenced)

    lines: List[str] = []
    for line in escaped_text.split("\n"):
        heading_match = re.match(r"^\s{0,3}#{1,6}\s+(.+)$", line)
        if heading_match:
            lines.append(f"<b>{heading_match.group(1).strip()}</b>")
        else:
            lines.append(line)
    transformed = "\n".join(lines)

    transformed = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        transformed,
    )
    transformed = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", transformed)
    transformed = re.sub(r"__(.+?)__", r"<b>\1</b>", transformed)
    transformed = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", transformed)

    for idx, block in enumerate(fenced_blocks):
        transformed = transformed.replace(f"@@TG_CODEBLOCK_{idx}@@", block)

    return transformed


def _should_fallback_from_rich_message_error(error: RuntimeError) -> bool:
    error_text = str(error).lower()
    return any(
        marker in error_text
        for marker in (
            "api error 404",
            "method not found",
            "can't parse",
            "cannot parse",
            "failed to parse",
            "can't find end",
            "unsupported start tag",
            "unsupported tag",
        )
    )


def _should_fallback_from_html_message_error(error: RuntimeError) -> bool:
    # Direct extraction of the predicate in `_send_legacy_message_part`.
    error_text = str(error).lower()
    return "parse entities" in error_text or "can't parse entities" in error_text


# --- test cases -------------------------------------------------------------

CHUNK_CASES = [
    ("empty", "", 10),
    ("shorter_than_chunk", "hello", 10),
    ("exactly_chunk_size", "0123456789", 10),
    ("one_over_no_newline", "0123456789A", 10),
    ("hard_cut_no_newlines", "A" * 25, 10),
    ("prefers_newline_break", "aaaa\nbbbb\ncccc\ndddd\neeee", 10),
    ("newline_at_window_start_not_used", "\n" + "B" * 15, 10),
    ("newline_only_string", "\n\n\n\n\n\n\n\n\n\n\n\n", 5),
    ("trailing_newline", "line one\nline two\nline three\n", 12),
    ("windows_crlf", "one\r\ntwo\r\nthree\r\nfour\r\nfive", 10),
    ("cyrillic_code_points", "привет мир это тест на кириллице " * 3, 20),
    ("emoji_astral_plane", "🚀🚀🚀 launch 🚀🚀🚀\n" * 4, 12),
    (
        "realistic_markdown",
        "# Title\n\nSome intro paragraph that is fairly long and rambles on.\n\n"
        "- bullet one\n- bullet two\n- bullet three\n\n"
        "```python\nprint('hello world')\n```\n\nClosing thoughts here.",
        60,
    ),
    ("long_single_line_then_newline", "X" * 30 + "\n" + "Y" * 5, 20),
    ("multibyte_boundary_stress", "ä" * 21, 10),
    ("chunk_size_one", "ab\ncd", 1),
]

MD_CASES = [
    ("empty", ""),
    ("plain_text", "Just a plain sentence."),
    ("html_escaping", 'Tags like <b> & "quotes" and \'single\' get escaped'),
    ("bold_asterisks", "This is **bold** text"),
    ("bold_underscores", "This is __also bold__ text"),
    ("bold_not_across_newline", "**not\nbold** stays literal asterisks"),
    ("inline_code", "Run `cargo test` to verify"),
    ("inline_code_escapes_inside", "Use `a < b && c > d` carefully"),
    ("heading_h1", "# Big Title"),
    ("heading_h3_leading_spaces", "   ### Indented heading   "),
    ("heading_too_indented_is_text", "    # four spaces is not a heading"),
    ("link_simple", "See [the docs](https://example.com/docs) for more"),
    ("link_with_query_amp", "Go to [search](https://example.com/a?b=1&c=2) now"),
    ("fenced_no_language", "before\n```\ncode here\n```\nafter"),
    ("fenced_with_language", "```python\nprint('hi')\n```"),
    (
        "fenced_preserves_literals",
        "```rust\nlet x = a < b && c > d; // **not bold** `not code`\n```",
    ),
    ("fenced_multiple_blocks", "```\nfirst\n```\nmiddle **bold**\n```\nsecond\n```"),
    ("fenced_unclosed_stays_raw", "```python\nprint('never closed')"),
    ("fenced_strips_outer_newlines", "```\n\n\npadded\n\n\n```"),
    # Adversarial cases for regex-engine semantics that a
    # hand-rolled parser can miss.
    ("quad_backtick_fence", "````\ncode\n```"),
    ("bracket_in_link_label", "[see [1]](https://x.org) end"),
    ("multiline_bold_second_open", "**a\nb** and **c**"),
    ("seven_hashes_not_a_heading", "####### seven"),
    ("heading_only_whitespace_after_hashes", "##   "),
    ("empty_bold_stays_literal", "****"),
    ("triple_underscore_bold", "___x___"),
    ("backtick_in_fence_language_no_fence", "```a`b\ncode\n```"),
    (
        "kitchen_sink",
        "# Report\n\nThe value of `x` is **42** — see [ref](https://x.org/p?a=1&b=2).\n\n"
        "```sh\necho \"a > b\" && ls\n```\n\n"
        "__Done__ & dusted <end>",
    ),
]

FALLBACK_CASES = [
    ("rich_api_404", "rich", "Telegram API error 404: not found"),
    ("rich_method_not_found", "rich", "Bad Request: method not found"),
    ("rich_cant_parse", "rich", "Bad Request: can't parse rich message"),
    ("rich_cannot_parse", "rich", "Bad Request: cannot parse payload"),
    ("rich_failed_to_parse", "rich", "Bad Request: failed to parse block"),
    ("rich_cant_find_end", "rich", "Bad Request: can't find end of entity"),
    ("rich_unsupported_start_tag", "rich", "Unsupported start tag \"foo\""),
    ("rich_unsupported_tag", "rich", "Unsupported tag \"foo\""),
    ("rich_case_insensitive", "rich", "METHOD NOT FOUND"),
    ("rich_marker_inside_context", "rich", "remote said: API ERROR 404; retry"),
    ("rich_html_entity_overlap", "rich", "Bad Request: can't parse entities"),
    ("rich_chat_not_found", "rich", "Bad Request: chat not found"),
    ("rich_api_400", "rich", "Telegram API error 400: bad request"),
    ("rich_unsupported_end_tag", "rich", "Unsupported end tag \"foo\""),
    ("rich_empty", "rich", ""),
    ("html_parse_entities", "html", "Bad Request: parse entities failed"),
    ("html_cant_parse_entities", "html", "Bad Request: can't parse entities"),
    ("html_case_insensitive", "html", "PARSE ENTITIES"),
    ("html_generic_parse_failure", "html", "Bad Request: failed to parse block"),
    ("html_chat_not_found", "html", "Bad Request: chat not found"),
    ("html_empty", "html", ""),
]


def main() -> None:
    chunk_vectors = [
        {
            "name": name,
            "input": text,
            "chunk_size": size,
            "expected": _chunk_text(text, size),
        }
        for name, text, size in CHUNK_CASES
    ]
    md_vectors = [
        {"name": name, "input": text, "expected": _telegram_markdown_to_html(text)}
        for name, text in MD_CASES
    ]
    fallback_vectors = []
    for name, classifier, text in FALLBACK_CASES:
        error = RuntimeError(text)
        if classifier == "rich":
            expected = _should_fallback_from_rich_message_error(error)
        elif classifier == "html":
            expected = _should_fallback_from_html_message_error(error)
        else:
            raise ValueError(f"unknown fallback classifier: {classifier}")
        fallback_vectors.append(
            {
                "name": name,
                "classifier": classifier,
                "input": text,
                "expected": expected,
            }
        )

    with open("chunk_text.json", "w", encoding="utf-8") as f:
        json.dump(chunk_vectors, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open("markdown_to_html.json", "w", encoding="utf-8") as f:
        json.dump(md_vectors, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open("fallback_classification.json", "w", encoding="utf-8") as f:
        json.dump(fallback_vectors, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"wrote {len(chunk_vectors)} chunk_text vectors, "
        f"{len(md_vectors)} markdown_to_html vectors, and "
        f"{len(fallback_vectors)} fallback classification vectors"
    )


if __name__ == "__main__":
    main()
