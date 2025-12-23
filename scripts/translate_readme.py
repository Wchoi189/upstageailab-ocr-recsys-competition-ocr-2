#!/usr/bin/env python3
"""Translate README.md into README.<lang>.md.

Notes:
- Skips fenced code blocks (``` / ~~~).
- Protects inline code, HTML tags, and link URLs from translation.
- Uses deep-translator (GoogleTranslator) which does not require API keys.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProtectedText:
    text: str
    replacements: dict[str, str]


_PLACEHOLDER_PREFIX = "⟦⟦"
_PLACEHOLDER_SUFFIX = "⟧⟧"


def _make_placeholder(index: int) -> str:
    return f"{_PLACEHOLDER_PREFIX}{index}{_PLACEHOLDER_SUFFIX}"


def _protect_patterns(text: str, patterns: list[re.Pattern[str]]) -> ProtectedText:
    replacements: dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        placeholder = _make_placeholder(len(replacements))
        replacements[placeholder] = match.group(0)
        return placeholder

    protected = text
    for pattern in patterns:
        protected = pattern.sub(_repl, protected)

    return ProtectedText(text=protected, replacements=replacements)


def _restore_placeholders(text: str, replacements: dict[str, str]) -> str:
    # Restore in reverse length order to avoid partial overlaps (defensive).
    for placeholder in sorted(replacements.keys(), key=len, reverse=True):
        text = text.replace(placeholder, replacements[placeholder])
    return text


_FENCE_START_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})(?P<lang>[^`]*)$", re.IGNORECASE)


def _split_markdown_fences(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """Return list of (is_code_fence_block, lines)."""
    blocks: list[tuple[bool, list[str]]] = []
    current: list[str] = []
    in_fence = False
    fence_token: str | None = None

    for line in lines:
        if not in_fence:
            m = _FENCE_START_RE.match(line.rstrip("\n"))
            if m:
                if current:
                    blocks.append((False, current))
                    current = []
                in_fence = True
                fence_token = m.group("fence")
                current.append(line)
            else:
                current.append(line)
        else:
            current.append(line)
            # Fence ends when a line starts with the same fence token.
            if fence_token is not None and line.rstrip("\n").startswith(fence_token):
                blocks.append((True, current))
                current = []
                in_fence = False
                fence_token = None

    if current:
        blocks.append((in_fence, current))

    return blocks


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Chunk by blank lines to stay under max_chars."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"(\n\s*\n)", text)  # keep delimiters
    chunks: list[str] = []
    buf = ""

    for part in paragraphs:
        if not part:
            continue
        if len(buf) + len(part) <= max_chars:
            buf += part
        else:
            if buf:
                chunks.append(buf)
                buf = ""
            if len(part) <= max_chars:
                buf = part
            else:
                # Hard split if a single paragraph is huge.
                for i in range(0, len(part), max_chars):
                    chunks.append(part[i : i + max_chars])
                buf = ""

    if buf:
        chunks.append(buf)

    return chunks


def translate_markdown(readme_text: str, target_lang: str, max_chunk_chars: int = 3500) -> str:
    try:
        from deep_translator import GoogleTranslator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: deep-translator. Install with: pip install deep-translator"
        ) from e

    translator = GoogleTranslator(source="auto", target=target_lang)

    lines = readme_text.splitlines(keepends=True)
    blocks = _split_markdown_fences(lines)

    # Protect inline code, HTML tags, markdown link URLs, and angle-bracket autolinks.
    patterns: list[re.Pattern[str]] = [
        re.compile(r"`[^`\n]+`"),  # inline code
        re.compile(r"<[^>]+>"),  # HTML tags / autolinks
        re.compile(r"\]\(([^)\s]+)\)"),  # markdown link target (includes images)
        re.compile(r"\((https?://[^)\s]+)\)"),  # raw URL in parens
        re.compile(r"https?://\S+"),  # bare URLs
    ]

    out_parts: list[str] = []

    for is_code, block_lines in blocks:
        block_text = "".join(block_lines)
        if is_code:
            out_parts.append(block_text)
            continue

        protected = _protect_patterns(block_text, patterns)

        translated_chunks: list[str] = []
        for chunk in _chunk_text(protected.text, max_chunk_chars):
            if chunk.strip() == "":
                translated_chunks.append(chunk)
                continue
            translated_chunks.append(translator.translate(chunk))

        translated = "".join(translated_chunks)
        restored = _restore_placeholders(translated, protected.replacements)
        out_parts.append(restored)

    return "".join(out_parts)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate README.md to README.<lang>.md")
    parser.add_argument("--input", default="README.md", help="Input markdown file")
    parser.add_argument("--target-lang", required=True, help="Target language code (e.g., ko, ja, es)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: README.<lang>.md in repo root)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    target_lang = str(args.target_lang).strip()
    if not target_lang:
        raise SystemExit("--target-lang must be non-empty")

    out_path = Path(args.output) if args.output else Path(f"README.{target_lang}.md")

    readme_text = in_path.read_text(encoding="utf-8")
    translated = translate_markdown(readme_text, target_lang=target_lang)
    out_path.write_text(translated, encoding="utf-8")

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
