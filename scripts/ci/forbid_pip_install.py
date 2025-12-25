#!/usr/bin/env python3
"""Fail if committed scripts reintroduce pip installs.

Scope is intentionally narrow (scripts + key Makefiles) so we don't break
Dockerfiles, generated site outputs, or legacy docs that may mention pip.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections.abc import Iterable

DISALLOWED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "pip-install",
        re.compile(
            r"(?<!\buv\s)(?<!\buvx\s)pip3?\s+install\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "python-m-pip",
        re.compile(
            r"(?<!\buv\s)(?<!\buvx\s)python3?\s+-m\s+pip\b",
            flags=re.IGNORECASE,
        ),
    ),
]


def _iter_text_lines(path: pathlib.Path) -> Iterable[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    yield from enumerate(text.splitlines(), start=1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Files to check")
    args = parser.parse_args(argv)

    offenders: list[str] = []

    for file_str in args.files:
        path = pathlib.Path(file_str)
        if not path.exists() or path.is_dir():
            continue

        for lineno, line in _iter_text_lines(path):
            for label, pattern in DISALLOWED_PATTERNS:
                if pattern.search(line):
                    offenders.append(f"{path}:{lineno}: found {label}: {line.strip()}")

    if offenders:
        print("ERROR: Disallowed pip usage detected. Use uv equivalents (uv sync / uv run / uv run --with).")
        for msg in offenders:
            print(msg)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
