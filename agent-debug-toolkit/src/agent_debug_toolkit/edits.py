"""
Agent Debug Toolkit - Edit Tools Module

Production-quality editing tools for AI agents to reliably modify code files.
Implements Aider-style fuzzy diff application and multi-mode search/replace.

Tools:
- apply_unified_diff: Apply git-style diffs with fuzzy matching
- smart_edit: Search/replace with exact, regex, or fuzzy modes
- read_file_slice: Read specific line ranges for targeted editing
- format_code: Post-edit formatting with black/ruff

Design Principles:
1. Graceful degradation: exact → whitespace-insensitive → fuzzy matching
2. Diagnostic feedback: always return what happened to each hunk
3. Non-destructive option: dry_run mode for testing
4. Minimal dependencies: core functions use only stdlib
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal


# Type aliases
DiffStrategy = Literal["exact", "whitespace_insensitive", "fuzzy"]
EditMode = Literal["exact", "regex", "fuzzy"]
FormatStyle = Literal["black", "ruff", "isort"]


class HunkStatus(Enum):
    """Status of a diff hunk application."""
    APPLIED = "applied"
    SKIPPED_NO_CHANGE = "skipped_no_change"
    FAILED_NOT_FOUND = "failed_not_found"
    FAILED_AMBIGUOUS = "failed_ambiguous"
    FAILED_CONFLICT = "failed_conflict"


@dataclass
class HunkResult:
    """Result of applying a single diff hunk."""
    file: str
    hunk_index: int
    status: HunkStatus
    original_start: int | None = None
    original_end: int | None = None
    message: str | None = None

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "hunk_index": self.hunk_index,
            "status": self.status.value,
            "original_start": self.original_start,
            "original_end": self.original_end,
            "message": self.message,
        }


@dataclass
class EditReport:
    """Report of an edit operation."""
    success: bool
    message: str
    files_modified: list[str] = field(default_factory=list)
    hunks: list[HunkResult] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message": self.message,
            "files_modified": self.files_modified,
            "hunks": [h.to_dict() for h in self.hunks],
            "dry_run": self.dry_run,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        status = "✅ Success" if self.success else "❌ Failed"
        lines = [
            "# Edit Report",
            "",
            f"**Status**: {status}",
            f"**Message**: {self.message}",
            f"**Dry Run**: {self.dry_run}",
            "",
        ]

        if self.files_modified:
            lines.append("## Files Modified")
            for f in self.files_modified:
                lines.append(f"- `{f}`")
            lines.append("")

        if self.hunks:
            lines.append("## Hunk Details")
            for h in self.hunks:
                icon = "✅" if h.status == HunkStatus.APPLIED else "❌"
                lines.append(f"- {icon} Hunk {h.hunk_index}: {h.status.value}")
                if h.message:
                    lines.append(f"  - {h.message}")
            lines.append("")

        return "\n".join(lines)


# --- File I/O Helpers ---

def _read_file(path: Path) -> str:
    """Read file with UTF-8 encoding."""
    return path.read_text(encoding="utf-8")


def _write_file(path: Path, content: str) -> None:
    """Write file with UTF-8 encoding."""
    path.write_text(content, encoding="utf-8")


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for comparison (preserves structure)."""
    lines = text.splitlines()
    normalized = [line.rstrip() for line in lines]
    return "\n".join(normalized)


def _collapse_whitespace(text: str) -> str:
    """Collapse all whitespace for fuzzy comparison."""
    return "".join(text.split())


# --- Diff Parsing ---

@dataclass
class DiffHunk:
    """Represents a single hunk from a unified diff."""
    file_path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    context_lines: list[str]
    remove_lines: list[str]
    add_lines: list[str]
    raw_hunk: str


def parse_unified_diff(diff_text: str) -> list[DiffHunk]:
    """
    Parse a unified diff into structured hunks.

    Handles standard git diff format:
    --- a/path/to/file
    +++ b/path/to/file
    @@ -start,count +start,count @@
    """
    hunks = []
    current_file = None

    lines = diff_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]

        # File headers
        if line.startswith("--- "):
            # Start of new file diff
            i += 1
            continue

        if line.startswith("+++ "):
            # Extract file path
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            current_file = path
            i += 1
            continue

        # Hunk header
        if line.startswith("@@"):
            # Parse hunk header: @@ -start,count +start,count @@
            match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match and current_file:
                old_start = int(match.group(1))
                old_count = int(match.group(2) or 1)
                new_start = int(match.group(3))
                new_count = int(match.group(4) or 1)

                # Collect hunk content
                context = []
                removes = []
                adds = []
                raw_lines = [line]

                i += 1
                while i < len(lines):
                    hline = lines[i]
                    if hline.startswith("@@") or hline.startswith("---") or hline.startswith("+++"):
                        break
                    raw_lines.append(hline)

                    if hline.startswith("-"):
                        removes.append(hline[1:])
                    elif hline.startswith("+"):
                        adds.append(hline[1:])
                    elif hline.startswith(" ") or hline == "":
                        context.append(hline[1:] if hline.startswith(" ") else "")
                    else:
                        # Could be context without leading space
                        context.append(hline)
                    i += 1

                hunks.append(DiffHunk(
                    file_path=current_file,
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    context_lines=context,
                    remove_lines=removes,
                    add_lines=adds,
                    raw_hunk="\n".join(raw_lines),
                ))
                continue

        i += 1

    return hunks


# --- Fuzzy Matching ---

def find_best_match(
    search_lines: list[str],
    file_lines: list[str],
    strategy: DiffStrategy,
    threshold: float = 0.8,
) -> tuple[int, int, float] | None:
    """
    Find the best matching location for search_lines in file_lines.

    Returns (start_index, end_index, similarity_score) or None if no match found.
    """
    if not search_lines:
        return None

    search_text = "\n".join(search_lines)
    search_len = len(search_lines)

    best_match = None
    best_score = 0.0

    for i in range(len(file_lines) - search_len + 1):
        candidate_lines = file_lines[i:i + search_len]
        candidate_text = "\n".join(candidate_lines)

        if strategy == "exact":
            if candidate_text == search_text:
                return (i, i + search_len, 1.0)

        elif strategy == "whitespace_insensitive":
            norm_search = _normalize_whitespace(search_text)
            norm_candidate = _normalize_whitespace(candidate_text)
            if norm_search == norm_candidate:
                return (i, i + search_len, 1.0)

        else:  # fuzzy
            # Use sequence matcher for similarity
            matcher = difflib.SequenceMatcher(None, search_text, candidate_text)
            score = matcher.ratio()

            if score > best_score and score >= threshold:
                best_score = score
                best_match = (i, i + search_len, score)

    return best_match


# --- Main Edit Functions ---

def apply_unified_diff(
    diff: str,
    strategy: DiffStrategy = "fuzzy",
    project_root: Path | str | None = None,
    dry_run: bool = False,
    fuzzy_threshold: float = 0.8,
) -> EditReport:
    """
    Apply a unified diff to files with configurable matching strategy.

    Args:
        diff: Unified diff text (git diff format)
        strategy: Matching strategy - "exact", "whitespace_insensitive", or "fuzzy"
        project_root: Base directory for relative file paths
        dry_run: If True, don't actually write changes
        fuzzy_threshold: Minimum similarity for fuzzy matching (0.0-1.0)

    Returns:
        EditReport with success status and details for each hunk
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    hunks = parse_unified_diff(diff)

    if not hunks:
        return EditReport(
            success=False,
            message="No valid hunks found in diff",
            dry_run=dry_run,
        )

    # Group hunks by file
    file_hunks: dict[str, list[DiffHunk]] = {}
    for hunk in hunks:
        file_hunks.setdefault(hunk.file_path, []).append(hunk)

    all_results: list[HunkResult] = []
    files_modified: list[str] = []

    for file_path, file_hunk_list in file_hunks.items():
        full_path = project_root / file_path

        if not full_path.exists():
            for idx, hunk in enumerate(file_hunk_list):
                all_results.append(HunkResult(
                    file=file_path,
                    hunk_index=idx,
                    status=HunkStatus.FAILED_NOT_FOUND,
                    message=f"File not found: {full_path}",
                ))
            continue

        try:
            content = _read_file(full_path)
            file_lines = content.splitlines()
            modified = False

            # Apply hunks in reverse order to preserve line numbers
            for idx, hunk in enumerate(reversed(file_hunk_list)):
                real_idx = len(file_hunk_list) - idx - 1

                # Build the search pattern from context + remove lines
                search_lines = hunk.context_lines + hunk.remove_lines
                if not search_lines:
                    search_lines = hunk.remove_lines

                # For simple case: use remove_lines as the search pattern
                if hunk.remove_lines:
                    search_lines = hunk.remove_lines

                if not search_lines:
                    # Nothing to search for - this is an insertion
                    # Insert at the specified line number
                    insert_at = max(0, hunk.old_start - 1)
                    for add_line in reversed(hunk.add_lines):
                        file_lines.insert(insert_at, add_line)

                    all_results.append(HunkResult(
                        file=file_path,
                        hunk_index=real_idx,
                        status=HunkStatus.APPLIED,
                        original_start=insert_at,
                        original_end=insert_at,
                        message="Inserted new lines",
                    ))
                    modified = True
                    continue

                # Try to find matching location with fallback
                match = None
                used_strategy = strategy

                # Try strategies in order
                strategies: list[DiffStrategy] = [strategy]
                if strategy == "exact":
                    strategies = ["exact", "whitespace_insensitive", "fuzzy"]
                elif strategy == "whitespace_insensitive":
                    strategies = ["whitespace_insensitive", "fuzzy"]

                for strat in strategies:
                    match = find_best_match(search_lines, file_lines, strat, fuzzy_threshold)
                    if match:
                        used_strategy = strat
                        break

                if not match:
                    all_results.append(HunkResult(
                        file=file_path,
                        hunk_index=real_idx,
                        status=HunkStatus.FAILED_NOT_FOUND,
                        message=f"Could not find matching content with {strategy} strategy",
                    ))
                    continue

                start, end, score = match

                # Check if content already matches (no change needed)
                current_content = "\n".join(file_lines[start:end])
                new_content = "\n".join(hunk.add_lines)

                if _collapse_whitespace(current_content) == _collapse_whitespace(new_content):
                    all_results.append(HunkResult(
                        file=file_path,
                        hunk_index=real_idx,
                        status=HunkStatus.SKIPPED_NO_CHANGE,
                        original_start=start + 1,
                        original_end=end,
                        message="Content already matches (no change needed)",
                    ))
                    continue

                # Apply the replacement
                file_lines[start:end] = hunk.add_lines
                modified = True

                all_results.append(HunkResult(
                    file=file_path,
                    hunk_index=real_idx,
                    status=HunkStatus.APPLIED,
                    original_start=start + 1,
                    original_end=end,
                    message=f"Applied with {used_strategy} matching (score: {score:.2f})",
                ))

            if modified and not dry_run:
                _write_file(full_path, "\n".join(file_lines))
                files_modified.append(file_path)
            elif modified:
                files_modified.append(file_path)

        except Exception as e:
            all_results.append(HunkResult(
                file=file_path,
                hunk_index=0,
                status=HunkStatus.FAILED_CONFLICT,
                message=f"Error processing file: {str(e)}",
            ))

    applied_count = sum(1 for r in all_results if r.status == HunkStatus.APPLIED)
    total_count = len(all_results)

    return EditReport(
        success=applied_count > 0,
        message=f"Applied {applied_count}/{total_count} hunks to {len(files_modified)} file(s)",
        files_modified=files_modified,
        hunks=all_results,
        dry_run=dry_run,
    )


def smart_edit(
    file: Path | str,
    search: str,
    replace: str,
    mode: EditMode = "exact",
    dry_run: bool = False,
    fuzzy_threshold: float = 0.7,
    all_occurrences: bool = False,
) -> EditReport:
    """
    Perform intelligent search/replace in a file.

    Args:
        file: Path to file to edit
        search: Text or pattern to search for
        replace: Replacement text
        mode: "exact" (literal), "regex" (re pattern), or "fuzzy" (approximate)
        dry_run: If True, don't actually write changes
        fuzzy_threshold: Minimum similarity for fuzzy mode (0.0-1.0)
        all_occurrences: Replace all matches (default: first only)

    Returns:
        EditReport with success status and details
    """
    file = Path(file)

    if not file.exists():
        return EditReport(
            success=False,
            message=f"File not found: {file}",
            dry_run=dry_run,
        )

    try:
        content = _read_file(file)
        original_content = content

        if mode == "exact":
            if search not in content:
                return EditReport(
                    success=False,
                    message="Search string not found in file",
                    dry_run=dry_run,
                )

            if all_occurrences:
                new_content = content.replace(search, replace)
            else:
                new_content = content.replace(search, replace, 1)

        elif mode == "regex":
            pattern = re.compile(search, re.MULTILINE | re.DOTALL)
            if not pattern.search(content):
                return EditReport(
                    success=False,
                    message="Regex pattern did not match",
                    dry_run=dry_run,
                )

            if all_occurrences:
                new_content = pattern.sub(replace, content)
            else:
                new_content = pattern.sub(replace, content, count=1)

        else:  # fuzzy
            # Split into lines for fuzzy matching
            search_lines = search.splitlines()
            file_lines = content.splitlines()

            match = find_best_match(search_lines, file_lines, "fuzzy", fuzzy_threshold)

            if not match:
                return EditReport(
                    success=False,
                    message=f"No fuzzy match found above threshold {fuzzy_threshold}",
                    dry_run=dry_run,
                )

            start, end, score = match
            replace_lines = replace.splitlines()
            file_lines[start:end] = replace_lines
            new_content = "\n".join(file_lines)

        if new_content == original_content:
            return EditReport(
                success=True,
                message="No changes needed (content already matches)",
                dry_run=dry_run,
            )

        if not dry_run:
            _write_file(file, new_content)

        return EditReport(
            success=True,
            message=f"Successfully applied {mode} edit",
            files_modified=[str(file)],
            dry_run=dry_run,
        )

    except re.error as e:
        return EditReport(
            success=False,
            message=f"Invalid regex pattern: {e}",
            dry_run=dry_run,
        )
    except Exception as e:
        return EditReport(
            success=False,
            message=f"Error during edit: {str(e)}",
            dry_run=dry_run,
        )


def read_file_slice(
    file: Path | str,
    start_line: int,
    end_line: int,
    context_lines: int = 0,
) -> str:
    """
    Read a specific line range from a file.

    Args:
        file: Path to file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, inclusive)
        context_lines: Additional lines to include before/after

    Returns:
        The requested lines as a string
    """
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"File not found: {file}")

    content = _read_file(file)
    lines = content.splitlines()

    # Adjust for 1-indexed and context
    actual_start = max(0, start_line - 1 - context_lines)
    actual_end = min(len(lines), end_line + context_lines)

    selected = lines[actual_start:actual_end]

    # Add line numbers for context
    numbered_lines = []
    for i, line in enumerate(selected, start=actual_start + 1):
        prefix = "→ " if start_line <= i <= end_line else "  "
        numbered_lines.append(f"{i:4d}{prefix}{line}")

    return "\n".join(numbered_lines)


def format_code(
    path: Path | str,
    style: FormatStyle = "black",
    check_only: bool = False,
) -> EditReport:
    """
    Format code using a specified formatter.

    Args:
        path: Path to file or directory
        style: Formatter to use - "black", "ruff", or "isort"
        check_only: If True, only check formatting without modifying

    Returns:
        EditReport with success status
    """
    path = Path(path)

    if not path.exists():
        return EditReport(
            success=False,
            message=f"Path not found: {path}",
        )

    cmd: list[str] = []

    if style == "black":
        cmd = ["black"]
        if check_only:
            cmd.append("--check")
        cmd.append(str(path))
    elif style == "ruff":
        cmd = ["ruff", "format"]
        if check_only:
            cmd.append("--check")
        cmd.append(str(path))
    elif style == "isort":
        cmd = ["isort"]
        if check_only:
            cmd.append("--check-only")
        cmd.append(str(path))
    else:
        return EditReport(
            success=False,
            message=f"Unknown formatter: {style}",
        )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return EditReport(
                success=True,
                message=f"{'Checked' if check_only else 'Formatted'} with {style}",
                files_modified=[] if check_only else [str(path)],
            )
        else:
            # For check_only, non-zero means needs formatting (not an error)
            if check_only and result.returncode == 1:
                return EditReport(
                    success=False,
                    message=f"File needs formatting: {result.stdout or result.stderr}",
                )
            return EditReport(
                success=False,
                message=f"Formatter error: {result.stderr or result.stdout}",
            )

    except FileNotFoundError:
        return EditReport(
            success=False,
            message=f"Formatter '{style}' not found. Install with: pip install {style}",
        )
    except subprocess.TimeoutExpired:
        return EditReport(
            success=False,
            message="Formatter timed out after 30 seconds",
        )
    except Exception as e:
        return EditReport(
            success=False,
            message=f"Error running formatter: {str(e)}",
        )


# --- Convenience Functions ---

def create_simple_diff(
    file_path: str,
    old_content: str,
    new_content: str,
) -> str:
    """
    Create a unified diff from old and new content.

    Useful for generating diffs programmatically.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )

    return "".join(diff_lines)
