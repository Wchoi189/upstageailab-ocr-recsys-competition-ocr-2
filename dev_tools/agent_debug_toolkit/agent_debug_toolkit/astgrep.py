"""
Agent Debug Toolkit - AST-Grep Integration Module

Provides structural code search capabilities using ast-grep CLI.
Wraps the `ast-grep` command-line tool for use by AI agents.

Tools:
    - sg_search: Pattern-based structural code search
    - sg_lint: YAML rule-based linting
    - dump_syntax_tree: Visualize AST structure
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


# Type aliases
OutputFormat = Literal["json", "text"]
Language = Literal[
    "python", "javascript", "typescript", "java", "go", "rust",
    "c", "cpp", "csharp", "kotlin", "swift", "ruby", "php",
    "html", "css", "yaml", "json", "bash", "lua"
]


@dataclass
class Match:
    """A single ast-grep match result."""
    file: str
    text: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    language: str
    meta_variables: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "text": self.text,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_column": self.start_column,
            "end_column": self.end_column,
            "language": self.language,
            "meta_variables": self.meta_variables,
        }


@dataclass
class SearchReport:
    """Report from an ast-grep search operation."""
    success: bool
    message: str
    pattern: str = ""
    matches: list[Match] = field(default_factory=list)
    match_count: int = 0
    files_searched: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "pattern": self.pattern,
            "match_count": self.match_count,
            "files_searched": self.files_searched,
            "matches": [m.to_dict() for m in self.matches],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        lines = [
            "# AST-Grep Search Report",
            "",
            f"**Pattern**: `{self.pattern}`",
            f"**Status**: {'✅ Success' if self.success else '❌ Failed'}",
            f"**Matches**: {self.match_count}",
            "",
        ]

        if self.matches:
            lines.append("## Matches")
            lines.append("")

            for i, match in enumerate(self.matches, 1):
                lines.append(f"### Match {i}: `{match.file}:{match.start_line}`")
                lines.append("")
                lines.append("```")
                lines.append(match.text)
                lines.append("```")
                lines.append("")

        return "\n".join(lines)


@dataclass
class LintReport:
    """Report from an ast-grep lint operation."""
    success: bool
    message: str
    rule_file: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)
    violation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "rule_file": self.rule_file,
            "violation_count": self.violation_count,
            "violations": self.violations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --- Helper Functions ---

def _find_ast_grep() -> str | None:
    """Find the ast-grep CLI executable."""
    # Try 'ast-grep' first (pip installed version)
    path = shutil.which("ast-grep")
    if path:
        return path

    # Try 'sg' as fallback (cargo installed version)
    # Note: 'sg' might conflict with other tools
    sg_path = shutil.which("sg")
    if sg_path:
        # Verify it's actually ast-grep
        try:
            result = subprocess.run(
                [sg_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "ast-grep" in result.stdout.lower():
                return sg_path
        except (subprocess.TimeoutExpired, OSError):
            pass

    return None


def _parse_match(raw: dict[str, Any]) -> Match:
    """Parse a raw JSON match from ast-grep into a Match object."""
    range_info = raw.get("range", {})
    start = range_info.get("start", {})
    end = range_info.get("end", {})

    # Extract meta variables
    meta_vars = {}
    raw_meta = raw.get("metaVariables", {})
    if "single" in raw_meta:
        for name, info in raw_meta["single"].items():
            meta_vars[name] = info.get("text", "")
    if "multi" in raw_meta:
        for name, items in raw_meta["multi"].items():
            meta_vars[name] = [item.get("text", "") for item in items]

    return Match(
        file=raw.get("file", ""),
        text=raw.get("text", ""),
        start_line=start.get("line", 0) + 1,  # Convert to 1-indexed
        end_line=end.get("line", 0) + 1,
        start_column=start.get("column", 0),
        end_column=end.get("column", 0),
        language=raw.get("language", ""),
        meta_variables=meta_vars,
    )


# --- Main Functions ---

def sg_search(
    pattern: str,
    path: Path | str = ".",
    lang: Language | str | None = None,
    max_results: int | None = None,
    output_format: OutputFormat = "json",
) -> SearchReport:
    """
    Search code using ast-grep structural patterns.

    Args:
        pattern: AST pattern to match (e.g., "def $NAME($$$)")
        path: Path to search (file or directory)
        lang: Language to search (auto-detected if not specified)
        max_results: Maximum number of matches to return
        output_format: Output format preference

    Returns:
        SearchReport with matches
    """
    ast_grep = _find_ast_grep()
    if not ast_grep:
        return SearchReport(
            success=False,
            message="ast-grep CLI not found. Install with: pip install ast-grep-cli",
            pattern=pattern,
        )

    path = Path(path)
    if not path.exists():
        return SearchReport(
            success=False,
            message=f"Path not found: {path}",
            pattern=pattern,
        )

    # Build command
    cmd = [ast_grep, "run", "--pattern", pattern, "--json"]

    if lang:
        cmd.extend(["--lang", lang])

    cmd.append(str(path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # ast-grep returns empty output for no matches
        if not result.stdout.strip():
            return SearchReport(
                success=True,
                message="No matches found",
                pattern=pattern,
                match_count=0,
            )

        # Parse JSON output
        raw_matches = json.loads(result.stdout)

        # Convert to Match objects
        matches = [_parse_match(m) for m in raw_matches]

        # Apply max_results limit
        if max_results and len(matches) > max_results:
            matches = matches[:max_results]

        return SearchReport(
            success=True,
            message=f"Found {len(matches)} match(es)",
            pattern=pattern,
            matches=matches,
            match_count=len(matches),
        )

    except subprocess.TimeoutExpired:
        return SearchReport(
            success=False,
            message="Search timed out after 60 seconds",
            pattern=pattern,
        )
    except json.JSONDecodeError as e:
        return SearchReport(
            success=False,
            message=f"Failed to parse ast-grep output: {e}",
            pattern=pattern,
        )
    except Exception as e:
        return SearchReport(
            success=False,
            message=f"Search failed: {e}",
            pattern=pattern,
        )


def sg_lint(
    path: Path | str,
    rule: str | dict[str, Any] | None = None,
    rule_file: Path | str | None = None,
) -> LintReport:
    """
    Run ast-grep lint rules against code.

    Args:
        path: Path to lint (file or directory)
        rule: YAML rule as string or dict
        rule_file: Path to YAML rule file

    Returns:
        LintReport with violations
    """
    ast_grep = _find_ast_grep()
    if not ast_grep:
        return LintReport(
            success=False,
            message="ast-grep CLI not found. Install with: pip install ast-grep-cli",
        )

    path = Path(path)
    if not path.exists():
        return LintReport(
            success=False,
            message=f"Path not found: {path}",
        )

    # Handle inline rule
    temp_rule_file = None
    if rule and not rule_file:
        import yaml

        temp_rule_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        )
        if isinstance(rule, dict):
            yaml.dump(rule, temp_rule_file)
        else:
            temp_rule_file.write(rule)
        temp_rule_file.close()
        rule_file = temp_rule_file.name

    if not rule_file:
        return LintReport(
            success=False,
            message="Either 'rule' or 'rule_file' must be provided",
        )

    rule_file = Path(rule_file)
    if not rule_file.exists():
        return LintReport(
            success=False,
            message=f"Rule file not found: {rule_file}",
        )

    try:
        cmd = [ast_grep, "scan", "--rule", str(rule_file), "--json", str(path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if not result.stdout.strip():
            return LintReport(
                success=True,
                message="No violations found",
                rule_file=str(rule_file),
                violation_count=0,
            )

        violations = json.loads(result.stdout)

        return LintReport(
            success=True,
            message=f"Found {len(violations)} violation(s)",
            rule_file=str(rule_file),
            violations=violations,
            violation_count=len(violations),
        )

    except subprocess.TimeoutExpired:
        return LintReport(
            success=False,
            message="Lint timed out after 120 seconds",
            rule_file=str(rule_file),
        )
    except json.JSONDecodeError as e:
        return LintReport(
            success=False,
            message=f"Failed to parse lint output: {e}",
            rule_file=str(rule_file),
        )
    except Exception as e:
        return LintReport(
            success=False,
            message=f"Lint failed: {e}",
            rule_file=str(rule_file),
        )
    finally:
        # Clean up temporary rule file
        if temp_rule_file:
            Path(temp_rule_file.name).unlink(missing_ok=True)


def dump_syntax_tree(
    code: str,
    lang: Language | str,
) -> str:
    """
    Dump the syntax tree for a code snippet.

    Args:
        code: Code snippet to parse
        lang: Language of the code

    Returns:
        String representation of the AST
    """
    ast_grep = _find_ast_grep()
    if not ast_grep:
        return "Error: ast-grep CLI not found. Install with: pip install ast-grep-cli"

    # Write code to temp file
    suffix_map = {
        "python": ".py",
        "javascript": ".js",
        "typescript": ".ts",
        "java": ".java",
        "go": ".go",
        "rust": ".rs",
    }
    suffix = suffix_map.get(lang.lower(), ".txt")

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        # Use --debug-query to dump tree
        cmd = [
            ast_grep, "run",
            "--pattern", "$$$",  # Match everything
            "--lang", lang,
            "--debug-query", "cst",
            temp_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return result.stderr or result.stdout or "No tree output"

    except subprocess.TimeoutExpired:
        return "Error: AST dump timed out"
    except Exception as e:
        return f"Error: {e}"


def is_available() -> bool:
    """Check if ast-grep CLI is available."""
    return _find_ast_grep() is not None


def get_version() -> str | None:
    """Get the ast-grep version if available."""
    ast_grep = _find_ast_grep()
    if not ast_grep:
        return None

    try:
        result = subprocess.run(
            [ast_grep, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None
