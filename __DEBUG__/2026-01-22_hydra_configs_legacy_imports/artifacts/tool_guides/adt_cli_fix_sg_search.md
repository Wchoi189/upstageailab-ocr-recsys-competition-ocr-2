# ADT CLI Fix: sg-search Pattern Argument Handling

**Issue**: `sg-search` command fails when pattern contains special shell characters
**Date**: 2026-01-23
**Status**: Fix needed in CLI argument parsing

---

## Problem Description

The `sg-search` command expects pattern as a positional argument, but shell interprets special characters before they reach the CLI:

```bash
# This fails - shell interprets $CFG as variable
adt sg-search "isinstance($CFG, dict)" --path ocr/file.py
# Result: Pattern becomes "isinstance(, dict)" - $CFG is empty

# This also fails - parentheses interpreted as subshell
adt sg-search isinstance($CFG, dict) --path ocr/file.py
```

---

## Root Cause Analysis

**Current CLI Definition** (line 432-437 in `cli.py`):
```python
@app.command("sg-search")
def sg_search_cmd(
    pattern: str = Argument(..., help="AST pattern to search for (e.g., 'def $NAME($$$)')"),
    path: str = Option(".", "--path", "-p", help="Path to search (file or directory)"),
    lang: Optional[str] = Option(None, "--lang", "-l", help="Language (python, javascript, etc.)"),
    # ...
):
```

**Issue**: Pattern is a positional `Argument`, which means:
1. Must be provided before options
2. Shell processes it before passing to CLI
3. Special characters (`$`, `(`, `)`, etc.) are interpreted by shell

---

## Shell Interpretation Problems

| Pattern                     | Shell Sees              | CLI Receives             | Result                                          |
| --------------------------- | ----------------------- | ------------------------ | ----------------------------------------------- |
| `isinstance($CFG, dict)`    | Variable `$CFG` (empty) | `isinstance(, dict)`     | ❌ Broken                                        |
| `"isinstance($CFG, dict)"`  | String with `$CFG`      | `isinstance(, dict)`     | ❌ Still broken (double quotes don't escape `$`) |
| `'isinstance($CFG, dict)'`  | Literal string          | `isinstance($CFG, dict)` | ✅ Works                                         |
| `isinstance\(\$CFG, dict\)` | Escaped chars           | `isinstance($CFG, dict)` | ✅ Works but ugly                                |

---

## Solutions

### Solution 1: Document Proper Quoting (Quick Fix)

**Update help text** to emphasize single quotes:

```python
@app.command("sg-search")
def sg_search_cmd(
    pattern: str = Argument(
        ...,
        help="AST pattern to search for. USE SINGLE QUOTES for patterns with $ or (). Example: 'isinstance($CFG, dict)'"
    ),
    # ...
):
```

**Pros**: No code changes
**Cons**: Users will still make mistakes

### Solution 2: Make Pattern an Option (Recommended)

**Change pattern from positional to named option**:

```python
@app.command("sg-search")
def sg_search_cmd(
    pattern: str = Option(..., "--pattern", "-p", help="AST pattern to search for"),
    path: str = Option(".", "--path", help="Path to search (file or directory)"),
    lang: Optional[str] = Option(None, "--lang", "-l", help="Language"),
    # ...
):
```

**Usage**:
```bash
# Now works with double quotes
adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/file.py

# Or with equals sign (bypasses some shell issues)
adt sg-search --pattern="isinstance($CFG, dict)" --path=ocr/file.py
```

**Pros**: More robust, clearer syntax
**Cons**: Breaking change for existing users

### Solution 3: Hybrid Approach (Best)

**Accept pattern as either positional OR option**:

```python
@app.command("sg-search")
def sg_search_cmd(
    pattern: Optional[str] = Argument(None, help="AST pattern (use single quotes for special chars)"),
    pattern_opt: Optional[str] = Option(None, "--pattern", "-p", help="AST pattern (alternative to positional)"),
    path: str = Option(".", "--path", help="Path to search"),
    lang: Optional[str] = Option(None, "--lang", "-l", help="Language"),
    # ...
):
    # Use whichever is provided
    actual_pattern = pattern_opt if pattern_opt else pattern

    if not actual_pattern:
        typer.echo("Error: Pattern is required (provide as argument or --pattern)", err=True)
        raise typer.Exit(1)
```

**Pros**: Backward compatible, flexible
**Cons**: Slightly more complex

---

## Recommended Implementation

**Use Solution 2** (make pattern an option) with clear migration path:

### Step 1: Update CLI (cli.py lines 431-464)

```python
@app.command("sg-search")
def sg_search_cmd(
    pattern: str = Option(..., "--pattern", "-p", help="AST pattern to search for (e.g., 'isinstance($CFG, dict)')"),
    path: str = Option(".", "--path", help="Path to search (file or directory)"),
    lang: Optional[str] = Option(None, "--lang", "-l", help="Language (python, javascript, etc.)"),
    max_results: Optional[int] = Option(None, "--max", "-m", help="Maximum number of results"),
    output: str = Option("markdown", "--output", "-o", help="Output format: json or markdown"),
):
    """
    Search code using ast-grep structural patterns.

    Examples:
      adt sg-search --pattern "def $NAME($$$)" --lang python
      adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/
      adt sg-search -p "function $NAME($$$)" --path src/ -l javascript
    """
    from agent_debug_toolkit.astgrep import sg_search

    report = sg_search(
        pattern=pattern,
        path=path,
        lang=lang,
        max_results=max_results,
        output_format=output,
    )

    if not report.success:
        typer.echo(f"Error: {report.message}", err=True)
        raise typer.Exit(1)

    if output == "markdown":
        typer.echo(report.to_markdown())
    else:
        typer.echo(report.to_json())
```

### Step 2: Update Documentation

Update all references to `sg-search` in:
- `adt_usage_patterns.md`
- `duplicate_file_detection.md`
- `instruction_patterns.md`
- AgentQMS standards

**Old**:
```bash
adt sg-search "isinstance($CFG, dict)" --path ocr/file.py
```

**New**:
```bash
adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/file.py
```

### Step 3: Add Migration Note

Add to CLI help or CHANGELOG:
```
BREAKING CHANGE (v0.x.x): sg-search pattern is now a named option (--pattern)
instead of positional argument for better shell compatibility.

Old: adt sg-search "pattern" --path file.py
New: adt sg-search --pattern "pattern" --path file.py
```

---

## Testing

After fix, verify these work:

```bash
# Basic pattern
adt sg-search --pattern "def $NAME($$$)" --path ocr/

# Pattern with special chars
adt sg-search --pattern "isinstance($CFG, dict)" --path ocr/core/

# Pattern with parentheses
adt sg-search --pattern "hydra.utils.instantiate($$$)" --path ocr/

# Using equals sign
adt sg-search --pattern="get_model_by_cfg($$$)" --path=ocr/

# Short form
adt sg-search -p "class $NAME" --path ocr/
```

---

## Alternative: Escape Helper

If keeping positional argument, add helper function:

```python
def escape_pattern_for_shell(pattern: str) -> str:
    """Escape special shell characters in AST pattern."""
    # Escape $ and ()
    escaped = pattern.replace('$', '\\$')
    escaped = escaped.replace('(', '\\(')
    escaped = escaped.replace(')', '\\)')
    return escaped
```

But this is less user-friendly than making it an option.

---

## Impact on Doc-Sync Audit Tool

The `doc_sync_audit.py` script also needs updating:

**Current** (broken):
```python
cmd = f"adt sg-search --pattern \"{escaped_pattern}\" --path {target_dir}"
```

**After fix**:
```python
cmd = f"adt sg-search --pattern \"{escaped_pattern}\" --path {target_dir}"
# Works correctly now that --pattern is an option
```

---

## Summary

**Recommended Fix**: Change pattern from positional `Argument` to named `Option`

**Files to Update**:
1. `agent-debug-toolkit/src/agent_debug_toolkit/cli.py` (lines 431-464)
2. Documentation in `__DEBUG__/*/artifacts/`
3. `AgentQMS/tools/compliance/doc_sync_audit.py`

**Benefit**: Robust handling of special characters without requiring users to remember quoting rules

**Migration**: One-line change in user commands (add `--pattern`)
