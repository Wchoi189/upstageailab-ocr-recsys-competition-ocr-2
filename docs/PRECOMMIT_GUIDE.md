# Pre-commit Hooks Guide

This guide explains how to use pre-commit hooks to catch linting errors before committing code.

## What are Pre-commit Hooks?

Pre-commit hooks are scripts that run automatically before each `git commit`. They check your code for issues and can automatically fix many problems, preventing linting errors from ever reaching CI.

## Current Configuration

The project already has pre-commit configured in `.pre-commit-config.yaml` with:

- **Ruff linter** - Catches and fixes Python linting errors
- **Ruff formatter** - Formats Python code consistently
- **Mypy** - Type checking for Python code
- **Trailing whitespace** - Removes trailing spaces
- **End of file fixer** - Ensures files end with newline
- **YAML checker** - Validates YAML syntax
- **Large file checker** - Prevents committing large files
- **Merge conflict checker** - Detects merge conflict markers
- **Debug statement checker** - Finds leftover debug statements

## Installation

### One-time Setup

```bash
# Install pre-commit hooks
make pre-commit-install

# Or manually:
pre-commit install
```

This installs the hooks into your local `.git/hooks/` directory.

### Verification

```bash
# Test that hooks are installed
pre-commit run --all-files
```

## Usage

### Automatic (Recommended)

Once installed, hooks run automatically on `git commit`:

```bash
# Make some changes
echo "import os" > myfile.py

# Try to commit
git add myfile.py
git commit -m "Add new file"

# Output:
# ruff-linter-fixer........................................Failed
# - hook id: ruff
# - exit code: 1
#
# myfile.py:1:8: F401 [*] `os` imported but unused
#
# [*] 1 fixable with the `--fix` option.
```

The hook will:
1. Detect the unused import
2. **Automatically fix it** (ruff has `--fix` enabled)
3. Show you what was fixed
4. **Abort the commit** so you can review changes

Then you can:
```bash
# Review the auto-fixes
git diff

# Add the fixes and commit again
git add myfile.py
git commit -m "Add new file"
```

### Manual Run

Run hooks manually without committing:

```bash
# Run on all files
make pre-commit-run

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run ruff

# Run on specific files
pre-commit run --files path/to/file.py
```

### Skipping Hooks

Sometimes you need to bypass hooks (use sparingly!):

```bash
# Skip all hooks for this commit
git commit --no-verify -m "Emergency fix"

# Or set environment variable
SKIP=ruff git commit -m "Skip only ruff"
```

## Updating Hooks

Pre-commit hook versions can become outdated:

```bash
# Update to latest versions
make pre-commit-update

# Or manually:
pre-commit autoupdate
```

## Troubleshooting

### "command not found: pre-commit"

Install pre-commit:
```bash
uv pip install pre-commit --system
# Or: pip install pre-commit
```

### Hooks not running

Reinstall hooks:
```bash
pre-commit uninstall
make pre-commit-install
```

### Hook fails on every commit

1. Check what's failing:
   ```bash
   pre-commit run --all-files --verbose
   ```

2. Fix the issues manually or let ruff auto-fix:
   ```bash
   make lint-fix
   ```

3. If a hook is broken, you can disable it temporarily in `.pre-commit-config.yaml`

### Slow hook execution

Some hooks (like mypy) can be slow. You can:

1. Run only fast hooks during commit:
   ```bash
   SKIP=mypy git commit -m "message"
   ```

2. Run slow hooks manually before pushing:
   ```bash
   pre-commit run mypy --all-files
   ```

## Integration with AI Linter

Pre-commit hooks work alongside the AI-powered linting system:

1. **Pre-commit** (local): Catches issues before commit
2. **CI linting** (GitHub): Catches issues that slip through
3. **AI autofix** (GitHub Actions): Fixes remaining issues automatically

### Best Practice Workflow

```bash
# 1. Make changes
vim myfile.py

# 2. Pre-commit catches issues on commit
git commit -m "Update myfile"
# → Ruff auto-fixes simple issues
# → Commit aborted for review

# 3. Review and re-commit
git add myfile.py
git commit -m "Update myfile"
# → All checks pass!

# 4. Push to GitHub
git push
# → CI runs, no errors
# → No need for AI autofix
```

## Excluded Paths

Pre-commit respects the same exclusions as the AI linter:
- `ocr/` - Core OCR pipeline (high-risk)
- `ocr-etl-pipeline/` - ETL pipeline
- `tests/ocr/` - OCR tests
- `runners/` - Training runners

These paths are excluded in `.pre-commit-config.yaml` via the `files` directive.

## Common Issues Fixed by Pre-commit

- ✅ Unused imports (F401)
- ✅ Undefined variables (F821)
- ✅ Multiple statements on one line (E701)
- ✅ Trailing whitespace
- ✅ Missing newline at end of file
- ✅ Inconsistent formatting
- ✅ Type errors (mypy)

## Customization

Edit `.pre-commit-config.yaml` to:
- Add new hooks
- Adjust hook arguments
- Change file patterns
- Update hook versions

After editing, run:
```bash
pre-commit install --install-hooks
```

## Resources

- [Pre-commit documentation](https://pre-commit.com/)
- [Ruff pre-commit](https://docs.astral.sh/ruff/integrations/#pre-commit)
- [Project's .pre-commit-config.yaml](file:///.pre-commit-config.yaml)

## Summary

**TL;DR:**
1. Run `make pre-commit-install` once
2. Hooks run automatically on `git commit`
3. Review auto-fixes before re-committing
4. Use `--no-verify` to skip in emergencies
