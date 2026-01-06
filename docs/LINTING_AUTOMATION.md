# Automated Linting System

This document describes the AI-powered automated linting system using xAI's Grok API.

## Overview

The automated linting system uses Grok AI to intelligently fix code quality issues reported by `ruff`. It can be run locally via Makefile targets or automatically via GitHub Actions.

## Components

### 1. Grok Linter Tool (`AgentQMS/tools/utilities/grok_linter.py`)

A Python script that:
- Accepts ruff JSON output as input
- Groups errors by file
- Calls xAI's Grok API to generate fixes
- Applies fixes with validation
- Supports dry-run mode for preview

### 2. GitHub Workflow (`.github/workflows/lint-autofix.yml`)

Automated workflow that:
- Can be manually triggered or called by other workflows
- Runs ruff to detect linting errors
- Uses Grok to generate fixes
- Creates a PR with the fixes
- Supports dry-run mode

### 3. Makefile Targets

Convenient local commands:
- `make lint-fix-ai` - Run AI-powered fixes (limit 5 files)
- `make lint-fix-ai-dry-run` - Preview fixes without applying
- `make lint-check-json` - Output ruff results as JSON

## Usage

### Local Usage

**Prerequisites:**
- Set `XAI_API_KEY` environment variable (available in `.env.local`)

**Run AI-powered fixes:**
```bash
# Fix up to 5 files
make lint-fix-ai

# Preview fixes without applying
make lint-fix-ai-dry-run

# Or use the tool directly for more control
export XAI_API_KEY="your-key-here"
uv run ruff check . --output-format=json > /tmp/lint_errors.json
python AgentQMS/tools/utilities/grok_linter.py --input /tmp/lint_errors.json --limit 10 --verbose
```

### GitHub Actions Usage

**Manual Trigger:**
1. Go to Actions tab in GitHub
2. Select "AI-Powered Lint Autofix" workflow
3. Click "Run workflow"
4. Configure options:
   - `limit`: Max files to fix (default: 5)
   - `dry_run`: Preview only (default: false)

**Automatic Trigger:**
The workflow can be called by other workflows when linting fails.

## Configuration

### Environment Variables

- `XAI_API_KEY`: Required. xAI API key for Grok access
  - Local: Set in `.env.local` or export in shell
  - GitHub: Set as repository secret

### Grok Linter Options

```bash
python grok_linter.py --help

Options:
  --input PATH        Path to ruff JSON output file
  --stdin             Read JSON from stdin
  --dry-run           Don't modify files, just show what would be done
  --limit N           Limit number of files to process
  --verbose, -v       Verbose output
  --api-key KEY       xAI API key (or set XAI_API_KEY env var)
```

### Workflow Inputs

- `limit`: Maximum number of files to process (default: 5)
- `dry_run`: Run in dry-run mode without creating PR (default: false)

## How It Works

1. **Error Detection**: Ruff scans the codebase and outputs errors in JSON format
2. **Error Grouping**: Errors are grouped by file for efficient processing
3. **AI Fix Generation**: For each file:
   - File content and error descriptions are sent to Grok
   - Grok generates corrected code
   - Response is cleaned and validated
4. **Fix Application**: Corrected code is written to files
5. **Verification**: Ruff re-runs to verify fixes
6. **PR Creation** (GitHub only): Changes are committed and a PR is created

## Best Practices

### When to Use

✅ **Good use cases:**
- Fixing style violations (E701, F401, UP012, etc.)
- Removing unused imports
- Updating deprecated syntax
- Batch fixing similar errors across multiple files

❌ **Not recommended for:**
- Complex logic errors
- Security vulnerabilities
- Performance issues requiring domain knowledge

### Review Guidelines

**Always review AI-generated fixes before merging:**
1. Check that functionality is preserved
2. Verify no unintended changes
3. Run tests to ensure nothing broke
4. Review the diff carefully

### Rate Limiting

The tool includes:
- Exponential backoff on rate limit errors
- 0.5s delay between file processing
- Configurable retry logic (max 3 attempts)

## Troubleshooting

### "XAI_API_KEY not found"

**Solution:** Set the API key:
```bash
export XAI_API_KEY="xai-your-key-here"
# Or add to .env.local
```

### "API error 429: Rate limited"

**Solution:** The tool automatically retries with exponential backoff. If persistent:
- Reduce `--limit` value
- Wait a few minutes before retrying
- Check API quota at https://console.x.ai

### Fixes not applying correctly

**Solution:**
1. Run in dry-run mode first: `make lint-fix-ai-dry-run`
2. Check verbose output for errors
3. Manually review the proposed fixes
4. Report issues to the team

### Verification fails after fix

This means the AI-generated fix didn't fully resolve the linting errors.

**Solution:**
1. Review the changes made
2. Manually fix remaining issues
3. Consider adjusting the prompt in `grok_linter.py`

## Examples

### Example 1: Fix unused imports

```bash
# Before: File has F401 errors (unused imports)
$ uv run ruff check myfile.py
myfile.py:1:8: F401 [*] `os` imported but unused
myfile.py:2:8: F401 [*] `sys` imported but unused

# Run AI fixer
$ make lint-fix-ai
🤖 Running AI-powered linting fixes...
Found 2 errors. Running Grok fixer...
[2026-01-06 18:40:00] [INFO] Processing myfile.py (2 errors)
[2026-01-06 18:40:05] [SUCCESS] Applied fix to myfile.py
[2026-01-06 18:40:06] [SUCCESS] ✓ Verified: myfile.py now passes linting

# After: Imports removed
$ uv run ruff check myfile.py
All checks passed!
```

### Example 2: Dry-run preview

```bash
$ make lint-fix-ai-dry-run
🤖 Running AI-powered linting fixes (DRY RUN)...
[2026-01-06 18:40:00] [INFO] Starting Grok Linter (dry_run=True)
[2026-01-06 18:40:00] [INFO] Processing scripts/test.py (3 errors)
[2026-01-06 18:40:05] [INFO] [DRY-RUN] Would write to scripts/test.py
================================================================================
Fixed content for scripts/test.py:
================================================================================
import json

def process_data(data):
    if not data:
        return None
    return json.dumps(data)
================================================================================
```

## Integration with AgentQMS

The Grok Linter is part of the AgentQMS utilities and follows the same patterns:
- Located in `AgentQMS/tools/utilities/`
- Integrated with GitHub Actions workflows
- Accessible via Makefile targets
- Documented in the AgentQMS standards

## Future Enhancements

Potential improvements:
- [ ] Support for other linters (mypy, pylint)
- [ ] Configurable AI models (GPT-4, Claude, etc.)
- [ ] Learning from user feedback
- [ ] Integration with pre-commit hooks
- [ ] Automatic PR review comments
- [ ] Cost tracking and budgeting

## References

- [xAI API Documentation](https://docs.x.ai/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [AgentQMS Standards](../AgentQMS/standards/)
