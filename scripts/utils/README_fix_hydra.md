# Hydra Override Fixer

Automatically add the `+` prefix to Hydra CLI overrides that introduce new config keys.

## The Problem

When using Hydra with struct mode enabled (default), you get errors like:

```bash
Could not override 'checkpoint_path'.
To append to your config use +checkpoint_path=...
Key 'checkpoint_path' is not in struct
```

Manually adding `+` prefixes through trial-and-error is tedious and wastes time.

## The Solution

This tool analyzes your Hydra command, loads the base config, and automatically determines which overrides need the `+` prefix.

## Usage

### Method 1: Direct Script (Recommended)

```bash
uv run python scripts/utils/fix_hydra_overrides.py "your command here"
```

### Method 2: Bash Wrapper

```bash
./fix-hydra.sh "your command here"
```

### Method 3: Makefile Target

```bash
make fix-hydra CMD="your command here"
```

### Method 4: Shell Alias

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias fix-hydra='python /workspaces/scripts/utils/fix_hydra_overrides.py'
```

Then use:

```bash
fix-hydra "your command here"
```

## Example

**Input command with errors:**

```bash
uv run python scripts/runners/train.py \
  mode=train \
  experiment=parseq_flash \
  checkpoint_path=outputs/checkpoints/best-acc-val/acc-0.8267.ckpt \
  trainer.max_epochs=40 \
  train.lr_scheduler._target_=torch.optim.lr_scheduler.ReduceLROnPlateau \
  train.lr_scheduler.mode=max
```

**Run through fixer:**

```bash
uv run python scripts/utils/fix_hydra_overrides.py \
  "uv run python scripts/runners/train.py mode=train experiment=parseq_flash ..." \
  --verbose
```

**Output (corrected command):**

```bash
uv run python scripts/runners/train.py \
  mode=train \
  experiment=parseq_flash \
  +checkpoint_path=outputs/checkpoints/best-acc-val/acc-0.8267.ckpt \
  trainer.max_epochs=40 \
  +train.lr_scheduler._target_=torch.optim.lr_scheduler.ReduceLROnPlateau \
  +train.lr_scheduler.mode=max
```

## Verbose Mode

Add `--verbose` or `-v` to see what's being changed:

```bash
uv run python scripts/utils/fix_hydra_overrides.py "your command" --verbose
```

Output shows:
- Which keys exist in the base config
- Which keys need `+` prefix added
- The final corrected command

## How It Works

1. **Parse** the command to extract overrides
2. **Load** the Hydra config with experiment-specific defaults
3. **Check** each override key against the loaded config
4. **Add** `+` prefix where keys don't exist
5. **Output** the corrected command

## Features

- ✅ Handles nested keys (e.g., `train.lr_scheduler.mode`)
- ✅ Preserves existing prefixes (`+`, `++`, `~`)
- ✅ Respects experiment-specific configs
- ✅ Verbose mode for transparency
- ✅ No false positives (actually loads your config)

## Limitations

- Requires valid Hydra configuration files
- Command must include the Python script path
- Override syntax must be valid (`key=value`)

## Advanced: Integrate into Your Workflow

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Validate Hydra commands in commit messages
grep -r "uv run python scripts/runners/train.py" . | while read -r line; do
    uv run python scripts/utils/fix_hydra_overrides.py "$line" --check
done
```

### VS Code Task

Add to `.vscode/tasks.json`:

```json
{
  "label": "Fix Hydra Overrides",
  "type": "shell",
  "command": "uv run python scripts/utils/fix_hydra_overrides.py",
  "args": ["${input:command}"],
  "problemMatcher": []
}
```

## Troubleshooting

### "Could not find .py script in command"

Make sure your command includes the full path to the Python script:

```bash
# ✓ Good
uv run python scripts/runners/train.py mode=train ...

# ✗ Bad (missing .py)
train mode=train ...
```

### "Invalid override format"

Overrides must be in `key=value` format:

```bash
# ✓ Good
trainer.max_epochs=40

# ✗ Bad
trainer.max_epochs 40
```

### Script Can't Find Your Config

The script assumes:
- Config path: `PROJECT_ROOT/configs`
- Config name: `main`

If your setup differs, modify line 75 in the script:

```python
config_path = Path(__file__).parent.parent.parent / "configs"
```

## Related Tools

- **Agent Debug Toolkit (ADT)**: AST-based config analysis
  - `uv run adt analyze-config <path>`
  - `uv run adt trace-merges <file>`
- **Hydra Docs**: https://hydra.cc/docs/advanced/override_grammar/package_directive/

## License

MIT
