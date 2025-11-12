# Configuration Protocols

**Purpose:** Concise instructions for Hydra configuration management. For detailed context, see `docs/maintainers/protocols/configuration/`.

## Hydra Configuration

**Paths:**
- Use relative paths from project root
- Config files in `configs/`
- Presets in `configs/presets/`

**Validation:**
```bash
uv run python scripts/agent_tools/validate_config.py --config-name <name>
```

**Resolution Troubleshooting:**
- Check config file exists in `configs/`
- Verify preset name matches file name
- Check for circular imports
- Validate YAML syntax

**Testing:**
```bash
# Test config resolution
uv run python runners/train.py --config-name <name> --dry-run

# Test with fast dev run
uv run python runners/train.py --config-name <name> trainer.fast_dev_run=True
```

## Command Builder

**Testing:**
- Test command generation with sample configs
- Validate command output
- Check for parameter conflicts

**Hydra Configuration Fixes:**
- Update command builder configs
- Test with multiple presets
- Validate parameter overrides

## Experiment Analysis

**Framework:**
- Use experiment analysis framework
- Collect metrics from W&B
- Generate comparison tables

**Tools:**
```bash
# Collect results
uv run python scripts/collect_results.py

# Generate ablation table
uv run python scripts/generate_ablation_table.py
```

## Configuration Refactoring

**Process:**
1. Identify common config patterns
2. Extract to shared configs
3. Update all references
4. Validate with smoke tests

**Best Practices:**
- Group related configs
- Use config composition
- Document config structure
- Test config changes

