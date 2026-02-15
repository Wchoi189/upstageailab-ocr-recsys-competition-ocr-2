#!/usr/bin/env python3
"""
Hydra Override Fixer - Automatically add + prefix to new config keys

This script analyzes a Hydra CLI command and determines which overrides
need the + prefix because the keys don't exist in the base config.

Usage:
    python scripts/utils/fix_hydra_overrides.py <your_command>

Example:
    python scripts/utils/fix_hydra_overrides.py \
        "uv run python scripts/runners/train.py \
         mode=train experiment=parseq_flash \
         checkpoint_path=outputs/checkpoints/best-acc-val/acc-0.8267.ckpt \
         trainer.max_epochs=40 \
         train.lr_scheduler._target_=torch.optim.lr_scheduler.ReduceLROnPlateau"
"""

import sys
import re
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra


def parse_cli_overrides(command: str) -> tuple[str, list[str]]:
    """Parse CLI command into base command and overrides."""
    # Remove the script invocation part
    parts = command.split()

    # Find where the Python script ends
    script_idx = -1
    for i, part in enumerate(parts):
        if part.endswith('.py'):
            script_idx = i
            break

    if script_idx == -1:
        raise ValueError("Could not find .py script in command")

    base_cmd = ' '.join(parts[:script_idx + 1])
    overrides = parts[script_idx + 1:]

    return base_cmd, overrides


def parse_override(override: str) -> tuple[str, str, str]:
    """Parse an override string into prefix, key, and value."""
    # Handle +key=value, ++key=value, ~key=value patterns
    prefix = ''
    if override.startswith('++'):
        prefix = '++'
        override = override[2:]
    elif override.startswith('+'):
        prefix = '+'
        override = override[1:]
    elif override.startswith('~'):
        prefix = '~'
        override = override[1:]

    if '=' not in override:
        raise ValueError(f"Invalid override format: {override}")

    key, value = override.split('=', 1)
    return prefix, key, value


def check_key_exists(cfg: DictConfig, key: str) -> bool:
    """Check if a nested key exists in the config."""
    keys = key.split('.')
    current = cfg

    try:
        for k in keys[:-1]:
            if k not in current:
                return False
            current = current[k]

        # Check the final key
        return keys[-1] in current
    except (KeyError, AttributeError, TypeError):
        return False


def fix_overrides(command: str, verbose: bool = False) -> str:
    """Analyze command and add + prefix where needed."""

    # Parse command
    base_cmd, overrides = parse_cli_overrides(command)

    if verbose:
        print(f"Base command: {base_cmd}")
        print(f"Found {len(overrides)} overrides")

    # Extract the config path from the training script
    # We know it uses PROJECT_ROOT / "configs" and config_name="main"
    config_path = Path(__file__).parent.parent.parent / "configs"

    # Initialize Hydra to load the base config
    GlobalHydra.instance().clear()

    # Collect experiment and other config overrides
    experiment_overrides = []
    other_overrides = []

    for override in overrides:
        prefix, key, value = parse_override(override)

        # Keep experiment and mode overrides for initial config loading
        if key in ['experiment', 'mode', 'domain', 'hardware']:
            experiment_overrides.append(f"{key}={value}")
        else:
            other_overrides.append((prefix, key, value))

    if verbose and experiment_overrides:
        print(f"\nLoading config with: {experiment_overrides}")

    try:
        # Initialize and compose config with experiment overrides
        with initialize_config_dir(config_dir=str(config_path.absolute()), version_base=None):
            cfg = compose(config_name="main", overrides=experiment_overrides)

            if verbose:
                print(f"\nConfig keys at root: {list(cfg.keys())}")

            # Check each override
            fixed_overrides = []

            # Add back the experiment overrides first
            fixed_overrides.extend(experiment_overrides)

            # Check other overrides
            warnings = []
            for prefix, key, value in other_overrides:
                needs_plus = not check_key_exists(cfg, key)

                # Warn about incomplete logger configs
                if 'train.logger.' in key and needs_plus:
                    # Check if we're trying to create a partial logger config
                    logger_name = key.split('.')[2] if len(key.split('.')) > 2 else None
                    if logger_name:
                        warnings.append(
                            f"⚠️  WARNING: Creating incomplete logger config for '{logger_name}'!\n"
                            f"   Key: {key}\n"
                            f"   This will fail at runtime because loggers need '_target_' and other required keys.\n"
                            f"   Solution: Define the logger in an experiment config file instead of CLI."
                        )

                if needs_plus and not prefix:
                    prefix = '+'
                    if verbose:
                        print(f"  ✓ Adding + to: {key}")
                elif not needs_plus and prefix == '+':
                    if verbose:
                        print(f"  ⚠ Removing unnecessary + from: {key}")
                    prefix = ''
                elif verbose:
                    status = "doesn't exist" if needs_plus else "exists"
                    print(f"  - {key} {status}, prefix={prefix!r}")

                fixed_overrides.append(f"{prefix}{key}={value}")
            
            # Print warnings after processing all overrides
            if warnings:
                print("\n" + "!"*80)
                for warning in warnings:
                    print(warning)
                print("!"*80 + "\n")

    finally:
        GlobalHydra.instance().clear()

    # Reconstruct command
    return f"{base_cmd} {' '.join(fixed_overrides)}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Join all arguments as they might have spaces
    command = ' '.join(sys.argv[1:])

    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    if verbose:
        command = command.replace('--verbose', '').replace('-v', '')

    print("Analyzing command...\n")

    try:
        fixed_command = fix_overrides(command, verbose=verbose)

        print("\n" + "="*80)
        print("FIXED COMMAND:")
        print("="*80)
        print(fixed_command)
        print("="*80)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
