#!/usr/bin/env python3
"""Validate Hydra configuration system for common issues."""

import sys
from pathlib import Path

import yaml

CONFIG_ROOT = Path("configs")


def validate_syntax():
    """Check all YAML files are valid."""
    errors = []
    for yaml_file in CONFIG_ROOT.rglob("*.yaml"):
        try:
            with open(yaml_file) as f:
                yaml.safe_load(f)
        except Exception as e:
            errors.append(f"Syntax error in {yaml_file}: {e}")
    return errors


def validate_packages():
    """Check all @package directives are valid."""
    valid_packages = {
        "_global_",
        "model",
        "model.encoder",
        "model.decoder",
        "model.head",
        "model.loss",
        "model.optimizer",
        "model.scheduler",
        "callbacks",
        "logger",
        "logger.wandb",
        "data",
        "trainer",
        "dataloaders",
        "hydra.job_logging",
        "_global_.datasets.val_dataset.config",
    }
    errors = []
    for yaml_file in CONFIG_ROOT.rglob("*.yaml"):
        with open(yaml_file) as f:
            for i, line in enumerate(f, 1):
                if "# @package" in line:
                    pkg = line.split("@package")[-1].strip()
                    if pkg not in valid_packages:
                        errors.append(f"{yaml_file}:{i} - Unknown @package: {pkg}")
    return errors


def validate_variables():
    """Check for undefined variable references."""
    # Load base config to get defined variables
    try:
        from ocr.utils.config_utils import load_config

        load_config("train")
        # If this succeeds, variables are resolved
        return []
    except Exception as e:
        return [f"Variable resolution failed: {e}"]


def main():
    print("Validating Hydra configuration system...\n")

    all_errors = []
    all_errors.extend(validate_syntax())
    all_errors.extend(validate_packages())
    all_errors.extend(validate_variables())

    if all_errors:
        print("❌ Validation failed:")
        for error in all_errors:
            print(f"  - {error}")
        return 1

    print("✅ All config validations passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
