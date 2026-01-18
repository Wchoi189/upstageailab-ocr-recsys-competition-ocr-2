#!/usr/bin/env python3
"""
Hydra v5.0 Configuration Pattern Verification Script

Validates that Hydra configurations follow "Domains First" v5.0 patterns:
- Flattening Rule compliance
- Absolute interpolation usage
- Logger aliasing structure
- Callback flattening
- Interpolation resolution

Usage:
    python scripts/utils/verify_hydra_flattening.py
    python scripts/utils/verify_hydra_flattening.py --domain recognition
"""

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig


def verify_logger_aliasing(cfg: DictConfig) -> tuple[bool, list[str]]:
    """Verify logger aliasing structure."""
    issues = []

    if not hasattr(cfg, 'train') or not hasattr(cfg.train, 'logger'):
        issues.append("❌ Missing train.logger configuration")
        return False, issues

    logger_cfg = cfg.train.logger

    # Check for aliased loggers
    has_aliased_loggers = False
    for key in logger_cfg:
        if key.endswith('_logger') and '_target_' in logger_cfg[key]:
            has_aliased_loggers = True
            print(f"✅ Logger Aliasing: Found 'train.logger.{key}'")

            # Verify _target_ is at alias root
            if '_target_' in logger_cfg[key]:
                print(f"✅ Logger Flattening: '_target_' at root of '{key}'")
            else:
                issues.append(f"❌ Logger '{key}' missing '_target_' at root")

    if not has_aliased_loggers:
        # Check if there's a single logger without aliasing
        if '_target_' in logger_cfg:
            print("✅ Single logger configured (no aliasing needed)")
        else:
            issues.append("⚠️  No aliased loggers found (may be intentional)")

    return len(issues) == 0, issues


def verify_callback_flattening(cfg: DictConfig) -> tuple[bool, list[str]]:
    """Verify callback flattening (no redundant wrapper keys)."""
    issues = []

    if not hasattr(cfg, 'train') or not hasattr(cfg.train, 'callbacks'):
        issues.append("⚠️  Missing train.callbacks configuration")
        return True, issues  # Not critical

    callbacks_cfg = cfg.train.callbacks

    for callback_name in callbacks_cfg:
        callback = callbacks_cfg[callback_name]

        # Check if callback has _target_ at root (flattened)
        if isinstance(callback, DictConfig) and '_target_' in callback:
            print(f"✅ Callback Flattening: 'train.callbacks.{callback_name}' is flattened")
        elif isinstance(callback, DictConfig):
            # Check for double nesting (e.g., early_stopping.early_stopping)
            if callback_name in callback:
                issues.append(
                    f"❌ Callback '{callback_name}' has double nesting "
                    f"(contains redundant '{callback_name}:' key)"
                )
            else:
                issues.append(
                    f"⚠️  Callback '{callback_name}' missing '_target_' "
                    f"(may be a group config)"
                )

    return len(issues) == 0, issues


def verify_interpolation_resolution(cfg: DictConfig) -> tuple[bool, list[str]]:
    """Verify all interpolations resolve successfully."""
    issues = []

    try:
        # Attempt to resolve all interpolations
        resolved = OmegaConf.to_container(cfg, resolve=True)
        print("✅ Interpolation: All absolute paths resolved successfully")
        return True, issues
    except Exception as e:
        error_msg = str(e)
        issues.append(f"❌ Interpolation Error: {error_msg}")

        # Provide helpful hints based on error type
        if "InterpolationKeyError" in error_msg:
            issues.append(
                "   💡 Hint: Use absolute interpolation paths like "
                "${data.transforms.train} instead of ${train}"
            )
        elif "InterpolationResolutionError" in error_msg:
            issues.append(
                "   💡 Hint: Check for circular dependencies in interpolations"
            )

        return False, issues


def verify_namespace_structure(cfg: DictConfig) -> tuple[bool, list[str]]:
    """Check for common namespace fragmentation issues."""
    issues = []

    # Check for double nesting in common namespaces
    double_nest_checks = [
        ('data', 'data'),
        ('train', 'train'),
        ('model', 'model'),
    ]

    for namespace, key in double_nest_checks:
        if hasattr(cfg, namespace):
            ns_cfg = getattr(cfg, namespace)
            if isinstance(ns_cfg, DictConfig) and key in ns_cfg:
                issues.append(
                    f"❌ Namespace Fragmentation: Found '{namespace}.{key}.*' "
                    f"(double nesting - violates Flattening Rule)"
                )

    if not issues:
        print("✅ Namespace Structure: No double nesting detected")

    return len(issues) == 0, issues


def main():
    """Run all verification checks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify Hydra v5.0 configuration patterns"
    )
    parser.add_argument(
        '--domain',
        default='detection',
        choices=['detection', 'recognition', 'kie', 'layout'],
        help='Domain to verify (default: detection)'
    )
    parser.add_argument(
        '--config-dir',
        default=None,
        help='Config directory (default: auto-detect from project root)'
    )
    args = parser.parse_args()

    # Determine config directory
    if args.config_dir:
        config_dir = Path(args.config_dir).resolve()
    else:
        # Auto-detect from script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        config_dir = project_root / "configs"

    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Hydra v5.0 Configuration Pattern Verification")
    print("=" * 70)
    print(f"Config Directory: {config_dir}")
    print(f"Domain: {args.domain}")
    print("=" * 70)
    print()

    # Initialize Hydra and compose configuration
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir)
        ):
            cfg = compose(
                config_name="main",
                overrides=[f"domain={args.domain}"]
            )
    except Exception as e:
        print(f"❌ Failed to compose configuration: {e}")
        sys.exit(1)

    # Run verification checks
    all_passed = True
    all_issues = []

    checks = [
        ("Logger Aliasing", verify_logger_aliasing),
        ("Callback Flattening", verify_callback_flattening),
        ("Namespace Structure", verify_namespace_structure),
        ("Interpolation Resolution", verify_interpolation_resolution),
    ]

    for check_name, check_func in checks:
        print(f"\n--- {check_name} Check ---")
        passed, issues = check_func(cfg)

        if not passed:
            all_passed = False
            all_issues.extend(issues)

        for issue in issues:
            print(issue)

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All checks passed! Configuration follows v5.0 patterns.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("❌ Some checks failed. Issues found:")
        print("=" * 70)
        for issue in all_issues:
            print(issue)
        print("\n📖 See AgentQMS/standards/tier2-framework/hydra-v5-patterns.yaml")
        print("   for pattern documentation and resolution strategies.")
        sys.exit(1)


if __name__ == "__main__":
    main()
