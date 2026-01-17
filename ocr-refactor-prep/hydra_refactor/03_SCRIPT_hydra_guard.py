#!/usr/bin/env python3
"""
Hydra Guard - Runtime Configuration Validator

Purpose: Validate resolved Hydra configurations against domain isolation rules
Usage:   python 03_SCRIPT_hydra_guard.py --domain recognition
Output:  Configuration health report + resolved YAML for AI context

This script performs RUNTIME validation by:
1. Instantiating Hydra configs in memory (sees actual resolved values)
2. Checking for domain leakage (ghost variables)
3. Verifying package directive correctness
4. Exporting resolved configs for AI agent consumption
"""

from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


@dataclass
class ValidationResult:
    """Result of a configuration validation check"""
    passed: bool
    message: str
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


class HydraGuard:
    """Runtime validator for Hydra configurations"""

    # Domain-specific keys that should NOT appear in other domains
    DOMAIN_EXCLUSIVE_KEYS = {
        "detection": {
            "max_polygons": "Maximum number of polygons for detection",
            "shrink_ratio": "Polygon shrink ratio",
            "thresh_min": "Minimum threshold for detection",
            "thresh_max": "Maximum threshold for detection",
        },
        "recognition": {
            "max_label_length": "Maximum character sequence length",
            "charset": "Character set for recognition",
            "case_sensitive": "Case sensitivity flag",
        },
        "kie": {
            "max_entities": "Maximum number of entities",
            "relation_types": "Types of relations to extract",
        }
    }

    # Keys that should NEVER be at root level (package violations)
    FORBIDDEN_ROOT_KEYS = [
        "batch_size",
        "num_workers",
        "learning_rate",
        "architecture",
        "backbone",
        "decoder",
    ]

    def __init__(self, config_path: str = "../configs"):
        self.config_path = config_path
        self.results: list[ValidationResult] = []

    def audit_domain(self, domain_name: str, overrides: list[str] | None = None) -> DictConfig:
        """
        Audit a specific domain configuration

        Args:
            domain_name: Domain to audit (detection, recognition, kie)
            overrides: Optional Hydra overrides

        Returns:
            Resolved configuration
        """
        print(f"🔍 Auditing domain: {domain_name}")

        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()

        # Initialize Hydra and compose config
        with initialize(config_path=self.config_path, version_base=None):
            overrides = overrides or []
            overrides.append(f"domain={domain_name}")

            cfg = compose(config_name="config", overrides=overrides)

            # Run validation checks
            self._check_domain_leakage(cfg, domain_name)
            self._check_package_violations(cfg)
            self._check_required_structure(cfg, domain_name)

            # Export resolved config for AI consumption
            self._export_resolved_config(cfg, domain_name)

            return cfg

    def _check_domain_leakage(self, cfg: DictConfig, current_domain: str):
        """Check for keys from other domains (ghost variables)"""
        print("  ├─ Checking domain leakage...")

        for other_domain, keys in self.DOMAIN_EXCLUSIVE_KEYS.items():
            if other_domain == current_domain:
                continue

            for key, description in keys.items():
                # Check if key exists and is NOT null
                if key in cfg and cfg[key] is not None:
                    self.results.append(ValidationResult(
                        passed=False,
                        message=f"Domain leakage: '{key}' ({description}) from {other_domain} is ACTIVE in {current_domain}",
                        severity="CRITICAL"
                    ))
                    print(f"    🚩 CRITICAL: {key} leaked from {other_domain}")

    def _check_package_violations(self, cfg: DictConfig):
        """Check for keys that should be namespaced but appear at root"""
        print("  ├─ Checking package violations...")

        for key in self.FORBIDDEN_ROOT_KEYS:
            if key in cfg:
                # Check if it's at root level (not nested)
                value = cfg[key]
                if not isinstance(value, DictConfig):
                    self.results.append(ValidationResult(
                        passed=False,
                        message=f"Package violation: '{key}' found at root level. Should be under appropriate group (data/model/train)",
                        severity="ERROR"
                    ))
                    print(f"    ⚠️  ERROR: {key} at root level")

    def _check_required_structure(self, cfg: DictConfig, domain_name: str):
        """Verify required configuration structure exists"""
        print("  ├─ Checking required structure...")

        # All domains should have these top-level groups
        required_groups = ["model", "data"]

        for group in required_groups:
            if group not in cfg:
                self.results.append(ValidationResult(
                    passed=False,
                    message=f"Missing required group: '{group}' not found in {domain_name} config",
                    severity="ERROR"
                ))
                print(f"    ⚠️  ERROR: Missing '{group}' group")

    def _export_resolved_config(self, cfg: DictConfig, domain_name: str):
        """Export resolved configuration for AI agent context"""
        output_file = f"resolved_{domain_name}.yaml"

        with open(output_file, "w") as f:
            f.write("# " + "=" * 78 + "\n")
            f.write(f"# RESOLVED CONFIGURATION: {domain_name}\n")
            f.write("# " + "=" * 78 + "\n")
            f.write("# This file shows the FINAL resolved values after all interpolations.\n")
            f.write("# Use this for AI agent context to eliminate ${variable} guessing.\n")
            f.write("# " + "=" * 78 + "\n\n")
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

        print(f"  ├─ Exported resolved config: {output_file}")

    def generate_health_report(self, output_file: str = "config_health_report.txt"):
        """Generate configuration health report"""
        # Categorize results
        critical = [r for r in self.results if r.severity == "CRITICAL"]
        errors = [r for r in self.results if r.severity == "ERROR"]
        warnings = [r for r in self.results if r.severity == "WARNING"]

        passed = len(self.results) - len(critical) - len(errors) - len(warnings)

        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HYDRA CONFIGURATION HEALTH REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Checks: {len(self.results)}\n")
            f.write(f"  ✅ Passed: {passed}\n")
            f.write(f"  🚩 Critical: {len(critical)}\n")
            f.write(f"  ⚠️  Errors: {len(errors)}\n")
            f.write(f"  ℹ️  Warnings: {len(warnings)}\n\n")

            if critical:
                f.write("\n" + "=" * 80 + "\n")
                f.write("CRITICAL ISSUES (Will Cause Runtime Failures)\n")
                f.write("=" * 80 + "\n")
                for r in critical:
                    f.write(f"🚩 {r.message}\n")

            if errors:
                f.write("\n" + "=" * 80 + "\n")
                f.write("ERRORS (Should Be Fixed)\n")
                f.write("=" * 80 + "\n")
                for r in errors:
                    f.write(f"⚠️  {r.message}\n")

            if warnings:
                f.write("\n" + "=" * 80 + "\n")
                f.write("WARNINGS (Review Recommended)\n")
                f.write("=" * 80 + "\n")
                for r in warnings:
                    f.write(f"ℹ️  {r.message}\n")

            if not critical and not errors:
                f.write("\n" + "=" * 80 + "\n")
                f.write("✅ CONFIGURATION IS HEALTHY\n")
                f.write("=" * 80 + "\n")

        print(f"\n📄 Health report saved to: {output_file}")

        return len(critical) == 0 and len(errors) == 0


def main():
    """Run Hydra Guard validation"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Hydra configuration at runtime")
    parser.add_argument("--domain", required=True, choices=["detection", "recognition", "kie"],
                       help="Domain to audit")
    parser.add_argument("--config-path", default="../configs",
                       help="Path to Hydra configs directory")
    parser.add_argument("--overrides", nargs="*", default=[],
                       help="Additional Hydra overrides")
    parser.add_argument("--output", default="config_health_report.txt",
                       help="Output health report file")

    args = parser.parse_args()

    guard = HydraGuard(config_path=args.config_path)

    try:
        guard.audit_domain(args.domain, overrides=args.overrides)
        is_healthy = guard.generate_health_report(output_file=args.output)

        if is_healthy:
            print("\n✅ Configuration is healthy!")
            return 0
        else:
            print("\n❌ Configuration has issues. See report for details.")
            return 1

    except Exception as e:
        print(f"\n❌ Failed to audit configuration: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    exit(main())
