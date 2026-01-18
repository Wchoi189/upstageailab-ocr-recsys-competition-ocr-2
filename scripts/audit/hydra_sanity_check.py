import sys
import os
from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig

def audit_config(domain_name):
    print(f"\n🔍 --- Auditing Domain: {domain_name} ---")
    try:
        with initialize(version_base=None, config_path="../../configs"):
            # Using overrides to simulate the domain selection
            cfg = compose(config_name="main", overrides=[f"domain={domain_name}"])

            violations = 0

            # 1. Check for Double-Wrap Bug (Flattening Rule)
            # If 'data' contains another 'data' key, it's a violation
            if "data" in cfg and isinstance(cfg.data, DictConfig) and "data" in cfg.data:
                print(f"❌ VIOLATION: Double-wrap detected in 'data' namespace (data.data)")
                violations += 1

            # 2. Check for Logger Flattening
            if "train" in cfg and "logger" in cfg.train:
                for logger_alias, content in cfg.train.logger.items():
                    # Heuristic: if the alias matches a top-level key inside the content
                    if isinstance(content, DictConfig) and logger_alias in content:
                         print(f"⚠️ WARNING: Logger '{logger_alias}' might be double-nested (contains key '{logger_alias}').")
                         # Double check if it has _target_ at root
                         if "_target_" not in content:
                             print(f"   -> Confirmed Violation: No _target_ at root of alias '{logger_alias}'.")
                             violations += 1

            # 3. Check for Absolute Path Anchors
            # Attempt to resolve all interpolations
            try:
                # We need to register resolvers if any custom ones are used
                # Assuming standard hydra resolvers or oc.env are automatic, but custom ones might fail if not registered.
                # Project seems to use oc.env.
                OmegaConf.to_container(cfg, resolve=True)
                print(f"✅ SUCCESS: All absolute interpolations resolved.")
            except Exception as e:
                print(f"❌ VIOLATION: Interpolation failure. Likely a relative path issue: {e}")
                violations += 1

            # 4. Check for Domain Isolation
            # In recognition, detection keys should be null
            if domain_name == "recognition":
                if cfg.get("detection") is not None:
                    print(f"❌ VIOLATION: 'detection' key is not null in Recognition domain.")
                    violations += 1
            if domain_name == "detection":
                if cfg.get("recognition") is not None:
                    print(f"❌ VIOLATION: 'recognition' key is not null in Detection domain.")
                    violations += 1

            if violations == 0:
                print(f"⭐ Domain '{domain_name}' is COMPLIANT with v5.0 Standards.")
            else:
                print(f"🚨 Found {violations} violations in '{domain_name}'.")

    except Exception as e:
        print(f"💥 CRITICAL: Config composition failed for {domain_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we are running from project root or correct relative path
    # Script is in scripts/audit/
    # If run via `uv run python scripts/audit/hydra_sanity_check.py`, cwd is project root.
    # initialize config_path="../configs" assumes cwd is scripts/audit/ OR relative to the python script file?
    # Hydra initialize is relative to the python script file location.
    # script is in scripts/audit/ -> ../../configs is correct (scripts/audit/ -> scripts/ -> root/configs)

    audit_config("detection")
    audit_config("recognition")
