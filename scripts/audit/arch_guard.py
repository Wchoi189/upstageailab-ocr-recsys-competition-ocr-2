import os
import sys

# v5.0 | Purpose: Consolidated Structural & Logical Enforcement
# Strategy: Tier Validation + Namespace Hardening + Domain Isolation

ALLOWED_TIERS = ["global", "hardware", "domain", "runtime", "model", "data", "train", "experiment"]

def audit():
    violations = []

    # 1. Structural Check (From your version)
    # Ensures no "Ghost Tiers" like _foundation remain.
    for item in os.listdir("configs"):
        path = os.path.join("configs", item)
        if os.path.isdir(path):
            if item not in ALLOWED_TIERS and not item.startswith("__"):
                violations.append(f"🚩 [STRUCT] Illegal Tier: configs/{item}/")

    # 2. Logical & Namespace Check (The Merge)
    for root, _, files in os.walk("configs"):
        for file in files:
            if not file.endswith(".yaml") or file == "main.yaml":
                continue

            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()

                # --- Namespace Hardening ---
                # Ensures _global_ isn't used as a shortcut in logic folders.
                is_global_tier = any(x in path for x in ["hardware", "experiment", "global"])
                if "@package _global_" in content and not is_global_tier:
                    violations.append(f"🚩 [NAMESPACE] Logic file {path} using _global_")

                # --- Domain Isolation (The "Anti-Leak" Check) ---
                # Ensures Recognition doesn't know about Detection variables.
                if "configs/domain/recognition.yaml" in path:
                    required_nulls = ["detection: null", "max_polygons: null"]
                    for r_null in required_nulls:
                        if r_null not in content:
                            violations.append(f"🚩 [ISOLATION] {path} missing nullification: '{r_null}'")

    return violations

if __name__ == "__main__":
    v = audit()
    if v:
        print("\n".join(v))
        sys.exit(1)
    print("✅ Architecture is fully compliant (Structure + Namespaces + Isolation).")
