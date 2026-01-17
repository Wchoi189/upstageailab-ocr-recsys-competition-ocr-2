import os
import sys

# The only allowed top-level directories in configs/
ALLOWED_TIERS = ["global", "hardware", "domain", "runtime", "model", "data", "train", "experiment"]

def audit():
    violations = []
    # 1. Structural Check
    for item in os.listdir("configs"):
        path = os.path.join("configs", item)
        if os.path.isdir(path):
            if item not in ALLOWED_TIERS and not item.startswith("__"):
                violations.append(f"🚩 [STRUCT] Illegal Tier: configs/{item}/")

    # 2. Package Directive Check
    for root, _, files in os.walk("configs"):
        for file in files:
            if not file.endswith(".yaml") or file == "main.yaml": continue
            path = os.path.join(root, file)
            with open(path, 'r') as f:
                content = f.read()
                # Global is ONLY for Tier 2 (Hardware) and Tier 7 (Experiment)
                # And configs/global/default.yaml which is the definition of global variables
                is_global_tier = any(x in path for x in ["hardware", "experiment", "global"])
                if "@package _global_" in content and not is_global_tier:
                    violations.append(f"🚩 [NAMESPACE] Logic file {path} using _global_")
    return violations

if __name__ == "__main__":
    v = audit()
    if v: print("\n".join(v)); sys.exit(1)
    print("✅ Architecture is compliant.")
