import os
import sys
import yaml
import importlib
from pathlib import Path

def get_target_definitions(data, file_path):
    """Recursively yield (key, value, path) for keys ending in _target_ or _partial_."""
    if isinstance(data, dict):
        for k, v in data.items():
            if k == "_target_":
                yield (k, v, file_path)
            elif isinstance(v, (dict, list)):
                yield from get_target_definitions(v, file_path)
    elif isinstance(data, list):
        for item in data:
            yield from get_target_definitions(item, file_path)

def check_import(target_path):
    """Attempt to import the target path."""
    try:
        parts = target_path.split(".")
        module_name = ".".join(parts[:-1])
        class_name = parts[-1]

        module = importlib.import_module(module_name)
        if not hasattr(module, class_name):
            return False, f"Module '{module_name}' has no attribute '{class_name}'"
        return True, "OK"
    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    root_dir = Path("configs")
    if not root_dir.exists():
        print("Config directory not found.")
        return

    print(f"🔍 Scanning {root_dir} for _target_ definitions...")

    broken_targets = []
    checked_count = 0

    for yaml_file in root_dir.rglob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                # Use base loader to avoid custom tag errors
                content = yaml.safe_load(f)
                if not content:
                    continue

                for key, target, path in get_target_definitions(content, str(yaml_file)):
                    checked_count += 1
                    is_valid, message = check_import(target)
                    if not is_valid:
                        # Ignore known interpolation patterns
                        if "${" in target:
                             continue

                        broken_targets.append({
                            "file": str(path),
                            "target": target,
                            "error": message
                        })
                        print(f"❌ BROKEN: {target} in {path} -> {message}")
                    # else:
                    #    print(f"✅ OK: {target}")

        except Exception as e:
            print(f"⚠️ Error reading {yaml_file}: {e}")

    print(f"\nAudit Complete. Checked {checked_count} targets.")
    if broken_targets:
        print(f"🚨 Found {len(broken_targets)} broken targets:")
        for t in broken_targets:
            print(f"  - {t['target']} ({t['file']}): {t['error']}")
        sys.exit(1)
    else:
        print("✅ All targets look valid.")
        sys.exit(0)

if __name__ == "__main__":
    main()
