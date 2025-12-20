#!/usr/bin/env python3
"""Validate OCR console documentation freshness.

Checks that YAML contracts match actual code:
- Port numbers in quickstart.yaml match main.py
- Module paths exist and are correct
- API endpoint paths match FastAPI decorators
"""

import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
OCR_CONSOLE = PROJECT_ROOT / "apps" / "ocr-inference-console"
AI_DOCS = OCR_CONSOLE / ".ai-instructions"


def validate_ports():
    """Verify port 8002 in quickstart.yaml matches main.py."""
    quickstart = yaml.safe_load((AI_DOCS / "quickstart.yaml").read_text())
    main_py = (OCR_CONSOLE / "backend" / "main.py").read_text()

    # Check quickstart.yaml
    backend_port = quickstart["startup"]["backend_only"]["port"]
    if backend_port != 8002:
        print(f"❌ quickstart.yaml backend port is {backend_port}, should be 8002")
        return False

    # Check main.py default port
    if 'port=int(os.getenv("BACKEND_PORT", "8002"))' not in main_py:
        print("❌ main.py default port is not 8002")
        return False

    print("✅ Port numbers are consistent (8002)")
    return True


def validate_module_paths():
    """Verify documented module paths exist."""
    index = yaml.safe_load((AI_DOCS / "INDEX.yaml").read_text())

    paths_to_check = {
        "backend_entry": index["paths"]["backend_entry"],
        "frontend_entry": index["paths"]["frontend_entry"],
        "services": index["paths"]["services"],
        "components": index["paths"]["components"],
        "context": index["paths"]["context"],
    }

    all_valid = True
    for name, path in paths_to_check.items():
        full_path = PROJECT_ROOT / path
        if not full_path.exists():
            print(f"❌ {name}: {path} does not exist")
            all_valid = False
        else:
            print(f"✅ {name}: {path}")

    return all_valid


def validate_api_endpoints():
    """Verify API endpoint paths match FastAPI decorators."""
    api_docs = yaml.safe_load((AI_DOCS / "contracts" / "api-endpoints.yaml").read_text())
    main_py = (OCR_CONSOLE / "backend" / "main.py").read_text()

    all_valid = True
    for endpoint_name, config in api_docs["endpoints"].items():
        path = config["path"]
        method = config["method"]

        # Extract endpoint suffix from path (after /api)
        endpoint_suffix = path.replace("/api", "")

        # Check if decorator exists in main.py
        decorator_pattern = f'@app.{method.lower()}(f"{{API_PREFIX}}{endpoint_suffix}"'

        if decorator_pattern in main_py:
            print(f"✅ {endpoint_name}: {method} {path}")
        else:
            print(f"❌ {endpoint_name}: {method} {path} not found in main.py")
            all_valid = False

    return all_valid


def main():
    """Run all validation checks."""
    print("🔍 Validating OCR console documentation freshness...\n")

    checks = [
        ("Port numbers", validate_ports),
        ("Module paths", validate_module_paths),
        ("API endpoints", validate_api_endpoints),
    ]

    all_passed = True
    for name, check_fn in checks:
        print(f"\n{name}:")
        if not check_fn():
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All documentation freshness checks passed!")
        return 0
    else:
        print("❌ Some documentation is stale - update .ai-instructions/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
