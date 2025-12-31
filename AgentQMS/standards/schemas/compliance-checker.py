#!/usr/bin/env python3
"""
ADS v1.0 Compliance Checker
Validates AI documentation files against the AI Documentation Standard v1.0
"""

import json
import sys
from pathlib import Path

import yaml

SCHEMA_PATH = Path(__file__).parent / "validation-rules.json"

# Prohibited content patterns (user-oriented phrases)
PROHIBITED_PHRASES = [
    "you should",
    "for example",
    "it is important",
    "it's important",
    "this is because",
    "let's",
    "we can",
    "we will",
    "tutorial",
    "step-by-step",
    "walkthrough",
]


def load_schema() -> dict:
    """Load JSON schema for validation"""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def validate_yaml_structure(file_path: Path) -> tuple[bool, str]:
    """Validate YAML is well-formed"""
    try:
        with open(file_path) as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            return False, "YAML root must be a dictionary"
        return True, "YAML structure valid"
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def validate_required_fields(content: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate required frontmatter fields are present"""
    required = schema.get("required", [])
    missing = [key for key in required if key not in content]

    if missing:
        return False, missing
    return True, []


def validate_field_values(content: dict, schema: dict) -> tuple[bool, list[str]]:
    """Validate field values match schema constraints"""
    errors = []
    properties = schema.get("properties", {})

    # Check ads_version
    if content.get("ads_version") != "1.0":
        errors.append(f"ads_version must be '1.0', got '{content.get('ads_version')}'")

    # Check type
    allowed_types = properties.get("type", {}).get("enum", [])
    if content.get("type") not in allowed_types:
        errors.append(f"Invalid type: '{content.get('type')}'")

    # Check agent
    agent_enum = ["claude", "copilot", "cursor", "gemini", "qwen", "all"]
    agent_val = content.get("agent")
    if isinstance(agent_val, str):
        if agent_val not in agent_enum:
            errors.append(f"Invalid agent: '{agent_val}'")
    elif isinstance(agent_val, list):
        invalid = [a for a in agent_val if a not in agent_enum]
        if invalid:
            errors.append(f"Invalid agent values: {invalid}")

    # Check tier
    tier = content.get("tier")
    if tier is not None and (not isinstance(tier, int) or tier < 1 or tier > 4):
        errors.append(f"tier must be 1-4, got '{tier}'")

    # Check priority
    priority_enum = ["critical", "high", "medium", "low"]
    if content.get("priority") not in priority_enum:
        errors.append(f"Invalid priority: '{content.get('priority')}'")

    return len(errors) == 0, errors


def check_prohibited_content(file_path: Path) -> tuple[bool, list[str]]:
    """Check for prohibited user-oriented content"""
    with open(file_path) as f:
        content_lower = f.read().lower()

    found = [phrase for phrase in PROHIBITED_PHRASES if phrase in content_lower]

    if found:
        return False, found
    return True, []


def validate_file(file_path: Path) -> dict:
    """Validate a single file against ADS v1.0"""
    result = {"file": str(file_path), "valid": False, "errors": [], "warnings": []}

    # 1. Validate YAML structure
    valid, msg = validate_yaml_structure(file_path)
    if not valid:
        result["errors"].append(f"YAML Structure: {msg}")
        return result

    # Load content
    with open(file_path) as f:
        content = yaml.safe_load(f)

    # Load schema
    schema = load_schema()

    # 2. Validate required fields
    valid, missing = validate_required_fields(content, schema)
    if not valid:
        result["errors"].append(f"Missing required fields: {', '.join(missing)}")

    # 3. Validate field values
    valid, errors = validate_field_values(content, schema)
    if not valid:
        result["errors"].extend(errors)

    # 4. Check for prohibited content
    valid, found = check_prohibited_content(file_path)
    if not valid:
        result["warnings"].append(f"Prohibited user-oriented phrases detected: {', '.join(found)}")

    # Set overall validity
    result["valid"] = len(result["errors"]) == 0

    return result


def main():
    """Main validation entry point"""
    if len(sys.argv) < 2:
        print("Usage: compliance-checker.py <file1.yaml> [file2.yaml ...]")
        sys.exit(1)

    files = [Path(f) for f in sys.argv[1:]]
    results = []

    for file_path in files:
        if not file_path.exists():
            print(f"❌ SKIP: {file_path} (not found)")
            continue

        result = validate_file(file_path)
        results.append(result)

        # Print result
        if result["valid"]:
            print(f"✅ PASS: {result['file']}")
            if result["warnings"]:
                for warn in result["warnings"]:
                    print(f"   ⚠️  {warn}")
        else:
            print(f"❌ FAIL: {result['file']}")
            for error in result["errors"]:
                print(f"   ❌ {error}")
            if result["warnings"]:
                for warn in result["warnings"]:
                    print(f"   ⚠️  {warn}")

    # Exit with error if any failed
    failed = [r for r in results if not r["valid"]]
    if failed:
        print(f"\n❌ {len(failed)}/{len(results)} files failed validation")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} files passed validation")
        sys.exit(0)


if __name__ == "__main__":
    main()
