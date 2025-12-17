#!/usr/bin/env python3
"""
EDS v1.0 Compliance Checker

Validates experiment artifact YAML frontmatter against EDS v1.0 specification.
Reuses patterns from AgentQMS ADS v1.0 compliance-checker.py.

Usage:
    python compliance-checker.py <file_or_directory>
    python compliance-checker.py experiments/20251217_024343_image_enhancements_implementation/
"""

import re
import sys
from pathlib import Path
from typing import Any

import yaml

# Prohibited user-oriented phrases (from ADS v1.0)
PROHIBITED_PHRASES = [
    r'\byou should\b',
    r'\byou can\b',
    r'\byou will\b',
    r'\bfor example\b',
    r'\bit is important\b',
    r'\btutorial\b',
    r'\bwalkthrough\b',
    r'\blet\'s\b',
    r'\bready to begin\b',
    r'\bfor more information\b',
]

# Prohibited emoji patterns
EMOJI_PATTERN = r'[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]'

# Required frontmatter fields
UNIVERSAL_REQUIRED = ['ads_version', 'type', 'experiment_id', 'status', 'created', 'updated', 'tags']

TYPE_SPECIFIC_REQUIRED = {
    'assessment': ['phase', 'priority', 'evidence_count'],
    'report': ['metrics', 'baseline', 'comparison'],
    'guide': ['commands', 'prerequisites'],
    'script': ['dependencies'],
}

# Valid enum values
VALID_TYPES = ['assessment', 'report', 'guide', 'script', 'manifest']
VALID_STATUSES = ['draft', 'active', 'complete', 'deprecated']
VALID_PHASES = ['phase_0', 'phase_1', 'phase_2', 'phase_3', 'phase_4']
VALID_PRIORITIES = ['critical', 'high', 'medium', 'low']
VALID_COMPARISONS = ['improvement', 'regression', 'neutral']


def extract_frontmatter(file_path: Path) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and body from markdown file."""
    content = file_path.read_text(encoding='utf-8')

    # Match YAML frontmatter between --- delimiters
    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}, content

    frontmatter_str, body = match.groups()

    try:
        frontmatter = yaml.safe_load(frontmatter_str)
        return frontmatter or {}, body
    except yaml.YAMLError as e:
        print(f"⚠️  YAML parse error: {e}")
        return {}, content


def validate_yaml_structure(frontmatter: dict[str, Any]) -> list[str]:
    """Validate YAML structure is well-formed."""
    errors = []

    if not isinstance(frontmatter, dict):
        errors.append("Frontmatter must be a YAML dictionary")
        return errors

    return errors


def validate_required_fields(frontmatter: dict[str, Any]) -> list[str]:
    """Validate all required frontmatter fields are present."""
    errors = []

    # Check universal required fields
    for field in UNIVERSAL_REQUIRED:
        if field not in frontmatter:
            errors.append(f"Missing required field: {field}")

    # Check type-specific required fields
    doc_type = frontmatter.get('type')
    if doc_type in TYPE_SPECIFIC_REQUIRED:
        for field in TYPE_SPECIFIC_REQUIRED[doc_type]:
            if field not in frontmatter:
                errors.append(f"Missing required field for type '{doc_type}': {field}")

    return errors


def validate_field_values(frontmatter: dict[str, Any]) -> list[str]:
    """Validate field values against EDS v1.0 constraints."""
    errors = []

    # Validate ads_version format
    if 'ads_version' in frontmatter:
        if not re.match(r'^[0-9]+\.[0-9]+$', str(frontmatter['ads_version'])):
            errors.append(f"Invalid ads_version format: {frontmatter['ads_version']} (expected X.Y)")

    # Validate type enum
    if 'type' in frontmatter:
        if frontmatter['type'] not in VALID_TYPES:
            errors.append(f"Invalid type: {frontmatter['type']} (expected one of {VALID_TYPES})")

    # Validate experiment_id pattern
    if 'experiment_id' in frontmatter:
        if not re.match(r'^[0-9]{8}_[0-9]{6}_[a-z0-9_]+$', frontmatter['experiment_id']):
            errors.append(f"Invalid experiment_id format: {frontmatter['experiment_id']}")

    # Validate status enum
    if 'status' in frontmatter:
        if frontmatter['status'] not in VALID_STATUSES:
            errors.append(f"Invalid status: {frontmatter['status']} (expected one of {VALID_STATUSES})")

    # Validate phase enum (if present)
    if 'phase' in frontmatter:
        if frontmatter['phase'] not in VALID_PHASES:
            errors.append(f"Invalid phase: {frontmatter['phase']} (expected one of {VALID_PHASES})")

    # Validate priority enum (if present)
    if 'priority' in frontmatter:
        if frontmatter['priority'] not in VALID_PRIORITIES:
            errors.append(f"Invalid priority: {frontmatter['priority']} (expected one of {VALID_PRIORITIES})")

    # Validate comparison enum (if present)
    if 'comparison' in frontmatter:
        if frontmatter['comparison'] not in VALID_COMPARISONS:
            errors.append(f"Invalid comparison: {frontmatter['comparison']} (expected one of {VALID_COMPARISONS})")

    # Validate tags are lowercase hyphenated
    if 'tags' in frontmatter:
        if isinstance(frontmatter['tags'], list):
            for tag in frontmatter['tags']:
                if not re.match(r'^[a-z0-9-]+$', str(tag)):
                    errors.append(f"Invalid tag format: {tag} (must be lowercase with hyphens)")
        else:
            errors.append("Tags must be an array")

    # Validate evidence_count is integer (if present)
    if 'evidence_count' in frontmatter:
        if not isinstance(frontmatter['evidence_count'], int) or frontmatter['evidence_count'] < 0:
            errors.append(f"Invalid evidence_count: {frontmatter['evidence_count']} (must be non-negative integer)")

    return errors


def check_prohibited_content(body: str) -> list[str]:
    """Check for prohibited user-oriented phrases and emoji."""
    warnings = []

    # Check prohibited phrases
    for pattern in PROHIBITED_PHRASES:
        if re.search(pattern, body, re.IGNORECASE):
            warnings.append(f"Prohibited user-oriented phrase detected: {pattern}")

    # Check emoji
    if re.search(EMOJI_PATTERN, body):
        warnings.append("Emoji usage detected (prohibited except in pre-commit error messages)")

    return warnings


def validate_file(file_path: Path) -> tuple[bool, list[str], list[str]]:
    """
    Validate a single file against EDS v1.0.

    Returns:
        (is_valid, errors, warnings)
    """
    if not file_path.suffix == '.md':
        return True, [], []  # Skip non-markdown files

    if file_path.name == 'README.md':
        return True, [], []  # Skip README files

    errors = []
    warnings = []

    # Extract frontmatter
    frontmatter, body = extract_frontmatter(file_path)

    if not frontmatter:
        errors.append("No YAML frontmatter found")
        return False, errors, warnings

    # Run validations
    errors.extend(validate_yaml_structure(frontmatter))
    errors.extend(validate_required_fields(frontmatter))
    errors.extend(validate_field_values(frontmatter))
    warnings.extend(check_prohibited_content(body))

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def validate_directory(dir_path: Path) -> tuple[int, int, list[tuple[Path, list[str], list[str]]]]:
    """
    Validate all markdown files in a directory.

    Returns:
        (passed_count, failed_count, failures)
    """
    passed = 0
    failed = 0
    failures = []

    for md_file in dir_path.rglob('*.md'):
        is_valid, errors, warnings = validate_file(md_file)

        if is_valid:
            passed += 1
            if warnings:
                print(f"✅ PASS: {md_file}")
                for warning in warnings:
                    print(f"   ⚠️  {warning}")
        else:
            failed += 1
            failures.append((md_file, errors, warnings))

    return passed, failed, failures


def main():
    if len(sys.argv) < 2:
        print("Usage: python compliance-checker.py <file_or_directory>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if not target.exists():
        print(f"❌ ERROR: Path not found: {target}")
        sys.exit(1)

    if target.is_file():
        # Validate single file
        is_valid, errors, warnings = validate_file(target)

        if is_valid:
            print(f"✅ PASS: {target}")
            if warnings:
                for warning in warnings:
                    print(f"   ⚠️  {warning}")
            sys.exit(0)
        else:
            print(f"❌ FAIL: {target}")
            for error in errors:
                print(f"   • {error}")
            if warnings:
                for warning in warnings:
                    print(f"   ⚠️  {warning}")
            sys.exit(1)

    elif target.is_dir():
        # Validate directory
        passed, failed, failures = validate_directory(target)

        # Print failures
        for file_path, errors, warnings in failures:
            print(f"❌ FAIL: {file_path}")
            for error in errors:
                print(f"   • {error}")
            if warnings:
                for warning in warnings:
                    print(f"   ⚠️  {warning}")

        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")

        if failed == 0:
            print(f"✅ All {passed} files passed validation")
            sys.exit(0)
        else:
            print(f"❌ {failed} file(s) failed validation")
            sys.exit(1)

    else:
        print(f"❌ ERROR: Invalid path type: {target}")
        sys.exit(1)


if __name__ == '__main__':
    main()
