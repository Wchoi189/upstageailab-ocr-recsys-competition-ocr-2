import re
from pathlib import Path

def validate_timestamp_format(filename: str, error_templates: dict = None) -> tuple[bool, str, re.Match | None]:
    """Validate timestamp format in filename."""
    timestamp_pattern = r"^\d{4}-\d{2}-\d{2}_\d{4}_"
    match = re.match(timestamp_pattern, filename)
    valid = bool(match)

    # Helper to get error message (simplified from original class method)
    def _get_msg(code, default):
        if error_templates and code in error_templates:
            tmpl = error_templates[code]
            return f"[{code}] {tmpl.get('message', default)}"
        return default

    if valid:
        msg = "Valid timestamp format"
    else:
        msg = _get_msg("E001", "Missing or invalid timestamp format (expected: YYYY-MM-DD_HHMM_)")

    match_result = match if valid else None
    return valid, msg, match_result

def _detect_intended_type(after_timestamp: str) -> tuple[str, str] | None:
    """Try to detect what artifact type the user intended based on partial match."""
    after_lower = after_timestamp.lower()

    # Check for common misspellings or format errors
    type_hints = [
        ("assessment", "assessment-", "-"),
        ("implementation_plan", "implementation_plan_", "_"),
        ("implementation-plan", "implementation_plan_", "_"),  # Common mistake
        ("bug", "BUG_", "_"),
        ("bug_report", "BUG_", "_"),
        ("session", "SESSION_", "_"),
        ("design", "design-", "-"),
        ("research", "research-", "-"),
        ("audit", "audit-", "-"),
        ("template", "template-", "-"),
    ]

    for keyword, correct_prefix, expected_sep in type_hints:
        if after_lower.startswith(keyword):
            return (correct_prefix, expected_sep)

    return None

def validate_naming_convention(
    file_path: Path,
    valid_artifact_types: dict[str, str],
    artifact_type_details: dict[str, dict],
    error_templates: dict = None
) -> tuple[bool, str]:
    """Validate artifact naming convention."""
    filename = file_path.name

    # Helper to get error message with formatting
    def _get_msg(code, default, **kwargs):
        if error_templates:
            # Find template by code
            template = None
            for name, tmpl in error_templates.items():
                if tmpl.get("code") == code:
                    template = tmpl
                    break

            if template:
                msg = template.get("message", "Validation error")
                expected = template.get("expected", "")
                hint = template.get("hint", "")

                try:
                    msg = msg.format(**kwargs) if kwargs else msg
                    expected = expected.format(**kwargs) if kwargs else expected
                    hint = hint.format(**kwargs) if kwargs else hint
                except (KeyError, ValueError):
                    pass

                parts = [f"[{code}] {msg}"]
                if expected:
                    parts.append(f"Expected: {expected}")
                if hint:
                    parts.append(f"Hint: {hint}")
                return " | ".join(parts)

        return default

    # Check timestamp format
    valid, msg, match = validate_timestamp_format(filename, error_templates)
    if not valid or match is None:
        return valid, msg

    after_timestamp = filename[match.end() :]

    # Check for valid artifact type after timestamp
    matched_type = None
    for artifact_type in valid_artifact_types:
        if after_timestamp.startswith(artifact_type):
            matched_type = artifact_type
            break

    if not matched_type:
        # Try to detect what type the user might have intended
        intended = _detect_intended_type(after_timestamp)
        if intended:
            correct_prefix, expected_sep = intended
            type_details = artifact_type_details.get(correct_prefix, {})
            example = type_details.get("example", "")

            msg = _get_msg(
                "E003",
                f"Invalid format for artifact type. Use '{correct_prefix}' prefix. Example: {example}",
                type=type_details.get("name", "artifact"),
                expected_pattern=example,
                separator=expected_sep,
                case=type_details.get("case", "lowercase"),
            )
            return (False, msg)
        else:
            # No match found - list valid options
            valid_types_str = ", ".join(valid_artifact_types.keys())
            msg = _get_msg(
                "E002",
                f"Missing valid artifact type. Valid artifact types: {valid_types_str}"
            )
            return (False, msg)

    # Check for kebab-case in descriptive part
    # Handle extensions (e.g., .md, .ko.md)
    if filename.endswith(".ko.md"):
        descriptive_part = after_timestamp[len(matched_type) : -6]
    else:
        descriptive_part = after_timestamp[len(matched_type) : -3]  # Remove .md
    type_details = artifact_type_details.get(matched_type, {})
    expected_case = type_details.get("case", "lowercase")

    # Check for ALL CAPS, uppercase, or mixed case words in descriptive part
    # Always validate - descriptive part must be lowercase for ALL types
    words = descriptive_part.replace("-", "_").split("_")
    has_uppercase_or_mixed = any((word.isupper() or (not word.islower() and not word.isdigit())) and len(word) > 1 for word in words)

    if descriptive_part.isupper() or has_uppercase_or_mixed:
        if expected_case == "uppercase_prefix":
            return (
                False,
                "Artifact filenames with uppercase prefix must have lowercase descriptive part. "
                f"Example: {matched_type}001_lowercase-name.md (not {matched_type}001_UPPERCASE-NAME.md)",
            )
        else:
            return (
                False,
                "Artifact filenames must be lowercase. No ALL CAPS allowed. Use kebab-case (lowercase with hyphens)",
            )

    if "_" in descriptive_part and not descriptive_part.replace("-", "").replace("_", "").isalnum():
        return (
            False,
            "Descriptive name should use kebab-case (hyphens, not underscores)",
        )

    return True, "Valid naming convention"
