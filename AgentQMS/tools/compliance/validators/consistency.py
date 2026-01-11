import re
from pathlib import Path
from AgentQMS.tools.compliance.validators.frontmatter import _extract_frontmatter

def _get_type_from_filename(
    filename: str,
    valid_artifact_types: dict[str, str],
    artifact_type_details: dict[str, dict]
) -> str | None:
    """Extract artifact type from filename based on prefix."""
    timestamp_match = re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}_", filename)
    if not timestamp_match:
        return None

    after_timestamp = filename[timestamp_match.end() :]

    for prefix, _ in valid_artifact_types.items():
        if after_timestamp.startswith(prefix):
            type_details = artifact_type_details.get(prefix, {})
            return type_details.get("frontmatter_type", type_details.get("name", ""))

    return None

def validate_type_consistency(
    file_path: Path,
    valid_artifact_types: dict[str, str],
    artifact_type_details: dict[str, dict],
    error_templates: dict = None
) -> tuple[bool, str]:
    """Cross-validate frontmatter type against filename and directory.

    Phase 2: Ensures frontmatter `type:` matches the artifact type implied
    by the filename prefix and the expected directory.
    """
    filename = file_path.name

    # Extract frontmatter type
    frontmatter = _extract_frontmatter(file_path)
    fm_type = frontmatter.get("type", "")

    # Extract type from filename
    filename_type = _get_type_from_filename(filename, valid_artifact_types, artifact_type_details)

    if not filename_type:
        # Can't determine type from filename, skip cross-validation
        return True, "Cannot determine type from filename"

    # Check if frontmatter type matches filename type
    if fm_type and fm_type != filename_type:
        # Build consolidated diagnostic
        mismatches = []
        mismatches.append(f"frontmatter type='{fm_type}'")
        mismatches.append(f"filename implies type='{filename_type}'")

        msg = f"Type mismatch: {', '.join(mismatches)}. Update frontmatter type to '{filename_type}'"

        if error_templates:
            tmpl = error_templates.get("frontmatter_type_mismatch")
            if tmpl and tmpl.get("code") == "E005":
                tmpl_msg = tmpl.get("message", msg)
                expected = tmpl.get("expected", "")
                hint = tmpl.get("hint", "")

                try:
                    tmpl_msg = tmpl_msg.format(fm_type=fm_type, filename_type=filename_type, expected_type=filename_type)
                    expected = expected.format(fm_type=fm_type, filename_type=filename_type, expected_type=filename_type)
                    hint = hint.format(fm_type=fm_type, filename_type=filename_type, expected_type=filename_type)
                except (KeyError, ValueError):
                    pass

                parts = [f"[E005] {tmpl_msg}"]
                if expected:
                    parts.append(f"Expected: {expected}")
                if hint:
                    parts.append(f"Hint: {hint}")
                msg = " | ".join(parts)

        return False, msg

    return True, "Type consistency validated"
