import re
from pathlib import Path

def validate_directory_placement(
    file_path: Path,
    artifacts_root: Path,
    valid_artifact_types: dict[str, str],
    artifact_type_details: dict[str, dict],
    error_templates: dict = None
) -> tuple[bool, str]:
    """Validate that file is in the correct directory."""
    filename = file_path.name
    try:
        relative_path = file_path.relative_to(artifacts_root)
    except ValueError:
        # File not relative to artifacts root, cannot validate directory placement this way
        return True, "File outside artifacts root, skipping directory placement check"

    current_dir = str(relative_path.parent)

    # Validate timestamp format first (needed to find the type prefix)
    timestamp_match = re.match(r"^\d{4}-\d{2}-\d{2}_\d{4}_", filename)
    if not timestamp_match:
        # File doesn't match timestamp-first format, can't validate directory
        return True, "Cannot validate directory (non-standard format)"

    # Extract everything after timestamp
    after_timestamp = filename[timestamp_match.end() :]

    # Find which artifact type this file matches by checking if it starts with a registered type
    expected_dir = None
    matched_prefix = None
    for artifact_type, directory in valid_artifact_types.items():
        if after_timestamp.startswith(artifact_type):
            expected_dir = directory.rstrip("/")
            matched_prefix = artifact_type
            break

    if expected_dir and current_dir != expected_dir:
        type_details = artifact_type_details.get(matched_prefix, {})
        type_name = type_details.get("name", "artifact")

        msg = f"File should be in '{expected_dir}/' directory, currently in '{current_dir}/' (detected type: {type_name})"

        # Try to use error template if available
        if error_templates:
            tmpl = error_templates.get("wrong_directory")
            if tmpl and tmpl.get("code") == "E004":
                tmpl_msg = tmpl.get("message", msg)
                expected = tmpl.get("expected", "")
                hint = tmpl.get("hint", "")

                try:
                    tmpl_msg = tmpl_msg.format(expected_dir=expected_dir)
                    expected = expected.format(expected_dir=expected_dir)
                    hint = hint.format(expected_dir=expected_dir)
                except (KeyError, ValueError):
                    pass

                parts = [f"[E004] {tmpl_msg}"]
                if expected:
                    parts.append(f"Expected: {expected}")
                if hint:
                    parts.append(f"Hint: {hint}")
                msg = " | ".join(parts) + f" (detected type: {type_name})"

        return (False, msg)

    return True, "Correct directory placement"

def validate_artifacts_root(file_path: Path) -> tuple[bool, str]:
    """Ensure artifacts are in docs/artifacts/ not root /artifacts/."""
    try:
        # Get the path relative to project root
        from AgentQMS.tools.utils.paths import get_project_root

        project_root = get_project_root()
        try:
            relative_path = file_path.relative_to(project_root)
        except ValueError:
             # File is outside project root
            return True, "Valid artifacts directory location"

        path_str = str(relative_path).replace("\\", "/")

        # Check if file starts with artifacts/ (without docs/ prefix matching standard)
        if path_str.startswith("artifacts/") and not path_str.startswith("docs/artifacts/"):
            return (
                False,
                f"Artifacts must be in 'docs/artifacts/' not root 'artifacts/'. Move file from '{path_str}' to 'docs/{path_str}'",
            )

        # Verify file is in docs/artifacts/ hierarchy (or AgentQMS for templates/examples)
        if not path_str.startswith("docs/artifacts/") and not path_str.startswith("AgentQMS/"):
             # If it's a markdown file we are validating, it should probably be in docs/artifacts
             # But if the user runs the script on a random file, we warn them.
            return (
                False,
                f"Artifacts must be in 'docs/artifacts/' directory. Current location: '{path_str}'",
            )

    except (ValueError, ImportError):
        pass

    return True, "Valid artifacts directory location"
