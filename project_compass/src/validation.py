"""
Project Compass Session Validation Module

Provides validation functions for session content and naming standards.
"""

import re
import yaml
from pathlib import Path


def validate_session_content(
    session_handover_path: Path,
    current_session_path: Path
) -> tuple[bool, list[str]]:
    """
    Validate that a session has meaningful content before export.

    Args:
        session_handover_path: Path to session_handover.md
        current_session_path: Path to current_session.yml

    Returns:
        Tuple of (is_valid, list of validation errors)
    """
    errors = []

    # Check 1: Session handover exists and has substance
    if not session_handover_path.exists():
        errors.append("session_handover.md does not exist")
    else:
        content = session_handover_path.read_text(encoding="utf-8")

        # Check for template/empty content
        if "No active session" in content:
            errors.append("session_handover.md contains template text 'No active session'")

        # Check minimum length (excluding markdown header)
        content_lines = [line for line in content.split('\n') if not line.startswith('#')]
        content_text = '\n'.join(content_lines).strip()

        if len(content_text) < 200:
            errors.append(
                f"session_handover.md is too short ({len(content_text)} chars). "
                f"Expected at least 200 characters of meaningful content."
            )

    # Check 2: Current session has real objective
    if not current_session_path.exists():
        errors.append("current_session.yml does not exist")
    else:
        try:
            with open(current_session_path) as f:
                session_data = yaml.safe_load(f)

            # Check for placeholder objective
            objective = session_data.get('objective', '')
            if isinstance(objective, dict):
                objective = objective.get('primary_goal', '')

            if '[Enter objective here]' in objective:
                errors.append("current_session.yml contains placeholder objective text")

            if not objective or len(str(objective).strip()) < 20:
                errors.append("current_session.yml has empty or trivial objective")

            # Check for at least some session work evidence
            phases = session_data.get('phases', {})
            references = session_data.get('working_references', {})

            if not phases and not references:
                errors.append(
                    "Session appears empty: no phases or working_references populated"
                )

        except yaml.YAMLError as e:
            errors.append(f"current_session.yml is invalid YAML: {e}")

    return len(errors) == 0, errors


def validate_session_name(session_id: str) -> tuple[bool, str]:
    """
    Validate that a session name is descriptive and follows standards.

    Args:
        session_id: Session identifier (may include timestamp prefix)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Strip timestamp prefix if present (format: YYYYMMDD_HHMMSS_)
    name = re.sub(r'^\d{8}_\d{6}_', '', session_id)

    # Also strip just date prefix (format: YYYYMMDD_)
    name = re.sub(r'^\d{8}_', '', name)

    # Check for banned generic terms
    banned_words = ['new', 'session', 'test', 'tmp', 'untitled', 'default']
    name_lower = name.lower()

    for word in banned_words:
        if word in name_lower:
            return False, (
                f"Session name '{name}' contains generic term '{word}'. "
                f"Use descriptive format: domain-action-target\n"
                f"Examples: hydra-refactor-domains, detection-optimize-preprocessing"
            )

    # Check minimum word count (words separated by hyphens or underscores)
    words = re.split(r'[-_]', name)
    words = [w for w in words if w and not w.isdigit()]  # Remove empty strings and pure numbers

    if len(words) < 3:
        return False, (
            f"Session name '{name}' is too vague ({len(words)} descriptive words). "
            f"Use at least 3 descriptive words.\n"
            f"Format: domain-action-target (e.g., 'ocr-implement-orchestrator')"
        )

    # Check that name isn't all numbers (after removing date prefix)
    if name.replace('-', '').replace('_', '').isdigit():
        return False, f"Session name '{name}' is all numbers. Use descriptive words."

    return True, ""
