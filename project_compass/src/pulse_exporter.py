"""
Project Compass V2 - Pulse Exporter

Strict export logic with staging audit.
Ensures disk-vs-manifest consistency before archiving.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from project_compass.src.state_schema import VesselState, Artifact


class ExportResult(TypedDict):
    status: str  # "SUCCESS" | "BLOCKED"
    message: str
    export_path: str | None
    action_required: str | None


def audit_staging(
    state: VesselState,
    staging_path: Path,
) -> tuple[set[str], set[str]]:
    """
    Compare physical staging files against manifest.

    Returns:
        Tuple of (stray_files, ghost_files)
        - stray_files: Files on disk not in manifest
        - ghost_files: Files in manifest not on disk
    """
    if not state.active_pulse:
        return set(), set()

    # Get registered artifact paths (relative to staging)
    registered = {a.path for a in state.active_pulse.artifacts}

    # Get actual files on disk
    artifacts_dir = staging_path / "artifacts"
    if not artifacts_dir.exists():
        actual = set()
    else:
        actual = {
            str(p.relative_to(artifacts_dir))
            for p in artifacts_dir.rglob("*")
            if p.is_file() and p.name != "scratchpad.md"  # Scratchpad is exempt
        }

    stray = actual - registered
    ghost = registered - actual

    return stray, ghost


def export_pulse(
    state_path: Path,
    staging_path: Path,
    history_path: Path,
) -> ExportResult:
    """
    Export current pulse with strict staging enforcement.

    Process:
    1. Load vessel_state.json
    2. Audit staging vs manifest
    3. Block if discrepancies exist
    4. Move artifacts to history/{pulse_id}/
    5. Archive manifest snapshot
    6. Clear active pulse

    Returns:
        ExportResult with status and details
    """
    # 1. Load state
    if not state_path.exists():
        return ExportResult(
            status="BLOCKED",
            message="vessel_state.json not found",
            export_path=None,
            action_required="Run pulse-init first",
        )

    state = VesselState.load(state_path)

    if not state.active_pulse:
        return ExportResult(
            status="BLOCKED",
            message="No active pulse to export",
            export_path=None,
            action_required="Run pulse-init to start a pulse",
        )

    # 2. Audit staging
    stray, ghost = audit_staging(state, staging_path)

    # 3. Block on discrepancies
    if stray:
        return ExportResult(
            status="BLOCKED",
            message=f"Unregistered artifacts in staging: {stray}",
            export_path=None,
            action_required="Register with pulse-sync or delete these files",
        )

    if ghost:
        return ExportResult(
            status="BLOCKED",
            message=f"Manifest references missing files: {ghost}",
            export_path=None,
            action_required="Create these files or remove from manifest",
        )

    # 4. Execute export
    pulse_id = state.active_pulse.pulse_id
    milestone_id = state.active_pulse.milestone_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Structure: history/{milestone_id}/{timestamp}_{pulse_id}
    export_dir = history_path / milestone_id / f"{timestamp}_{pulse_id}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Move artifacts
    artifacts_dir = staging_path / "artifacts"
    if artifacts_dir.exists():
        for artifact in state.active_pulse.artifacts:
            src = artifacts_dir / artifact.path
            if src.exists():
                dst = export_dir / artifact.path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    # Delete scratchpad if exists
    scratchpad = artifacts_dir / "scratchpad.md"
    if scratchpad.exists():
        scratchpad.unlink()

    # 5. Archive manifest snapshot
    manifest_snapshot = {
        "pulse_id": pulse_id,
        "objective": state.active_pulse.objective,
        "milestone_id": state.active_pulse.milestone_id,
        "artifacts": [a.model_dump(mode="json") for a in state.active_pulse.artifacts],
        "started_at": str(state.active_pulse.started_at),
        "exported_at": datetime.now().isoformat(),
    }

    with open(export_dir / "pulse_manifest.json", "w") as f:
        json.dump(manifest_snapshot, f, indent=2)

    # 6. Clear active pulse
    state.active_pulse = None
    state.save(state_path)

    return ExportResult(
        status="SUCCESS",
        message=f"Pulse {pulse_id} exported successfully",
        export_path=str(export_dir),
        action_required=None,
    )


def register_artifact(
    state_path: Path,
    artifact_path: str,
    artifact_type: str,
    milestone_id: str | None = None,
) -> tuple[bool, str]:
    """
    Register a new artifact in the manifest.

    Args:
        state_path: Path to vessel_state.json
        artifact_path: Relative path within pulse_staging/artifacts/
        artifact_type: One of ArtifactType enum values
        milestone_id: Optional override (defaults to pulse milestone)

    Returns:
        Tuple of (success, message)
    """
    from project_compass.src.state_schema import ArtifactType

    state = VesselState.load(state_path)

    if not state.active_pulse:
        return False, "No active pulse. Run pulse-init first."

    # Validate artifact type
    try:
        art_type = ArtifactType(artifact_type)
    except ValueError:
        valid = [t.value for t in ArtifactType]
        return False, f"Invalid artifact type '{artifact_type}'. Valid: {valid}"

    # Use pulse milestone if not specified
    if not milestone_id:
        milestone_id = state.active_pulse.milestone_id

    # Check for duplicates
    existing = [a.path for a in state.active_pulse.artifacts]
    if artifact_path in existing:
        return False, f"Artifact already registered: {artifact_path}"

    # Create and add artifact
    artifact = Artifact(
        path=artifact_path,
        artifact_type=art_type,
        milestone_id=milestone_id,
    )
    state.active_pulse.artifacts.append(artifact)
    state.save(state_path)

    return True, f"Registered: {artifact_path}"
