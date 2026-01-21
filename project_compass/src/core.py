"""
Project Compass V2 - Vessel Core Module

Unified path resolution and Pulse lifecycle management.
REPLACES: Legacy SessionManager, compass.json management.

PRINCIPLE: "If it's in the Vessel, it's a rule."
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from project_compass.src.state_schema import (
    VesselState,
    ProjectHealth,
    PipelinePhase,
    create_pulse,
)
from project_compass.src.rule_injector import inject_rules


class VesselPaths:
    """
    Resolve paths to Vessel V2 directories and files.

    REPLACES: CompassPaths (legacy)
    """

    def __init__(self, project_root: Path | None = None):
        if project_root:
            self.project_root = project_root
        else:
            # Auto-detect project root by finding project_compass/
            current = Path.cwd()
            while current != current.parent:
                if (current / "project_compass").exists():
                    self.project_root = current
                    break
                current = current.parent
            else:
                self.project_root = Path.cwd()

        self.compass_dir = self.project_root / "project_compass"

        # V2 Directories
        self.vessel_dir = self.compass_dir / ".vessel"
        self.vault_dir = self.compass_dir / "vault"
        self.staging_dir = self.compass_dir / "pulse_staging"
        self.history_dir = self.compass_dir / "history"

        # V2 Files
        self.vessel_state = self.vessel_dir / "vessel_state.json"

        # Legacy (for migration only)
        self.environments_dir = self.compass_dir / "environments"
        self.uv_lock_state = self.environments_dir / "uv_lock_state.yml"

    def ensure_structure(self) -> None:
        """Create required directories if they don't exist."""
        for d in [self.vessel_dir, self.vault_dir, self.staging_dir / "artifacts", self.history_dir]:
            d.mkdir(parents=True, exist_ok=True)


class PulseManager:
    """
    Pulse Lifecycle Manager.

    Handles:
    - Pulse initialization with rule injection
    - State loading/saving
    - Artifact registration

    REPLACES: SessionManager (legacy)
    """

    def __init__(self, paths: VesselPaths | None = None):
        self.paths = paths or VesselPaths()
        self.paths.ensure_structure()

    def load_state(self) -> VesselState:
        """Load or create vessel state."""
        if self.paths.vessel_state.exists():
            return VesselState.load(self.paths.vessel_state)

        # Initialize empty state
        state = VesselState()
        state.save(self.paths.vessel_state)
        return state

    def save_state(self, state: VesselState) -> None:
        """Atomically save vessel state."""
        state.save(self.paths.vessel_state)

    def init_pulse(
        self,
        pulse_id: str,
        objective: str,
        milestone_id: str,
        phase: PipelinePhase | str = PipelinePhase.KIE,
    ) -> tuple[bool, str]:
        """
        Initialize a new pulse (work cycle).

        Args:
            pulse_id: Format domain-action-target (e.g., "recognition-optimize-vocab")
            objective: 20-500 char description of the work
            milestone_id: Link to star-chart milestone (e.g., "rec-opt")
            phase: Active pipeline phase

        Returns:
            Tuple of (success, message)
        """
        state = self.load_state()

        # Check for existing pulse
        if state.active_pulse:
            return False, f"Active pulse exists: {state.active_pulse.pulse_id}. Export it first with pulse-export."

        # Validate inputs
        try:
            pulse = create_pulse(
                pulse_id=pulse_id,
                objective=objective,
                milestone_id=milestone_id,
            )
        except ValueError as e:
            return False, str(e)

        # Set pulse and phase
        state.active_pulse = pulse
        if isinstance(phase, str):
            try:
                state.current_phase = PipelinePhase(phase)
            except ValueError:
                state.current_phase = PipelinePhase.KIE
        else:
            state.current_phase = phase

        # Inject rules from vault
        state = inject_rules(state, self.paths.vault_dir)

        # Save state
        self.save_state(state)

        return True, f"Pulse initialized: {pulse_id} (milestone: {milestone_id})"

    def get_pulse_status(self) -> dict[str, Any]:
        """Get current pulse status summary."""
        state = self.load_state()

        if not state.active_pulse:
            return {
                "active": False,
                "message": "No active pulse. Run pulse-init to start.",
            }

        return {
            "active": True,
            "pulse_id": state.active_pulse.pulse_id,
            "objective": state.active_pulse.objective,
            "milestone_id": state.active_pulse.milestone_id,
            "artifact_count": len(state.active_pulse.artifacts),
            "instructions_count": len(state.active_pulse.instructions),
            "token_burden": state.active_pulse.token_burden,
        }

    def update_health(self, health: ProjectHealth | str) -> bool:
        """Update project health status."""
        state = self.load_state()
        if isinstance(health, str):
            try:
                state.health = ProjectHealth(health)
            except ValueError:
                return False
        else:
            state.health = health
        self.save_state(state)
        return True

    def add_blocker(self, blocker: str) -> bool:
        """Add an active blocker (max 5)."""
        state = self.load_state()
        if len(state.active_blockers) >= 5:
            # Remove oldest
            state.active_blockers.pop(0)
        state.active_blockers.append(blocker[:200])  # Truncate long blockers
        self.save_state(state)
        return True


class EnvironmentChecker:
    """
    Environment Guard: Validates current environment against lock state.

    Lightweight version - checks basics before pulse init.
    """

    def __init__(self, paths: VesselPaths | None = None):
        self.paths = paths or VesselPaths()
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check_all(self) -> tuple[bool, list[str], list[str]]:
        """
        Run environment checks.

        Returns:
            Tuple of (all_passed, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        self._check_uv()
        self._check_python()

        return len(self.errors) == 0, self.errors, self.warnings

    def _check_uv(self) -> None:
        """Verify UV is available."""
        try:
            result = subprocess.run(["which", "uv"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                self.errors.append("UV not found in PATH")
        except Exception as e:
            self.errors.append(f"Failed to check UV: {e}")

    def _check_python(self) -> None:
        """Verify Python version."""
        try:
            result = subprocess.run(
                ["uv", "run", "python", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            version = result.stdout.strip().replace("Python ", "")
            if not version.startswith("3."):
                self.errors.append(f"Python 3.x required, got: {version}")
        except Exception as e:
            self.warnings.append(f"Failed to check Python version: {e}")


# Convenience functions
def get_pulse_manager() -> PulseManager:
    """Factory for PulseManager with default paths."""
    return PulseManager()


def get_vessel_state() -> VesselState:
    """Quick access to current vessel state."""
    return PulseManager().load_state()


def format_uv_command(command: str) -> str:
    """
    Prefix a command with 'uv run' for UV-managed environments.

    Example:
        format_uv_command("python train.py") -> "uv run python train.py"
    """
    if command.startswith("uv "):
        return command
    return f"uv run {command}"
