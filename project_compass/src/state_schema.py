"""
Project Compass V2 - Vessel State Schema

Unified Pydantic model replacing fragmented state files:
- compass.json
- current_session.yml
- session_handover.md

THE Source of Truth for project state.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ProjectHealth(str, Enum):
    """Project health status indicators."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BLOCKED = "blocked"


class PipelinePhase(str, Enum):
    """Active pipeline domain."""
    DETECTION = "detection"
    RECOGNITION = "recognition"
    KIE = "kie"
    INTEGRATION = "integration"


class ArtifactType(str, Enum):
    """Valid artifact types for staging."""
    DESIGN = "design"
    RESEARCH = "research"
    WALKTHROUGH = "walkthrough"
    IMPLEMENTATION_PLAN = "implementation_plan"
    BUG_REPORT = "bug_report"
    AUDIT = "audit"
    # Spec Kit artifacts
    SPECIFICATION = "specification"
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"


class Artifact(BaseModel):
    """Registered artifact in pulse staging."""
    path: str = Field(..., description="Relative path within pulse_staging/artifacts/")
    artifact_type: ArtifactType
    milestone_id: str
    created_at: datetime = Field(default_factory=datetime.now)

    @field_validator("path")
    @classmethod
    def validate_staging_path(cls, v: str) -> str:
        """Ensure artifact path is within staging area."""
        if v.startswith("/") or ".." in v:
            raise ValueError(f"Artifact path must be relative and within staging: {v}")
        return v


class Pulse(BaseModel):
    """
    A Pulse represents a single work cycle.

    Replaces legacy 'Session' concept with strict naming and rule injection.
    """
    pulse_id: str = Field(..., min_length=5, description="Format: domain-action-target")
    objective: str = Field(..., min_length=20, max_length=500)
    milestone_id: str = Field(..., description="Link to star-chart milestone")
    instructions: list[str] = Field(default_factory=list, description="Injected rules from vault")
    artifacts: list[Artifact] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    token_burden: str = Field(default="low", pattern="^(low|medium|high)$")

    @field_validator("pulse_id")
    @classmethod
    def validate_pulse_id(cls, v: str) -> str:
        """Enforce domain-action-target naming."""
        # At least 2 words separated by hyphens, no generic terms
        parts = v.split("-")
        if len(parts) < 2:
            raise ValueError(f"Pulse ID must be domain-action format: {v}")

        banned = {"new", "session", "test", "tmp", "untitled", "default"}
        for part in parts:
            if part.lower() in banned:
                raise ValueError(f"Pulse ID contains banned term '{part}': {v}")
        return v


class Milestone(BaseModel):
    """Star-chart milestone definition."""
    milestone_id: str
    title: str
    status: str = Field(default="pending", pattern="^(pending|active|complete|blocked)$")
    tasks: list[str] = Field(default_factory=list)


class VesselState(BaseModel):
    """
    THE Single Source of Truth for Project Compass.

    Replaces:
    - compass.json (project status)
    - current_session.yml (active work)
    - session_handover.md (narrative context)

    AI agents interact ONLY through tools that validate against this schema.
    """
    version: str = "2.0.0"
    last_updated: datetime = Field(default_factory=datetime.now)
    health: ProjectHealth = ProjectHealth.HEALTHY
    current_phase: PipelinePhase = PipelinePhase.KIE

    # Active work cycle (None = no active pulse)
    active_pulse: Pulse | None = None

    # Star-chart: Roadmap milestones
    star_chart: list[Milestone] = Field(default_factory=list)

    # Persistent notes (max 5, ultra-concise)
    active_blockers: list[str] = Field(default_factory=list, max_length=5)

    def model_post_init(self, __context: Any) -> None:
        """Auto-update timestamp on any state change."""
        self.last_updated = datetime.now()

    @classmethod
    def load(cls, path: Path) -> VesselState:
        """Load state from JSON file."""
        import json
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Atomically save state to JSON file."""
        import json
        import tempfile
        import os

        self.last_updated = datetime.now()
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write pattern
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            delete=False
        ) as tmp:
            json.dump(self.model_dump(mode="json"), tmp, indent=2, default=str)
            tmp_path = tmp.name

        os.replace(tmp_path, path)


# Factory functions for common operations
def create_empty_state() -> VesselState:
    """Create a new empty vessel state."""
    return VesselState()


def create_pulse(
    pulse_id: str,
    objective: str,
    milestone_id: str,
) -> Pulse:
    """Create a new pulse with validation."""
    return Pulse(
        pulse_id=pulse_id,
        objective=objective,
        milestone_id=milestone_id,
    )
