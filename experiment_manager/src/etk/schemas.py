from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    BACKLOG = "backlog"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class ExperimentStatus(str, Enum):
    ACTIVE = "active"
    COMPLETE = "complete"
    PAUSED = "paused"
    DEPRECATED = "deprecated"


class InsightType(str, Enum):
    INSIGHT = "insight"
    DECISION = "decision"
    FAILURE = "failure"


class Task(BaseModel):
    id: str
    description: str
    status: TaskStatus
    completed_at: datetime | None = None


class Insight(BaseModel):
    type: InsightType
    content: str
    # Using datetime directly, Pydantic handles ISO 8601 validation
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ArtifactType(str, Enum):
    ASSESSMENT = "assessment"
    REPORT = "report"
    GUIDE = "guide"
    SCRIPT = "script"
    OTHER = "other"


class Artifact(BaseModel):
    path: str
    type: ArtifactType
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExperimentManifest(BaseModel):
    experiment_id: str = Field(pattern=r"^[0-9]{8}_[0-9]{6}_[a-z0-9_]+$")
    name: str = Field(min_length=3)
    status: ExperimentStatus
    created_at: datetime
    updated_at: datetime | None = None
    intention: str | None = None
    tasks: list[Task] = Field(default_factory=list)
    insights: list[Insight] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
