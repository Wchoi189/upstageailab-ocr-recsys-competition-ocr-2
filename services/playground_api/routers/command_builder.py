"""Command Builder API router."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ui.apps.command_builder.services.overrides import build_additional_overrides, maybe_suffix_exp_name
from ui.apps.command_builder.services.recommendations import UseCaseRecommendationService
from ui.utils.command import CommandBuilder, CommandValidator
from ui.utils.config_parser import ConfigParser
from ui.utils.ui_generator import compute_overrides

from ..utils.paths import PROJECT_ROOT

router = APIRouter()

SCHEMA_DIR = PROJECT_ROOT / "ui" / "apps" / "command_builder" / "schemas"

SchemaId = Literal["train", "test", "predict"]

SCHEMA_REGISTRY: dict[SchemaId, dict[str, Any]] = {
    "train": {
        "path": SCHEMA_DIR / "command_builder_train.yaml",
        "script": "train.py",
        "label": "Training",
        "description": "Build metadata-aware training commands.",
    },
    "test": {
        "path": SCHEMA_DIR / "command_builder_test.yaml",
        "script": "test.py",
        "label": "Testing",
        "description": "Generate commands for evaluation runs.",
    },
    "predict": {
        "path": SCHEMA_DIR / "command_builder_predict.yaml",
        "script": "predict.py",
        "label": "Prediction",
        "description": "Prepare batch prediction commands.",
    },
}


class SchemaSummary(BaseModel):
    """Metadata about the available command builder schemas."""

    id: SchemaId
    label: str
    script: str
    description: str | None = None


class CommandBuildRequest(BaseModel):
    """Request body for building a CLI command from schema values."""

    schema_id: SchemaId = Field(description="Identifier of the schema to use (train, test, predict).")
    values: dict[str, Any] = Field(default_factory=dict, description="Form values keyed by schema element key.")
    append_model_suffix: bool = Field(
        default=True,
        description="Mirror Command Builder behavior of appending architecture/backbone info to experiment name.",
    )


class CommandBuildResponse(BaseModel):
    """Response data returned after building a command."""

    command: str
    overrides: list[str]
    constant_overrides: list[str]
    validation_error: str | None = None


class RecommendationResponse(BaseModel):
    """Response model for use case recommendations."""

    id: str
    title: str
    description: str
    architecture: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def _get_command_builder() -> CommandBuilder:
    return CommandBuilder(project_root=str(PROJECT_ROOT))


@lru_cache(maxsize=1)
def _get_validator() -> CommandValidator:
    return CommandValidator()


@lru_cache(maxsize=1)
def _get_config_parser() -> ConfigParser:
    return ConfigParser()


@lru_cache(maxsize=1)
def _get_recommendation_service() -> UseCaseRecommendationService:
    return UseCaseRecommendationService(_get_config_parser())


@lru_cache(maxsize=8)
def _load_schema_data(schema_id: SchemaId) -> dict[str, Any]:
    """Load a schema definition from disk."""
    entry = SCHEMA_REGISTRY[schema_id]
    schema_path = entry["path"]
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with open(schema_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@router.get("/schemas", response_model=list[SchemaSummary])
def list_schemas() -> list[SchemaSummary]:
    """List available command builder schemas."""
    summaries: list[SchemaSummary] = []
    for schema_id, entry in SCHEMA_REGISTRY.items():
        summaries.append(
            SchemaSummary(
                id=schema_id,
                label=entry["label"],
                script=entry["script"],
                description=entry.get("description"),
            )
        )
    return summaries


@router.get("/schemas/{schema_id}")
def get_schema(schema_id: SchemaId) -> dict[str, Any]:
    """Get full schema definition with UI elements."""
    if schema_id not in SCHEMA_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown schema_id '{schema_id}'")
    schema_data = _load_schema_data(schema_id)
    # Populate options for selectboxes
    schema_data = _populate_options(schema_data)
    return schema_data


def _populate_options(schema: dict[str, Any]) -> dict[str, Any]:
    """Populate options for selectbox elements."""
    config_parser = _get_config_parser()

    for element in schema.get("ui_elements", []):
        if element.get("type") == "selectbox" and element.get("options_source"):
            source = element["options_source"]
            element["options"] = _get_options_from_source(source, config_parser)

    return schema


def _get_options_from_source(source: str, config_parser: ConfigParser) -> list[str]:
    """Get dynamic options list from config parser."""
    model_source_map = {
        "models.backbones": "backbones",
        "models.encoders": "encoders",
        "models.decoders": "decoders",
        "models.heads": "heads",
        "models.optimizers": "optimizers",
        "models.losses": "losses",
    }

    if source in model_source_map:
        models = config_parser.get_available_models()
        return models.get(model_source_map[source], [])
    if source == "models.architectures":
        return config_parser.get_available_architectures()
    if source == "checkpoints":
        return config_parser.get_available_checkpoints()
    if source == "datasets":
        return config_parser.get_available_datasets()

    return []


@router.post("/build", response_model=CommandBuildResponse)
def build_command(payload: CommandBuildRequest) -> CommandBuildResponse:
    """Build and validate a CLI command from the provided values."""
    if payload.schema_id not in SCHEMA_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown schema_id '{payload.schema_id}'")

    schema = _load_schema_data(payload.schema_id)
    overrides, constant_overrides = compute_overrides(schema, payload.values)

    config_parser = _get_config_parser()
    additional_overrides = build_additional_overrides(payload.values, config_parser)
    merged_overrides = overrides + additional_overrides
    merged_overrides = maybe_suffix_exp_name(merged_overrides, payload.values, payload.append_model_suffix)

    schema_entry = SCHEMA_REGISTRY[payload.schema_id]
    command_builder = _get_command_builder()
    command = command_builder.build_command_from_overrides(
        script=schema_entry["script"],
        overrides=merged_overrides,
        constant_overrides=constant_overrides,
    )

    validator = _get_validator()
    is_valid, error = validator.validate_command(command)
    validation_error = None if is_valid else error or "Unknown validation error"

    return CommandBuildResponse(
        command=command,
        overrides=merged_overrides,
        constant_overrides=constant_overrides,
        validation_error=validation_error,
    )


@router.get("/recommendations", response_model=list[RecommendationResponse])
def get_recommendations(architecture: str | None = None) -> list[RecommendationResponse]:
    """Get use case recommendations, optionally filtered by architecture."""
    service = _get_recommendation_service()
    recommendations = service.for_architecture(architecture)

    return [
        RecommendationResponse(
            id=rec.id,
            title=rec.title,
            description=rec.description,
            architecture=rec.architecture,
            parameters=rec.parameters,
        )
        for rec in recommendations
    ]


