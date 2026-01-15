from __future__ import annotations

from typing import Any
from pydantic import BaseModel

class TrainingInfo(BaseModel):
    epoch: int
    global_step: int
    training_phase: str
    max_epochs: int | None = None

class EncoderInfo(BaseModel):
    model_name: str
    pretrained: bool
    frozen: bool

class DecoderInfo(BaseModel):
    name: str
    in_channels: list[int]
    inner_channels: int | None
    output_channels: int | None
    params: dict[str, Any]

class HeadInfo(BaseModel):
    name: str
    in_channels: Any | None
    params: dict[str, Any]

class LossInfo(BaseModel):
    name: str
    params: dict[str, Any]

class ModelInfo(BaseModel):
    architecture: str
    encoder: EncoderInfo
    decoder: DecoderInfo
    head: HeadInfo
    loss: LossInfo

class MetricsInfo(BaseModel):
    precision: float | None
    recall: float | None
    hmean: float | None
    validation_loss: float | None
    additional_metrics: dict[str, float]

class CheckpointingConfig(BaseModel):
    monitor: str
    mode: str
    save_top_k: int
    save_last: bool

class CheckpointMetadataV1(BaseModel):
    schema_version: str = "1.0"
    checkpoint_path: str
    exp_name: str
    created_at: str
    training: TrainingInfo
    model: ModelInfo
    metrics: MetricsInfo
    checkpointing: CheckpointingConfig
    hydra_config_path: str | None
    wandb_run_id: str | None
