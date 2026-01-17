"""Logger factory for creating appropriate logger from configuration.

Provides a centralized factory function for creating Lightning loggers
(WandB or TensorBoard) based on configuration settings.
"""

from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from ocr.core.utils.config_utils import is_config


def create_logger(config: DictConfig) -> Logger:
    """Create appropriate logger (W&B or TensorBoard) from configuration.

    Args:
        config: Hydra configuration containing logger settings

    Returns:
        Lightning Logger instance (WandbLogger or TensorBoardLogger)

    Logic:
        - If config.logger.wandb.enabled is explicitly False → TensorBoard
        - Otherwise → W&B (default, enabled by default)
    """
    logger_config = config.get("logger", {})
    wandb_cfg = logger_config.get("wandb", {})

    # Type-safe check: if explicitly disabled, use TensorBoard
    # Check both dict and DictConfig types
    wandb_enabled = True
    if hasattr(wandb_cfg, "get"):
        # Works for both dict and DictConfig
        enabled_value = wandb_cfg.get("enabled")
        if enabled_value is False:
            wandb_enabled = False
    elif isinstance(wandb_cfg, bool):
        wandb_enabled = wandb_cfg

    if not wandb_enabled:
        return _create_tensorboard_logger(config, wandb_cfg)

    # Default: use W&B
    return _create_wandb_logger(config, wandb_cfg)


def _create_tensorboard_logger(config: DictConfig, wandb_cfg: dict) -> TensorBoardLogger:
    """Create TensorBoard logger for when W&B is disabled."""
    exp_version = wandb_cfg.get("exp_version", "v1.0") if is_config(wandb_cfg) else "v1.0"

    return TensorBoardLogger(
        save_dir=config.paths.log_dir,
        name=config.exp_name,
        version=exp_version,
        default_hp_metric=False,
    )


def _create_wandb_logger(config: DictConfig, wandb_cfg: dict) -> WandbLogger:
    """Create Weights & Biases logger (default)."""
    from ocr.core.utils.wandb_base import generate_run_name, load_env_variables

    # Load environment variables for W&B API key
    load_env_variables()

    # Resolve interpolations before generating run name
    OmegaConf.resolve(config)
    run_name = generate_run_name(config)

    # Serialize config for W&B, handling Hydra interpolations
    # If resolution fails, it's a config problem - let it propagate
    wandb_config = OmegaConf.to_container(config, resolve=True)

    project_name = wandb_cfg.get("project_name", "ocr-training") if is_config(wandb_cfg) else "ocr-training"

    return WandbLogger(
        name=run_name,
        project=project_name,
        config=wandb_config,
    )
