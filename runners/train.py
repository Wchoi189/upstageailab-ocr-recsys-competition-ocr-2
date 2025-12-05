import logging
import math
import sys
import warnings

# Setup project paths automatically
import hydra
from omegaconf import DictConfig

# Heavy imports (torch, lightning, transformers) are deferred until inside train()
# to enable fast config validation without loading 85s of dependencies

# Suppress known wandb Pydantic compatibility warnings
# This is a known issue where wandb uses incorrect Field() syntax in Annotated types
# The warnings come from Pydantic when processing wandb's type annotations
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*(repr|frozen).*Field.*function.*no effect", category=UserWarning)

# Also suppress by category for more reliable filtering
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass  # In case the warning class is not available in future pydantic versions

# wandb imported lazily inside train() to avoid slow imports

from ocr.utils.callbacks import build_callbacks
from ocr.utils.path_utils import ensure_output_dirs, setup_project_paths

setup_project_paths()

# ocr.lightning_modules imported inside train() to avoid loading models/datasets at module level

# PyTorch Lightning handles SIGINT/SIGTERM gracefully by default
# No custom signal handlers needed (and they cause threading issues in Streamlit)


# Setup consistent logging configuration
# Create a custom handler that flushes immediately
class ImmediateFlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
    force=True,  # Override any existing configuration
    handlers=[ImmediateFlushHandler(sys.stdout)],
)

# Reduce verbosity of some noisy loggers
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def train(config: DictConfig):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (DictConfig): A dictionary containing configuration settings for training.
    """
    # No global state needed - all cleanup handled by Lightning

    # Disable struct mode to allow Hydra to populate runtime fields dynamically
    # This fixes "Key 'mode' is not in struct" errors with Hydra 1.3.2
    from omegaconf import OmegaConf
    OmegaConf.set_struct(config, False)
    if hasattr(config, 'hydra') and config.hydra is not None:
        OmegaConf.set_struct(config.hydra, False)

    # === LAZY IMPORTS: Heavy ML libraries loaded here after Hydra config validation ===
    # This defers ~85s of import time until config is validated, enabling:
    # - Fast config validation (<5s instead of 85s)
    # - Quick error feedback for config typos
    # - Improved development iteration speed (3-5x faster)
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import LearningRateMonitor

    import wandb
    from ocr.lightning_modules import get_pl_modules_by_cfg
    # === END LAZY IMPORTS ===

    # Clean up any lingering W&B session to prevent warnings
    wandb.finish()

    pl.seed_everything(config.get("seed", 42), workers=True)

    # Enable Tensor Core utilization for better GPU performance
    torch.set_float32_matmul_precision("high")

    # GPU device selection is handled by PyTorch Lightning automatically
    # For multi-GPU training, set trainer.devices=2 and trainer.strategy=ddp in config

    model_module, data_module = get_pl_modules_by_cfg(config)

    # Ensure key output directories exist before creating callbacks
    output_dirs = [config.paths.log_dir, config.paths.checkpoint_dir]
    if hasattr(config.paths, "submission_dir"):
        output_dirs.append(config.paths.submission_dir)

    ensure_output_dirs(output_dirs)

    # Create appropriate logger (W&B or TensorBoard) based on configuration
    from ocr.utils.logger_factory import create_logger

    logger = create_logger(config)

    callbacks = build_callbacks(config)

    # Always add LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Preprocess trainer config for PyTorch Lightning compatibility
    trainer_config = dict(config.trainer)
    if trainer_config.get("max_steps") is None:
        trainer_config["max_steps"] = -1  # PyTorch Lightning expects -1 for unlimited steps

    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume", None),
    )

    # Run test evaluation unless explicitly skipped
    if not config.get("skip_test", False):
        trainer.test(
            model_module,
            data_module,
        )

    # Finalize wandb run if wandb was used
    if config.logger.wandb:
        from ocr.utils.wandb_utils import finalize_run

        metrics: dict[str, float] = {}

        def _to_float(value) -> float | None:  # Changed from float | None to Optional[float]
            try:
                if isinstance(value, torch.Tensor):
                    return float(value.detach().cpu().item())
                if hasattr(value, "item"):
                    item_val = value.item()
                    return float(item_val)
                return float(value)
            except (TypeError, ValueError):
                return None

        for key, value in trainer.callback_metrics.items():
            cast_value = _to_float(value)
            if cast_value is not None and math.isfinite(cast_value):
                metrics[key] = cast_value

        finalize_run(metrics)


if __name__ == "__main__":
    train()
