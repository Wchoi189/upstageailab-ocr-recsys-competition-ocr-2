import logging
import math
import multiprocessing as mp
import os
import signal
import sys
import warnings

# Fix for CUDA multiprocessing conflict between wandb and PyTorch DataLoader
# Use 'forkserver' instead of 'fork' to avoid CUDA context corruption
# forkserver is faster than 'spawn' but still CUDA-safe
# See: docs/sessions/2025-11-10-cuda-debugging-session-handover.md
try:
    mp.set_start_method('forkserver', force=False)
    print("[Multiprocessing] Set start method to 'forkserver' for CUDA safety")
except RuntimeError:
    # Already set (e.g., by another module or previous run)
    current_method = mp.get_start_method()
    print(f"[Multiprocessing] Start method already set to '{current_method}'")

# Setup project paths automatically
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig

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

# Defer wandb import to prevent hanging during module import
# wandb will be imported lazily when needed in the train() function
# This prevents wandb from trying to connect to server during import

from ocr.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()

from ocr.lightning_modules import get_pl_modules_by_cfg  # noqa: E402

_shutdown_in_progress = False
trainer = None
data_module = None


def _safe_wandb_finish():
    """
    Safely finish any lingering wandb sessions without blocking on import.

    BUG-20251109-002: Fixed wandb import hanging during module import.
    Changed from top-level import to lazy import helper function.
    See: docs/bug_reports/BUG-20251109-002-code-changes.md
    """
    try:
        # Set environment variables to prevent hanging during import
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = "disabled"
        if "WANDB_SILENT" not in os.environ:
            os.environ["WANDB_SILENT"] = "true"

        import wandb
        wandb.finish()
    except (ImportError, Exception):
        # wandb not available or failed to import - ignore
        pass


def signal_handler(signum, frame):
    """Handle interrupt signals to ensure graceful shutdown without recursion."""
    global _shutdown_in_progress
    if _shutdown_in_progress:
        return
    _shutdown_in_progress = True

    print(f"Received signal {signum}. Shutting down gracefully...")

    try:
        if trainer is not None:
            print("Stopping trainer...")
            # Lightning handles SIGINT/SIGTERM for graceful shutdown
    except Exception as e:
        print(f"Error during trainer shutdown: {e}")

    try:
        if data_module is not None:
            print("Cleaning up data module...")
            # DataLoader workers will be cleaned up by process shutdown
    except Exception as e:
        print(f"Error during data module cleanup: {e}")

    # Do not send SIGTERM to our own process group to avoid recursive signals
    print("Shutdown complete.")
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Avoid creating a new process group here; the caller (UI) manages process groups


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
# Reduce verbosity of polygon_utils to reduce blank lines from progress bar interruptions
logging.getLogger("ocr.utils.polygon_utils").setLevel(logging.WARNING)


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (DictConfig): A dictionary containing configuration settings for training.
    """
    global trainer, data_module

    # Setup CUDA debugging environment variables if not already set
    # These help debug CUDA errors by making operations synchronous
    if "CUDA_LAUNCH_BLOCKING" not in os.environ:
        # Check if debugging is enabled via config or environment
        debug_cuda = config.get("runtime", {}).get("debug_cuda", False) or os.environ.get("DEBUG_CUDA", "0") == "1"
        if debug_cuda:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            print("[CUDA Debug] Enabled CUDA_LAUNCH_BLOCKING=1 for synchronous CUDA operations")

    if "TORCH_USE_CUDA_DSA" not in os.environ:
        debug_cuda = config.get("runtime", {}).get("debug_cuda", False) or os.environ.get("DEBUG_CUDA", "0") == "1"
        if debug_cuda:
            os.environ["TORCH_USE_CUDA_DSA"] = "1"
            print("[CUDA Debug] Enabled TORCH_USE_CUDA_DSA=1 for device-side assertions")

    # Clean up any lingering wandb sessions before starting
    # This is done here (lazily) instead of at module import time to prevent hanging
    _safe_wandb_finish()

    print("[DEBUG] Step 1: Starting training function")
    print("[DEBUG] Step 1.1: Setting random seed")
    pl.seed_everything(config.get("seed", 42), workers=True)
    print("[DEBUG] Step 1.2: Random seed set")

    # Enable Tensor Core utilization for better GPU performance
    import torch
    print("[DEBUG] Step 1.3: PyTorch imported")

    # Check CUDA availability and print diagnostic info
    if torch.cuda.is_available():
        print(f"[CUDA Info] CUDA available: {torch.cuda.is_available()}")
        print(f"[CUDA Info] CUDA device count: {torch.cuda.device_count()}")
        print(f"[CUDA Info] Current device: {torch.cuda.current_device()}")
        print(f"[CUDA Info] Device name: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA Info] CUDA version: {torch.version.cuda}")
        print(f"[CUDA Info] cuDNN version: {torch.backends.cudnn.version()}")

        # Check for CUDA errors before starting
        try:
            torch.cuda.empty_cache()
            # Test a simple CUDA operation
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.synchronize()
            print("[CUDA Info] CUDA initialization test passed")
        except Exception as e:
            print(f"[CUDA Warning] CUDA initialization test failed: {e}")
            print("[CUDA Warning] Continuing anyway, but errors may occur")

    torch.set_float32_matmul_precision("high")

    runtime_cfg = config.get("runtime") or {}
    auto_gpu_devices = runtime_cfg.get("auto_gpu_devices", True)
    preferred_strategy = runtime_cfg.get("ddp_strategy", "ddp_find_unused_parameters_false")
    min_auto_devices = runtime_cfg.get("min_auto_devices", 2)

    def _normalize_device_request(value):
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    if auto_gpu_devices and config.trainer.get("accelerator", "cpu") == "gpu" and torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        requested_devices = _normalize_device_request(config.trainer.get("devices"))
        if available_gpus >= max(1, min_auto_devices):
            if requested_devices in (None, 1):
                config.trainer.devices = available_gpus
                strategy_cfg = config.trainer.get("strategy")
                if strategy_cfg in (None, "auto"):
                    config.trainer.strategy = preferred_strategy
                print(f"[AutoParallel] Scaling to {available_gpus} GPUs with strategy='{config.trainer.strategy}'.")
            elif isinstance(requested_devices, int) and requested_devices > available_gpus:
                config.trainer.devices = available_gpus
                print(
                    f"[AutoParallel] Requested {requested_devices} GPUs, but only {available_gpus} detected. "
                    f"Falling back to {available_gpus}."
                )

    print("[DEBUG] Step 2: Before model and data module creation")
    model_module, data_module = get_pl_modules_by_cfg(config)
    print("[DEBUG] Step 2.1: Model and data module created")

    # Ensure key output directories exist before creating callbacks
    try:
        os.makedirs(config.paths.log_dir, exist_ok=True)
        os.makedirs(config.paths.checkpoint_dir, exist_ok=True)
        # Some workflows also expect a submission dir
        if hasattr(config.paths, "submission_dir"):
            os.makedirs(config.paths.submission_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to ensure output directories exist: {e}")

    from lightning.pytorch.loggers import Logger

    logger: Logger

    if "wandb" in config.logger and config.logger.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger  # noqa: E402
        from omegaconf import OmegaConf  # noqa: E402

        from ocr.utils.wandb_utils import generate_run_name, load_env_variables  # noqa: E402

        # Load environment variables from .env.local/.env
        load_env_variables()

        # Resolve interpolations before generating run name
        OmegaConf.resolve(config)

        run_name = generate_run_name(config)

        # Properly serialize config for wandb, handling hydra interpolations
        try:
            # Try to resolve interpolations for cleaner config
            wandb_config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            # Fall back to unresolved config if resolution fails
            wandb_config = OmegaConf.to_container(config, resolve=False)

        logger = WandbLogger(
            name=run_name,
            project=config.logger.project_name,
            config=wandb_config,
        )
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger  # noqa: E402

        logger = TensorBoardLogger(
            save_dir=config.paths.log_dir,
            name=config.exp_name,
            version=config.logger.exp_version,
            default_hp_metric=False,
        )

    # Ensure no default logger is created by explicitly setting logger
    # This prevents lightning_logs from being created in the root directory

    # --- Callback Configuration ---
    # This is the new, Hydra-native way to handle callbacks.
    # It iterates through the 'callbacks' config group and instantiates each one.
    callbacks = []
    if config.get("callbacks"):
        for _, cb_conf in config.callbacks.items():
            if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
                # Only instantiate enabled callbacks
                if cb_conf.get("enabled", True):
                    # Commented out: Callback instantiation logging to reduce noise
                    # print(f"Instantiating callback <{cb_conf._target_}>")
                    callback = hydra.utils.instantiate(cb_conf)

                    # Pass resolved config to checkpoint callback for saving alongside checkpoints
                    if hasattr(callback, "_resolved_config"):
                        from omegaconf import OmegaConf

                        resolved_config = OmegaConf.to_container(config, resolve=True)
                        callback._resolved_config = resolved_config

                    callbacks.append(callback)

    # Always add LearningRateMonitor
    callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Preprocess trainer config for PyTorch Lightning compatibility
    trainer_config = dict(config.trainer)
    if trainer_config.get("max_steps") is None:
        trainer_config["max_steps"] = -1  # PyTorch Lightning expects -1 for unlimited steps
    # Disable model summary to reduce verbosity and whitespace
    if "enable_model_summary" not in trainer_config:
        trainer_config["enable_model_summary"] = False
    # Optionally disable progress bar to reduce blank lines from Rich progress bar redraws
    # This can be overridden in config if progress bar is needed
    if "enable_progress_bar" not in trainer_config and config.get("disable_progress_bar", False):
        trainer_config["enable_progress_bar"] = False

    print("[DEBUG] Step 3: Before trainer creation")
    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=callbacks)
    print("[DEBUG] Step 3.1: Trainer created")

    print("[DEBUG] Step 4: Before trainer.fit()")
    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume", None),
    )
    print("[DEBUG] Step 4.1: trainer.fit() completed")

    # Run test evaluation unless explicitly skipped
    if not config.get("skip_test", False):
        trainer.test(
            model_module,
            data_module,
        )

    # Finalize wandb run if wandb was used
    if "wandb" in config.logger and config.logger.wandb.enabled:
        from ocr.utils.wandb_utils import finalize_run  # noqa: E402

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
