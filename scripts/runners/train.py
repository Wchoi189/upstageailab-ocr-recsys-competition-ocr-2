# MULTIPROCESSING METHOD CONFIGURATION
# ============================================================================
# SOLUTION (2026-02-08): Use 'spawn' method for CUDA + multiprocessing in Docker
#
# ISSUE: Default 'fork' method causes CUDA initialization errors with num_workers > 0
#   - Fork inherits CUDA context from main process
#   - Workers cannot properly initialize their own CUDA contexts
#   - Results in: "CUDA error: initialization error" during tensor cleanup
#
# FIX: Spawn creates fresh Python processes (no state inheritance)
#   - Each worker initializes its own clean CUDA context
#   - Compatible with pin_memory=true and persistent_workers=true
#   - Enables safe multi-worker data loading with CUDA
#
# VERIFIED: num_workers=0,1,2 all work with spawn method (2026-02-08)
# ============================================================================
import torch
import torch.multiprocessing as mp
# ATTEMPT 1: Use spawn method (best for CUDA + multiprocessing)
try:
    mp.set_start_method('spawn', force=True)
    print("[MULTIPROCESSING] Using 'spawn' start method (CUDA-safe)")
except RuntimeError as e:
    # Start method already set
    print(f"[MULTIPROCESSING] Start method already set: {mp.get_start_method()}")

# Enable Tensor Cores for RTX 3090
torch.set_float32_matmul_precision('medium')


import logging
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf

# Suppress known wandb warning
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)

log = logging.getLogger(__name__)

from ocr.core.utils.path_utils import PROJECT_ROOT

@hydra.main(config_path=str(PROJECT_ROOT / "configs"), config_name="main", version_base=None)
def train(config: DictConfig):
    """
    Entry point for OCR Training/Evaluation.
    Delegates entirely to the OCRProjectOrchestrator.
    """

    # Lazy import - defers torch/Lightning loading until function execution
    from ocr.pipelines.orchestrator import OCRProjectOrchestrator

    # 1. Disable struct mode to allow runtime injection
    OmegaConf.set_struct(config, False)
    if hasattr(config, "hydra") and config.hydra is not None:
        OmegaConf.set_struct(config.hydra, False)

    # 2. Instantiate the Bridge / Orchestrator
    # The Orchestrator handles:
    # - Domain logic (Detection vs Recognition)
    # - Dependency Injection (Tokenizer -> Model)
    # - Trainer setup (Callbacks, Loggers)
    orchestrator = OCRProjectOrchestrator(config)

    # 3. Execute
    orchestrator.run()

if __name__ == "__main__":
    train()
