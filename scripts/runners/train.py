# MULTIPROCESSING METHOD CONFIGURATION
# ============================================================================
# ISSUE: Spawn method (required for CUDA + multiprocessing) fails with:
#   "TypeError: cannot pickle 'Environment' object"
#
# ROOT CAUSE: Hydra Environment objects are stored somewhere in the dataset/dataloader chain.
# With spawn method, all objects must be picklable to send to worker processes.
#
# TEMPORARY WORKAROUND: Use fork method (default) with num_workers=0
# - Fork works with num_workers=0 (no child processes)
# - Fork + num_workers>0 causes SIGABRT/CUDA crashes
#
# TODO: Fix pickle issue by:
#   1. Remove Hydra Environment references from dataset/dataloader
#   2. Use OmegaConf.to_container() to convert configs to plain dicts
#   3. Then re-enable spawn method for proper CUDA multiprocessing
# ============================================================================
import torch
import torch.multiprocessing as mp
# try:
#     mp.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

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
