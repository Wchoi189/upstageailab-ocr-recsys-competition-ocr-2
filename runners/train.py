import logging
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
from ocr.pipelines.orchestrator import OCRProjectOrchestrator

# Suppress known wandb warning
warnings.filterwarnings("ignore", message=r"The '(repr|frozen)' attribute.*Field.*function.*no effect", category=UserWarning)

log = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def train(config: DictConfig):
    """
    Entry point for OCR Training/Evaluation.
    Delegates entirely to the OCRProjectOrchestrator.
    """

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
