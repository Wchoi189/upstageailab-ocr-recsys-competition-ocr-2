
import hydra
from omegaconf import OmegaConf
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_path="../../configs", config_name="main", version_base="1.1")
def dry_run(cfg):
    logger.info("Initializing Hydra...")

    # 1. Instantiate Model
    try:
        if cfg.model.get("architectures"):
             logger.info("Directly instantiating model from 'model.architectures' (Rec pattern)")
             model = hydra.utils.instantiate(cfg.model.architectures)
        else:
             logger.info("Instantiating model from 'model' (Det pattern)")
             model = hydra.utils.instantiate(cfg.model)

        logger.info(f"Model instantiated: {type(model).__name__}")

    except Exception as e:
        logger.error(f"Failed to instantiate model: {e}")
        return

    # 2. Check Parameters
    try:
        params = list(model.parameters())
        param_count = sum(p.numel() for p in params)
        logger.info(f"Parameter count: {param_count}")
        if param_count == 0:
            logger.error("❌ Model has 0 parameters! (This explains optimizer error)")
        else:
            logger.info("✅ Model has parameters.")
    except Exception as e:
        logger.error(f"Failed to inspect parameters: {e}")

    # 3. Check Components
    components = ['encoder', 'decoder', 'head', 'loss']
    for comp in components:
        if hasattr(model, comp):
            val = getattr(model, comp)
            status = "✅ Present" if val is not None else "❌ Missing (None)"
            logger.info(f"Component '{comp}': {status}")
        else:
            logger.info(f"Component '{comp}': ❌ Not defined in class")

    # 4. Simulate Forward Pass
    logger.info("Simulating forward pass...")
    try:
        # Create dummy input [B, C, H, W]
        dummy_input = torch.randn(2, 3, 32, 128)

        # Recognition usually needs output targets for training/loss?
        # Let's try inference mode first (no targets)
        logger.info("Attempting Inference Forward Pass (no targets)...")
        output = model(dummy_input, return_loss=False)
        logger.info("✅ Inference Forward Pass Successful")

        # Try Training Forward Pass (needs targets)
        logger.info("Attempting Training Forward Pass (with targets)...")
        # Targets: [B, MaxLen]
        dummy_targets = torch.randint(0, 50, (2, 25))
        output = model(dummy_input, return_loss=True, text_tokens=dummy_targets)
        logger.info("✅ Training Forward Pass Successful")

    except Exception as e:
        logger.error(f"❌ Forward Pass Failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    dry_run()
