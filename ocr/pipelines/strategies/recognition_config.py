import logging
import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

class RecognitionConfigStrategy:
    """Strategy for handling recognition-specific configuration injection."""

    @staticmethod
    def inject_vocab_size(cfg: DictConfig) -> None:
        """Inject vocab_size into recognition model config."""
        if "tokenizer" not in cfg.data:
            return

        logger.info("💉 Injecting vocab_size for recognition model...")
        tokenizer = hydra.utils.instantiate(cfg.data.tokenizer)
        vocab_size = tokenizer.vocab_size

        # Disable struct mode safely
        OmegaConf.set_struct(cfg, False)

        # 1. Update Global/Model var
        if "vocab_size" in cfg.model:
            cfg.model.vocab_size = vocab_size

        # 2. Inject into Head
        if "head" in cfg.model:
            cfg.model.head.out_features = vocab_size
            cfg.model.head.num_classes = vocab_size
            cfg.model.head.out_channels = vocab_size

            # Legacy Params support
            if "params" in cfg.model.head:
                cfg.model.head.params.out_channels = vocab_size
                cfg.model.head.params.out_features = vocab_size

        # 3. Inject into Decoder
        if "decoder" in cfg.model:
            cfg.model.decoder.vocab_size = vocab_size
            if "params" in cfg.model.decoder:
                cfg.model.decoder.params.vocab_size = vocab_size

        # 4. Component Overrides
        if "component_overrides" not in cfg.model:
            cfg.model.component_overrides = {}

        overrides = cfg.model.component_overrides
        if "head" not in overrides:
             overrides.head = {}
        overrides.head.out_channels = vocab_size
        overrides.head.out_features = vocab_size
        overrides.head.num_classes = vocab_size

        if "decoder" not in overrides:
            overrides.decoder = {}
        overrides.decoder.vocab_size = vocab_size

        if "params" not in overrides.decoder:
            overrides.decoder.params = {}
        overrides.decoder.params.vocab_size = vocab_size

        logger.info(f"   ✓ Vocab size {vocab_size} injected into model config")
