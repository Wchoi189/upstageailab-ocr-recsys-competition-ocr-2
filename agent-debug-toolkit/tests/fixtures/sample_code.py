"""
Sample code fixtures for testing analyzers.

This module contains example Python code patterns that the analyzers
should be able to detect.
"""

# Sample code that uses configuration patterns
SAMPLE_CONFIG_ACCESS = """
from omegaconf import OmegaConf, DictConfig

class OCRModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.architecture_config_obj = cfg.architecture_name

        # Direct config access
        encoder_cfg = cfg.encoder
        decoder_cfg = cfg.decoder

        # Nested access
        model_head = cfg.model.head

        # Subscript access
        loss_name = cfg["loss"]["name"]

        # getattr pattern
        has_encoder = hasattr(cfg, "encoder")
        encoder = getattr(cfg, "encoder", None)

    def _prepare_component_configs(self, cfg):
        # Multiple merge operations
        merged_config = OmegaConf.create({})

        if self.architecture_config_obj is not None:
            merged_config = OmegaConf.merge(merged_config, self.architecture_config_obj)

        # Top level overrides
        top_level = {}
        for key in ["encoder", "decoder", "head", "loss"]:
            if hasattr(cfg, key):
                top_level[key] = getattr(cfg, key)

        merged_config = OmegaConf.merge(merged_config, top_level)

        # Final override
        if hasattr(cfg, "component_overrides"):
            merged_config = OmegaConf.merge(merged_config, cfg.component_overrides)

        return merged_config
"""

SAMPLE_HYDRA_USAGE = """
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # Instantiate model from config
    model = instantiate(cfg.model)

    # Create optimizer
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Pattern with _target_
    custom_config = {
        "_target_": "my_module.MyClass",
        "_recursive_": False,
        "param1": 42,
    }

    return model, optimizer

if __name__ == "__main__":
    main()
"""

SAMPLE_COMPONENT_INSTANTIATION = """
from ocr.core.models.encoder import get_encoder_by_cfg
from ocr.core.models.decoder import get_decoder_by_cfg
from ocr.core.models.head import get_head_by_cfg
from ocr.core import get_registry

class ModelFactory:
    def create_model(self, cfg):
        registry = get_registry()

        # Factory pattern calls
        self.encoder = get_encoder_by_cfg(cfg.encoder)
        self.decoder = get_decoder_by_cfg(cfg.decoder)
        self.head = get_head_by_cfg(cfg.head)

        # Registry pattern
        components = registry.create_architecture_components(
            cfg.architecture_name,
            encoder_config=cfg.encoder,
            decoder_config=cfg.decoder
        )

        # Direct instantiation
        loss = FPNLoss(in_channels=256)

        return components
"""
