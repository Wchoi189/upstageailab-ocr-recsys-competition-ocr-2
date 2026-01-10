from ocr.data.datasets import get_datasets_by_cfg
from ocr.core.models import get_model_by_cfg

from .ocr_pl import OCRDataPLModule, OCRPLModule


def get_pl_modules_by_cfg(config):
    import hydra
    # Inject vocab size into model config if needed
    if "tokenizer" in config.data:
        tokenizer = hydra.utils.instantiate(config.data.tokenizer)
        if "component_overrides" in config.model and config.model.component_overrides:
            if "head" in config.model.component_overrides:
                if "params" not in config.model.component_overrides.head:
                    config.model.component_overrides.head.params = {}
                config.model.component_overrides.head.params.out_channels = tokenizer.vocab_size

            # Inject vocab_size into Decoder (specifically for PARSeq/Transformers with Embeddings)
            if "decoder" in config.model.component_overrides:
                if "params" not in config.model.component_overrides.decoder:
                    config.model.component_overrides.decoder.params = {}
                config.model.component_overrides.decoder.params.vocab_size = tokenizer.vocab_size

    model = get_model_by_cfg(config.model)
    data_config = getattr(config, "data", None)
    dataset = get_datasets_by_cfg(config.datasets, data_config, config)
    metric_cfg = None
    if "metrics" in config and "eval" in config.metrics:
        metric_cfg = config.metrics.eval

    modules = OCRPLModule(model=model, dataset=dataset, config=config, metric_cfg=metric_cfg)
    data_modules = OCRDataPLModule(dataset=dataset, config=config)
    return modules, data_modules
