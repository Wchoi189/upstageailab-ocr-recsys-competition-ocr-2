from ocr.datasets import get_datasets_by_cfg
from ocr.models import get_model_by_cfg

from .ocr_pl import OCRDataPLModule, OCRPLModule


def get_pl_modules_by_cfg(config):
    print("[DEBUG] Step 2.1: Inside get_pl_modules_by_cfg")
    print("[DEBUG] Step 2.2: Before model creation")
    model = get_model_by_cfg(config.model)
    print("[DEBUG] Step 2.3: Model created")
    data_config = getattr(config, "data", None)
    print("[DEBUG] Step 2.4: Before dataset creation")
    dataset = get_datasets_by_cfg(config.datasets, data_config, config)
    print("[DEBUG] Step 2.5: Dataset created")
    metric_cfg = None
    if "metrics" in config and "eval" in config.metrics:
        metric_cfg = config.metrics.eval

    print("[DEBUG] Step 2.6: Before OCRPLModule creation")
    modules = OCRPLModule(model=model, dataset=dataset, config=config, metric_cfg=metric_cfg)
    print("[DEBUG] Step 2.7: OCRPLModule created")
    print("[DEBUG] Step 2.8: Before OCRDataPLModule creation")
    data_modules = OCRDataPLModule(dataset=dataset, config=config)
    print("[DEBUG] Step 2.9: OCRDataPLModule created")
    return modules, data_modules
