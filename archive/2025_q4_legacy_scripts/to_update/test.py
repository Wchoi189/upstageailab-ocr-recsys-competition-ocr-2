import warnings

import hydra

from ocr.core.utils.callbacks import build_callbacks
from ocr.core.utils.logger_factory import create_logger
from ocr.core.utils.path_utils import get_path_resolver, setup_project_paths

setup_project_paths()

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
    # Pydantic v1 doesn't have this warning class
    pass

# # from ocr.core.lightning import get_pl_modules_by_cfg  # TODO: Use OCRProjectOrchestrator like train.py  # TODO: Use OCRProjectOrchestrator like train.py  # noqa: E402


@hydra.main(config_path=str(get_path_resolver().config.config_dir), config_name="eval", version_base=None)
def test(config):
    """
    Train a OCR model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for test.
    """
    # Lazy import to keep startup light for config validation
    import lightning.pytorch as pl

    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module, data_module = get_pl_modules_by_cfg(config)

    callbacks = build_callbacks(config)

    logger = create_logger(config)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
    )

    ckpt_path = config.get("checkpoint_path")
    assert ckpt_path is not None, "checkpoint_path must be provided for test"

    trainer.test(
        model_module,
        data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    test()
