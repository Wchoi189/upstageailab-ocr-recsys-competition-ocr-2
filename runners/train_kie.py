import logging
import warnings
import sys

# Setup project paths automatically
import hydra
from omegaconf import DictConfig, OmegaConf

# Determine project root and add to path if not already there
# This mimics setup_project_paths from ocr.utils.path_utils but we verify imports first
from pathlib import Path

# Setup logging with RichHandler
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    force=True,
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

# Reduce verbosity
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
# Suppress specific warnings if needed
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")

@hydra.main(config_path="../configs", config_name="train_kie", version_base=None)
def main(config: DictConfig):
    """
    Train a KIE model using the provided configuration.
    """
    # Disable struct mode to allow Hydra to populate runtime fields dynamically
    OmegaConf.set_struct(config, False)
    if hasattr(config, "hydra") and config.hydra is not None:
        OmegaConf.set_struct(config.hydra, False)

    # === LAZY IMPORTS: Heavy ML libraries loaded here after Hydra config validation ===
    import torch
    import lightning.pytorch as pl
    from transformers import LayoutLMv3Processor, AutoTokenizer

    from ocr.data.datasets.kie_dataset import KIEDataset
    from ocr.models.kie_models import LayoutLMv3Wrapper, LiLTWrapper
    from ocr.lightning_modules.kie_pl import KIEPLModule, KIEDataPLModule
    from ocr.utils.path_utils import ensure_output_dirs
    # === END LAZY IMPORTS ===

    # Import config utils
    from ocr.utils.config_utils import ensure_dict

    pl.seed_everything(config.get("seed", 42), workers=True)

    # Enable Tensor Core utilization for better GPU performance
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # Ensure output directories exist
    output_dirs = [config.paths.log_dir, config.paths.checkpoint_dir]
    ensure_output_dirs(output_dirs)

    # 1. Prepare Processor/Tokenizer
    # Use ensure_dict to safely convert to primitive dict
    model_config = ensure_dict(config.model)

    model_type = model_config.get("type", "layoutlmv3")
    pretrained_path = model_config.get("pretrained_model_name_or_path", "microsoft/layoutlmv3-base")

    logger = logging.getLogger(__name__)
    logger.info(f"Initializing {model_type} model from {pretrained_path}")

    if model_type == "layoutlmv3":
        processor = LayoutLMv3Processor.from_pretrained(pretrained_path, apply_ocr=False)
        tokenizer = None
    elif model_type == "lilt":
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        processor = None
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 2. Prepare Datasets
    data_config = ensure_dict(config.data)
    train_path = data_config.get("train_path")
    val_path = data_config.get("val_path")

    if not train_path or not val_path:
        raise ValueError("train_path and val_path must be specified in data config")

    # Ensure label_list is a list, handling ListConfig via ensure_dict check/conversion
    # ensure_dict recursively converts ListConfig to list
    label_list = data_config.get("label_list", ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"])
    if not isinstance(label_list, list):
         # Just in case ensure_dict missed it (it shouldn't if data_config was converted)
         # But allows robust fallback
         label_list = ensure_dict(label_list) if hasattr(label_list, "_content") else list(label_list)

    train_dataset = KIEDataset(
        parquet_file=train_path,
        processor=processor,
        tokenizer=tokenizer,
        image_dir=data_config.get("image_dir", None),
        max_length=model_config.get("max_length", 512),
        label_list=label_list
    )

    val_dataset = KIEDataset(
        parquet_file=val_path,
        processor=processor,
        tokenizer=tokenizer,
        image_dir=data_config.get("image_dir", None),
        max_length=model_config.get("max_length", 512),
        label_list=label_list
    )

    train_config = ensure_dict(config.train)
    data_module = KIEDataPLModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=train_config.get("batch_size", 4),
        num_workers=train_config.get("num_workers", 4)
    )

    # 3. Prepare Model
    # Explicitly pass required config to model wrapper
    # ensure_dict already called on model_config

    if model_type == "layoutlmv3":
        model = LayoutLMv3Wrapper(model_config)
    elif model_type == "lilt":
        model = LiLTWrapper(model_config)

    # 4. Prepare Lightning Module
    # Ensure train_config is passed correctly
    pl_module = KIEPLModule(model, train_config, label_list)

    # 5. Prepare Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename="{epoch}-{val_f1:.2f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3
    )

    # Configure Logger
    logger_type = None
    if config.get("use_wandb", False):
        from lightning.pytorch.loggers import WandbLogger
        logger_type = WandbLogger(
            project=config.get("project_name", "ocr-kie"),
            name=config.get("run_name", "kie-run")
        )

    trainer = pl.Trainer(
        max_epochs=train_config.get("max_epochs", 10),
        accelerator=train_config.get("accelerator", "auto"),
        devices=train_config.get("devices", 1),
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=logger_type,
        strategy="ddp_find_unused_parameters_true" if train_config.get("devices", 1) > 1 else "auto"
    )

    # 6. Train
    trainer.fit(pl_module, datamodule=data_module)

if __name__ == "__main__":
    main()
