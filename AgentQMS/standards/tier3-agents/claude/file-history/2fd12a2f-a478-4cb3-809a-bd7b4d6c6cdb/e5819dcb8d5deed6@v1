import logging
import sys
import warnings

# Determine project root and add to path if not already there
from pathlib import Path

# Setup project paths automatically
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

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
# Suppress Transformers FutureWarning about device argument
warnings.filterwarnings("ignore", category=FutureWarning, message="The `device` argument is deprecated")
# Suppress LR Scheduler warning (common false positive in PL)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

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
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import ConcatDataset
    from transformers import AutoTokenizer, LayoutLMv3Processor

    from ocr.kie.data import KIEDataset
    from ocr.lightning_modules.callbacks.kie_wandb_image_logging import WandBKeyInformationExtractionImageLogger
    from ocr.kie.trainer import KIEDataPLModule, KIEPLModule
    from ocr.kie.models import LayoutLMv3Wrapper, LiLTWrapper

    # === END LAZY IMPORTS ===
    # Import config utils
    from ocr.utils.config_utils import ensure_dict
    from ocr.utils.path_utils import ensure_output_dirs

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

    def create_dataset(path_or_list, img_dir, label_list, processor, tokenizer, max_length):
        if isinstance(path_or_list, (list, pd.Series, np.ndarray)) or (isinstance(path_or_list, (DictConfig, list))):
            # It's a list of datasets
            # If it's a list from Hydra, it might be a ListConfig. ensure_dict converts it to list.
            # But here `path_or_list` comes from data_config.get(), which is already processed by ensure_dict?
            # data_config is from ensure_dict(config.data).

            datasets = []
            # Check if it's a list of dicts (advanced config) or list of strings (simple paths)
            items = path_or_list if isinstance(path_or_list, list) else [path_or_list]

            for item in items:
                if isinstance(item, dict):
                    # {path: ..., image_dir: ...}
                    dataset_path = item.get("path") or item.get("parquet")
                    p = dataset_path
                    i_d = item.get("image_dir", img_dir) # Fallback to global image_dir
                else:
                    # just a string path
                    p = item
                    i_d = img_dir

                if not p: continue

                ds = KIEDataset(
                    parquet_file=p,
                    processor=processor,
                    tokenizer=tokenizer,
                    image_dir=i_d,
                    max_length=max_length,
                    label_list=label_list
                )
                datasets.append(ds)

            if not datasets:
                raise ValueError("No valid datasets found in list")
            return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

        else:
            # Single path
            return KIEDataset(
                parquet_file=path_or_list,
                processor=processor,
                tokenizer=tokenizer,
                image_dir=img_dir,
                max_length=max_length,
                label_list=label_list
            )

    # Allow "train_datasets" list in config, fallback to "train_path"
    train_input = data_config.get("train_datasets", data_config.get("train_path"))
    val_input = data_config.get("val_datasets", data_config.get("val_path"))
    global_image_dir = data_config.get("image_dir", None)
    max_len = model_config.get("max_length", 512)

    if not train_input or not val_input:
        raise ValueError("train_datasets/train_path and val_datasets/val_path must be specified")

    train_dataset = create_dataset(train_input, global_image_dir, label_list, processor, tokenizer, max_len)
    val_dataset = create_dataset(val_input, global_image_dir, label_list, processor, tokenizer, max_len)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")


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
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoint_dir,
        filename="{epoch}-{val_f1:.2f}",
        monitor="val_f1",
        mode="max",
        save_top_k=3
    )

    # Configure Logger
    logger_type = None
    callbacks_list = [checkpoint_callback, LearningRateMonitor(logging_interval="step")]

    if config.get("use_wandb", False):
        from lightning.pytorch.loggers import WandbLogger
        logger_type = WandbLogger(
            project=config.get("project_name", "ocr-kie"),
            name=config.get("run_name", "kie-run")
        )
        # Add WandB Image Logger
        callbacks_list.append(WandBKeyInformationExtractionImageLogger(label_list=label_list))


    # Filter out known keys that we handle explicitly or that might conflict if passed twice
    # warmup_steps is used by KIEPLModule.configure_optimizers(), not Trainer
    trainer_args = {k: v for k, v in train_config.items() if k not in ["batch_size", "num_workers", "max_epochs", "accelerator", "devices", "strategy", "warmup_steps"]}

    trainer = pl.Trainer(
        max_epochs=train_config.get("max_epochs", 10),
        accelerator=train_config.get("accelerator", "auto"),
        devices=train_config.get("devices", 1),
        callbacks=callbacks_list,
        logger=logger_type,

        strategy="ddp_find_unused_parameters_true" if train_config.get("devices", 1) > 1 else "auto",
        **trainer_args
    )

    # 6. Train
    trainer.fit(pl_module, datamodule=data_module)

if __name__ == "__main__":
    main()
