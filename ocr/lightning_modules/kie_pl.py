import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, classification_report
import numpy as np

# Use centralized config utilities
from ocr.utils.config_utils import ensure_dict, is_config

# Import data contracts
# from ocr.core.kie_validation import KIEDataItem

class KIEDataPLModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset=None, predict_dataset=None, batch_size=8, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        return None

    def predict_dataloader(self):
        if self.predict_dataset:
            return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        return None

    def collate_fn(self, batch):
        # Strictly typed collation
        batch_out = {}
        if not batch:
            return batch_out

        keys = batch[0].keys()
        for k in keys:
             # Stack tensors
             items = [item[k] for item in batch]
             if isinstance(items[0], torch.Tensor):
                 batch_out[k] = torch.stack(items)
             elif isinstance(items[0], (int, float)):
                 batch_out[k] = torch.tensor(items)
             else:
                 batch_out[k] = items # Keep as list (e.g. image_path)

        return batch_out


class KIEPLModule(pl.LightningModule):
    def __init__(self, model, config: dict | object, label_list: list[str]):
        super().__init__()
        self.model = model
        # Safe conversion to primitive dict using project standard
        self.config = ensure_dict(config)

        self.label_list = label_list
        self.save_hyperparameters(ignore=["model"])

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):
        # Validate batch existence usually covered by collate, but double check keys
        required_keys = {"input_ids", "attention_mask", "bbox", "labels"}
        if not all(k in batch for k in required_keys):
            raise ValueError(f"Batch missing required keys: {required_keys - batch.keys()}")

        outputs = self.model(**batch)
        loss = outputs["loss"]

        # Log carefully
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        outputs_model = self.model(**batch)
        loss = outputs_model["loss"]
        logits = outputs_model["logits"]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        predictions = torch.argmax(logits, dim=2)
        labels = batch["labels"]

        true_predictions = [
            [self.label_list[p.item()] for (p, l) in zip(prediction, label) if l.item() != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l.item()] for (p, l) in zip(prediction, label) if l.item() != -100]
            for prediction, label in zip(predictions, labels)
        ]

        self.validation_step_outputs.append({"predictions": true_predictions, "labels": true_labels})
        return loss

    def on_validation_epoch_end(self):
        preds = []
        labels = []
        for x in self.validation_step_outputs:
            preds.extend(x["predictions"])
            labels.extend(x["labels"])

        if preds:
            f1 = f1_score(labels, preds)
            report = classification_report(labels, preds)

            self.log("val_f1", f1, prog_bar=True)
            # Log classification report
            # print("\n" + report) # Avoid printing in production/CI
        else:
            self.log("val_f1", 0.0, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # Safe optimizer access
        optimizers = self.model.get_optimizers()
        if not optimizers or not optimizers[0]:
             # Fallback
             optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr", 5e-5))
        else:
             optimizer = optimizers[0][0]

        if self.trainer.estimated_stepping_batches:
             num_training_steps = self.trainer.estimated_stepping_batches
        else:
             num_training_steps = 1000

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.get("warmup_steps", 0),
            num_training_steps=num_training_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
