import torch
import torch.nn as nn
from transformers import LayoutLMv3ForTokenClassification, LiltForTokenClassification
from ocr.utils.config_utils import ensure_dict

class LayoutLMv3Wrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use project standard to ensure primitive dict
        self.cfg = ensure_dict(config)

        num_labels = self.cfg.get("num_labels", 7)

        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.cfg.get("pretrained_model_name_or_path", "microsoft/layoutlmv3-base"),
            num_labels=num_labels
        )

    def forward(self, input_ids, bbox, attention_mask, pixel_values, labels=None, return_loss=True, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        predictions = {
            "logits": outputs.logits,
            "loss": outputs.loss if return_loss else None
        }

        # Add loss_dict for logging if loss is present
        if return_loss and outputs.loss is not None:
             predictions["loss_dict"] = {"ce_loss": outputs.loss}

        return predictions

    def get_optimizers(self):
        # Check if optimizer config exists
        lr = 5e-5
        optimizer_cfg = self.cfg.get("optimizer", {})
        if optimizer_cfg:
             # ensure_dict recursively converts, so optimizer_cfg is a dict if it exists
             lr = optimizer_cfg.get("lr", 5e-5)

        # If we are in PL module, it might override this, but this is default model optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return [optimizer], []

class LiLTWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = ensure_dict(config)
        num_labels = self.cfg.get("num_labels", 7)

        self.model = LiltForTokenClassification.from_pretrained(
            self.cfg.get("pretrained_model_name_or_path", "nielsr/lilt-xlm-roberta-base"),
            num_labels=num_labels
        )

    def forward(self, input_ids, bbox, attention_mask, labels=None, return_loss=True, **kwargs):
        # LiLT might check for pixel_values but usually doesn't need them if not fine-tuning layout?
        # Check standard usage. LiLT uses bbox and text.

        outputs = self.model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            labels=labels
        )

        predictions = {
            "logits": outputs.logits,
            "loss": outputs.loss if return_loss else None
        }

        if return_loss and outputs.loss is not None:
             predictions["loss_dict"] = {"ce_loss": outputs.loss}

        return predictions

    def get_optimizers(self):
        lr = 5e-5
        optimizer_cfg = self.cfg.get("optimizer", {})
        if optimizer_cfg:
             lr = optimizer_cfg.get("lr", 5e-5)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        return [optimizer], []
