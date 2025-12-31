import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, ProcessorMixin

from ocr.utils.image_utils import load_pil_image
from ocr.core.kie_validation import KIEDataItem

logger = logging.getLogger(__name__)

class KIEDataset(Dataset):
    """
    Dataset for Key Information Extraction (KIE) using models like LayoutLMv3/LiLT.
    Loads data from Parquet files following KIEStorageItem schema.
    """

    def __init__(
        self,
        parquet_file: Union[str, Path],
        processor: Optional[ProcessorMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        image_dir: Optional[Union[str, Path]] = None,
        max_length: int = 512,
        pad_to_max_length: bool = True,
        label_list: Optional[List[str]] = None,
    ):
        """
        Args:
            parquet_file: Path to input parquet file.
            processor: HuggingFace Processor (e.g. LayoutLMv3Processor).
            tokenizer: HuggingFace Tokenizer (if processor is not used, e.g. LiLT).
            image_dir: Root directory for images (if paths in parquet are relative).
            max_length: Max sequence length.
            pad_to_max_length: Whether to pad to max_length.
            label_list: List of all possible labels (for mapping to IDs).
        """
        self.parquet_path = Path(parquet_file)
        # Validation of input args
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError(f"max_length must be positive int, got {max_length}")

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        try:
            self.df = pd.read_parquet(self.parquet_path)
            logger.info(f"Loaded {len(self.df)} samples from {self.parquet_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet file {self.parquet_path}: {e}")

        self.processor = processor
        self.tokenizer = tokenizer
        self.image_dir = Path(image_dir) if image_dir else None
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.label_list = label_list

        if self.label_list:
            if not isinstance(self.label_list, list):
                raise TypeError("label_list must be a list of strings")
            self.label2id = {label: i for i, label in enumerate(self.label_list)}
            self.id2label = {i: label for i, label in enumerate(self.label_list)}
        else:
            self.label2id = None
            self.id2label = None

        # Basic validation
        required_cols = ["image_path", "texts", "polygons"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Returns a dictionary validated by KIEDataItem.
        """
        try:
            row = self.df.iloc[idx]

            # Load Image
            image_path = row["image_path"]
            if self.image_dir and not Path(image_path).is_absolute():
                full_image_path = self.image_dir / image_path
            else:
                full_image_path = Path(image_path)

            # Use safe image loading if possible, or PIL open
            # KIE models usually need RGB
            try:
                if not full_image_path.exists():
                     raise FileNotFoundError(f"Image not found: {full_image_path}")
                image = Image.open(full_image_path).convert("RGB")
            except Exception as e:
                logger.error(f"Failed to load image {full_image_path}: {e}")
                raise e

            # Get Text, Boxes, Labels
            words = row["texts"]
            boxes = row["polygons"]

            # Check for numpy array from parquet
            if isinstance(words, np.ndarray):
                words = words.tolist()

            # Ensure words is list of str
            words = [str(w) for w in words]

            # Convert polygons to bounding boxes [x1, y1, x2, y2]
            formatted_boxes = []
            width, height = image.size

            for poly in boxes:
                # Handle possible numpy array or list
                if isinstance(poly, (np.ndarray, list)):
                    try:
                        # Ensure we have a clean numpy array of floats
                        # If poly is a list of arrays (or object array), vstack or strict conversion
                        if isinstance(poly, np.ndarray) and poly.dtype == object:
                            pts = np.vstack(poly).astype(np.float32)
                        else:
                            pts = np.array(poly, dtype=np.float32)

                        pts = pts.reshape(-1, 2)

                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)

                        # Denormalize/Scale to 0-1000
                        # LayoutLMv3 expects boxes in [0, 1000]

                        # Check if already normalized (0-1)
                        if x_max <= 1.05 and y_max <= 1.05:
                            # Scale 0-1 to 0-1000
                            x_min = int(x_min * 1000)
                            x_max = int(x_max * 1000)
                            y_min = int(y_min * 1000)
                            y_max = int(y_max * 1000)
                        else:
                            # Assume pixels or already 0-1000?
                            # If pixels, we need to normalize to 1000 using image size
                            # But if they are just big integers...
                            # Safest to normalize using width/height if available
                            if width > 0 and height > 0:
                                x_min = int((x_min / width) * 1000)
                                x_max = int((x_max / width) * 1000)
                                y_min = int((y_min / height) * 1000)
                                y_max = int((y_max / height) * 1000)
                            else:
                                # Fallback if no image size (shouldn't happen)
                                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

                        # Clamp to [0, 1000] strictly
                        x_min = max(0, min(1000, x_min))
                        x_max = max(0, min(1000, x_max))
                        y_min = max(0, min(1000, y_min))
                        y_max = max(0, min(1000, y_max))

                        formatted_boxes.append([x_min, y_min, x_max, y_max])
                    except Exception as e:
                        logger.warning(f"Error processing polygon at idx {idx}: {e}")
                        formatted_boxes.append([0, 0, 0, 0])
                else:
                    # Fallback or error
                    logger.warning(f"Invalid polygon format at idx {idx}")
                    formatted_boxes.append([0, 0, 0, 0])

            # Labels - try standard column names
            labels = row.get("labels") if "labels" in row else row.get("kie_labels", None)

            label_ids = None
            if labels is not None and self.label2id:
                if isinstance(labels, np.ndarray):
                    labels = labels.tolist()
                label_ids = [self.label2id.get(l, 0) for l in labels]

            # Processing
            encoding = {}
            if self.processor:
                # LayoutLMv3 Processor
                encoding = self.processor(
                    images=image,
                    text=words,
                    boxes=formatted_boxes,
                    word_labels=label_ids,
                    return_tensors="pt",
                    padding="max_length" if self.pad_to_max_length else False,
                    truncation=True,
                    max_length=self.max_length
                )

                # Squeeze batch dimension
                for k, v in encoding.items():
                    encoding[k] = v.squeeze(0)

            elif self.tokenizer:
                # LiLT
                width, height = image.size
                norm_boxes = []
                for box in formatted_boxes:
                    norm_boxes.append([
                        int(1000 * box[0] / max(1, width)),
                        int(1000 * box[1] / max(1, height)),
                        int(1000 * box[2] / max(1, width)),
                        int(1000 * box[3] / max(1, height))
                    ])

                # If label_ids provided but not supported by tokenizer direct call (usually tokenizer is text only)
                # We need to manually align labels or pass to tokenizer if it supports it (mostly for word classification)
                # LiltForTokenClassification expects labels aligned with tokens.
                # Tokenizer usually handles label alignment if `label_ids` passed to `text_target`? No.
                # Standard practice: tokenize, then map word labels to token labels.

                logger.warning("LiLT tokenizer label alignment not fully implemented in this basic block - assuming 1-1 word-token or external handling if labels provided.")
                # For now, simplistic

                encoding = self.tokenizer(
                    text=words,
                    boxes=norm_boxes,
                    return_tensors="pt",
                    padding="max_length" if self.pad_to_max_length else False,
                    truncation=True,
                    max_length=self.max_length
                )
                 # Squeeze
                for k, v in encoding.items():
                    encoding[k] = v.squeeze(0)

                # Fixme: Manually handle labels for LiLT if needed here or return word_labels
                if label_ids:
                    # Very rough approximation for now to match interface
                    # Ideally we align labels with subwords
                    logger.warning("LiLT label alignment skipped (TODO needs strict implementation)")
                    # Placeholder to prevent crash
                    encoding["labels"] = torch.zeros(self.max_length, dtype=torch.long)

            else:
                 raise ValueError("Neither processor nor tokenizer provided")

            # Validate using KIEDataItem
            data_item = KIEDataItem(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                bbox=encoding["bbox"],
                pixel_values=encoding.get("pixel_values"),
                labels=encoding.get("labels"),
                image_path=str(full_image_path)
            )

            # Return dict with explicit tensors to avoid model_dump serialization risks
            return {
                "input_ids": data_item.input_ids,
                "attention_mask": data_item.attention_mask,
                "bbox": data_item.bbox,
                "pixel_values": data_item.pixel_values,
                "labels": data_item.labels,
                "image_path": data_item.image_path
            }

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {e}")
            raise e
