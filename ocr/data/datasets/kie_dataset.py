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
        max_img_size: int = 1024, # Optimized default size
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
        self.max_img_size = max_img_size

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

    def __getitem__(self, idx) -> Optional[Dict[str, Any]]:
        """
        Returns a dictionary validated by KIEDataItem. Returns None if image loading fails.
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
                     # Return None to be filtered by collate_fn
                     logger.warning(f"Image not found (skipping): {full_image_path}")
                     return None
                image = Image.open(full_image_path).convert("RGB")

                # Resize if too large to speed up processing
                if self.max_img_size and (image.width > self.max_img_size or image.height > self.max_img_size):
                    image.thumbnail((self.max_img_size, self.max_img_size))

            except Exception as e:
                logger.error(f"Failed to load image {full_image_path}: {e}")
                return None

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

            # Labels - conversion to BIO
            labels = row.get("labels") if "labels" in row else row.get("kie_labels", None)

            label_ids = None
            if labels is not None and self.label2id:
                if isinstance(labels, np.ndarray):
                    labels = labels.tolist()

                # Dynamic BIO Conversion
                # Assumes input labels are simple ("store_name") and config label_list is BIO ("B-store_name", "I-store_name", "O")
                label_ids = []
                prev_label = "O"

                # Check if label_list actually has BIO tags
                has_bio = any(l.startswith("B-") for l in self.label_list)

                for curr_label in labels:
                    if not has_bio:
                        # Fallback for simple labels
                        label_ids.append(self.label2id.get(curr_label, self.label2id.get("O", 0)))
                        continue

                    if curr_label == "O" or curr_label not in self.label2id: # Handle unknowns as O? Or strictly check?
                        # If curr_label is "group_0", it might be in list as "B-group_0"
                        # If curr_label is NOT in label2id (simple vs BIO), we need to prefix it
                        target_label = "O"

                        # Try to construct B/I tags
                        if curr_label != "O":
                            if curr_label != prev_label:
                                attempt = f"B-{curr_label}"
                            else:
                                attempt = f"I-{curr_label}"

                            if attempt in self.label2id:
                                target_label = attempt
                            else:
                                # Fallback: maybe it's just meant to be O or it's a label mismatch
                                # logger.warning(f"Label {curr_label} -> {attempt} not found in label_list. Mapping to O.")
                                target_label = "O"

                        label_ids.append(self.label2id[target_label])

                        # Update prev_label only if it was a valid entity
                        if target_label != "O":
                            # Strip B/I to store base label for continuity check
                            prev_label = curr_label
                        else:
                            prev_label = "O"
                    else:
                         # If the dataset ALREADY has BIO tags (unlikely given description), use them
                         label_ids.append(self.label2id[curr_label])
                         prev_label = curr_label # simplified

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
                # ... (LiLT logic preserved but heavily abridged for brevity if not used)
                 logger.warning("LiLT path triggered but not fully refactored for BIO.")
                 encoding = self.tokenizer(
                    text=words,
                    boxes=formatted_boxes, # Needs normalization? Logic above handles formatted_boxes as 0-1000
                    return_tensors="pt",
                    padding="max_length" if self.pad_to_max_length else False,
                    truncation=True,
                    max_length=self.max_length
                )
                 for k, v in encoding.items():
                    encoding[k] = v.squeeze(0)
                 if label_ids:
                     encoding["labels"] = torch.tensor(label_ids[:self.max_length] + [0]*(self.max_length-len(label_ids))) # Rough pad

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
            return None
