"""LMDB-based dataset for text recognition tasks."""
import io
import logging
from pathlib import Path
from typing import Any
from collections.abc import Callable

import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LMDBRecognitionDataset(Dataset):
    """
    LMDB dataset for text recognition (e.g., PARSeq training).

    LMDB format expected:
        image-{idx:09d} -> JPEG bytes
        label-{idx:09d} -> UTF-8 text
        num-samples -> total count

    Usage:
        from ocr.recognition.data.tokenizer import KoreanOCRTokenizer

        tokenizer = KoreanOCRTokenizer("ocr/data/charset.json")
        dataset = LMDBRecognitionDataset(
            lmdb_path="data/processed/aihub_lmdb_validation",
            tokenizer=tokenizer,
            transform=transforms,
        )
    """

    def __init__(
        self,
        lmdb_path: str | Path,
        tokenizer: Any,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        max_len: int = 25,
    ):
        """
        Initialize LMDB recognition dataset.

        Args:
            lmdb_path: Path to LMDB directory
            tokenizer: Tokenizer with encode() method
            transform: Optional image transform (PIL -> Tensor)
            max_len: Maximum sequence length for tokenization
        """
        self.lmdb_path = str(lmdb_path)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        # Open LMDB environment (lazy, readonly, no locking for multi-worker)
        self.env = None
        self._num_samples: int | None = None

        # Initialize to get num_samples
        self._init_env()

    def _init_env(self) -> None:
        """Initialize LMDB environment."""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        if self._num_samples is None:
            with self.env.begin() as txn:
                num_samples_raw = txn.get(b"num-samples")
                if num_samples_raw:
                    self._num_samples = int(num_samples_raw.decode())
                else:
                    # Fallback
                    self._num_samples = self.env.stat()["entries"] // 2

            logger.info("LMDBRecognitionDataset: %d samples from %s", self._num_samples, self.lmdb_path)

    def __len__(self) -> int:
        """Return number of samples."""
        if self._num_samples is None:
            self._init_env()
        return self._num_samples  # type: ignore

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample by index.

        Returns:
            dict with keys:
                - image: Tensor [C, H, W] or PIL Image if no transform
                - text_tokens: Tensor [max_len]
                - label: str (raw text)
        """
        if self.env is None:
            self._init_env()

        # LMDB uses 1-based indexing
        lmdb_idx = idx + 1

        with self.env.begin() as txn:  # type: ignore
            image_key = f"image-{lmdb_idx:09d}".encode()
            label_key = f"label-{lmdb_idx:09d}".encode()

            image_bytes = txn.get(image_key)
            label_bytes = txn.get(label_key)

        if image_bytes is None or label_bytes is None:
            raise IndexError(f"Sample {idx} (LMDB idx {lmdb_idx}) not found")

        # Decode image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)

        # Decode label
        label = label_bytes.decode("utf-8")

        # Tokenize
        text_tokens = self.tokenizer.encode(label)
        text_tokens_tensor = torch.tensor(text_tokens, dtype=torch.long)

        return {
            "image": image,
            "text_tokens": text_tokens_tensor,
            "label": label,
        }

    def __del__(self):
        """Clean up LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
