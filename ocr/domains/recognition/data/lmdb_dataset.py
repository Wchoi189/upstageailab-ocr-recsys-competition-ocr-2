import six
import sys
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from typing import Union, Optional

from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer

class LMDBRecognitionDataset(Dataset):
    """
    Dataset interface for LMDB files used in text recognition.

    Expected LMDB structure:
    - num-samples: Total number of samples
    - image-{index}: Image binary data (1-indexed string)
    - label-{index}: Text label (1-indexed string)
    """

    def __init__(
        self,
        lmdb_path,
        tokenizer: Union[KoreanOCRTokenizer, dict],
        max_len=25,
        transform=None,
        config=None,
        **kwargs
    ):
        """
        Initialize LMDB dataset.

        Args:
            lmdb_path: Path to LMDB directory
            tokenizer: Either a KoreanOCRTokenizer instance OR a dict with keys:
                       - charset_path: Path to charset.json
                       - max_len: Maximum sequence length
                       If dict, will use cached tokenizer via get_or_create()
            max_len: Maximum sequence length (for backward compatibility)
            transform: Image transform pipeline
            config: Deprecated config parameter
        """
        self.lmdb_path = str(lmdb_path)
        self.max_len = max_len
        self.transform = transform
        self.env = None

        # Handle tokenizer: support both instance and config dict
        if isinstance(tokenizer, dict):
            # Use cached tokenizer via get_or_create()
            self.tokenizer = KoreanOCRTokenizer.get_or_create(
                charset_path=tokenizer.get("charset_path"),
                max_len=tokenizer.get("max_len", max_len)
            )
        else:
            # Use provided tokenizer instance directly
            self.tokenizer = tokenizer

        # Open temporarily to read dataset length
        env = lmdb.open(
            self.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not env:
            print(f"cannot open lmdb from {self.lmdb_path}")
            sys.exit(0)

        with env.begin(write=False) as txn:
            self.nSamples = int(txn.get(b"num-samples"))

        env.close()

    def _open_env(self):
        self.env = lmdb.open(
            self.lmdb_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index += 1

        if self.env is None:
            self._open_env()

        with self.env.begin(write=False) as txn:
            label_key = f"label-{index:09d}".encode()
            label = txn.get(label_key).decode("utf-8")

            img_key = f"image-{index:09d}".encode()
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = Image.open(buf).convert("RGB")
            except OSError:
                print(f"Corrupted image for index {index}")
                # return self[index + 1] # naive retry, better to fail or return dummy?
                # For robustness, we might want to return a dummy or fail.
                # Let's fail for now to detect data issues.
                raise

            if self.transform:
                # Transform expects an image (PIL or Tensor)
                # Our transforms in recognition usually expect PIL or Tensor
                img = self.transform(img)

            # Tokenize
            text_tokens = self.tokenizer.encode(label)

            # Return dict compatible with recognition_collate_fn
            return {
                "image": img,
                "text_tokens": text_tokens,
                "label": label
            }
