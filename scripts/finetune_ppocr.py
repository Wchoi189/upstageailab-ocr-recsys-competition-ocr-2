"""Fine-tune PP-OCRv5 on receipt dataset.

This script downloads and prepares the mychen76/invoices-and-receipts_ocr_v1
dataset and fine-tunes PP-OCRv5 recognition model on receipt images.

Usage:
    uv run python scripts/finetune_ppocr.py --output-dir data/receipts --epochs 50

Requirements:
    - datasets: HuggingFace datasets library
    - paddleocr: PaddleOCR library for fine-tuning

Note:
    The mychen76/invoices-and-receipts_ocr_v1 dataset is for research use.
    Verify licensing compliance before production deployment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def prepare_dataset(output_dir: Path) -> Path:
    """Download and prepare mychen76/invoices-and-receipts dataset.

    Converts HuggingFace dataset to PaddleOCR format:
    - Images saved to train/ directory
    - Annotations written to label.txt (image_path TAB text)

    Args:
        output_dir: Output directory for prepared dataset

    Returns:
        Path to training data directory
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets library not installed. Install with: uv pip install datasets"
        ) from e

    LOGGER.info("Downloading mychen76/invoices-and-receipts_ocr_v1 dataset...")
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1")

    # Convert to PaddleOCR format
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    label_file = train_dir / "label.txt"
    LOGGER.info("Preparing dataset in PaddleOCR format: %s", train_dir)

    with open(label_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset["train"]):
            # Extract text regions and annotations
            image_path = train_dir / f"{i:06d}.jpg"
            sample["image"].save(image_path)

            # Write annotations (format: image_path TAB text)
            annotations = sample.get("annotations", [])
            if annotations:
                for annotation in annotations:
                    text = annotation.get("text", "")
                    if text:
                        f.write(f"{image_path}\t{text}\n")
            else:
                # If no annotations, write empty line
                f.write(f"{image_path}\t\n")

            if (i + 1) % 100 == 0:
                LOGGER.info("Processed %d samples", i + 1)

    LOGGER.info("Dataset prepared: %d samples written", i + 1)
    return train_dir


def finetune_ppocr(
    train_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
) -> None:
    """Fine-tune PP-OCRv5 recognition model.

    Args:
        train_dir: Directory containing training data
        epochs: Number of training epochs
        batch_size: Batch size per GPU

    Note:
        This requires PaddleOCR to be installed with training dependencies.
        Install with: pip install paddlepaddle-gpu paddleocr
    """
    import subprocess

    LOGGER.info("Starting PP-OCRv5 fine-tuning...")
    LOGGER.info("Training directory: %s", train_dir)
    LOGGER.info("Epochs: %d, Batch size: %d", epochs, batch_size)

    # PaddleOCR fine-tuning command
    # Note: This assumes PP-OCRv5 config is available
    # Adjust config path based on your PaddleOCR installation
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "paddleocr.tools.train",
                "-c",
                "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml",
                "-o",
                f"Train.dataset.data_dir={train_dir}",
                "-o",
                f"Global.epoch_num={epochs}",
                "-o",
                f"Train.loader.batch_size_per_card={batch_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        LOGGER.info("Training completed successfully")
        LOGGER.info("Output: %s", result.stdout)
    except subprocess.CalledProcessError as e:
        LOGGER.error("Training failed: %s", e.stderr)
        raise
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "PaddleOCR training script not found. "
            "Ensure PaddleOCR is installed with: pip install paddleocr"
        ) from e


def main():
    """Main entry point for fine-tuning script."""
    parser = argparse.ArgumentParser(
        description="Fine-tune PP-OCRv5 on receipt dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/receipts"),
        help="Output directory for dataset (default: data/receipts)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU (default: 32)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare dataset, skip training",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Prepare dataset
    train_dir = prepare_dataset(args.output_dir)
    LOGGER.info("Dataset prepared at: %s", train_dir)

    # Fine-tune (unless prepare-only mode)
    if not args.prepare_only:
        finetune_ppocr(
            train_dir=train_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    else:
        LOGGER.info("Prepare-only mode: Skipping training")

    LOGGER.info("Done!")


if __name__ == "__main__":
    main()
