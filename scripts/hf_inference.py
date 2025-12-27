#!/usr/bin/env python3
"""
Hugging Face Model Inference Script

Loads a model from Hugging Face and performs inference on sample images.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


def load_model(model_name: str):
    """Load model and processor from Hugging Face."""
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def run_inference(model, processor, image_path: Path):
    """Run inference on a single image."""
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Assuming text detection model, adjust inputs as needed
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Placeholder for output processing
    print(f"Inference results: {outputs}")
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Run HF model inference on images")
    parser.add_argument("--model", default="wchoi189/receipt-text-detection_kr-pan_resnet18", help="HF model name")
    parser.add_argument("--image", type=Path, help="Path to image file")
    parser.add_argument("--samples-dir", type=Path, default="data/samples_for_ai/original", help="Directory with sample images")

    args = parser.parse_args()

    model, processor = load_model(args.model)

    if args.image:
        run_inference(model, processor, args.image)
    else:
        # Run on all webp in samples dir
        for img_path in Path(args.samples_dir).glob("*.webp"):
            run_inference(model, processor, img_path)

if __name__ == "__main__":
    main()
