#!/usr/bin/env python3
"""
Hugging Face Model Inference Script

Loads a model from Hugging Face and performs inference on sample images.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from omegaconf import OmegaConf
from PIL import Image

# Import from project
from ocr.core.inference.model_loader import instantiate_model, load_state_dict, register_safe_globals
from ocr.core.utils.config_utils import load_config


def visualize_detection(image_path: Path, outputs: dict, save_path: Path = None):
    """Visualize text detection results on the image."""
    # Load original image
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    # Get binary map
    binary_map = outputs["binary_map"].squeeze(0).squeeze(0).cpu().numpy()  # (640, 640)

    # Resize to original image size
    h, w = img_array.shape[:2]
    binary_resized = cv2.resize(binary_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # Find contours
    binary_uint8 = (binary_resized * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours to reduce file size
    min_area = 100  # Minimum area in pixels
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw contours on image
    img_with_contours = img_array.copy()
    cv2.drawContours(img_with_contours, filtered_contours, -1, (0, 255, 0), 2)  # Green contours

    # Save or return
    if save_path:
        # Change extension to jpg for smaller size
        save_path = save_path.with_suffix(".jpg")
        Image.fromarray(img_with_contours).save(save_path, "JPEG", quality=90)
        print(f"Visualization saved to {save_path}")
    else:
        # Display (but since script, just print)
        print("Visualization created (not saved)")

    return img_with_contours


def load_model(model_name: str):
    """Load model from HF using project's pipeline."""
    print(f"Loading model: {model_name}")

    # List available files
    try:
        files = list_repo_files(model_name)
        print(f"Available files in HF repo: {files}")
    except Exception as e:
        print(f"Error listing files: {e}")
        files = []

    # Load config from HF if available
    config = None
    config_files = [f for f in files if f.endswith(".config.json")]
    if config_files:
        config_file = config_files[0]  # Take the first one
        try:
            config_path = hf_hub_download(model_name, config_file)
            with open(config_path) as f:
                config_data = json.load(f)
            print(f"Loaded config from {config_file}")
            config = OmegaConf.create(config_data)
        except Exception as e:
            print(f"Failed to load config: {e}")

    if config is None:
        # Fallback to default
        config_path = "configs/_base/model.yaml"
        config = load_config(config_path)

    model = instantiate_model(config.model)

    # Possible model file names
    possible_files = ["model.safetensors", "pytorch_model.bin", "model.bin", "checkpoint.pth", "model.ckpt", "epoch-18_step-001957.ckpt"]

    model_file = None
    for filename in possible_files:
        if filename in files:
            try:
                model_file = hf_hub_download(model_name, filename)
                print(f"Downloaded {filename}")
                break
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue

    if model_file is None:
        raise ValueError(f"No suitable model file found in HF repo. Available files: {files}")

    # Load checkpoint
    register_safe_globals()
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load into model
    load_state_dict(model, {"state_dict": state_dict})
    model.eval()

    return model


def run_inference(model, image_path: Path):
    """Run inference on a single image."""
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # Simple preprocessing: resize to 640x640, normalize
    image = image.resize((640, 640))
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 640, 640)

    with torch.no_grad():
        outputs = model(img_tensor, return_loss=False)

    print(f"Inference results shape: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

    # Visualize and save
    viz_path = image_path.parent / f"{image_path.stem}_detection_overlay.jpg"
    visualize_detection(image_path, outputs, viz_path)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run HF model inference on images")
    parser.add_argument("--model", default="wchoi189/receipt-text-detection_kr-pan_resnet18", help="HF model name")
    parser.add_argument("--image", type=Path, help="Path to image file")
    parser.add_argument("--samples-dir", type=Path, default="data/samples_for_ai/original", help="Directory with sample images")

    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        run_inference(model, args.image)
    else:
        for img_path in Path(args.samples_dir).glob("*.webp"):
            run_inference(model, img_path)


if __name__ == "__main__":
    main()
