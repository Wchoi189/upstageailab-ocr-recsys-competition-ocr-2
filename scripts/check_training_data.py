#!/usr/bin/env python3
"""Quick check of training data validity."""

import json
import numpy as np
from pathlib import Path
from PIL import Image

# Paths
data_dir = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets")
train_json = data_dir / "jsons" / "train.json"
train_images = data_dir / "images" / "train"

print("=" * 80)
print("Training Data Validation Check")
print("=" * 80)

# Load annotations
print(f"\nLoading annotations from: {train_json}")
with open(train_json, 'r') as f:
    data = json.load(f)

print(f"Total annotations: {len(data)}")

# Check first 10 samples
print("\nChecking first 10 samples:")
for i, (key, value) in enumerate(list(data.items())[:10]):
    img_path = train_images / key

    # Check image exists
    if not img_path.exists():
        print(f"  {i+1}. {key}: ✗ IMAGE NOT FOUND")
        continue

    # Load image
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
    except Exception as e:
        print(f"  {i+1}. {key}: ✗ ERROR LOADING IMAGE: {e}")
        continue

    # Check annotation
    words = value.get('words', [])
    if not words:
        print(f"  {i+1}. {key}: ✗ NO WORDS (empty annotation)")
        continue

    # Check polygons
    valid_polygons = 0
    total_polygons = len(words)

    for word in words:
        points = word.get('points', [])
        if len(points) >= 3:  # At least 3 points for a valid polygon
            # Check if points are within image bounds
            points_array = np.array(points)
            if (points_array[:, 0] >= 0).all() and (points_array[:, 0] <= img_array.shape[1]).all() and \
               (points_array[:, 1] >= 0).all() and (points_array[:, 1] <= img_array.shape[0]).all():
                valid_polygons += 1

    if valid_polygons == 0:
        print(f"  {i+1}. {key}: ✗ NO VALID POLYGONS ({total_polygons} total, all invalid/out-of-bounds)")
    elif valid_polygons < total_polygons:
        print(f"  {i+1}. {key}: ⚠  PARTIAL ({valid_polygons}/{total_polygons} valid polygons)")
    else:
        print(f"  {i+1}. {key}: ✓ OK ({valid_polygons} valid polygons, image {img_array.shape})")

# Overall statistics
print("\n" + "=" * 80)
print("Overall Statistics:")
print("=" * 80)

total_samples = len(data)
samples_with_no_words = 0
samples_with_no_valid_polygons = 0
total_words = 0
total_valid_polygons = 0

for key, value in data.items():
    words = value.get('words', [])
    if not words:
        samples_with_no_words += 1
        continue

    total_words += len(words)

    img_path = train_images / key
    if not img_path.exists():
        continue

    try:
        img = Image.open(img_path)
        img_array = np.array(img)
    except:
        continue

    valid_polygons_in_sample = 0
    for word in words:
        points = word.get('points', [])
        if len(points) >= 3:
            points_array = np.array(points)
            if (points_array[:, 0] >= 0).all() and (points_array[:, 0] <= img_array.shape[1]).all() and \
               (points_array[:, 1] >= 0).all() and (points_array[:, 1] <= img_array.shape[0]).all():
                valid_polygons_in_sample += 1

    total_valid_polygons += valid_polygons_in_sample

    if valid_polygons_in_sample == 0:
        samples_with_no_valid_polygons += 1

print(f"Total samples: {total_samples}")
print(f"Samples with NO words: {samples_with_no_words} ({100*samples_with_no_words/total_samples:.1f}%)")
print(f"Samples with NO valid polygons: {samples_with_no_valid_polygons} ({100*samples_with_no_valid_polygons/total_samples:.1f}%)")
print(f"Total words/polygons: {total_words}")
print(f"Total VALID polygons: {total_valid_polygons}")
print(f"Average valid polygons per sample: {total_valid_polygons/total_samples:.2f}")

print("\n" + "=" * 80)
if samples_with_no_valid_polygons > total_samples * 0.1:  # More than 10% invalid
    print("✗ CRITICAL: More than 10% of samples have no valid polygons!")
    print("  This will cause training failures.")
    print("  Action required: Re-run polygon correction script or revert to original data.")
elif samples_with_no_valid_polygons > 0:
    print(f"⚠  WARNING: {samples_with_no_valid_polygons} samples have no valid polygons")
    print("  Training may encounter issues with these batches.")
else:
    print("✓ All samples have valid polygons")
print("=" * 80)
