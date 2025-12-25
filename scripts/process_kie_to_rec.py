#!/usr/bin/env python3
"""
Script to process KIE datasets into PaddleOCR text recognition format.
Extracts cropped text lines from bounding boxes and creates rec_gt.txt.
"""

import os
import ast
import json
from pathlib import Path
from PIL import Image
from datasets import load_dataset

DATA_DIR = Path('/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data')
OUTPUT_DIR = DATA_DIR / 'paddleocr_rec_datasets'

def process_kie_dataset(dataset_name, output_subdir):
    """Process a KIE dataset into text recognition format."""
    output_path = OUTPUT_DIR / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / 'images'
    images_dir.mkdir(exist_ok=True)
    
    ds = load_dataset(dataset_name, split='train')
    
    gt_lines = []
    crop_count = 0
    
    for i, item in enumerate(ds):
        image = item['image']
        raw_data = item['raw_data']
        
        # Parse raw_data
        try:
            data = ast.literal_eval(raw_data)
            ocr_boxes = ast.literal_eval(data['ocr_boxes'])
            if i == 0:
                print(f"Sample ocr_boxes: {ocr_boxes[:1]}")
        except Exception as e:
            print(f"Skipping item {i}, parse error: {e}")
            continue
        
        for j, box in enumerate(ocr_boxes):
            try:
                points, (text, conf) = box
                if conf < 0.5:  # Skip low confidence
                    continue
                # Get bbox from points
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                left, top = min(xs), min(ys)
                right, bottom = max(xs), max(ys)
                
                # Crop
                crop = image.crop((left, top, right, bottom))
                crop_path = images_dir / f"{i:06d}_{j:02d}.jpg"
                crop.save(crop_path)
                
                gt_lines.append(f"{crop_path}\t{text}")
                crop_count += 1
            except:
                continue
        
        if i % 100 == 0:
            print(f"Processed {i} images, {crop_count} crops")
    
    gt_file = output_path / 'rec_gt.txt'
    with open(gt_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(gt_lines))
    
    print(f"Processed {dataset_name}: {len(gt_lines)} crops saved to {output_path}")

if __name__ == "__main__":
    process_kie_dataset('mychen76/wildreceipts_ocr_v1', 'wildreceipts_rec')
    process_kie_dataset('mychen76/receipt_cord_ocr_v2', 'cord_rec')
    # process_kie_dataset('mychen76/ds_receipts_v2_train', 'ds_receipts_rec')  # No raw_data
    process_kie_dataset('mychen76/invoices-and-receipts_ocr_v1', 'invoices_rec')