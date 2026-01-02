import pandas as pd
import numpy as np
import os
import re
from difflib import SequenceMatcher
from tqdm import tqdm

def normalize_filename(path):
    return os.path.basename(str(path))

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags if present (DP outputs HTML)
    text = re.sub(r'<[^>]+>', '', text)
    # Simple normalization
    return re.sub(r'\s+', ' ', text).strip().lower()

def align_single_sample(row):
    """
    Aligns KIE labels to DP text blocks for a single sample.
    Returns aligned (texts, boxes, labels) vectors.
    """
    # DP data (Has geometry)
    dp_texts = row['texts_dp']
    dp_polygons = row['polygons']

    # KIE data (Has entity labels)
    # KIE 'texts' and 'labels' are token-level
    kie_labels = row['labels_kie'] # labels only in KIE so no suffix usually, unless DP has labels too.
    # DP has labels too! So it might be labels_kie. Let's check the merge.
    kie_tokens = row['texts_kie']

    # Check if we have KIE texts. The previous merge kept them as 'texts_kie'?
    # In the simple merge script, we dropped them. We need them!
    # Let's assume the input dataframe has suffixes _kie and _dp

    # If dp_texts or dp_polygons are empty, return empty
    if len(dp_texts) == 0:
        return [], [], []

    # Prepare output vectors (length = len(dp_texts))
    # Default label is "O" (Outside)
    # Since we are using a list of labels in config, we should use the string labels.
    aligned_labels = ["O"] * len(dp_texts)

    # Create normalized DP texts for matching
    dp_texts_norm = [normalize_text(t) for t in dp_texts]

    # Iterate through KIE entities
    # kie_tokens and kie_labels should be same length
    if isinstance(kie_tokens, np.ndarray): kie_tokens = kie_tokens.tolist()
    if isinstance(kie_labels, np.ndarray): kie_labels = kie_labels.tolist()

    # Map valid entities (ignore 'O' or 'group_0' if they are irrelevant?
    # Actually 'group_0' is in the label list, so we treat it as an entity if needed,
    # but usually we care about specific fields like 'store_name')

    for k_token, k_label in zip(kie_tokens, kie_labels):
        if k_label == 'O':
            continue

        k_norm = normalize_text(k_token)
        if not k_norm:
            continue

        # Find best matching DP block
        best_idx = -1
        best_score = 0.0

        for i, dp_norm in enumerate(dp_texts_norm):
            # Check for containment or high similarity
            if k_norm in dp_norm:
                # favor exact containment
                score = 1.0 + (len(k_norm) / len(dp_norm)) # Boost by coverage
                if score > best_score:
                    best_score = score
                    best_idx = i
            else:
                # Sequence matcher (slower)
                # Only check if lengths are comparable to avoid noise
                score = SequenceMatcher(None, k_norm, dp_norm).ratio()
                if score > 0.7 and score > best_score: # Threshold
                    best_score = score
                    best_idx = i

        if best_idx != -1:
            # Assign label to this block
            # Note: This is a simplification. A block might contain multiple entities.
            # But LayoutLMv3 usually takes segment-level labels.
            # If a block is "Starbucks Coffee", and "Starbucks" is store_name,
            # we label the whole block as store_name for now.
            # Or strict BIO tagging? The config uses simple labels, not BIO?
            # Config label_list has "store_name", not "B-store_name".
            # So we assign the label directly.

            # Conflict resolution: prioritize more specific labels over 'group_0' etc?
            current = aligned_labels[best_idx]
            if current == "O" or current == "group_0":
                aligned_labels[best_idx] = k_label

    return dp_texts, dp_polygons, aligned_labels

def process_datasets():
    splits = ['train', 'val']

    for split in splits:
        print(f"\nProcessing {split} split...")
        kie_path = f"/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_kie/{split}.parquet"
        dp_path = f"/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_dp/{split}.parquet"
        output_path = f"/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/processed/aligned/baseline_kie_dp_{split}.parquet"

        if not os.path.exists(kie_path) or not os.path.exists(dp_path):
            print(f"Skipping {split}: Files not found")
            continue

        print("Loading datasets...")
        df_kie = pd.read_parquet(kie_path)
        df_dp = pd.read_parquet(dp_path)

        df_kie['filename'] = df_kie['image_path'].apply(normalize_filename)
        df_dp['filename'] = df_dp['image_path'].apply(normalize_filename)

        # Merge to get both sets of data
        # Note: both have 'labels' too!
        merged = pd.merge(
            df_kie[['filename', 'texts', 'labels']],
            df_dp[['filename', 'texts', 'polygons', 'width', 'height', 'labels']], # DP also has labels (layout)
            on='filename',
            how='inner',
            suffixes=('_kie', '_dp')
        )

        print(f"Merged count: {len(merged)}")

        # Apply alignment
        new_rows = []
        for idx, row in tqdm(merged.iterrows(), total=len(merged), desc="Aligning"):
            texts, boxes, labels = align_single_sample(row)

            # Fix image path to use canonical absolute path data/raw/...
            # Old: data/datasets/images/train/file.jpg (relative or absolute)
            # New: /workspaces/.../data/raw/competition/baseline_text_detection/images/...

            raw_path = df_dp[df_dp['filename'] == row['filename']].iloc[0]['image_path']
            # Normalize to string
            raw_path = str(raw_path)

            # Legacy path replacement
            # Check most specific first!
            if "data/datasets/images_val_canonical" in raw_path:
                 new_path = raw_path.replace("data/datasets/images_val_canonical", "data/raw/competition/baseline_text_detection/images/val")
            elif "data/datasets/images" in raw_path:
                new_path = raw_path.replace("data/datasets/images", "data/raw/competition/baseline_text_detection/images")
            else:
                 new_path = raw_path

            # Ensure absolute if not already, assuming project root relative
            if not new_path.startswith("/"):
                 # if relative, prepend project root? No, dataset class handles relative.
                 # But we want absolute to be safe.
                 # data/raw... is relative.
                 # Let's keep it relative if it was relative, just swapped.
                 pass

            new_rows.append({
                'image_path': new_path,
                'texts': texts,
                'polygons': boxes,
                'labels': labels,
                'width': row['width'],
                'height': row['height'],
                'filename': row['filename']
            })

        result_df = pd.DataFrame(new_rows)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_parquet(output_path)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    process_datasets()
