
import pandas as pd
import glob
import os

def normalize_filename(path):
    return os.path.basename(str(path))

def recover_data():
    split_dir = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/archive/output/splits_train"
    files = glob.glob(os.path.join(split_dir, "*.parquet"))

    print("Loading splits...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Add implicit filename column if checking is needed
            if 'image_path' in df.columns:
                df['filename'] = df['image_path'].apply(normalize_filename)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows: {len(full_df)}")

    # Deduplicate by filename
    if 'filename' in full_df.columns:
        dedup_df = full_df.drop_duplicates(subset=['filename'])
        print(f"Deduplicated count: {len(dedup_df)}")

        # Check if missing images are present
        # Load missing list if available or just check count against DP (3272)
        if len(dedup_df) >= 3272:
            print("SUCCESS: Recovered all 3272 images (or more)!")

            # Save this recovered dataset temporarily
            output_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/processed/kie/recovered_baseline_kie_train.parquet"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dedup_df.to_parquet(output_path)
            print(f"Saved recovered dataset to {output_path}")
        else:
            print(f"Still missing images. Count: {len(dedup_df)} (Target: 3272)")

    else:
        print("Could not deduplicate, 'filename' extraction failed.")

if __name__ == "__main__":
    recover_data()
