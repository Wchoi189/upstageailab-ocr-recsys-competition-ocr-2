import pandas as pd
from pathlib import Path

def convert_to_canonical(dataset, subdir):
    # Read the original input or the local one - doesn't matter much as long as we get the indices/metadata
    # Let's read the local one we just made to be sure of the format
    input_path = Path(f"data/input/{dataset}_local.parquet")
    if not input_path.exists():
         input_path = Path(f"data/input/{dataset}.parquet")
         
    output_path = Path(f"data/input/{dataset}_canonical.parquet")
    
    # Path to canonical images: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images_val_canonical
    canonical_base = Path(f"/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/{subdir}").resolve()
    
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    def transform_path(current_path):
        filename = Path(current_path).name
        # Construct new path
        canonical_p = canonical_base / filename
        return str(canonical_p)
    
    # Apply transformation
    print(f"Converting paths to canonical: {canonical_base}")
    df['image_path'] = df['image_path'].apply(transform_path)
    
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    convert_to_canonical("baseline_val", "images_val_canonical")
