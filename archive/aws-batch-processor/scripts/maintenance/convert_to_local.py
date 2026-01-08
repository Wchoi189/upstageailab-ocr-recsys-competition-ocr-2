import pandas as pd
from pathlib import Path

def convert_to_local(dataset, subdir):
    input_path = Path(f"data/input/{dataset}.parquet")
    output_path = Path(f"data/input/{dataset}_local.parquet")
    
    # Path to local images: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/{subdir}
    local_base = Path(f"/workspaces/upstageailab-ocr-recsys-competition-ocr-2/data/datasets/images/{subdir}").resolve()
    
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    def transform_path(s3_path):
        filename = Path(s3_path).name
        local_p = local_base / filename
        return str(local_p)
    
    # Check if we need to transform
    sample = df['image_path'].iloc[0]
    if str(sample).startswith("s3://"):
        print(f"Converting s3 paths to local: {local_base}")
        df['image_path'] = df['image_path'].apply(transform_path)
    else:
        print("Paths might already be local or different format. Sample:", sample)
        
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path)
    print("Done.")

if __name__ == "__main__":
    convert_to_local("baseline_train", "train")
    convert_to_local("baseline_val", "val")
