import pandas as pd
from pathlib import Path

def clean_val_output():
    path = Path("data/output/baseline_val_doc_parse.parquet")
    if not path.exists():
        print(f"File not found: {path}")
        return
        
    print(f"Reading {path}...")
    df = pd.read_csv(path) if path.suffix == '.csv' else pd.read_parquet(path)
    
    initial_count = len(df)
    print(f"Initial Rows: {initial_count}")
    
    # Check for unique images
    unique_count = df['image_path'].nunique()
    print(f"Unique Paths: {unique_count}")

    # Inspect paths
    print("Sample Paths:")
    print(df['image_path'].head().tolist())
    print(df['image_path'].tail().tolist())

    # Create a normalized filename column for deduplication check
    df['filename'] = df['image_path'].apply(lambda x: Path(x).name)
    unique_filenames = df['filename'].nunique()
    print(f"Unique Filenames: {unique_filenames}")
    
    if unique_filenames < len(df):
        print("Duplicates detected by filename. Deduplicating...")
        # Keep the last entry (likely the most recent local/canonical one)
        df = df.drop_duplicates(subset=['filename'], keep='last')
        
        # Verify count
        final_count = len(df)
        print(f"Final Rows: {final_count}")
        
        # Drop the temp filename column
        df = df.drop(columns=['filename'])
        
        # Save
        if final_count == 404:
             print("SUCCESS: Matches expected count (404).")
        
        df.to_parquet(path)
        print(f"Saved cleaned file to {path}")
    else:
        print("No filename duplicates found. Investigating why count is high...")

if __name__ == "__main__":
    clean_val_output()
