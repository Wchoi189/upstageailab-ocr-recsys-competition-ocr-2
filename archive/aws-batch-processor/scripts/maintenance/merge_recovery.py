import pandas as pd
from pathlib import Path

# Paths
main_path = Path("data/output/baseline_val_doc_parse.parquet")
recovery_path = Path("data/output/missing_recovery/baseline_val_missing_doc_parse.parquet")

if not main_path.exists() or not recovery_path.exists():
    print("One of the files is missing!")
    exit(1)

# Load
print("Loading main...")
main_df = pd.read_parquet(main_path)
print(f"Main size: {len(main_df)}")

print("Loading recovery...")
rec_df = pd.read_parquet(recovery_path)
print(f"Recovery size: {len(rec_df)}")

# Concatenate
combined_df = pd.concat([main_df, rec_df])

# Deduplicate just in case (by image_path)
combined_df = combined_df.drop_duplicates(subset=['image_path'], keep='last')
print(f"Combined size: {len(combined_df)}")

# Save
print("Saving...")
combined_df.to_parquet(main_path)
print("Done!")
