import pandas as pd
import argparse
from pathlib import Path

def check_missing():
    parser = argparse.ArgumentParser(description="Find missing images between input and output parquet files")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--save-missing", help="Save missing rows to this parquet file")
    args = parser.parse_args()

    # Load input
    print(f"Loading input: {args.input}")
    input_df = pd.read_parquet(args.input)
    input_paths = set(input_df['image_path'])
    print(f"Input Unique Paths: {len(input_paths)}")
    
    # Load output
    print(f"Loading output: {args.output}")
    try:
        output_df = pd.read_parquet(args.output)
        output_paths = set(output_df['image_path'])
        print(f"Output Unique Paths: {len(output_paths)}")
    except Exception as e:
        print(f"Error loading output: {e}")
        output_paths = set()
    
    # Find diff
    missing = input_paths - output_paths
    
    if missing:
        print(f"\nMISSING {len(missing)} IMAGES:")
        missing_list = sorted(list(missing))
        for p in missing_list[:10]:
            print(f" - {p}")
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more")
            
        if args.save_missing:
            print(f"\nSaving missing rows to {args.save_missing}...")
            missing_df = input_df[input_df['image_path'].isin(missing)]
            missing_df.to_parquet(args.save_missing)
            print("Done.")
            
    else:
        print("\nNo missing images! Sets match.")

if __name__ == "__main__":
    check_missing()
