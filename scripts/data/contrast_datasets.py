import os

import pandas as pd


def normalize_filename(path):
    return os.path.basename(str(path))


def find_in_archive(filenames, archive_root):
    found = {}
    print(f"Searching for {len(filenames)} files in {archive_root}...")
    # Walk through the archive
    for root, dirs, files in os.walk(archive_root):
        for file in files:
            if file in filenames:
                found[file] = os.path.join(root, file)
    return found


def main():
    kie_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_kie/train.parquet"
    dp_path = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/export/baseline_dp/train.parquet"
    archive_root = "/workspaces/upstageailab-ocr-recsys-competition-ocr-2/aws-batch-processor/data/archive"

    # Load Datasets
    print("Loading datasets...")
    df_kie = pd.read_parquet(kie_path)
    df_dp = pd.read_parquet(dp_path)

    # Get Filenames
    kie_filenames = set(df_kie["image_path"].apply(normalize_filename))
    dp_filenames = set(df_dp["image_path"].apply(normalize_filename))

    print(f"KIE Count: {len(kie_filenames)}")
    print(f"DP Count: {len(dp_filenames)}")

    # Contrast
    missing_in_kie = dp_filenames - kie_filenames
    missing_in_dp = kie_filenames - dp_filenames

    print(f"\nMissing in KIE (present in DP): {len(missing_in_kie)}")
    print(f"Missing in DP (present in KIE): {len(missing_in_dp)}")

    # Report Missing in KIE
    if missing_in_kie:
        print("\n--- Investigating images missing from KIE ---")
        sample = list(missing_in_kie)[:5]
        print(f"Sample missing files: {sample}")

        # Check archive
        found_in_archive = find_in_archive(missing_in_kie, archive_root)
        print(f"Found {len(found_in_archive)} / {len(missing_in_kie)} in archive.")

        if found_in_archive:
            print("Sample found paths:")
            for k in list(found_in_archive.keys())[:3]:
                print(f"  {k}: {found_in_archive[k]}")

        # Save list
        with open("missing_in_kie.txt", "w") as f:
            for item in missing_in_kie:
                f.write(f"{item}\n")
        print("Saved list to missing_in_kie.txt")

    # Report Missing in DP (Should be 0 based on previous checks, but good to verify)
    if missing_in_dp:
        print("\n--- Investigating images missing from DP ---")
        print(f"Sample missing files: {list(missing_in_dp)[:5]}")


if __name__ == "__main__":
    main()
