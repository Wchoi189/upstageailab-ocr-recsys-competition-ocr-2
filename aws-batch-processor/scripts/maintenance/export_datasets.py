import pandas as pd
from pathlib import Path
import shutil
import sys

def strip_polygons_for_kie(df: pd.DataFrame) -> pd.DataFrame:
    """Remove polygon columns from KIE dataset.

    KIE datasets should contain only key-value pairs, not layout/polygon information.
    This function creates a copy of the dataframe with polygons removed.
    """
    df_kie = df.copy()

    # Remove polygon column if it exists
    if "polygons" in df_kie.columns:
        df_kie = df_kie.drop(columns=["polygons"])
        print(f"  ⚠ Removed 'polygons' column from KIE data ({len(df_kie)} rows)")

    # Validate that polygons are removed
    if "polygons" in df_kie.columns:
        raise ValueError("Failed to remove polygons column from KIE data")

    return df_kie


def export_datasets():
    export_base = Path("data/export")
    if export_base.exists():
        shutil.rmtree(export_base)
    export_base.mkdir(parents=True)

    # 1. Baseline DP
    dp_dir = export_base / "baseline_dp"
    dp_dir.mkdir()

    print("Exporting Baseline DP...")
    # Train
    src_train = Path("data/output/baseline_train_doc_parse.parquet")
    if src_train.exists():
        shutil.copy2(src_train, dp_dir / "train.parquet")
        print(f"✓ Train: {src_train} -> {dp_dir}/train.parquet")
    else:
        print(f"❌ Missing Train: {src_train}")

    # Val
    src_val = Path("data/output/baseline_val_doc_parse.parquet")
    if src_val.exists():
        shutil.copy2(src_val, dp_dir / "val.parquet")
        print(f"✓ Val: {src_val} -> {dp_dir}/val.parquet")
    else:
        print(f"❌ Missing Val: {src_val}")

    # Test (Input only)
    src_test = Path("data/input/baseline_test.parquet")
    if src_test.exists():
        shutil.copy2(src_test, dp_dir / "test.parquet")
        print(f"✓ Test: {src_test} -> {dp_dir}/test.parquet")
    else:
        print(f"❌ Missing Test: {src_test}")

    # 2. Baseline KIE
    kie_dir = export_base / "baseline_kie"
    kie_dir.mkdir()

    print("\nExporting Baseline KIE...")
    print("  Note: KIE data will have polygons removed (KIE = key-value pairs only)")

    # Train - Use same as DP for now as it contains the pseudo-labels
    # If merged_baseline_train exists, prefer that, else use doc_parse
    src_kie_train = Path("data/output/merged_baseline_train.parquet")
    if not src_kie_train.exists():
        src_kie_train = src_train # Fallback

    if src_kie_train.exists():
        # Read, strip polygons, and save
        df_kie_train = pd.read_parquet(src_kie_train)
        df_kie_train = strip_polygons_for_kie(df_kie_train)
        df_kie_train.to_parquet(kie_dir / "train.parquet", index=False)
        print(f"✓ Train: {src_kie_train} -> {kie_dir}/train.parquet (polygons removed)")
    else:
        print(f"❌ Missing KIE Train: {src_kie_train}")

    # Val - Strip polygons from DP data
    if src_val.exists():
        df_kie_val = pd.read_parquet(src_val)
        df_kie_val = strip_polygons_for_kie(df_kie_val)
        df_kie_val.to_parquet(kie_dir / "val.parquet", index=False)
        print(f"✓ Val: {src_val} -> {kie_dir}/val.parquet (polygons removed)")

    # Test - Strip polygons from DP data
    if src_test.exists():
        df_kie_test = pd.read_parquet(src_test)
        df_kie_test = strip_polygons_for_kie(df_kie_test)
        df_kie_test.to_parquet(kie_dir / "test.parquet", index=False)
        print(f"✓ Test: {src_test} -> {kie_dir}/test.parquet (polygons removed)")

if __name__ == "__main__":
    export_datasets()
