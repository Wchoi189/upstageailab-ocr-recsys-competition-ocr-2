import pandas as pd
from pathlib import Path
import shutil
import sys

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
    # Train - Use same as DP for now as it contains the pseudo-labels
    # If merged_baseline_train exists, prefer that, else use doc_parse
    src_kie_train = Path("data/output/merged_baseline_train.parquet")
    if not src_kie_train.exists():
        src_kie_train = src_train # Fallback
        
    if src_kie_train.exists():
        shutil.copy2(src_kie_train, kie_dir / "train.parquet")
        print(f"✓ Train: {src_kie_train} -> {kie_dir}/train.parquet")
    else:
        print(f"❌ Missing KIE Train: {src_kie_train}")
        
    # Val - Same as DP
    if src_val.exists():
        shutil.copy2(src_val, kie_dir / "val.parquet")
        print(f"✓ Val: {src_val} -> {kie_dir}/val.parquet")
    
    # Test - Same as DP
    if src_test.exists():
        shutil.copy2(src_test, kie_dir / "test.parquet")
        print(f"✓ Test: {src_test} -> {kie_dir}/test.parquet")

if __name__ == "__main__":
    export_datasets()
