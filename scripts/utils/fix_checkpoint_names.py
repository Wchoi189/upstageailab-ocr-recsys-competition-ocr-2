#!/usr/bin/env python3
"""
Rename existing checkpoints to include their metric scores.

This script reads the metrics from checkpoint files and renames them
to follow the pattern: best-acc-0.8301.ckpt

Usage:
    # Show what would be renamed (dry run)
    uv run python scripts/utils/fix_checkpoint_names.py

    # Actually rename files
    uv run python scripts/utils/fix_checkpoint_names.py --apply
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch


def extract_metric_from_checkpoint(ckpt_path: Path, monitor: str = "val/acc") -> float | None:
    """
    Extract metric value from checkpoint callbacks state.

    Args:
        ckpt_path: Path to checkpoint file
        monitor: Metric to extract (e.g., "val/acc")

    Returns:
        Metric value or None if not found
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Try to get from callbacks state first
        if "callbacks" in ckpt:
            for cb_key, cb_state in ckpt["callbacks"].items():
                if "ModelCheckpoint" in cb_key and "best_model_score" in cb_state:
                    score = cb_state["best_model_score"]
                    if isinstance(score, torch.Tensor):
                        return score.item()
                    return float(score)

        # Fallback: try to get from saved metrics (less reliable)
        # Note: This usually doesn't contain the actual best score
        return None

    except Exception as e:
        print(f"  ⚠️  Error reading {ckpt_path.name}: {e}")
        return None


def rename_checkpoints(checkpoint_dir: Path, monitor: str = "val/acc", apply: bool = False):
    """
    Rename checkpoint files to include metric scores.

    Args:
        checkpoint_dir: Directory containing checkpoints
        monitor: Metric being monitored (e.g., "val/acc")
        apply: If True, rename files; if False, dry-run only
    """
    print("=" * 70)
    print("Checkpoint Renaming Tool")
    print("=" * 70)
    print(f"\nDirectory: {checkpoint_dir}")
    print(f"Monitor metric: {monitor}")
    print(f"Mode: {'APPLY CHANGES' if apply else 'DRY RUN (use --apply to rename)'}\n")

    # Find checkpoint files to rename
    # Pattern: best.ckpt, best-v1.ckpt, best-v2.ckpt, etc. (without metric scores)
    best_checkpoints = list(checkpoint_dir.glob("best*.ckpt"))

    # Filter out files that already have scores (contain monitor metric name)
    metric_name = monitor.split("/")[-1]  # "val/acc" -> "acc"
    to_rename = []

    for ckpt_path in best_checkpoints:
        # Skip if already has metric in filename
        if f"-{metric_name}-" in ckpt_path.name:
            print(f"⏭️  Skipping (already has metric): {ckpt_path.name}")
            continue

        # Skip 'last' checkpoint
        if ckpt_path.name.startswith("last"):
            print(f"⏭️  Skipping (last checkpoint): {ckpt_path.name}")
            continue

        to_rename.append(ckpt_path)

    if not to_rename:
        print("\n✓ No checkpoints need renaming")
        return

    print(f"\nFound {len(to_rename)} checkpoint(s) to rename:\n")

    renamed_count = 0
    failed_count = 0

    for ckpt_path in sorted(to_rename):
        print(f"Processing: {ckpt_path.name}")

        # Extract metric value
        metric_val = extract_metric_from_checkpoint(ckpt_path, monitor)

        if metric_val is None:
            print(f"  ⚠️  Could not extract metric value - skipping")
            failed_count += 1
            continue

        # Build new filename
        # Extract version if present (best-v1.ckpt -> v1)
        stem = ckpt_path.stem
        version_suffix = ""
        if "-v" in stem:
            parts = stem.rsplit("-v", 1)
            if len(parts) == 2 and parts[1].isdigit():
                version_suffix = f"_v{parts[1]}"

        new_name = f"best-{metric_name}-{metric_val:.4f}{version_suffix}.ckpt"
        new_path = ckpt_path.parent / new_name

        print(f"  Metric value: {metric_val:.4f}")
        print(f"  New name: {new_name}")

        if apply:
            # Create backup first
            backup_name = f"{ckpt_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt"
            backup_path = ckpt_path.parent / backup_name

            try:
                # Check if target already exists
                if new_path.exists():
                    print(f"  ⚠️  Target already exists: {new_name}")
                    failed_count += 1
                    continue

                # Create backup
                shutil.copy2(ckpt_path, backup_path)
                print(f"  ✓ Backup created: {backup_name}")

                # Rename
                ckpt_path.rename(new_path)
                print(f"  ✓ Renamed successfully")
                renamed_count += 1

            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed_count += 1
                # Restore backup if rename failed
                if backup_path.exists() and not ckpt_path.exists():
                    backup_path.rename(ckpt_path)
                    print(f"  ✓ Restored from backup")
        else:
            print(f"  → Would rename to: {new_name}")
            renamed_count += 1

        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    if apply:
        print(f"✓ Renamed: {renamed_count}")
        print(f"✗ Failed: {failed_count}")
        if renamed_count > 0:
            print(f"\n✓ Backups created with '_backup_TIMESTAMP' suffix")
    else:
        print(f"Would rename: {renamed_count}")
        print(f"Would skip: {failed_count}")
        print(f"\nAdd --apply flag to perform the renaming")


def main():
    parser = argparse.ArgumentParser(
        description="Rename checkpoint files to include metric scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory containing checkpoints (default: outputs/checkpoints)",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val/acc",
        help="Metric being monitored (default: val/acc)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run only)",
    )

    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        print(f"Error: Directory not found: {args.checkpoint_dir}")
        return 1

    rename_checkpoints(args.checkpoint_dir, args.monitor, args.apply)
    return 0


if __name__ == "__main__":
    exit(main())
