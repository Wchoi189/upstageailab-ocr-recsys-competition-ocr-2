#!/usr/bin/env python3
"""
Checkpoint Cleanup Tool

Clean up chaotic checkpoint files from previous training runs.

Features:
- Remove backup files
- Remove checkpoints with 0.0 scores (failed training)
- Remove duplicate versioned checkpoints (v1, v2, v3)
- Keep only the best checkpoints
"""

import argparse
import shutil
from pathlib import Path

import torch


def get_checkpoint_score(ckpt_path: Path, monitor: str = "val/acc") -> float | None:
    """Extract metric score from checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "callbacks" in ckpt:
            for cb_key, cb_state in ckpt["callbacks"].items():
                if "ModelCheckpoint" in cb_key and "best_model_score" in cb_state:
                    score = cb_state["best_model_score"]
                    if isinstance(score, torch.Tensor):
                        return score.item()
                    return float(score)
        return None
    except Exception:
        return None


def cleanup_checkpoints(
    checkpoint_dir: Path,
    monitor: str = "val/acc",
    keep_top_k: int = 3,
    dry_run: bool = True,
    remove_backups: bool = True,
    remove_zero_scores: bool = True,
    remove_versioned: bool = False,
):
    """
    Clean up checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        monitor: Metric being monitored
        keep_top_k: Number of best checkpoints to keep
        dry_run: If True, only show what would be done
        remove_backups: Remove backup files
        remove_zero_scores: Remove checkpoints with 0.0 scores
        remove_versioned: Remove versioned checkpoints (v1, v2, v3)
    """
    print("=" * 70)
    print("Checkpoint Cleanup Tool")
    print("=" * 70)
    print(f"\nDirectory: {checkpoint_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'APPLY CHANGES'}")
    print(f"Keep top {keep_top_k} checkpoints\n")

    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))

    to_remove = []
    to_keep = []

    for ckpt in checkpoints:
        # Get score first for all checkpoints
        score = get_checkpoint_score(ckpt, monitor)

        # Handle 'last' checkpoints
        if ckpt.name.startswith("last"):
            # Remove versioned last checkpoints (last-v1, last-v2, etc.)
            if "_v" in ckpt.stem or (score is not None and score == 0.0):
                print(f"🗑️  Remove (last, invalid): {ckpt.name}")
                to_remove.append(ckpt)
            else:
                print(f"✓ Keep (last): {ckpt.name} (score={score:.4f})")
            continue

        # Check for backup files
        if remove_backups and "backup" in ckpt.name:
            print(f"🗑️  Remove (backup): {ckpt.name}")
            to_remove.append(ckpt)
            continue

        # Get score
        score = get_checkpoint_score(ckpt, monitor)

        # Check for zero scores
        if remove_zero_scores and score is not None and score == 0.0:
            print(f"🗑️  Remove (score=0.0): {ckpt.name}")
            to_remove.append(ckpt)
            continue

        # Check for versioned checkpoints
        if remove_versioned and "_v" in ckpt.stem:
            print(f"⚠️  Review (versioned): {ckpt.name} (score={score})")
            # Don't auto-remove, add to review list
            continue

        # Add to keep list with score
        if score is not None:
            to_keep.append((ckpt, score))
        else:
            print(f"⚠️  No score found: {ckpt.name}")

    # Sort by score and keep top-k
    to_keep.sort(key=lambda x: x[1], reverse=True)

    if len(to_keep) > keep_top_k:
        print(f"\n📊 Found {len(to_keep)} valid checkpoints, keeping top {keep_top_k}:")
        for ckpt, score in to_keep[:keep_top_k]:
            print(f"  ✓ Keep: {ckpt.name} (score={score:.4f})")

        for ckpt, score in to_keep[keep_top_k:]:
            print(f"  🗑️  Remove (not top-{keep_top_k}): {ckpt.name} (score={score:.4f})")
            to_remove.append(ckpt)
    else:
        print(f"\n📊 Found {len(to_keep)} valid checkpoint(s) - keeping all")
        for ckpt, score in to_keep:
            print(f"  ✓ Keep: {ckpt.name} (score={score:.4f})")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Keep: {len(to_keep[:keep_top_k])}")
    print(f"Remove: {len(to_remove)}")

    if not dry_run and to_remove:
        print("\nDeleting files...")
        for ckpt in to_remove:
            try:
                ckpt.unlink()
                print(f"  ✓ Deleted: {ckpt.name}")

                # Also delete metadata and config files if they exist
                for suffix in [".metadata.json", ".config.json"]:
                    related_file = ckpt.with_suffix(ckpt.suffix + suffix)
                    if related_file.exists():
                        related_file.unlink()
                        print(f"  ✓ Deleted: {related_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to delete {ckpt.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up checkpoint directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--monitor",
        type=str,
        default="val/acc",
        help="Metric being monitored",
    )
    parser.add_argument(
        "--keep-top-k",
        type=int,
        default=3,
        help="Number of best checkpoints to keep",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry run (default: True)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (sets dry_run=False)",
    )
    parser.add_argument(
        "--remove-backups",
        action="store_true",
        default=True,
        help="Remove backup files",
    )
    parser.add_argument(
        "--remove-zero-scores",
        action="store_true",
        default=True,
        help="Remove checkpoints with 0.0 scores",
    )
    parser.add_argument(
        "--remove-versioned",
        action="store_true",
        default=False,
        help="Remove versioned checkpoints (v1, v2, v3) - use with caution",
    )

    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        print(f"Error: Directory not found: {args.checkpoint_dir}")
        return 1

    cleanup_checkpoints(
        args.checkpoint_dir,
        args.monitor,
        args.keep_top_k,
        dry_run=not args.apply,
        remove_backups=args.remove_backups,
        remove_zero_scores=args.remove_zero_scores,
        remove_versioned=args.remove_versioned,
    )
    return 0


if __name__ == "__main__":
    exit(main())
