#!/usr/bin/env python3
"""
Repair checkpoint loop state (epoch, global_step) without modifying model weights.

This tool fixes checkpoints that have correct model weights but incorrect training
loop state (e.g., epoch=0, global_step=2) due to early training interruption or
corrupt checkpoint saving.

Usage:
    # Dry run (show what would be changed)
    uv run python scripts/utils/repair_checkpoint_state.py \\
        outputs/checkpoints/best-acc-0.8620.ckpt \\
        --epoch 39 --global-step 213768

    # Apply changes
    uv run python scripts/utils/repair_checkpoint_state.py \\
        outputs/checkpoints/best-acc-0.8620.ckpt \\
        --epoch 39 --global-step 213768 \\
        --apply

    # Backup is created automatically at <checkpoint>_backup_<timestamp>.ckpt
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch


def inspect_checkpoint(ckpt_path: Path) -> dict:
    """Load and inspect checkpoint state."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    info = {
        "epoch": ckpt.get("epoch", "N/A"),
        "global_step": ckpt.get("global_step", "N/A"),
        "pytorch_lightning_version": ckpt.get("pytorch-lightning_version", "N/A"),
        "has_state_dict": "state_dict" in ckpt,
        "has_optimizer_states": "optimizer_states" in ckpt,
        "has_lr_schedulers": "lr_schedulers" in ckpt,
        "has_loops": "loops" in ckpt,
        "has_callbacks": "callbacks" in ckpt,
    }

    return info


def repair_checkpoint(
    ckpt_path: Path,
    new_epoch: int,
    new_global_step: int,
    apply: bool = False,
) -> None:
    """
    Repair checkpoint loop state.

    Args:
        ckpt_path: Path to checkpoint file
        new_epoch: New epoch value
        new_global_step: New global_step value
        apply: If True, save the repaired checkpoint; if False, dry-run only
    """
    print(f"\n{'='*70}")
    print(f"Checkpoint Loop State Repair Tool")
    print(f"{'='*70}\n")

    # Inspect current state
    print("Current checkpoint state:")
    info = inspect_checkpoint(ckpt_path)
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"\nProposed changes:")
    print(f"  epoch: {ckpt.get('epoch')} → {new_epoch}")
    print(f"  global_step: {ckpt.get('global_step')} → {new_global_step}")

    if not apply:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("   Add --apply flag to save changes")
        return

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = ckpt_path.parent / f"{ckpt_path.stem}_backup_{timestamp}.ckpt"
    print(f"\nCreating backup: {backup_path}")
    shutil.copy2(ckpt_path, backup_path)

    # Update loop state
    ckpt["epoch"] = new_epoch
    ckpt["global_step"] = new_global_step

    # Update loop state in the 'loops' dict (Lightning internal state)
    if "loops" in ckpt and "fit_loop" in ckpt["loops"]:
        if "epoch_loop.state_dict" in ckpt["loops"]["fit_loop"]:
            ckpt["loops"]["fit_loop"]["epoch_loop.state_dict"]["epoch_progress"] = {
                "current": {"completed": new_epoch, "ready": new_epoch + 1, "started": new_epoch + 1, "processed": new_epoch + 1},
                "total": {"completed": new_epoch, "ready": new_epoch + 1, "started": new_epoch + 1, "processed": new_epoch + 1},
            }
            print("  ✓ Updated fit_loop epoch progress")

        if "epoch_loop.batch_progress" in ckpt["loops"]["fit_loop"]:
            # Reset batch progress (will restart from beginning of epoch)
            ckpt["loops"]["fit_loop"]["epoch_loop.batch_progress"] = {
                "current": {"completed": 0, "ready": 0, "started": 0, "processed": 0},
                "total": {"completed": 0, "ready": 0, "started": 0, "processed": 0},
            }
            print("  ✓ Reset batch progress")

    # Save repaired checkpoint
    print(f"\nSaving repaired checkpoint: {ckpt_path}")
    torch.save(ckpt, ckpt_path)

    print("\n✅ Checkpoint repair complete!")
    print(f"\nBackup location: {backup_path}")
    print(f"Repaired checkpoint: {ckpt_path}")

    # Verify
    print("\nVerifying repaired checkpoint...")
    repaired_info = inspect_checkpoint(ckpt_path)
    print(f"  New epoch: {repaired_info['epoch']}")
    print(f"  New global_step: {repaired_info['global_step']}")


def main():
    parser = argparse.ArgumentParser(
        description="Repair checkpoint loop state without modifying model weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint file to repair",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        required=True,
        help="New epoch value",
    )
    parser.add_argument(
        "--global-step",
        type=int,
        required=True,
        help="New global_step value",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default is dry-run only)",
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1

    repair_checkpoint(
        args.checkpoint,
        args.epoch,
        args.global_step,
        apply=args.apply,
    )

    return 0


if __name__ == "__main__":
    exit(main())
