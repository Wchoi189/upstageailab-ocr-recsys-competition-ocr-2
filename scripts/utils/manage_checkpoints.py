#!/usr/bin/env python3
"""
Checkpoint Manager - Clean up old checkpoints and rename with metrics

Usage:
    python scripts/utils/manage_checkpoints.py --list          # List all checkpoints with metadata
    python scripts/utils/manage_checkpoints.py --cleanup       # Remove old/redundant checkpoints
    python scripts/utils/manage_checkpoints.py --rename        # Rename checkpoints with metrics
    python scripts/utils/manage_checkpoints.py --recommend     # Show recommended checkpoint for next run
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import torch


def load_checkpoint_metadata(ckpt_path: Path) -> dict:
    """Load checkpoint and extract metadata."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        metadata = {
            'path': str(ckpt_path),
            'size_mb': ckpt_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(ckpt_path.stat().st_mtime),
            'epoch': ckpt.get('epoch', 'N/A'),
            'global_step': ckpt.get('global_step', 'N/A'),
        }

        # Try to get callback metadata
        callbacks = ckpt.get('callbacks', {})
        if isinstance(callbacks, dict):
            model_checkpoint = callbacks.get('ModelCheckpoint', {})
            if isinstance(model_checkpoint, dict):
                metadata['best_score'] = model_checkpoint.get('best_model_score', 'N/A')
                metadata['monitor'] = model_checkpoint.get('monitor', 'N/A')

        return metadata
    except Exception as e:
        return {
            'path': str(ckpt_path),
            'error': str(e),
            'size_mb': ckpt_path.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(ckpt_path.stat().st_mtime),
        }


def list_checkpoints(checkpoint_dir: Path):
    """List all checkpoints with metadata."""
    checkpoints = []

    for pattern in ['*.ckpt', 'best-acc-val/*.ckpt']:
        for ckpt_path in checkpoint_dir.glob(pattern):
            metadata = load_checkpoint_metadata(ckpt_path)
            checkpoints.append(metadata)

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.get('modified', datetime.min), reverse=True)

    print("\n" + "="*80)
    print("CHECKPOINT INVENTORY")
    print("="*80)

    total_size = 0
    for ckpt in checkpoints:
        print(f"\n📦 {Path(ckpt['path']).name}")
        print(f"   Size: {ckpt['size_mb']:.1f} MB")
        print(f"   Modified: {ckpt.get('modified', 'N/A')}")
        print(f"   Epoch: {ckpt.get('epoch', 'N/A')} | Step: {ckpt.get('global_step', 'N/A')}")

        if 'best_score' in ckpt and ckpt['best_score'] != 'N/A':
            print(f"   Best Score: {ckpt['best_score']}")

        if 'error' in ckpt:
            print(f"   ⚠️  Error: {ckpt['error']}")

        total_size += ckpt['size_mb']

    print("\n" + "="*80)
    print(f"Total: {len(checkpoints)} checkpoints, {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print("="*80 + "\n")


def recommend_checkpoint(checkpoint_dir: Path):
    """Recommend the best checkpoint for next training run."""
    best_checkpoints = list(checkpoint_dir.glob('best-v*.ckpt'))
    best_checkpoints.extend(checkpoint_dir.glob('best.ckpt'))

    if not best_checkpoints:
        print("❌ No 'best' checkpoints found!")
        return

    # Sort by modification time to get latest
    best_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_best = best_checkpoints[0]

    metadata = load_checkpoint_metadata(latest_best)

    print("\n" + "="*80)
    print("✅ RECOMMENDED CHECKPOINT FOR NEXT RUN")
    print("="*80)
    print(f"\nFile: {latest_best.name}")
    print(f"Path: {latest_best}")
    print(f"Size: {metadata['size_mb']:.1f} MB")
    print(f"Modified: {metadata.get('modified', 'N/A')}")
    print(f"Epoch: {metadata.get('epoch', 'N/A')}")
    print(f"Global Step: {metadata.get('global_step', 'N/A')}")

    if 'best_score' in metadata and metadata['best_score'] != 'N/A':
        print(f"Best Score: {metadata['best_score']}")

    print("\nCommand to use:")
    print(f"uv run python scripts/runners/train.py \\")
    print(f"  mode=train \\")
    print(f"  experiment=parseq_flash_plateau \\")
    print(f"  +checkpoint_path={latest_best} \\")
    print(f"  trainer.max_epochs=60 \\")
    print(f"  trainer.val_check_interval=1.0 \\")
    print(f"  train.optimizer.lr=5e-5")
    print("\n" + "="*80 + "\n")


def cleanup_checkpoints(checkpoint_dir: Path, dry_run: bool = True):
    """Remove redundant checkpoints, keeping only best performers."""
    print("\n" + "="*80)
    print("CHECKPOINT CLEANUP" + (" (DRY RUN)" if dry_run else ""))
    print("="*80 + "\n")

    # Collect all checkpoints
    all_checkpoints = []
    for pattern in ['best*.ckpt', 'last*.ckpt', 'best-acc-val/*.ckpt']:
        all_checkpoints.extend(checkpoint_dir.glob(pattern))

    # Sort by modification time
    all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Keep rules
    keep = set()

    # Keep top 3 most recent 'best' checkpoints
    best_checkpoints = [c for c in all_checkpoints if 'best' in c.stem.lower()]
    keep.update(best_checkpoints[:3])
    print(f"✓ Keeping top 3 'best' checkpoints")

    # Keep most recent 'last' checkpoint
    last_checkpoints = [c for c in all_checkpoints if 'last' in c.stem.lower()]
    if last_checkpoints:
        keep.add(last_checkpoints[0])
        print(f"✓ Keeping most recent 'last' checkpoint")

    # Keep manually renamed checkpoint (fallback)
    manual_checkpoint = checkpoint_dir / 'best-acc-val' / 'acc-0.8267.ckpt'
    if manual_checkpoint.exists():
        keep.add(manual_checkpoint)
        print(f"✓ Keeping manually renamed checkpoint (fallback)")

    # Calculate what to remove
    to_remove = set(all_checkpoints) - keep

    if not to_remove:
        print("\n✅ No checkpoints to remove!")
        return

    print(f"\n📋 Will remove {len(to_remove)} checkpoints:")
    total_freed = 0

    for ckpt in sorted(to_remove, key=lambda x: x.stat().st_mtime, reverse=True):
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        total_freed += size_mb
        print(f"   ❌ {ckpt.name} ({size_mb:.1f} MB)")

        if not dry_run:
            ckpt.unlink()

    print(f"\n💾 Total space freed: {total_freed:.1f} MB ({total_freed/1024:.2f} GB)")

    if dry_run:
        print("\n⚠️  This was a dry run. Use --cleanup --confirm to actually delete.")
    else:
        print("\n✅ Cleanup complete!")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Manage training checkpoints")
    parser.add_argument('--list', action='store_true', help='List all checkpoints')
    parser.add_argument('--recommend', action='store_true', help='Recommend best checkpoint')
    parser.add_argument('--cleanup', action='store_true', help='Clean up old checkpoints')
    parser.add_argument('--confirm', action='store_true', help='Confirm cleanup (not dry run)')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints',
                        help='Checkpoint directory (default: outputs/checkpoints)')

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return 1

    if args.list:
        list_checkpoints(checkpoint_dir)
    elif args.recommend:
        recommend_checkpoint(checkpoint_dir)
    elif args.cleanup:
        cleanup_checkpoints(checkpoint_dir, dry_run=not args.confirm)
    else:
        # Default: show recommendation
        recommend_checkpoint(checkpoint_dir)

    return 0


if __name__ == '__main__':
    exit(main())
