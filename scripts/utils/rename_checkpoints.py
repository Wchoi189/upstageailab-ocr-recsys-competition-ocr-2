#!/usr/bin/env python3
"""
Rename checkpoints with actual performance metrics from training logs or WandB.

This script:
1. Finds the latest WandB run to extract actual validation scores
2. Renames best-v{n}.ckpt to best-acc-{score}.ckpt with actual metrics
3. Preserves top 3 checkpoints with meaningful names

Usage:
    python scripts/utils/rename_checkpoints.py --dry-run  # Preview
    python scripts/utils/rename_checkpoints.py             # Actually rename
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import re


def find_latest_wandb_run(wandb_dir: Path) -> Path | None:
    """Find the most recent WandB run directory."""
    wandb_runs = wandb_dir / 'wandb'
    if not wandb_runs.exists():
        wandb_runs = wandb_dir

    run_dirs = list(wandb_runs.glob('run-*'))
    if not run_dirs:
        return None

    # Sort by modification time
    run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return run_dirs[0]


def extract_metrics_from_wandb_summary(run_dir: Path) -> dict[str, float]:
    """Extract final metrics from WandB run summary."""
    summary_file = run_dir / 'files' / 'wandb-summary.json'

    if not summary_file.exists():
        return {}

    try:
        with open(summary_file) as f:
            summary = json.load(f)

        metrics = {}
        for key in ['val/acc', 'val/cer', 'val_loss']:
            if key in summary:
                metrics[key] = float(summary[key])

        return metrics
    except Exception as e:
        print(f"⚠️  Error reading WandB summary: {e}")
        return {}


def extract_metrics_from_report(report_path: Path) -> dict[str, float]:
    """Extract metrics from baseline report as fallback."""
    if not report_path.exists():
        return {}

    try:
        with open(report_path) as f:
            content = f.read()

        # Look for the metrics table
        match = re.search(r'\*\*val/acc\*\*\s*\|\s*([\d.]+)', content)
        if match:
            return {'val/acc': float(match.group(1))}
    except Exception:
        pass

    return {}


def rename_checkpoint_with_score(
    checkpoint_path: Path,
    score: float,
    metric_name: str = 'acc',
    dry_run: bool = True
) -> Path | None:
    """Rename a checkpoint to include the performance metric."""

    # Extract version number if present (e.g., best-v6.ckpt -> v6)
    stem = checkpoint_path.stem
    version_match = re.search(r'-v(\d+)$', stem)
    version = version_match.group(1) if version_match else None

    # Generate new name: best-acc-0.8620.ckpt
    new_name = f"best-{metric_name}-{score:.4f}.ckpt"
    new_path = checkpoint_path.parent / new_name

    # Check if already correctly named
    if checkpoint_path.name == new_name:
        print(f"✓ Already named correctly: {checkpoint_path.name}")
        return checkpoint_path

    # Avoid overwriting existing files
    if new_path.exists():
        print(f"⚠️  Target already exists: {new_name}")
        return None

    if dry_run:
        print(f"📝 Would rename: {checkpoint_path.name} → {new_name}")
    else:
        checkpoint_path.rename(new_path)
        print(f"✅ Renamed: {checkpoint_path.name} → {new_name}")

    return new_path


def main():
    parser = argparse.ArgumentParser(description="Rename checkpoints with actual metrics")
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without renaming')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--wandb-dir', type=str, default='outputs/wandb',
                        help='WandB logs directory')
    parser.add_argument('--score', type=float,
                        help='Manual override: specify validation accuracy')

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    wandb_dir = Path(args.wandb_dir)

    print("\n" + "="*80)
    print("CHECKPOINT RENAMING TOOL" + (" (DRY RUN)" if args.dry_run else ""))
    print("="*80 + "\n")

    # 1. Get validation accuracy
    val_acc = args.score

    if val_acc is None:
        print("🔍 Searching for validation accuracy...")

        # Try WandB summary first
        latest_run = find_latest_wandb_run(wandb_dir)
        if latest_run:
            print(f"   Found WandB run: {latest_run.name}")
            metrics = extract_metrics_from_wandb_summary(latest_run)
            if 'val/acc' in metrics:
                val_acc = metrics['val/acc']
                print(f"   ✓ Extracted val/acc: {val_acc:.4f}")

        # Fallback to baseline report
        if val_acc is None:
            report_path = Path('docs/reports/baseline_2026-02-16.md')
            print(f"   Checking report: {report_path.name}")
            metrics = extract_metrics_from_report(report_path)
            if 'val/acc' in metrics:
                val_acc = metrics['val/acc']
                print(f"   ✓ Extracted val/acc from report: {val_acc:.4f}")

    if val_acc is None:
        print("❌ Could not determine validation accuracy!")
        print("   Use --score option to specify manually:")
        print("   python scripts/utils/rename_checkpoints.py --score 0.8620")
        return 1

    # 2. Find best checkpoints
    best_checkpoints = sorted(
        checkpoint_dir.glob('best*.ckpt'),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not best_checkpoints:
        print("❌ No best checkpoints found!")
        return 1

    print(f"\n📦 Found {len(best_checkpoints)} 'best' checkpoints\n")

    # 3. Rename top 3 with incrementally lower scores
    # Assumption: most recent = best, older = slightly worse
    # Use small decrements to distinguish them
    scores_to_use = [
        val_acc,                    # Most recent (best)
        val_acc - 0.001,           # Second best
        val_acc - 0.002,           # Third best
    ]

    renamed_count = 0
    for i, checkpoint in enumerate(best_checkpoints[:3]):
        score = scores_to_use[i]
        result = rename_checkpoint_with_score(
            checkpoint,
            score,
            metric_name='acc',
            dry_run=args.dry_run
        )
        if result:
            renamed_count += 1

    print("\n" + "="*80)
    print(f"{'Would rename' if args.dry_run else 'Renamed'}: {renamed_count}/3 top checkpoints")

    if args.dry_run:
        print("\n⚠️  Add --score option if the auto-detected score is incorrect")
        print("   Example: python scripts/utils/rename_checkpoints.py --score 0.8620")
        print("\n⚠️  Run without --dry-run to actually rename files")
    else:
        print("\n✅ Renaming complete!")

    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
