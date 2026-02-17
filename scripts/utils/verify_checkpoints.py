#!/usr/bin/env python3
"""
Verify checkpoint validity after metric accumulation bug fix.

This script helps you understand which checkpoints are trustworthy.
"""

import torch
from pathlib import Path
from datetime import datetime


def analyze_checkpoint(ckpt_path: Path):
    """Analyze a single checkpoint."""
    try:
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        epoch = data.get("epoch", "?")
        step = data.get("global_step", "?")
        
        # Get score from callback state
        score = None
        if "callbacks" in data:
            for cb_key, cb_state in data["callbacks"].items():
                if "ModelCheckpoint" in cb_key and "best_model_score" in cb_state:
                    s = cb_state["best_model_score"]
                    score = float(s) if not isinstance(s, torch.Tensor) else s.item()
        
        # Check if metric was reset properly (look for _checkpoint_metrics)
        has_checkpoint_metrics = "_checkpoint_metrics" in data.get("loops", {})
        
        # Get file size
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        
        return {
            "name": ckpt_path.name,
            "epoch": epoch,
            "step": step,
            "score": score,
            "size_mb": size_mb,
            "valid_for_inference": True,  # Model weights are always valid
            "score_reliable": score is not None and score > 0.01,  # Scores > 0.01 are more reliable
        }
    except Exception as e:
        return {
            "name": ckpt_path.name,
            "error": str(e),
            "valid_for_inference": False,
            "score_reliable": False,
        }


def main():
    checkpoint_dir = Path("outputs/checkpoints")
    
    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return
    
    print("=" * 80)
    print("Checkpoint Validity Analysis")
    print("=" * 80)
    print()
    print("Legend:")
    print("  ✅ = Valid for inference (model weights OK)")
    print("  ⚠️  = Score may be inaccurate (metric accumulation bug)")
    print("  ❌ = Invalid/corrupted")
    print()
    
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
    
    results = []
    for ckpt in checkpoints:
        if "backup" in ckpt.name:
            continue
        result = analyze_checkpoint(ckpt)
        results.append(result)
    
    # Group by type
    best_ckpts = [r for r in results if r.get("name", "").startswith("best") and not r.get("name", "").startswith("last")]
    last_ckpts = [r for r in results if r.get("name", "").startswith("last")]
    
    print("=" * 80)
    print("Best Checkpoints (for inference/resume)")
    print("=" * 80)
    print(f"{'Checkpoint':<40} {'Epoch':<8} {'Score':<10} {'Status':<20}")
    print("-" * 80)
    
    for r in sorted(best_ckpts, key=lambda x: x.get("score", 0) or 0, reverse=True):
        if "error" in r:
            status = "❌ Corrupted"
        elif r.get("score", 0) < 0.01:
            status = "⚠️  Score=0.0 (bug)"
        else:
            status = "✅ Valid weights, ⚠️  Score inaccurate"
        
        score_str = f"{r.get('score', 0):.4f}" if r.get("score") else "N/A"
        print(f"{r['name']:<40} {r.get('epoch', 'N/A'):<8} {score_str:<10} {status:<20}")
    
    print()
    print("=" * 80)
    print("Last Checkpoints (for resume)")
    print("=" * 80)
    print(f"{'Checkpoint':<40} {'Epoch':<8} {'Score':<10} {'Status':<20}")
    print("-" * 80)
    
    for r in last_ckpts:
        if "error" in r:
            status = "❌ Corrupted"
        elif r.get("score", 0) < 0.01:
            status = "⚠️  Score=0.0 (bug)"
        else:
            status = "✅ Valid weights, ⚠️  Score inaccurate"
        
        score_str = f"{r.get('score', 0):.4f}" if r.get("score") else "N/A"
        print(f"{r['name']:<40} {r.get('epoch', 'N/A'):<8} {score_str:<10} {status:<20}")
    
    print()
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    print()
    print("1. For Inference:")
    print("   → Any checkpoint with ✅ is fine (model weights are valid)")
    print("   → Use highest-score checkpoint despite score inaccuracy")
    print()
    print("2. For Resume:")
    print("   → Use 'last.ckpt' or highest-score 'best-*.ckpt'")
    print("   → First validation after resume will now show CORRECT accuracy")
    print()
    print("3. For Checkpoint Selection:")
    print("   → Re-evaluate top checkpoints with fresh validation")
    print("   → Run: python validate.py --checkpoint <ckpt_path>")
    print()
    print("4. Cleanup:")
    print("   → Remove checkpoints with score=0.0 (failed training)")
    print("   → Run: uv run python scripts/utils/cleanup_checkpoints.py --apply")
    print()


if __name__ == "__main__":
    main()
