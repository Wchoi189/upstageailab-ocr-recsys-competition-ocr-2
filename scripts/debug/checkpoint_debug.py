#!/usr/bin/env python3
"""Debug script to analyze checkpoint scoring issues."""

import torch
from pathlib import Path


def analyze_checkpoint(ckpt_path: Path):
    """Analyze a single checkpoint file."""
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"\n{'='*70}")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"{'='*70}")
    print(f"Epoch: {data.get('epoch')}")
    print(f"Global step: {data.get('global_step')}")
    
    if 'callbacks' in data:
        for cb_key, cb_state in data['callbacks'].items():
            if 'ModelCheckpoint' in cb_key:
                print(f"\nModelCheckpoint state:")
                print(f"  monitor: {cb_state.get('monitor')}")
                print(f"  mode: {cb_state.get('mode')}")
                print(f"  best_model_score: {cb_state.get('best_model_score')}")
                print(f"  current_score: {cb_state.get('current_score')}")
                print(f"  best_k_models:")
                for path, score in cb_state.get('best_k_models', {}).items():
                    print(f"    {Path(path).name}: {score}")
    
    if 'loops' in data:
        print(f"\nLoop state:")
        for loop_name, loop_state in data['loops'].items():
            if hasattr(loop_state, 'state_dict'):
                print(f"  {loop_name}: {loop_state.state_dict()}")


def main():
    checkpoint_dir = Path("outputs/checkpoints")
    
    # Analyze checkpoints without metric in name
    checkpoints = [
        checkpoint_dir / "best-v3.ckpt",
        checkpoint_dir / "best.ckpt",
        checkpoint_dir / "best-v1.ckpt",
        checkpoint_dir / "best-v2.ckpt",
    ]
    
    for ckpt_path in checkpoints:
        if ckpt_path.exists():
            analyze_checkpoint(ckpt_path)
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    print("""
The issue with best-v3.ckpt having score=0.0:

1. The checkpoint was saved at epoch 0, global step 1
2. The val/acc metric was 0.0 at the time of saving
3. This indicates either:
   a) Validation ran but all predictions were wrong (0% accuracy)
   b) Validation didn't run properly and the metric defaulted to 0.0
   c) The checkpoint was saved before validation completed

Root cause: PyTorch Lightning's ModelCheckpoint saves whatever value
is in trainer.callback_metrics for the monitored metric. If validation
fails or produces 0.0 accuracy, that value is saved as best_model_score.

Recommendation: Add validation to skip checkpoints with score=0.0 or
investigate why validation produced 0.0 accuracy at epoch 0.
""")


if __name__ == "__main__":
    main()
