"""Quick loss diagnostic - check if loss is decreasing."""
import torch
import sys
import os
sys.path.insert(0, '/workspaces')

print("=" * 60)
print("QUICK TRAINING LOSS CHECK")
print("=" * 60)

# Find latest training output
outputs_dir = "/workspaces/outputs"
if os.path.exists(outputs_dir):
    runs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    if runs:
        # Get most recent
        latest = sorted(runs)[-1]
        log_file = os.path.join(outputs_dir, latest, "train.log")

        if os.path.exists(log_file):
            print(f"\nReading: {log_file}\n")

            # Extract loss values
            losses = []
            with open(log_file) as f:
                for line in f:
                    if "train/loss" in line:
                        try:
                            # Parse loss value from log
                            parts = line.split("train/loss")
                            if len(parts) > 1:
                                value_str = parts[1].strip().split()[0]
                                loss_val = float(value_str)
                                losses.append(loss_val)
                        except:
                            pass

            if losses:
                print(f"Found {len(losses)} loss values")
                print(f"\nFirst 10: {losses[:10]}")
                print(f"Last  10: {losses[-10:]}")

                print(f"\n=== Statistics ===")
                print(f"First loss: {losses[0]:.4f}")
                print(f"Last loss:  {losses[-1]:.4f}")
                print(f"Min loss:   {min(losses):.4f}")
                print(f"Max loss:   {max(losses):.4f}")

                # Check if changing
                if abs(losses[-1] - losses[0]) < 0.01:
                    print(f"\n❌ PROBLEM: Loss is FROZEN (change < 0.01)")
                    print(f"   This confirms optimizer is not updating weights")
                elif losses[-1] < losses[0]:
                    print(f"\n✅ Loss is decreasing ({losses[0]:.4f} → {losses[-1]:.4f})")
                else:
                    print(f"\n⚠️  Loss is increasing ({losses[0]:.4f} → {losses[-1]:.4f})")
            else:
                print("❌ No loss values found in log")
        else:
            print(f"❌ No train.log found at {log_file}")
    else:
        print("❌ No training runs found in outputs/")
else:
    print("❌ outputs/ directory not found")

# Also check if we can find global step info
print("\n" + "=" * 60)
print("Checking config dimensionality mismatch...")
print("=" * 60)

print("\nPARSeq config shows:")
print("  encoder output_indices: 4")
print("  decoder in_channels: 512")
print("  decoder d_model: 512")

print("\nBut ResNet18 at index 4 outputs 512 channels")
print("The decoder in_channels should match encoder output!")

print("\n⚠️  Potential config issue:")
print("   If encoder outputs different dimension than decoder.in_channels,")
print("   the model architecture has incompatible components!")
