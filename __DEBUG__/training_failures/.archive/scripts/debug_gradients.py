"""Debug script to check gradient flow and parameter training status."""
import torch
from hydra import initialize, compose

# Load config
with initialize(config_path="../configs", version_base=None):
    cfg = compose(config_name="main", overrides=["experiment=rec_baseline_v1", "+trainer=debug_safe"])

# Build model
from ocr.pipelines.orchestrator import OCRProjectOrchestrator
orch = OCRProjectOrchestrator(cfg)
orch.build_model_and_datasets()

model = orch.pl_module.model

# Check which parameters require gradients
print("\n=== Parameter Gradient Status ===")
gradient_status = {}
for name, param in model.named_parameters():
    requires_grad = param.requires_grad
    gradient_status[name] = requires_grad
    status_symbol = "✅" if requires_grad else "❌"
    print(f"{status_symbol} {name:60s} requires_grad={requires_grad}")

# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
frozen = total - trainable

print(f"\n=== Summary ===")
print(f"Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
print(f"Frozen params:    {frozen:,} ({100*frozen/total:.1f}%)")
print(f"Total params:     {total:,}")

# Check individual components
encoder_params = sum(p.numel() for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad)
decoder_params = sum(p.numel() for n, p in model.named_parameters() if 'decoder' in n and p.requires_grad)
head_params = sum(p.numel() for n, p in model.named_parameters() if 'head' in n and p.requires_grad)

print(f"\n=== Component Breakdown ===")
print(f"Encoder trainable: {encoder_params:,}")
print(f"Decoder trainable: {decoder_params:,}")
print(f"Head trainable:    {head_params:,}")

# Diagnosis
if encoder_params == 0:
    print("\n🚨 PROBLEM FOUND: Encoder is completely frozen!")
    print("   This means the model cannot learn visual features.")
    print("   Fix: Unfreeze encoder weights in config or code.")
elif trainable < total * 0.5:
    print(f"\n⚠️  WARNING: Only {100*trainable/total:.1f}% of parameters are trainable.")
    print("    This might limit learning capacity.")
else:
    print("\n✅ All parameters appear to be trainable.")
