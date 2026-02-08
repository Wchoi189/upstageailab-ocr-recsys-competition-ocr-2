"""Check if optimizer is being created and configured correctly."""
import torch
from hydra import initialize, compose
import sys
sys.path.insert(0, '/workspaces')

# Load config - simplified
with initialize(config_path="../../../configs", version_base=None):
    cfg = compose(config_name="main", overrides=["experiment=rec_baseline_v1"])

# Build model
from ocr.pipelines.orchestrator import OCRProjectOrchestrator

print("Building model...")
orch = OCRProjectOrchestrator(cfg)
# Use internal method to build
from ocr.core.factory import ModelFactory, DatasetFactory
from ocr.domains.recognition.module import RecognitionPLModule
from lightning.pytorch import LightningDataModule

# Simplified build
print("Loading tokenizer...")
from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer
tokenizer = KoreanOCRTokenizer(
    charset_path="/workspaces/ocr/data/charset.json",
    max_len=25
)

print("Building model from config...")
cfg.model.vocab_size = tokenizer.vocab_size
model = ModelFactory.create(cfg.model, cfg.global_paths)

print(f"\n✓ Model created: {model.__class__.__name__}")
print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

# Create PL Module
pl_module = RecognitionPLModule(model=model, dataset={}, config=cfg)

print("\n=== Optimizer Configuration ===")
try:
    optimizer = pl_module.configure_optimizers()
    print(f"✅ Optimizer created successfully")
    print(f"   Type: {optimizer.__class__.__name__}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Number of parameter groups: {len(optimizer.param_groups)}")
    print(f"   Parameters in optimizer: {sum(len(g['params']) for g in optimizer.param_groups)}")

    # Test gradient flow manually
    print("\n=== Manual Gradient Test ===")
    dummy_input = torch.randn(2, 3, 32, 128).cuda()
    dummy_targets = torch.randint(0, 100, (2, 25)).cuda()

    # Forward
    output = model(images=dummy_input, text_tokens=dummy_targets, return_loss=True)
    loss = output['loss']
    print(f"Loss value: {loss.item():.4f}")
    print(f"Loss requires_grad: {loss.requires_grad}")

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms[name] = grad_norm

    if grad_norms:
        print(f"\n✅ Gradients computed for {len(grad_norms)} parameters")
        print(f"   Max gradient norm: {max(grad_norms.values()):.6f}")
        print(f"   Min gradient norm: {min(grad_norms.values()):.6f}")

        # Show top 5 gradients
        print("\n   Top 5 gradient norms:")
        for name, norm in sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {name:50s} {norm:.6f}")
    else:
        print("❌ NO GRADIENTS computed!")

    # Test optimizer step
    print("\n=== Optimizer Step Test ===")
    param_before = {name: param.clone() for name, param in model.named_parameters()}
    optimizer.step()

    changes = 0
    for name, param in model.named_parameters():
        if not torch.equal(param, param_before[name]):
            changes += 1

    if changes > 0:
        print(f"✅ {changes} parameters updated after optimizer.step()")
    else:
        print("❌ NO parameters changed after optimizer.step()")
        print("   This indicates optimizer is not updating weights!")

except Exception as e:
    print(f"❌ Optimizer configuration failed: {e}")
    import traceback
    traceback.print_exc()
