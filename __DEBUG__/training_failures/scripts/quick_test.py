"""Quick dimension compatibility test using orchestrator."""
import torch
import sys
sys.path.insert(0, '/workspaces')

print("=" * 70)
print("DIMENSION COMPATIBILITY TEST (via Orchestrator)")
print("=" * 70)

from hydra import initialize, compose

with initialize(config_path="../../../configs", version_base=None):
    cfg = compose(config_name="main", overrides=["experiment=rec_baseline_v1"])

print("\n[1/3] Building model via orchestrator...")
from ocr.pipelines.orchestrator import OCRProjectOrchestrator

orch = OCRProjectOrchestrator(cfg)
orch.run_model_training()  # This builds everything

model = orch.pl_module.model.cuda()
tokenizer = orch.datasets['val'].tokenizer

print(f"✓ Model built successfully")
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total_params:,}")

print("\n[2/3] Testing forward pass...")
dummy_img = torch.randn(4, 3, 32, 128).cuda()
dummy_tokens = torch.randint(4, tokenizer.vocab_size, (4, 25)).cuda()
dummy_tokens[:, 0] = tokenizer.BOS

try:
    model.train()
    output = model(images=dummy_img, text_tokens=dummy_tokens, return_loss=True)
    print(f"✓ Forward pass successful!")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits shape: {output['logits'].shape}")

    if torch.isnan(output['loss']):
        print(f"  ❌ Loss is NaN - check initialization")
        exit(1)
    else:
        print(f"  ✓ Loss is finite")

except RuntimeError as e:
    print(f"❌ Dimension error: {e}")
    exit(1)

print("\n[3/3] Testing inference...")
try:
    model.eval()
    with torch.no_grad():
        inf_out = model(images=dummy_img, return_loss=False)

    pred_texts = tokenizer.batch_decode(inf_out['tokens'].cpu().tolist())
    print(f"✓ Inference successful")
    print(f"  Sample predictions: {pred_texts[:2]}")

except Exception as e:
    print(f"❌ Inference failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED - Ready for training!")
print("=" * 70)
