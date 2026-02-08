"""Verify encoder-decoder dimension compatibility after config fix."""
import torch
import sys
sys.path.insert(0, '/workspaces')

print("=" * 70)
print("DIMENSION COMPATIBILITY TEST")
print("=" * 70)

# Load config
from hydra import initialize, compose
from omegaconf import OmegaConf
with initialize(config_path="../../../configs", version_base=None):
    cfg = compose(config_name="main", overrides=["experiment=rec_baseline_v1"])

print("\n[1/4] Loading tokenizer...")
from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer
tokenizer = KoreanOCRTokenizer("/workspaces/ocr/data/charset.json", max_len=25)
print(f"✓ Vocab size: {tokenizer.vocab_size}")

print("\n[2/4] Building model from corrected config...")
cfg.model.vocab_size = tokenizer.vocab_size

from ocr.core.factory import ModelFactory
try:
    # Get paths from config - handle different config structures
    paths = getattr(cfg, 'global', OmegaConf.create({}))
    if hasattr(paths, 'paths'):
        paths = paths.paths
    else:
        paths = OmegaConf.create({})  # Use empty if not found

    model = ModelFactory.create(cfg.model, paths).cuda()
    print(f"✓ Model created successfully")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    exit(1)

print("\n[3/4] Testing forward pass with dummy data...")
dummy_img = torch.randn(4, 3, 32, 128).cuda()
dummy_tokens = torch.randint(4, tokenizer.vocab_size, (4, 25)).cuda()
dummy_tokens[:, 0] = tokenizer.BOS  # Set BOS token
dummy_tokens[:, -1] = tokenizer.PAD  # Set PAD token

try:
    output = model(images=dummy_img, text_tokens=dummy_tokens, return_loss=True)
    print(f"✓ Forward pass successful!")
    print(f"  Loss: {output['loss'].item():.4f}")
    print(f"  Logits shape: {output['logits'].shape}")

    if torch.isnan(output['loss']) or torch.isinf(output['loss']):
        print(f"  ⚠️  WARNING: Loss is {output['loss'].item()}, check initialization")
    else:
        print(f"  ✓ Loss is finite")

except RuntimeError as e:
    print(f"❌ Forward pass failed with dimension error:")
    print(f"   {e}")
    exit(1)

print("\n[4/4] Testing inference mode...")
try:
    model.eval()
    with torch.no_grad():
        inference_out = model(images=dummy_img, return_loss=False)

    print(f"✓ Inference successful!")
    print(f"  Tokens shape: {inference_out['tokens'].shape}")
    print(f"  Logits shape: {inference_out['logits'].shape}")

    # Decode predictions
    pred_texts = tokenizer.batch_decode(inference_out['tokens'].cpu().tolist())
    print(f"\n  Sample predictions (should vary, not all same char):")
    for i, text in enumerate(pred_texts[:3]):
        print(f"    [{i}]: '{text}'")

    # Check diversity
    all_same = all(t == pred_texts[0] for t in pred_texts)
    if all_same:
        print(f"\n  ⚠️  WARNING: All predictions are identical!")
        print(f"     This is normal for untrained model, will improve with training")
    else:
        print(f"\n  ✓ Predictions show some diversity")

except Exception as e:
    print(f"❌ Inference failed: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ ALL DIMENSION TESTS PASSED")
print("=" * 70)
print("\nModel is ready for training!")
print("Run: uv run python scripts/runners/train.py experiment=rec_baseline_v1")
