"""Inspect why model predicts repeated '장' character."""
import torch
from hydra import initialize, compose
import sys
sys.path.insert(0, '/workspaces')

# Load config
with initialize(config_path="../../../configs", version_base=None):
    cfg = compose(config_name="main", overrides=["experiment=rec_baseline_v1"])

print("Building model and tokenizer...")
from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer
from ocr.core.factory import ModelFactory

tokenizer = KoreanOCRTokenizer(
    charset_path="/workspaces/ocr/data/charset.json",
    max_len=25
)

cfg.model.vocab_size = tokenizer.vocab_size
model = ModelFactory.create(cfg.model, cfg["global"].paths).cuda().eval()

print(f"\n=== Tokenizer Info ===")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"BOS token: {tokenizer.BOS} -> '{tokenizer.id_to_char[tokenizer.BOS]}'")
print(f"EOS token: {tokenizer.EOS} -> '{tokenizer.id_to_char[tokenizer.EOS]}'")
print(f"PAD token: {tokenizer.PAD} -> '{tokenizer.id_to_char[tokenizer.PAD]}'")

# Check what '장' is
jang_id = tokenizer.char_to_id.get('장', None)
print(f"\n'장' character: token_id = {jang_id}")

print(f"\n=== Model Inference Test ===")
# Create dummy image
dummy_img = torch.randn(1, 3, 32, 128).cuda()

with torch.no_grad():
    output = model(images=dummy_img, return_loss=False)

print(f"Output keys: {output.keys()}")
print(f"Tokens shape: {output['tokens'].shape}")
print(f"Predicted tokens: {output['tokens'][0].tolist()}")

pred_text = tokenizer.decode(output['tokens'][0].tolist())
print(f"Decoded text: '{pred_text}'")

# Check logits statistics
print(f"\n=== Logits Analysis ===")
print(f"Logits shape: {output['logits'].shape}")  # [B, T, V]

logits = output['logits'][0]  # [T, V]
probs = torch.softmax(logits, dim=-1)

print(f"\nFirst 5 predictions:")
for i in range(min(5, logits.shape[0])):
    top_prob, top_id = probs[i].max(dim=-1)
    top_char = tokenizer.id_to_char.get(top_id.item(), '?')
    print(f"  Step {i}: token {top_id.item():4d} ('{top_char}') prob={top_prob.item():.4f}")

    # Show top 3 candidates
    top3_probs, top3_ids = probs[i].topk(3)
    print(f"    Top 3: ", end='')
    for prob, tid in zip(top3_probs, top3_ids):
        char = tokenizer.id_to_char.get(tid.item(), '?')
        print(f"{tid.item()}('{char}')={prob.item():.3f}", end=' ')
    print()

# Check if all predictions are the same
unique_preds = output['tokens'][0].unique()
print(f"\n=== Prediction Diversity ===")
print(f"Unique tokens predicted: {len(unique_preds)}")
print(f"Unique token IDs: {unique_preds.tolist()}")

if len(unique_preds) <= 3:  # BOS, one char, EOS
    print("⚠️  Model is stuck predicting same token!")
    print("   This indicates head initialization issue or dead gradients")
else:
    print("✓ Model generates diverse tokens")

# Check head weights
print(f"\n=== Head Weight Statistics ===")
head_weight = model.head.weight  # [vocab_size, d_model]
print(f"Head weight shape: {head_weight.shape}")
print(f"Head weight mean: {head_weight.mean().item():.6f}")
print(f"Head weight std: {head_weight.std().item():.6f}")
print(f"Head weight max: {head_weight.max().item():.6f}")
print(f"Head weight min: {head_weight.min().item():.6f}")

if head_weight.std().item() < 0.01:
    print("⚠️  Head weights have very small variance!")
    print("   This could indicate poor initialization")
