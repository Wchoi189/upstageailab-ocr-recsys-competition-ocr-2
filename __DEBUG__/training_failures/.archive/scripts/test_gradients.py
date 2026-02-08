"""Simplified gradient and optimizer check without full orchestrator."""
import torch
import sys
sys.path.insert(0, '/workspaces')

print("=" * 60)
print("GRADIENT & OPTIMIZER DIAGNOSTIC")
print("=" * 60)

# 1. Build model components directly
print("\n[1/5] Loading tokenizer...")
from ocr.domains.recognition.data.tokenizer import KoreanOCRTokenizer
tokenizer = KoreanOCRTokenizer(
    charset_path="/workspaces/ocr/data/charset.json",
    max_len=25
)
print(f"✓ Vocab size: {tokenizer.vocab_size}")

# 2. Build model
print("\n[2/5] Building model...")
from ocr.domains.recognition.models.architecture import PARSeq
from ocr.core.models.encoder import TimmBackbone
from ocr.domains.recognition.models.decoder import PARSeqDecoder
from ocr.domains.recognition.models.head import PARSeqHead
from ocr.domains.recognition.models.loss.cross_entropy_loss import CrossEntropyLoss

encoder = TimmBackbone(
    model_name='resnet18',
    pretrained=True,
    features_only=True,
    out_indices=[3]
)

decoder = PARSeqDecoder(
    in_channels=512,  # ResNet18 final layer
    d_model=384,
    nhead=12,
    num_layers=1,  # Reduced for testing
    vocab_size=tokenizer.vocab_size,
    max_len=25,
    pad_token_id=tokenizer.PAD,
    bos_token_id=tokenizer.BOS,
    eos_token_id=tokenizer.EOS
)

head = PARSeqHead(
    in_channels=384,
    out_channels=tokenizer.vocab_size
)

loss_fn = CrossEntropyLoss()

model = PARSeq(
    encoder=encoder,
    decoder=decoder,
    head=head,
    loss=loss_fn
).cuda()

print(f"✓ Model built")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}")
print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

# 3. Create optimizer
print("\n[3/5] Creating optimizer...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print(f"✓ Optimizer: {optimizer.__class__.__name__} (lr=0.001)")
print(f"  Param groups: {len(optimizer.param_groups)}")

# 4. Test forward + backward
print("\n[4/5] Testing gradient flow...")
model.train()
optimizer.zero_grad()

# Dummy batch
dummy_images = torch.randn(4, 3, 32, 128).cuda()
dummy_targets = torch.randint(4, tokenizer.vocab_size, (4, 25)).cuda()
# Ensure valid sequence: [BOS, ..., EOS, PAD, ...]
dummy_targets[:, 0] = tokenizer.BOS
dummy_targets[:, -1] = tokenizer.PAD

output = model(images=dummy_images, text_tokens=dummy_targets, return_loss=True)
loss = output['loss']

print(f"✓ Forward pass complete")
print(f"  Loss: {loss.item():.4f}")
print(f"  Loss requires_grad: {loss.requires_grad}")

loss.backward()
print(f"✓ Backward pass complete")

# Check gradients
encoder_grads = [p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None]
decoder_grads = [p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None]
head_grads = [p.grad.norm().item() for p in model.head.parameters() if p.grad is not None]

print(f"\n  Encoder gradients: {len(encoder_grads)} params")
if encoder_grads:
    print(f"    Max: {max(encoder_grads):.6f}, Mean: {sum(encoder_grads)/len(encoder_grads):.6f}")
else:
    print(f"    ❌ NO GRADIENTS!")

print(f"  Decoder gradients: {len(decoder_grads)} params")
if decoder_grads:
    print(f"    Max: {max(decoder_grads):.6f}, Mean: {sum(decoder_grads)/len(decoder_grads):.6f}")
else:
    print(f"    ❌ NO GRADIENTS!")

print(f"  Head gradients: {len(head_grads)} params")
if head_grads:
    print(f"    Max: {max(head_grads):.6f}, Mean: {sum(head_grads)/len(head_grads):.6f}")
else:
    print(f"    ❌ NO GRADIENTS!")

# 5. Test optimizer step
print("\n[5/5] Testing optimizer step...")
param_snapshots = {name: param.data.clone() for name, param in model.named_parameters()}
optimizer.step()

changes = 0
max_change = 0.0
for name, param in model.named_parameters():
    diff = (param.data - param_snapshots[name]).abs().max().item()
    if diff > 1e-10:
        changes += 1
        max_change = max(max_change, diff)

print(f"✓ Optimizer step complete")
print(f"  Parameters changed: {changes}/{len(list(model.named_parameters()))}")
print(f"  Max parameter change: {max_change:.6e}")

# Diagnosis
print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

if not encoder_grads and not decoder_grads and not head_grads:
    print("❌ CRITICAL: No gradients computed anywhere!")
    print("   → Loss may not be connected to model outputs")
elif changes == 0:
    print("❌ CRITICAL: Gradients computed but optimizer didn't update!")
    print("   → Possible issues:")
    print("     - Learning rate too small")
    print("     - Gradient clipping issue")
    print("     - Optimizer bug")
elif changes < len(list(model.named_parameters())) * 0.5:
    print(f"⚠️  WARNING: Only {changes} params updated (expected {len(list(model.named_parameters()))})")
else:
    print("✅ Gradient flow and optimizer working correctly!")
    print("   → Problem must be elsewhere (learning rate, data, architecture)")
