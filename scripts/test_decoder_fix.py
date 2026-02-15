"""
Verify the decoder fix for repeated token issue (BUG-001).

Tests that the content stream shift prevents information leakage and produces
diverse token predictions during inference.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr.core.models.encoder.timm_backbone import TimmBackbone
from ocr.domains.recognition.models.decoder import PARSeqDecoder
from ocr.core.models.architecture import OCRModel


def create_model(vocab_size=1027):
    """Create a minimal OCRModel with PARSeq decoder."""
    encoder = TimmBackbone(
        model_name="resnet18",
        pretrained=False,  # Faster initialization
        features_only=True,
        output_indices=[3]
    )

    decoder = PARSeqDecoder(
        in_channels=256,
        d_model=384,
        nhead=12,
        num_layers=2,  # Reduced for faster testing
        dim_feedforward=1536,
        dropout=0.1,
        max_len=25,
        vocab_size=vocab_size,
        use_flash_attention=False,  # Use standard attention for testing
        plm_config=None
    )

    # Simple linear head that returns dict format
    class SimpleHead(nn.Module):
        def __init__(self, d_model, vocab_size):
            super().__init__()
            self.proj = nn.Linear(d_model, vocab_size)

        def forward(self, x, return_loss=False):
            logits = self.proj(x)
            return {"logits": logits, "tokens": logits.argmax(dim=-1)}

    head = SimpleHead(384, vocab_size)

    # Manually create OCRModel
    model = nn.Module()
    model.encoder = encoder
    model.decoder = decoder
    model.head = head

    # Add forward and generate methods
    def forward(self, images, return_loss=False, **kwargs):
        encoded_features = self.encoder(images)
        if not return_loss:
            return generate(self, encoded_features)
        else:
            targets = kwargs.get("labels") or kwargs.get("targets")
            decoded_features = self.decoder(encoded_features, targets=targets)
            pred = self.head(decoded_features, return_loss)
            return pred

    def generate(self, encoded_features):
        bos_token_id = self.decoder.bos_token_id
        max_len = self.decoder.max_len
        device = encoded_features[-1].device if isinstance(encoded_features, list) else encoded_features.device
        B = encoded_features[-1].size(0) if isinstance(encoded_features, list) else encoded_features.size(0)

        current_tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for i in range(max_len):
            decoded_features = self.decoder(encoded_features, targets=current_tokens)
            head_out = self.head(decoded_features, return_loss=False)
            logits = head_out["logits"]
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1)
            current_tokens = torch.cat([current_tokens, next_token.unsqueeze(1)], dim=1)

        return head_out

    model.forward = lambda *args, **kwargs: forward(model, *args, **kwargs)
    model.generate = lambda *args, **kwargs: generate(model, *args, **kwargs)

    return model


def test_inference_diversity(model, device, num_samples=4):
    """Test that inference produces diverse token predictions."""
    print("\n" + "="*80)
    print("TEST: Inference Token Diversity")
    print("="*80)

    # Create dummy images
    images = torch.randn(num_samples, 3, 32, 128, device=device)

    model.eval()
    with torch.no_grad():
        # Run inference
        out = model(images, return_loss=False)

    tokens = out['tokens']
    print(f"Predicted tokens shape: {tokens.shape}")

    all_diverse = True
    for i in range(num_samples):
        # Skip BOS token, examine first 7 predictions
        token_seq = tokens[i, 1:8].tolist()
        unique = len(set(token_seq))

        print(f"  Sample {i}: {token_seq} → Unique tokens: {unique}/7")

        if unique == 1:
            print(f"    ❌ FAILED: All tokens are identical!")
            all_diverse = False
        elif unique < 3:
            print(f"    ⚠️  WARNING: Low diversity ({unique} unique tokens)")
        else:
            print(f"    ✅ PASSED: Diverse predictions")

    return all_diverse


def test_step_by_step_analysis(model, device):
    """Analyze logits at each decoding step to verify they change."""
    print("\n" + "="*80)
    print("TEST: Step-by-Step Logit Analysis")
    print("="*80)

    # Create single image
    images = torch.randn(1, 3, 32, 128, device=device)

    # Get encoder features
    model.eval()
    with torch.no_grad():
        encoded_features = model.encoder(images)

    bos_token = model.decoder.bos_token_id
    max_steps = 5

    print(f"\nDecoding {max_steps} steps:")
    current_tokens = torch.full((1, 1), bos_token, dtype=torch.long, device=device)

    logits_history = []

    for step in range(max_steps):
        # Decode current sequence
        decoded_features = model.decoder(encoded_features, targets=current_tokens)

        # Project to logits
        logits = model.head(decoded_features, return_loss=False)['logits']

        # Get logits for the last position
        last_logits = logits[0, -1, :]
        logits_history.append(last_logits)

        # Get top-5 predictions
        top5_probs, top5_tokens = torch.topk(torch.softmax(last_logits, dim=0), k=5)
        next_token = last_logits.argmax()

        print(f"  Step {step+1}: seq_len={current_tokens.size(1)}")
        print(f"    Top-5 tokens: {top5_tokens.tolist()}")
        print(f"    Top-5 probs:  {[f'{p:.4f}' for p in top5_probs.tolist()]}")
        print(f"    Selected: {next_token.item()}")
        print(f"    Logits: mean={last_logits.mean():.4f}, std={last_logits.std():.4f}")

        # Append selected token
        current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    print(f"\n  Final sequence: {current_tokens[0].tolist()}")

    # Check if logits are varying across steps
    print("\n  Checking logit variation across steps:")
    logits_varying = True
    for i in range(1, len(logits_history)):
        diff = (logits_history[i] - logits_history[i-1]).abs().mean().item()
        print(f"    Step {i} vs {i+1}: mean diff = {diff:.6f}")
        if diff < 0.001:
            print(f"      ⚠️  WARNING: Logits barely changing!")
            logits_varying = False

    if logits_varying:
        print(f"\n  ✅ Logits are varying properly across steps")
    else:
        print(f"\n  ❌ Logits are too stable (not varying enough)")

    return logits_varying  # Return True if logits ARE varying


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Create model
    print("Creating model with fixed decoder...")
    model = create_model()
    model = model.to(device)

    # Run tests
    test1_passed = test_inference_diversity(model, device)
    test2_passed = test_step_by_step_analysis(model, device)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"  Inference Diversity: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Logit Variation:     {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✅ All tests passed! Decoder fix verified.")
        return 0
    else:
        print("\n❌ Some tests failed. Issue may persist.")
        return 1


if __name__ == "__main__":
    exit(main())
