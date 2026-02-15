"""
Test different inference strategies to fix the repeated token issue.

Tests:
1. Current broken implementation (explicit causal mask)
2. Fix 1: Use is_causal=True instead of explicit mask
3. Fix 2: Disable Flash Attention for inference only
4. Fix 3: Use standard attention throughout
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4]))

from ocr.domains.recognition.models.architecture import PARSeq
from ocr.core.models.encoder.timm_backbone import TimmBackbone
from ocr.domains.recognition.models.decoder import PARSeqDecoder
from ocr.domains.recognition.models.flash_attention import enable_flash_attention_kernel


def create_model(use_flash=True):
    """Create a PARSeq model."""
    encoder = TimmBackbone(
        model_name="resnet18",
        pretrained=True,
        features_only=True,
        output_indices=[3]
    )

    decoder = PARSeqDecoder(
        in_channels=256,
        d_model=384,
        nhead=12,
        num_layers=12,
        dim_feedforward=1536,
        dropout=0.1,
        max_len=25,
        vocab_size=1027,
        use_flash_attention=use_flash,
        plm_config=None
    )

    head = torch.nn.Linear(384, 1027)
    loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    model = PARSeq(
        encoder=encoder,
        decoder=decoder,
        head=head,
        loss=loss
    )

    return model


def test_current_broken(model, images, device):
    """Test current broken implementation."""
    print("\n" + "="*80)
    print("TEST 1: Current Implementation (BROKEN)")
    print("="*80)

    model.eval()
    with torch.no_grad():
        with enable_flash_attention_kernel():
            out = model(images, return_loss=False)

    tokens = out['tokens']
    print(f"Predicted tokens shape: {tokens.shape}")

    for i in range(min(4, tokens.shape[0])):
        token_seq = tokens[i, 1:8].tolist()  # Skip BOS, take 7 tokens
        unique = len(set(token_seq))
        print(f"  Sample {i}: {token_seq[:7]} → Unique tokens: {unique}/7")

        if unique == 1:
            print(f"    ❌ FAILED: All tokens are identical!")
            return False

    print(f"  ✅ PASSED: Diverse token predictions")
    return True


def test_fix_is_causal(model, images, device):
    """Test Fix 1: Use is_causal=True in decoder forward."""
    print("\n" + "="*80)
    print("TEST 2: Fix 1 - Use is_causal=True Flag")
    print("="*80)
    print("Modifying decoder to use is_causal=True...")

    # Monkey-patch the decoder forward to use is_causal=True
    original_forward = model.decoder.forward

    def patched_forward(features, targets=None, memory_key_padding_mask=None,
                        tgt_mask=None, tgt_query_mask=None, **kwargs):
        # Same logic but use is_causal in decoder call
        if isinstance(features, list):
            visual_feat = features[-1]
            memory = visual_feat.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            memory = features

        device = memory.device
        B, S, C = memory.shape
        memory = model.decoder.input_proj(memory)

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.zeros(B, S, dtype=torch.bool, device=device)

        if targets is None:
            raise ValueError("Targets required")

        device = memory.device
        B, T = targets.shape

        tgt_emb = model.decoder.embed_tokens(targets) * (model.decoder.d_model ** 0.5)
        pos_emb = model.decoder.pos_encoder[:, :T, :] * (model.decoder.d_model ** 0.5)
        tgt = tgt_emb + pos_emb

        # KEY FIX: Don't generate explicit mask, rely on is_causal=True
        # The decoder layers should support tgt_is_causal parameter
        tgt_key_padding_mask = (targets == model.decoder.pad_token_id)

        # Call decoder WITHOUT explicit tgt_mask, but we need to modify decoder call
        # For now, test if removing the mask helps
        output = model.decoder.decoder(
            tgt, memory,
            tgt_mask=None,  # Don't pass explicit mask
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True,  # Use built-in causal attention
        )

        return model.decoder.norm(output)

    # Apply patch
    model.decoder.forward = patched_forward

    model.eval()
    with torch.no_grad():
        try:
            with enable_flash_attention_kernel():
                out = model(images, return_loss=False)

            tokens = out['tokens']
            print(f"Predicted tokens shape: {tokens.shape}")

            all_unique = True
            for i in range(min(4, tokens.shape[0])):
                token_seq = tokens[i, 1:8].tolist()
                unique = len(set(token_seq))
                print(f"  Sample {i}: {token_seq[:7]} → Unique tokens: {unique}/7")

                if unique == 1:
                    print(f"    ❌ Still failing with repeated tokens")
                    all_unique = False

            # Restore original
            model.decoder.forward = original_forward

            if all_unique:
                print(f"  ✅ PASSED: Diverse predictions with is_causal=True")
                return True
            else:
                print(f"  ❌ FAILED: Still producing repeated tokens")
                return False

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            model.decoder.forward = original_forward
            return False


def test_without_flash_inference(model_flash, images, device):
    """Test Fix 2: Disable Flash Attention for inference only."""
    print("\n" + "="*80)
    print("TEST 3: Fix 2 - Standard Attention for Inference")
    print("="*80)

    # Create a model with standard attention
    print("Creating model with standard attention...")
    model_standard = create_model(use_flash=False)
    model_standard = model_standard.to(device)

    # Copy weights from Flash model to standard model
    print("Copying weights from Flash model...")
    model_standard.load_state_dict(model_flash.state_dict(), strict=False)

    model_standard.eval()
    with torch.no_grad():
        out = model_standard(images, return_loss=False)

    tokens = out['tokens']
    print(f"Predicted tokens shape: {tokens.shape}")

    all_diverse = True
    for i in range(min(4, tokens.shape[0])):
        token_seq = tokens[i, 1:8].tolist()
        unique = len(set(token_seq))
        print(f"  Sample {i}: {token_seq[:7]} → Unique tokens: {unique}/7")

        if unique == 1:
            print(f"    ❌ FAILED: Still repeating with standard attention!")
            all_diverse = False

    if all_diverse:
        print(f"  ✅ PASSED: Standard attention produces diverse predictions")
        return True
    else:
        print(f"  ❌ FAILED: Issue persists even without Flash Attention")
        return False


def analyze_step_by_step(model, images, device):
    """Analyze the decoding process step by step."""
    print("\n" + "="*80)
    print("STEP-BY-STEP ANALYSIS")
    print("="*80)

    # Get visual memory
    with torch.no_grad():
        features = model.encoder(images)
        if isinstance(features, (list, tuple)):
            visual_feat = features[-1]
        else:
            visual_feat = features

        if visual_feat.ndim == 4:
            b, c, h, w = visual_feat.shape
            pos_embed = model._generate_2d_sincos_pos_embed(h, w, c, device=visual_feat.device)
            visual_feat_normalized = visual_feat.permute(0, 2, 3, 1)
            visual_feat_normalized = model.visual_norm(visual_feat_normalized)
            visual_feat = visual_feat_normalized.permute(0, 3, 1, 2)
            pos_embed = pos_embed * (c ** 0.5) * 0.1
            visual_feat = visual_feat + pos_embed
            visual_memory = visual_feat.permute(0, 2, 3, 1).flatten(1, 2)

    B = visual_memory.size(0)
    bos_token = model.decoder.bos_token_id

    print(f"\nDecoding sample 0 step by step:")
    tgt_tokens = torch.full((1, 1), bos_token, dtype=torch.long, device=device)

    for step in range(10):
        # Decode one step
        with enable_flash_attention_kernel():
            step_logits = model._decode_step(visual_memory[:1], tgt_tokens)

        # Get top 5 predictions
        top5_probs, top5_tokens = torch.topk(torch.softmax(step_logits[0, 0], dim=0), k=5)

        next_token = step_logits.argmax(dim=-1)

        print(f"  Step {step+1}: Current length={tgt_tokens.size(1)}")
        print(f"    Top 5 predictions: {top5_tokens.tolist()}")
        print(f"    Top 5 probs: {[f'{p:.4f}' for p in top5_probs.tolist()]}")
        print(f"    Selected: {next_token.item()}")
        print(f"    Logits stats: mean={step_logits.mean():.4f}, std={step_logits.std():.4f}")

        # Check if logits are degenerate (very low std)
        if step_logits.std() < 0.01:
            print(f"    ⚠️  WARNING: Logits have very low variance!")

        tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

    print(f"\n  Final sequence: {tgt_tokens[0].tolist()}")
    unique = len(set(tgt_tokens[0, 1:].tolist()))
    print(f"  Unique tokens (excluding BOS): {unique}/{tgt_tokens.size(1)-1}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create dummy input
    B = 4
    images = torch.randn(B, 3, 32, 128, device=device)

    # Create Flash Attention model
    print("\nCreating Flash Attention model...")
    model_flash = create_model(use_flash=True)
    model_flash = model_flash.to(device)

    # Run tests
    test_current_broken(model_flash, images, device)
    analyze_step_by_step(model_flash, images, device)
    test_without_flash_inference(model_flash, images, device)

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
