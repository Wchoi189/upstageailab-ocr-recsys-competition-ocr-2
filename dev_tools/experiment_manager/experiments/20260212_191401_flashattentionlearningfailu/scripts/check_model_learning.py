"""
Check if the model is actually learning or if weights are stuck.

This script loads the model and checks:
1. Head output distribution (should not be constant)
2. Decoder embeddings are being used correctly
3. Attention patterns during forward pass
"""

import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[4]))

from ocr.domains.recognition.models.flash_attention import enable_flash_attention_kernel


def check_head_initialization(model):
    """Check if the classification head is properly initialized."""
    print("\n" + "="*80)
    print("HEAD INITIALIZATION CHECK")
    print("="*80)

    head = model.head
    weight = head.weight  # [vocab_size, d_model]
    bias = head.bias if hasattr(head, 'bias') and head.bias is not None else None

    print(f"Head weight shape: {weight.shape}")
    print(f"Head weight stats: mean={weight.mean():.6f}, std={weight.std():.6f}")
    print(f"Head weight range: [{weight.min():.6f}, {weight.max():.6f}]")

    if bias is not None:
        print(f"Head bias shape: {bias.shape}")
        print(f"Head bias stats: mean={bias.mean():.6f}, std={bias.std():.6f}")
        print(f"Head bias range: [{bias.min():.6f}, {bias.max():.6f}]")

        # Check if bias is uniform (bad initialization)
        unique_bias = bias.unique()
        print(f"Unique bias values: {len(unique_bias)}/{len(bias)}")
        if len(unique_bias) < 10:
            print(f"⚠️  WARNING: Very few unique bias values detected!")
            print(f"   Top 10 bias values: {bias.sort()[0][:10]}")

    # Check for degenerate weight patterns
    weight_per_class_std = weight.std(dim=1)  # std across features for each class
    print(f"\nPer-class weight std: mean={weight_per_class_std.mean():.6f}, min={weight_per_class_std.min():.6f}")

    if weight_per_class_std.min() < 0.001:
        zero_std_classes = (weight_per_class_std < 0.001).sum().item()
        print(f"⚠️  WARNING: {zero_std_classes} classes have near-zero weight std (degenerate)")


def check_embeddings(model):
    """Check token embeddings initialization."""
    print("\n" + "="*80)
    print("TOKEN EMBEDDINGS CHECK")
    print("="*80)

    embeddings = model.decoder.embed_tokens.weight  # [vocab_size, d_model]

    print(f"Embedding weight shape: {embeddings.shape}")
    print(f"Embedding stats: mean={embeddings.mean():.6f}, std={embeddings.std():.6f}")
    print(f"Embedding range: [{embeddings.min():.6f}, {embeddings.max():.6f}]")

    # Check for duplicate embeddings (bad initialization)
    unique_embeddings = torch.unique(embeddings, dim=0).shape[0]
    print(f"Unique embeddings: {unique_embeddings}/{embeddings.shape[0]}")

    if unique_embeddings < embeddings.shape[0]:
        print(f"⚠️  WARNING: Duplicate embeddings detected!")

    # Check BOS, EOS, PAD token embeddings
    bos_idx = model.decoder.bos_token_id
    eos_idx = model.decoder.eos_token_id
    pad_idx = model.decoder.pad_token_id

    print(f"\nSpecial tokens:")
    print(f"  BOS (idx={bos_idx}): mean={embeddings[bos_idx].mean():.6f}, std={embeddings[bos_idx].std():.6f}")
    print(f"  EOS (idx={eos_idx}): mean={embeddings[eos_idx].mean():.6f}, std={embeddings[eos_idx].std():.6f}")
    print(f"  PAD (idx={pad_idx}): mean={embeddings[pad_idx].mean():.6f}, std={embeddings[pad_idx].std():.6f}")


def check_positional_encodings(model):
    """Check positional encodings."""
    print("\n" + "="*80)
    print("POSITIONAL ENCODING CHECK")
    print("="*80)

    pos_enc = model.decoder.pos_queries  # [1, max_len+1, d_model]

    print(f"Pos encoding shape: {pos_enc.shape}")
    print(f"Pos encoding stats: mean={pos_enc.mean():.6f}, std={pos_enc.std():.6f}")
    print(f"Pos encoding range: [{pos_enc.min():.6f}, {pos_enc.max():.6f}]")

    # Check if positional encodings are different for each position
    pos_enc_squeezed = pos_enc.squeeze(0)  # [max_len+1, d_model]
    pos_diffs = (pos_enc_squeezed[1:] - pos_enc_squeezed[:-1]).abs().mean(dim=1)

    print(f"Mean difference between consecutive positions: {pos_diffs.mean():.6f}")
    print(f"Min difference: {pos_diffs.min():.6f}, Max difference: {pos_diffs.max():.6f}")

    if pos_diffs.mean() < 0.01:
        print(f"⚠️  WARNING: Positional encodings are too similar!")


def check_forward_pass(model, device):
    """Check a forward pass with dummy input."""
    print("\n" + "="*80)
    print("FORWARD PASS CHECK")
    print("="*80)

    # Create dummy input
    B = 4
    images = torch.randn(B, 3, 32, 128, device=device)
    targets = torch.tensor([
        [1, 5, 10, 15, 20, 2, 0, 0],  # BOS, chars, EOS, PAD
        [1, 7, 12, 17, 22, 25, 2, 0],
        [1, 3, 8, 13, 18, 2, 0, 0],
        [1, 4, 9, 14, 19, 24, 2, 0],
    ], device=device)

    model.eval()
    with torch.no_grad():
        with enable_flash_attention_kernel():
            # Training mode forward (with targets)
            train_out = model(images, text_tokens=targets, return_loss=True)

            # Inference mode forward
            infer_out = model(images, return_loss=False)

    print(f"Training mode:")
    print(f"  Logits shape: {train_out['logits'].shape}")
    print(f"  Logits stats: mean={train_out['logits'].mean():.6f}, std={train_out['logits'].std():.6f}")
    print(f"  Loss: {train_out['loss'].item():.6f}")

    # Check if logits are diverse
    logit_stds_per_position = train_out['logits'].std(dim=2)  # std across vocab
    print(f"  Logit std per position (mean): {logit_stds_per_position.mean():.6f}")
    print(f"  Logit std per position (min): {logit_stds_per_position.min():.6f}")

    if logit_stds_per_position.min() < 0.1:
        print(f"  ⚠️  WARNING: Some positions have very low logit variance!")

    print(f"\nInference mode:")
    print(f"  Predicted tokens shape: {infer_out['tokens'].shape}")
    print(f"  Logits shape: {infer_out['logits'].shape}")

    # Check predicted tokens
    pred_tokens = infer_out['tokens']
    print(f"\n  Predicted token sequences:")
    for i in range(min(B, 4)):
        tokens = pred_tokens[i].tolist()
        print(f"    Sample {i}: {tokens[:10]}")  # First 10 tokens

        # Check if all tokens are the same (repetition issue)
        unique_tokens = set(tokens[1:8])  # Skip BOS, check next 7 tokens
        if len(unique_tokens) == 1:
            print(f"      ⚠️  WARNING: All predicted tokens are the same! (Token ID: {list(unique_tokens)[0]})")
        elif len(unique_tokens) <= 2:
            print(f"      ⚠️  WARNING: Very few unique tokens ({len(unique_tokens)})")


def main():
    print("="*80)
    print("MODEL LEARNING CHECK - Flash Attention Training")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Manually create model using the Flash Attention config
    from ocr.domains.recognition.models.architecture import PARSeq
    from ocr.core.models.encoder.timm_backbone import TimmBackbone
    from ocr.domains.recognition.models.decoder import PARSeqDecoder

    print("\nInstantiating model components...")

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
        vocab_size=1027,  # Korean charset
        use_flash_attention=True,
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

    model = model.to(device)

    print(f"Model instantiated: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Run checks
    check_head_initialization(model)
    check_embeddings(model)
    check_positional_encodings(model)
    check_forward_pass(model, device)

    print("\n" + "="*80)
    print("CHECK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
