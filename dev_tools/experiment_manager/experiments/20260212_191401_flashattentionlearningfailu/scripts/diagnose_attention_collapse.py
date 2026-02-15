"""
Diagnostic script for Flash Attention learning failure.

This script investigates why the model produces repeated single tokens
during validation, suggesting attention mechanism collapse.

Key Hypotheses to Test:
1. Attention mask formatting issues with Flash Attention backend
2. Gradient flow problems through Flash Attention layers
3. Mask dtype/device mismatches causing silent failures
4. Incremental decoding mask issues during greedy inference
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def test_attention_mask_formats():
    """Test different attention mask formats with Flash Attention."""
    print("="*80)
    print("TEST 1: Attention Mask Format Compatibility")
    print("="*80)

    # Simulate model dimensions
    B, H, T, S, D = 2, 12, 5, 5, 32  # batch, heads, query_len, key_len, head_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy Q, K, V
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    # Test 1: Causal mask from generate_square_subsequent_mask
    print("\n1a. Testing causal mask from nn.Transformer.generate_square_subsequent_mask:")
    causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=device)
    print(f"   Mask shape: {causal_mask.shape}, dtype: {causal_mask.dtype}")
    print(f"   Mask values (unique): {causal_mask.unique()}")
    print(f"   Mask:\n{causal_mask}")

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            out1 = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask, is_causal=False)
        print(f"   ✓ Success with explicit mask, is_causal=False")
        print(f"   Output stats: mean={out1.mean():.4f}, std={out1.std():.4f}")
        print(f"   Output has NaN: {out1.isnan().any()}")
        print(f"   Output has Inf: {out1.isinf().any()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 2: Using is_causal=True without explicit mask
    print("\n1b. Testing with is_causal=True (no explicit mask):")
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            out2 = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        print(f"   ✓ Success with is_causal=True")
        print(f"   Output stats: mean={out2.mean():.4f}, std={out2.std():.4f}")
        print(f"   Outputs are equivalent: {torch.allclose(out1, out2, atol=1e-3)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Test 3: Mask dtype conversion issue
    print("\n1c. Testing mask dtype issue (float32 -> float16):")
    causal_mask_fp16 = causal_mask.to(dtype=torch.float16)
    print(f"   Mask shape: {causal_mask_fp16.shape}, dtype: {causal_mask_fp16.dtype}")
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            out3 = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask_fp16, is_causal=False)
        print(f"   ✓ Success with fp16 mask")
        print(f"   Output stats: mean={out3.mean():.4f}, std={out3.std():.4f}")
        print(f"   Outputs are equivalent to fp32: {torch.allclose(out1, out3, atol=1e-3)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")


def test_incremental_decoding():
    """Test incremental decoding with growing sequence lengths."""
    print("\n" + "="*80)
    print("TEST 2: Incremental Decoding with Growing Masks")
    print("="*80)

    B, H, D = 2, 12, 32
    max_len = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate encoder memory (fixed)
    S = 16
    memory_k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    memory_v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    print(f"\nSimulating {max_len} decoding steps:")
    all_outputs = []

    for t in range(1, max_len + 1):
        # Current sequence length
        q = torch.randn(B, H, t, D, device=device, dtype=torch.float16)

        # Create causal self-attention mask
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=device)

        # Self-attention
        try:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                self_attn_out = F.scaled_dot_product_attention(
                    q, q, q,
                    attn_mask=causal_mask,
                    is_causal=False
                )

            # Cross-attention to memory
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
                cross_attn_out = F.scaled_dot_product_attention(
                    q, memory_k, memory_v,
                    attn_mask=None,
                    is_causal=False
                )

            all_outputs.append(self_attn_out[:, :, -1:, :])  # Last position only

            if t <= 3 or t == max_len:
                print(f"  Step {t:2d}: Self-Attn mean={self_attn_out.mean():.4f}, std={self_attn_out.std():.4f}, "
                      f"Cross-Attn mean={cross_attn_out.mean():.4f}, std={cross_attn_out.std():.4f}")

        except Exception as e:
            print(f"  Step {t:2d}: ✗ Failed: {e}")
            break

    if len(all_outputs) == max_len:
        print(f"\n  ✓ All {max_len} steps succeeded")

        # Check for attention collapse (all outputs converging to same value)
        output_tensor = torch.cat(all_outputs, dim=2)  # [B, H, T, D]
        output_std_per_position = output_tensor.std(dim=2)  # [B, H, D]
        avg_std = output_std_per_position.mean()
        print(f"  Average std across positions: {avg_std:.6f}")
        if avg_std < 0.01:
            print(f"  ⚠️  WARNING: Low variance suggests attention collapse!")
    else:
        print(f"\n  ✗ Failed at step {len(all_outputs) + 1}")


def test_plm_mask_format():
    """Test PLM-style custom masks with Flash Attention."""
    print("\n" + "="*80)
    print("TEST 3: PLM Custom Masks")
    print("="*80)

    B, H, T, D = 2, 12, 5, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy Q, K, V
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    # Simulate PLM-style boolean mask (from plm.py)
    print("\n3a. Testing PLM boolean mask conversion:")
    plm_bool_mask = torch.zeros((T, T), dtype=torch.bool, device=device)
    # Example: mask some positions based on permutation
    plm_bool_mask[0, [2, 3, 4]] = True
    plm_bool_mask[1, [3, 4]] = True
    plm_bool_mask[2, [4]] = True
    print(f"   Boolean mask shape: {plm_bool_mask.shape}")
    print(f"   Boolean mask:\n{plm_bool_mask.int()}")

    # Convert to additive mask (as done in architecture.py:247-249)
    plm_additive_mask = plm_bool_mask.float()
    plm_additive_mask = plm_additive_mask.masked_fill(plm_additive_mask == 1.0, float('-inf'))
    plm_additive_mask = plm_additive_mask.masked_fill(plm_additive_mask == 0.0, 0.0)

    print(f"   Additive mask unique values: {plm_additive_mask.unique()}")

    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
            out_plm = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=plm_additive_mask,
                is_causal=False
            )
        print(f"   ✓ Success with PLM mask")
        print(f"   Output stats: mean={out_plm.mean():.4f}, std={out_plm.std():.4f}")
        print(f"   Output has NaN: {out_plm.isnan().any()}")
        print(f"   Output has Inf: {out_plm.isinf().any()}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")


def test_backend_selection():
    """Test which backend is actually being used."""
    print("\n" + "="*80)
    print("TEST 4: Backend Selection Verification")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        major, minor = torch.cuda.get_device_capability()
        compute_cap = major * 10 + minor
        print(f"   CUDA Compute Capability: sm_{compute_cap}")
        print(f"   Flash Attention supported: {compute_cap >= 80}")
    else:
        print(f"   Running on CPU - Flash Attention not available")

    print(f"   PyTorch version: {torch.__version__}")

    # Test backend selection
    B, H, T, D = 2, 12, 5, 32
    q = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.float16)

    print("\n   Testing backend availability:")
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    backend_names = ["FLASH_ATTENTION", "MATH", "EFFICIENT_ATTENTION"]

    for backend, name in zip(backends, backend_names):
        try:
            with sdpa_kernel([backend]):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            print(f"   ✓ {name}: Available")
        except Exception as e:
            print(f"   ✗ {name}: Not available - {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Flash Attention Diagnostic Suite")
    print("Investigating repeated token prediction issue")
    print("="*80)

    test_backend_selection()
    test_attention_mask_formats()
    test_incremental_decoding()
    test_plm_mask_format()

    print("\n" + "="*80)
    print("Diagnostic Complete")
    print("="*80)
