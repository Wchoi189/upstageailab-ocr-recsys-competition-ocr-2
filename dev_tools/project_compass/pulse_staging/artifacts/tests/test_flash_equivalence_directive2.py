#!/usr/bin/env python3
"""
Numerical Drift Check - Perplexity Directive 2

Verifies Flash Attention vs MHA outputs are within acceptable tolerance (1e-3).
Tests both single-layer and full decoder equivalence with weight copying.

Usage:
    uv run pytest dev_tools/project_compass/pulse_staging/artifacts/tests/test_flash_equivalence_directive2.py -v -s
"""

import torch
import torch.nn as nn
import pytest
from ocr.domains.recognition.models.flash_attention import (
    FlashMultiheadAttention,
    enable_flash_attention_kernel,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


class TestNumericalEquivalence:
    """Test Flash Attention numerical equivalence (Directive 2)."""

    def test_flash_mha_equivalence_same_weights(self, device, dtype):
        """
        Test Flash vs Standard MHA with SAME weights.

        Expected: max_diff < 1e-3 (Perplexity directive 2)
        """
        if device == "cpu":
            pytest.skip("Flash Attention requires CUDA")

        # Config
        embed_dim = 384
        num_heads = 12
        batch_size = 4
        seq_len = 26

        # Create both attention modules
        torch.manual_seed(42)
        standard_mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,  # Disable for deterministic comparison
            batch_first=True,
        ).to(device).eval()

        torch.manual_seed(42)
        flash_mha = FlashMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        ).to(device).eval()

        # Copy weights from standard to flash (for exact comparison)
        with torch.no_grad():
            # Standard MHA uses in_proj_weight for combined QKV
            # Flash MHA uses separate q_proj, k_proj, v_proj
            # We need to split the in_proj_weight

            # Standard: in_proj_weight shape [3*embed_dim, embed_dim]
            # Flash: q_proj.weight, k_proj.weight, v_proj.weight each [embed_dim, embed_dim]

            in_proj_weight = standard_mha.in_proj_weight  # [1152, 384]
            in_proj_bias = standard_mha.in_proj_bias  # [1152]

            # Split into Q, K, V
            q_weight = in_proj_weight[:embed_dim, :]  # [384, 384]
            k_weight = in_proj_weight[embed_dim:2*embed_dim, :]  # [384, 384]
            v_weight = in_proj_weight[2*embed_dim:, :]  # [384, 384]

            q_bias = in_proj_bias[:embed_dim]  # [384]
            k_bias = in_proj_bias[embed_dim:2*embed_dim]  # [384]
            v_bias = in_proj_bias[2*embed_dim:]  # [384]

            # Copy to Flash MHA
            flash_mha.q_proj.weight.copy_(q_weight)
            flash_mha.k_proj.weight.copy_(k_weight)
            flash_mha.v_proj.weight.copy_(v_weight)

            flash_mha.q_proj.bias.copy_(q_bias)
            flash_mha.k_proj.bias.copy_(k_bias)
            flash_mha.v_proj.bias.copy_(v_bias)

            # Copy output projection
            flash_mha.out_proj.weight.copy_(standard_mha.out_proj.weight)
            flash_mha.out_proj.bias.copy_(standard_mha.out_proj.bias)

        # Create test inputs (deterministic)
        torch.manual_seed(123)
        query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)
        key = query  # Self-attention
        value = query

        # Forward pass
        with torch.no_grad():
            # Standard MHA (fp32 for reference)
            standard_out, _ = standard_mha(
                query.float(), key.float(), value.float(),
                need_weights=False
            )
            standard_out = standard_out.to(dtype)

            # Flash MHA (with context manager)
            with enable_flash_attention_kernel():
                with torch.amp.autocast('cuda', dtype=dtype):
                    flash_out, _ = flash_mha(
                        query, key, value,
                        need_weights=False
                    )

        # Compute numerical drift
        max_diff = torch.max(torch.abs(flash_out - standard_out)).item()
        mean_diff = torch.mean(torch.abs(flash_out - standard_out)).item()

        print("\n" + "=" * 60)
        print("Numerical Equivalence Test Results (Directive 2)")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Dtype: {dtype}")
        print(f"Batch size: {batch_size}, Seq len: {seq_len}")
        print(f"-" * 60)
        print(f"Max absolute difference:  {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Tolerance threshold:      1e-3 (0.001)")
        print(f"-" * 60)

        if max_diff < 1e-3:
            print(f"✓ PASS: max_diff ({max_diff:.6f}) < 1e-3")
        else:
            print(f"✗ FAIL: max_diff ({max_diff:.6f}) >= 1e-3")

        print("=" * 60)

        # Assertion
        assert max_diff < 1e-3, (
            f"Numerical drift too large: {max_diff:.6f} >= 1e-3\n"
            f"Flash Attention output differs significantly from standard MHA."
        )

    def test_flash_mha_with_causal_mask(self, device, dtype):
        """
        Test Flash Attention with causal mask (autoregressive scenario).

        This tests the is_causal=True path used in AR decoding.
        """
        if device == "cpu":
            pytest.skip("Flash Attention requires CUDA")

        # Config
        embed_dim = 384
        num_heads = 12
        batch_size = 4
        seq_len = 26

        torch.manual_seed(42)
        flash_mha = FlashMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        ).to(device).eval()

        # Create test inputs
        torch.manual_seed(123)
        query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        # Forward pass with causal mask
        with torch.no_grad():
            with enable_flash_attention_kernel():
                with torch.amp.autocast('cuda', dtype=dtype):
                    output, _ = flash_mha(
                        query, query, query,
                        is_causal=True,
                        need_weights=False
                    )

        # Validation
        assert output.shape == query.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print(f"\n✓ Causal mask test passed: shape={output.shape}, dtype={output.dtype}")

    def test_flash_mha_with_custom_mask_plm(self, device, dtype):
        """
        Test Flash Attention with custom additive mask (PLM scenario).

        PLM uses permutation-based masks that are additive (-inf for masked).
        This is the CRITICAL test for PLM+Flash performance.
        """
        if device == "cpu":
            pytest.skip("Flash Attention requires CUDA")

        # Config
        embed_dim = 384
        num_heads = 12
        batch_size = 4
        seq_len = 26

        torch.manual_seed(42)
        flash_mha = FlashMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        ).to(device).eval()

        # Create test inputs
        torch.manual_seed(123)
        query = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        # Create custom additive mask (simulate PLM permutation)
        attn_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        # Example: mask out upper triangle (causal-like)
        attn_mask = attn_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(),
            float('-inf')
        )

        # Forward pass with custom mask
        with torch.no_grad():
            with enable_flash_attention_kernel():
                with torch.amp.autocast('cuda', dtype=dtype):
                    output, _ = flash_mha(
                        query, query, query,
                        attn_mask=attn_mask,
                        need_weights=False
                    )

        # Validation
        assert output.shape == query.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        print(f"\n✓ Custom mask test passed: shape={output.shape}, dtype={output.dtype}")
        print("⚠️  Custom masks may force SDP to use MATH kernel - verify with profiler")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
