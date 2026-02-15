#!/usr/bin/env python3
"""
Backend Confirmation Diagnostic - Perplexity Directive 1

Logs which SDP backend is actually selected during forward pass.
Expected: FLASH on RTX 3090 (sm_86)
Problem: If MATH is selected, no speedup will be achieved.

Usage:
    uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/diagnostic_backend_check.py
"""

import torch
import torch.nn.functional as F
from ocr.domains.recognition.models.flash_attention import (
    check_flash_attention_support,
    enable_flash_attention_kernel,
    FlashDecoderLayer,
)


def check_sdp_backend():
    """Check which backend F.scaled_dot_product_attention selects."""
    print("=" * 60)
    print("SDP Backend Diagnostic")
    print("=" * 60)

    # Check device capability
    supported, message = check_flash_attention_support()
    print(f"Flash Support: {supported} - {message}")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - Flash Attention requires CUDA")
        return

    device = "cuda"
    dtype = torch.bfloat16

    # Create test inputs
    B, H, L, S, D = 2, 12, 26, 26, 32  # batch, heads, query_len, key_len, head_dim
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    print("\n" + "-" * 60)
    print("Test 1: Default SDP (no context manager)")
    print("-" * 60)

    # Check default backend selection
    with torch.amp.autocast('cuda', dtype=dtype):
        try:
            # PyTorch 2.0+ has sdp_kernel_* functions to check backend
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True
            ):
                _ = F.scaled_dot_product_attention(q, k, v)

            # Try to introspect which backend was used
            # NOTE: PyTorch doesn't expose this directly, so we check implicitly
            print("✓ SDP executed successfully")
            print("⚠️  Backend selection is automatic - could be FLASH, MATH, or MEM_EFF")
            print("    Without profiling, we cannot determine which backend was used")

        except Exception as e:
            print(f"✗ SDP failed: {e}")

    print("\n" + "-" * 60)
    print("Test 2: SDP with enable_flash_attention_kernel() context")
    print("-" * 60)

    with torch.amp.autocast('cuda', dtype=dtype):
        try:
            with enable_flash_attention_kernel():
                _ = F.scaled_dot_product_attention(q, k, v)
            print("✓ SDP with Flash-only context executed successfully")
            print("✓ FLASH backend should be forced (MATH/MEM_EFF disabled)")

        except Exception as e:
            print(f"✗ SDP with Flash context failed: {e}")
            print("    This suggests Flash Attention is not available")

    print("\n" + "-" * 60)
    print("Test 3: Check with attention mask (PLM scenario)")
    print("-" * 60)

    # PLM uses custom additive masks
    attn_mask = torch.zeros(L, S, device=device, dtype=dtype)
    attn_mask[5:, 5:] = float('-inf')  # Example causal-like mask

    with torch.amp.autocast('cuda', dtype=dtype):
        try:
            with enable_flash_attention_kernel():
                _ = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            print("✓ SDP with custom mask executed successfully")
            print("⚠️  Custom masks may force fallback to MATH kernel")
            print("    Need profiling to confirm actual backend")

        except Exception as e:
            print(f"✗ SDP with custom mask failed: {e}")

    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print("1. Run training with torch.profiler to confirm backend")
    print("2. Ensure enable_flash_attention_kernel() is used in training loop")
    print("3. Verify custom PLM masks don't force MATH fallback")
    print("4. Consider using is_causal=True instead of custom masks if possible")
    print("=" * 60)


if __name__ == "__main__":
    check_sdp_backend()
