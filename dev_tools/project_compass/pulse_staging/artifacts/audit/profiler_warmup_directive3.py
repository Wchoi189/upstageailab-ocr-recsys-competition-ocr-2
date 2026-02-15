#!/usr/bin/env python3
"""
Warmup Profiler - Perplexity Directive 3

Profiles training loop to identify:
1. JIT compilation overhead (first few steps)
2. Kernel execution time vs Python overhead
3. Whether FLASH or MATH backend is actually used

Usage:
    uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/profiler_warmup_directive3.py

Output:
    - Console trace showing kernel times
    - Chrome trace JSON for visualization (chrome://tracing)
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule

from ocr.domains.recognition.models.decoder import PARSeqDecoder
from ocr.domains.recognition.models.flash_attention import enable_flash_attention_kernel


def profile_flash_attention_warmup():
    """Profile Flash Attention with warmup to identify compilation overhead."""

    print("=" * 80)
    print("Flash Attention Warmup Profiler (Directive 3)")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16

    # PARSeq decoder config
    decoder_config = {
        "in_channels": 384,
        "d_model": 384,
        "nhead": 12,
        "num_layers": 12,  # Full decoder
        "dim_feedforward": 1536,
        "dropout": 0.0,  # Disable for profiling
        "vocab_size": 100,
        "max_len": 25,
        "use_flash_attention": True,
    }

    print("\nInitializing decoder...")
    decoder = PARSeqDecoder(**decoder_config).to(device).eval()

    # Create sample batch
    batch_size = 64
    seq_len_encoder = 384  # 12x32 visual features
    seq_len_decoder = 26   # max_len + 1

    features = torch.randn(batch_size, seq_len_encoder, 384, device=device, dtype=dtype)
    targets = torch.randint(1, 100, (batch_size, seq_len_decoder), device=device)

    print(f"Batch size: {batch_size}")
    print(f"Seq len: {seq_len_decoder}")
    print(f"Dtype: {dtype}")

    # Profiler configuration
    # - wait=0: Start profiling immediately
    # - warmup=5: First 5 steps are warmup (JIT compilation happens here)
    # - active=10: Profile next 10 steps (should be faster after warmup)
    # - repeat=1: Run once
    prof_schedule = schedule(wait=0, warmup=5, active=10, repeat=1)

    print("\n" + "-" * 80)
    print("Starting profiler...")
    print("  Warmup: 5 steps (JIT compilation)")
    print("  Active: 10 steps (profiled)")
    print("-" * 80)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            with enable_flash_attention_kernel():
                with torch.amp.autocast('cuda', dtype=dtype):
                    for step in range(15):  # 5 warmup + 10 active
                        _ = decoder(features, targets=targets)
                        prof.step()  # Signal profiler to move to next step

                        if step < 5:
                            print(f"  Warmup step {step + 1}/5")
                        else:
                            print(f"  Active step {step - 4}/10")

    print("\n" + "-" * 80)
    print("Profiling complete!")
    print("-" * 80)

    # Print key averages
    print("\nTop 10 CUDA kernels by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nTop 10 CPU operations by total time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # Export Chrome trace
    trace_file = "./profiler_logs/flash_attention_warmup_trace.json"
    prof.export_chrome_trace(trace_file)
    print(f"\n✓ Chrome trace exported to: {trace_file}")
    print("  Open chrome://tracing and load this file for visualization")

    # Search for Flash Attention kernel in trace
    print("\n" + "=" * 80)
    print("Analyzing kernel usage...")
    print("=" * 80)

    events = prof.key_averages()
    flash_kernels = [e for e in events if 'flash' in e.key.lower() or 'sdpa' in e.key.lower()]
    math_kernels = [e for e in events if 'gemm' in e.key.lower() or 'matmul' in e.key.lower()]

    if flash_kernels:
        print("✓ Flash Attention kernels detected:")
        for kernel in flash_kernels[:5]:
            print(f"  - {kernel.key}: {kernel.cuda_time_total / 1000:.2f} ms")
    else:
        print("⚠️  No Flash Attention kernels detected in trace")
        print("    This suggests SDP may be using MATH or MEM_EFF backend")

    if math_kernels:
        print("\nMatrix multiplication kernels (MATH backend?):")
        for kernel in math_kernels[:5]:
            print(f"  - {kernel.key}: {kernel.cuda_time_total / 1000:.2f} ms")

    print("\n" + "=" * 80)
    print("Recommendations:")
    print("=" * 80)
    print("1. Check if Flash kernels appear in trace")
    print("2. Compare warmup vs active step times")
    print("3. If MATH backend used, custom masks may be forcing fallback")
    print("4. Run longer benchmark (1000 steps) to amortize compilation cost")
    print("=" * 80)


if __name__ == "__main__":
    profile_flash_attention_warmup()
