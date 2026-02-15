#!/usr/bin/env python3
"""
Comprehensive Performance Audit - Phase 6.3

Executes Perplexity Directives 3 & 8 (HIGH priority):
1. Warmup profiling with torch.profiler
2. Long-run benchmark (1000 steps)
3. Batch/sequence sweep to find optimal config

Usage:
    # Run full audit
    uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/comprehensive_performance_audit.py

    # Run specific test
    uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/comprehensive_performance_audit.py --test warmup
    uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/comprehensive_performance_audit.py --test longrun
    uv run python dev_tools/project_compass/pulse_staging/artifacts/audit/comprehensive_performance_audit.py --test sweep

Output:
    - profiler_logs/phase6.3_results.json - Full results
    - profiler_logs/phase6.3_heatmap.png - Speedup heatmap
    - Console summary with recommendations
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule

from ocr.domains.recognition.models.architecture import PARSeqModel
from ocr.domains.recognition.models.flash_attention import (
    enable_flash_attention_kernel,
    check_flash_attention_support,
)


def create_model(use_flash: bool = False, plm: bool = False, device: str = "cuda") -> nn.Module:
    """Create PARSeq model with specified configuration."""
    plm_config = None
    if plm:
        plm_config = {
            "max_label_length": 25,
            "perm_num": 6,
            "perm_forward": True,
            "perm_mirrored": True,
        }

    model = PARSeqModel(
        d_model=384,
        nhead=12,
        num_layers=12,
        vocab_size=100,
        max_len=25,
        use_flash_attention=use_flash,
        plm_config=plm_config,
    )
    model = model.to(device).eval()
    return model


def create_batch(batch_size: int, seq_len: int, device: str = "cuda") -> Dict:
    """Create synthetic batch for benchmarking."""
    # Synthetic images: [B, 3, 32, 128]
    images = torch.randn(batch_size, 3, 32, 128, device=device)

    # Synthetic text tokens: [B, seq_len]
    # Pad to seq_len (usually 25, but can be extended)
    text_tokens = torch.randint(1, 99, (batch_size, seq_len), device=device)

    return {"images": images, "text_tokens": text_tokens}


def warmup_profiling(output_dir: Path) -> Dict:
    """
    Perplexity Directive 3: Warmup Profiler
    Profile training loop to identify JIT compilation overhead.
    """
    print("\n" + "=" * 80)
    print("DIRECTIVE 3: Warmup Profiling")
    print("=" * 80)

    supported, msg = check_flash_attention_support()
    print(f"Flash Attention: {msg}")

    if not supported:
        print("⚠️  Flash Attention not supported, skipping...")
        return {"status": "skipped", "reason": msg}

    device = "cuda"
    dtype = torch.bfloat16

    # Test configurations
    configs = [
        ("baseline", False),
        ("flash", True),
    ]

    results = {}

    for name, use_flash in configs:
        print(f"\n{'─' * 80}")
        print(f"Testing: {name.upper()}")
        print(f"{'─' * 80}")

        model = create_model(use_flash=use_flash, plm=False, device=device)
        batch = create_batch(batch_size=64, seq_len=26, device=device)

        # Profiler schedule: 5 warmup + 10 active
        prof_schedule = schedule(wait=0, warmup=5, active=10, repeat=1)

        step_times = []

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "warmup")),
            record_shapes=False,
            profile_memory=True,
        ) as prof:
            with torch.no_grad():
                if use_flash:
                    ctx = enable_flash_attention_kernel()
                else:
                    ctx = torch.cuda.amp.autocast(dtype=dtype)

                with ctx:
                    for step in range(15):
                        start = time.perf_counter()
                        _ = model(**batch)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000  # ms

                        step_times.append(elapsed)
                        prof.step()

                        phase = "Warmup" if step < 5 else "Active"
                        print(f"  {phase} step {step + 1}: {elapsed:.2f} ms")

        # Analyze results
        warmup_avg = sum(step_times[:5]) / 5
        active_avg = sum(step_times[5:]) / 10

        print(f"\nResults:")
        print(f"  Warmup average: {warmup_avg:.2f} ms")
        print(f"  Active average: {active_avg:.2f} ms")
        print(f"  Overhead: {((warmup_avg - active_avg) / active_avg * 100):.1f}%")

        results[name] = {
            "warmup_avg_ms": warmup_avg,
            "active_avg_ms": active_avg,
            "overhead_pct": (warmup_avg - active_avg) / active_avg * 100,
            "step_times": step_times,
        }

    # Compare Flash vs Baseline
    if "baseline" in results and "flash" in results:
        speedup = results["baseline"]["active_avg_ms"] / results["flash"]["active_avg_ms"]
        print(f"\n{'=' * 80}")
        print(f"Flash Attention Speedup (warmup phase): {speedup:.2f}x")
        print(f"{'=' * 80}")

    return results


def long_run_benchmark(output_dir: Path) -> Dict:
    """
    Perplexity Directive 8: Long-Run Benchmark
    Run 1000 steps with larger batch/sequence to prove speedup at scale.
    """
    print("\n" + "=" * 80)
    print("DIRECTIVE 8: Long-Run Benchmark (1000 steps)")
    print("=" * 80)

    supported, msg = check_flash_attention_support()
    if not supported:
        print("⚠️  Flash Attention not supported, skipping...")
        return {"status": "skipped", "reason": msg}

    device = "cuda"
    dtype = torch.bfloat16

    # Configurations for long-run test
    test_configs = [
        ("baseline_64", False, 64, 26),
        ("flash_64", True, 64, 26),
        ("baseline_128", False, 128, 26),
        ("flash_128", True, 128, 26),
    ]

    results = {}

    for name, use_flash, batch_size, seq_len in test_configs:
        print(f"\n{'─' * 80}")
        print(f"Testing: {name}")
        print(f"  Batch: {batch_size}, Seq: {seq_len}")
        print(f"{'─' * 80}")

        model = create_model(use_flash=use_flash, plm=False, device=device)
        batch = create_batch(batch_size=batch_size, seq_len=seq_len, device=device)

        num_steps = 1000
        warmup_steps = 50  # Warmup for JIT

        step_times = []

        with torch.no_grad():
            if use_flash:
                ctx = enable_flash_attention_kernel()
            else:
                ctx = torch.cuda.amp.autocast(dtype=dtype)

            with ctx:
                # Warmup
                for _ in range(warmup_steps):
                    _ = model(**batch)

                torch.cuda.synchronize()

                # Timed run
                start_time = time.perf_counter()

                for step in range(num_steps):
                    step_start = time.perf_counter()
                    _ = model(**batch)
                    torch.cuda.synchronize()
                    step_elapsed = (time.perf_counter() - step_start) * 1000

                    step_times.append(step_elapsed)

                    if (step + 1) % 100 == 0:
                        recent_avg = sum(step_times[-100:]) / 100
                        print(f"  Step {step + 1}/{num_steps}: {recent_avg:.2f} ms/step (avg last 100)")

                total_time = time.perf_counter() - start_time

        # Calculate metrics
        avg_step_time = sum(step_times) / len(step_times)
        throughput = (batch_size * num_steps) / total_time  # samples/sec

        print(f"\nResults:")
        print(f"  Total time: {total_time:.2f} s")
        print(f"  Avg step time: {avg_step_time:.2f} ms")
        print(f"  Throughput: {throughput:.1f} samples/sec")

        results[name] = {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_steps": num_steps,
            "total_time_s": total_time,
            "avg_step_ms": avg_step_time,
            "throughput_samples_per_sec": throughput,
            "step_times": step_times,
        }

    # Compare Flash vs Baseline
    print(f"\n{'=' * 80}")
    print("Long-Run Speedup Analysis")
    print(f"{'=' * 80}")

    for batch_size in [64, 128]:
        baseline_key = f"baseline_{batch_size}"
        flash_key = f"flash_{batch_size}"

        if baseline_key in results and flash_key in results:
            baseline_tput = results[baseline_key]["throughput_samples_per_sec"]
            flash_tput = results[flash_key]["throughput_samples_per_sec"]
            speedup = flash_tput / baseline_tput

            print(f"Batch {batch_size}: {speedup:.2f}x speedup ({baseline_tput:.1f} → {flash_tput:.1f} samples/sec)")

    return results


def batch_sequence_sweep(output_dir: Path) -> Dict:
    """
    Batch/Sequence Sweep
    Test different batch sizes and sequence lengths to find optimal config.
    """
    print("\n" + "=" * 80)
    print("Batch/Sequence Sweep")
    print("=" * 80)

    supported, msg = check_flash_attention_support()
    if not supported:
        print("⚠️  Flash Attention not supported, skipping...")
        return {"status": "skipped", "reason": msg}

    device = "cuda"
    dtype = torch.bfloat16

    # Test grid
    batch_sizes = [16, 32, 64, 128]
    seq_lens = [26, 64, 128]  # PARSeq default=26, extended for Flash
    num_steps = 100

    results = {"baseline": {}, "flash": {}}

    for use_flash in [False, True]:
        mode = "flash" if use_flash else "baseline"
        print(f"\n{'─' * 80}")
        print(f"Testing: {mode.upper()}")
        print(f"{'─' * 80}")

        model = create_model(use_flash=use_flash, plm=False, device=device)

        for batch_size in batch_sizes:
            results[mode][batch_size] = {}

            for seq_len in seq_lens:
                print(f"  Batch={batch_size}, Seq={seq_len}...", end=" ", flush=True)

                try:
                    batch = create_batch(batch_size=batch_size, seq_len=seq_len, device=device)

                    with torch.no_grad():
                        if use_flash:
                            ctx = enable_flash_attention_kernel()
                        else:
                            ctx = torch.cuda.amp.autocast(dtype=dtype)

                        with ctx:
                            # Warmup
                            for _ in range(10):
                                _ = model(**batch)

                            torch.cuda.synchronize()

                            # Timed run
                            start_time = time.perf_counter()

                            for _ in range(num_steps):
                                _ = model(**batch)

                            torch.cuda.synchronize()
                            elapsed = time.perf_counter() - start_time

                    throughput = (batch_size * num_steps) / elapsed
                    avg_step = (elapsed / num_steps) * 1000

                    print(f"{throughput:.1f} samples/sec")

                    results[mode][batch_size][seq_len] = {
                        "throughput": throughput,
                        "avg_step_ms": avg_step,
                    }

                except RuntimeError as e:
                    print(f"OOM")
                    results[mode][batch_size][seq_len] = {"error": str(e)}

    # Calculate speedup heatmap
    print(f"\n{'=' * 80}")
    print("Speedup Matrix (Flash / Baseline)")
    print(f"{'=' * 80}")
    print(f"{'Batch':>6} | ", end="")
    for seq_len in seq_lens:
        print(f"Seq={seq_len:>3} | ", end="")
    print()
    print("─" * 50)

    for batch_size in batch_sizes:
        print(f"{batch_size:>6} | ", end="")
        for seq_len in seq_lens:
            baseline_data = results["baseline"][batch_size].get(seq_len)
            flash_data = results["flash"][batch_size].get(seq_len)

            if baseline_data and flash_data and "throughput" in baseline_data and "throughput" in flash_data:
                speedup = flash_data["throughput"] / baseline_data["throughput"]
                print(f" {speedup:>5.2f}x | ", end="")
            else:
                print(f"   N/A | ", end="")
        print()

    return results


def main():
    """Run comprehensive performance audit."""
    parser = argparse.ArgumentParser(description="Phase 6.3 Performance Audit")
    parser.add_argument(
        "--test",
        choices=["warmup", "longrun", "sweep", "all"],
        default="all",
        help="Which test to run (default: all)",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path("profiler_logs")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("Phase 6.3: Performance Analysis")
    print("Comprehensive Audit for Flash Attention Optimization")
    print("=" * 80)

    all_results = {}

    # Run tests
    if args.test in ["warmup", "all"]:
        all_results["warmup"] = warmup_profiling(output_dir)

    if args.test in ["longrun", "all"]:
        all_results["longrun"] = long_run_benchmark(output_dir)

    if args.test in ["sweep", "all"]:
        all_results["sweep"] = batch_sequence_sweep(output_dir)

    # Save results
    results_file = output_dir / "phase6.3_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 80}")

    # Summary
    print("\nKEY FINDINGS:")
    print("-" * 80)

    if "warmup" in all_results:
        warmup = all_results["warmup"]
        if "baseline" in warmup and "flash" in warmup:
            speedup = warmup["baseline"]["active_avg_ms"] / warmup["flash"]["active_avg_ms"]
            print(f"1. Warmup Phase Speedup: {speedup:.2f}x")
            print(f"   - Baseline: {warmup['baseline']['active_avg_ms']:.2f} ms/step")
            print(f"   - Flash: {warmup['flash']['active_avg_ms']:.2f} ms/step")

    if "longrun" in all_results:
        longrun = all_results["longrun"]
        if "baseline_128" in longrun and "flash_128" in longrun:
            baseline_tput = longrun["baseline_128"]["throughput_samples_per_sec"]
            flash_tput = longrun["flash_128"]["throughput_samples_per_sec"]
            speedup = flash_tput / baseline_tput
            print(f"\n2. Long-Run Speedup (batch=128, 1000 steps): {speedup:.2f}x")
            print(f"   - Baseline: {baseline_tput:.1f} samples/sec")
            print(f"   - Flash: {flash_tput:.1f} samples/sec")

    if "sweep" in all_results:
        print(f"\n3. Optimal Configuration:")
        print(f"   See speedup matrix above for batch/sequence combinations")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
