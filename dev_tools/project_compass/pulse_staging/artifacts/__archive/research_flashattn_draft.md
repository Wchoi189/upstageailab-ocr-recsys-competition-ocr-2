**Flash Attention research for PARSeq Phase 3 is complete, focusing on PyTorch integration for RTX 3090.** Key findings cover API updates, precision choices, and benchmarks to guide your implementation. [docs.pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

## PyTorch Flash Attention API
PyTorch's `F.scaled_dot_product_attention` (introduced in 2.0+) integrates Flash Attention 2 automatically on supported hardware like Ampere GPUs (RTX 3090, sm_86), selecting the optimal kernel via `torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)` context manager. No major API changes noted through 2026; use keyword args like `attn_mask`, `is_causal=False` (for custom PLM masks), `scale=None` (defaults to 1/sqrt(d)), and `enable_gqa=False` unless GQA needed. Wrap in custom `FlashDecoderLayer` replacing `nn.MultiheadAttention`, with fallback: check `torch.cuda.get_device_capability() >= (8,6)`. [github](https://github.com/Dao-AILab/flash-attention/issues/190)

## Precision Recommendations
Use bfloat16 over fp16 for Flash Attention on RTX 3090; bf16 offers better numerical stability with the same tensor core throughput, avoiding fp16's gradient underflow in long-sequence PLM permutations. Enable via `torch.autocast(dtype=torch.bfloat16)`; RTX 3090 natively supports bf16 since Ampere. Acceptable drift threshold: ε ≤ 1e-3 (max abs diff) vs baseline attention, as Flash shows ~10x more deviation than eager attention at bf16 but bounded impact on weights (< low-precision training). [discuss.pytorch](https://discuss.pytorch.org/t/fp16-and-bf16-way-slower-than-fp32-and-tf32/162778)

## Mixed Precision Practices
Follow PyTorch AMP: `torch.cuda.amp.GradScaler` with autocast for forward/backward; master weights in fp32. For PLM loop, apply autocast inside permutation iterations; monitor drift with paired forward passes (Flash vs standard). RTX 3090 tip: Prefetch batches to pin_memory=True, batch_size=12-64 for seq_len~384 (PARSeq d_model). [pypi](https://pypi.org/project/flash-attn/0.2.4/)

## RTX 3090 Performance
Expect 2.5-4.5x speedup vs standard MHA at batch_size=12, heads=12, due to tiled I/O reducing GDDR6X bandwidth limits (~900GB/s). Memory savings match A100 (~50% less peak VRAM); target ≤18GB baseline for PARSeq. Throughput scales with batch_size up to VRAM limit; test img/sec on your dataset (e.g., 240-300 target). [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/62467903/be99ece8-1ecd-41fd-b6b4-40ac9b7e3a0c/2026-02-12_0316_design_session-handover-initial.md)

## Benchmarking Guidance
| Metric | Baseline (MHA) | Flash (bf16) | Notes |
|--------|----------------|--------------|-------|
| Speedup | 1x | 2.5-4.5x | Batch=12, seq=384 [pypi](https://pypi.org/project/flash-attn/0.2.4/) |
| VRAM Peak | ~18GB | ≤ baseline | 50% savings [pypi](https://pypi.org/project/flash-attn/0.2.4/) |
| Drift (ε) | 0 | ≤1e-3 | Max abs diff [arxiv](https://arxiv.org/html/2405.02803v1) |
| Throughput | Varies | Higher at large batch | Pin memory, AMP [discuss.pytorch](https://discuss.pytorch.org/t/fp16-and-bf16-way-slower-than-fp32-and-tf32/162778) |

Implement `tests/benchmarks/test_flash_attention.py` with `torch.profiler`, compare `torch.norm(flash_out - std_out) < 1e-3`; profile VRAM via `torch.cuda.max_memory_allocated()`. [docs.pytorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

**Attached file feedback:** Mostly relevant for project context (design, phases); trim future handovers to "Next Tasks" + "Technical Details" (~50% shorter) to avoid redundancy.
