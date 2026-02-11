# Deferred: Multiprocessing CUDA Context Investigation

> **Status:** Deferred - Not currently blocking training
> **Priority:** Low (working around with `num_workers=0`)
> **Related Log:** [`manual_test1_error.log`](file:///workspaces/__DEBUG__/training_failures/logs/manual_test1_error.log)

---

## What This Addresses

This investigation targets the **CUDA initialization error** that occurs when training with **default multiprocessing settings** (num_workers > 0).

### Symptoms
- Error: `CUDA error: initialization error` during tensor cleanup
- Occurs after Step 0 when using multiprocessing workers
- Full stacktrace shows `c10_cuda_check_implementation` failure in `ExchangeDevice`

### Current Status
✅ **Workaround in place:** Training works with `num_workers=0` and `pin_memory=false`
⏸️ **Not actively investigated:** Performance penalty accepted for now
📋 **Revisit when:** Need to optimize training speed (after model works correctly)

### Why Deferred
1. **Higher priority bugs fixed first**:
   - ✅ Encoder-decoder dimension mismatch (resolved)
   - ✅ Token index out of bounds (resolved)

2. **Workaround is stable**:
   - Single-worker training completes successfully
   - Only performance impact, not correctness

3. **Complex to debug**:
   - Requires CUDA context analysis
   - May involve PyTorch/LMDB/multiprocessing interaction
   - Time-consuming investigation

---

## Original Perplexity Analysis


This is a classic **CUDA context corruption** error during tensor cleanup after the first forward pass. Your model successfully loads, tokenizer works, and Step 0 runs (debug preds show), but the `PARSeq` forward/backward corrupts the CUDA context—subsequent tensor destructors (`~TensorImpl`) trigger `c10_cuda_check_implementation` failure in `ExchangeDevice`. [discuss.pytorch](https://discuss.pytorch.org/t/what-does-runtimeerror-cuda-driver-error-initialization-error-mean/87505)

## Root Causes (Most Likely)
1. **Tensor device mismatch** in PARSeq forward: Encoder outputs on `cuda:0`, but decoder/head or attention creates CPU tensors (e.g., `torch.eye`, masks, positional embeddings). Explains "initialization error" during dealloc—PyTorch tries recreating CUDA context for mixed-device ops. [youtube](https://www.youtube.com/watch?v=6Xm96V2YOAE)
2. **Invalid NaN/inf values** from dimension mismatch remnants: Even post-config fix, uninitialized decoder weights or attention scales produce NaNs → kernel panic → context crash. [discuss.pytorch](https://discuss.pytorch.org/t/cuda-out-of-memory-error-during-forward-pass/123423)
3. **Async kernel failure**: Custom OCR transforms/kernels (e.g., in dataset collation) launch bad CUDA code; error surfaces later during cleanup. [github](https://github.com/pytorch/pytorch/issues/67978)
4. **RTX 3090 driver quirk**: Rare with older PyTorch/CUDA on high-VRAM cards—context init fails after first heavy forward. [forums.developer.nvidia](https://forums.developer.nvidia.com/t/solved-cuda-driver-initialization-failed-2x-rtx-5090/334578)

## Immediate Fixes (Test in Order)
### 1. Force Synchronous CUDA (Pinpoint Trigger)
```bash
CUDA_LAUNCH_BLOCKING=1 uv run python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer.precision=32 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10
```
- Error will crash **exactly** at the bad kernel line (not async). [discuss.pytorch](https://discuss.pytorch.org/t/what-does-runtimeerror-cuda-driver-error-initialization-error-mean/87505)
- If still Step 0, check `PARSeq` forward for CPU tensors.

### 2. CPU-Only Debug (Isolate CUDA)
```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_v1 \
  trainer.devices=0 \
  trainer.accelerator=null \
  trainer.precision=32 \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10
```
- Succeeds? → CUDA-specific (device mismatch/kernel). [github](https://github.com/pytorch/pytorch/issues/21092)
- Fails? → Model logic (NaNs, shapes).

### 3. Enable DSA + Strict Checks
```bash
TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 uv run python ...
```
- Catches device asserts/NaNs immediately. [discuss.pytorch](https://discuss.pytorch.org/t/what-does-runtimeerror-cuda-driver-error-initialization-error-mean/87505)

### 4. Fix Common PARSeq Device Bugs
In `ocr/domains/recognition/models/parseq.py` (or equivalent), ensure **all** tensors go to device:

```python
class PARSeqDecoder(nn.Module):
    def forward(self, x):  # x from encoder: [B, C=256, H, W]
        # Global avg pool if needed
        feats = x.mean(dim=[-2, -1])  # [B, 256]
        feats = feats.to(x.device)    # Explicit!

        # Positional embeddings, masks → device!
        pos_emb = self.pos_emb[:feats.size(1)].to(feats.device)
        attn_mask = self.attn_mask.to(feats.device) if hasattr(self, 'attn_mask') else None

        # Rest of decoder...
        return decoded  # Verify no CPU ops
```

**Quick audit**:
```python
# Add to PARSeq forward end:
print(f"feats device: {feats.device}, dtype: {feats.dtype}")
print(f"Any NaN? {torch.isnan(feats).any()}")
torch.cuda.synchronize()  # Flush async errors
```

### 5. Reset CUDA Environment
```bash
# Kill lingering processes
nvidia-smi --gpu-reset  # Or reboot container

# Verify fresh init
python -c "import torch; print(torch.cuda.get_device_properties(0)); torch.randn(10).cuda()"
```
RTX 3090 reports clean? Run train. [forums.developer.nvidia](https://forums.developer.nvidia.com/t/solved-cuda-driver-initialization-failed-2x-rtx-5090/334578)

## Model-Specific Clues
- **Preds `'<<<<<<<<<<<<<<<<<<<<<<<<<'`**: `<` is likely blank/pad token (index 0?). Decoder stuck on eos/bos due to bad attention (mismatched dims → inf scales). [arxiv](https://arxiv.org/html/2503.16184v1)
- **40.5M params**: Post-fix looks good (ResNet18 ~11M + decoder ~29M).
- **No OOM**: 161MB model fine on 3090.

## Next Steps
1. Run **CUDA_LAUNCH_BLOCKING=1** → share new stacktrace (exact kernel).
2. If device mismatch suspected, grep codebase: `grep -r "torch\.eye\|torch\.arange\|torch\.zeros.*(device=" *.py`
3. Confirm config applied: Log `self.encoder.output_indices`, `self.decoder.in_channels` in model init.

This gets 95% of "initialization error" cases—usually a sneaky CPU tensor in forward. [discuss.pytorch](https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390)
