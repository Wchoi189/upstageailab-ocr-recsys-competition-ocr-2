# Training Failures Debug Workspace

**Last Updated:** 2026-02-08 02:52
**Status:** ✅ GPU Training Stable

This directory contains investigation materials for training issues encountered during recognition model development.

---

## 🎯 Current Status: READY FOR PRODUCTION

### ✅ Recently Resolved (2026-02-08)

**CUDA Context Errors - RESOLVED**
- **Cause:** Docker GPU isolation incompatible with multiprocessing + pin_memory IPC
- **Solution:** `num_workers=0, pin_memory=false` (applied to global config)
- **Status:** ✅ GPU training stable (Run 48 successful)
- **Details:** [`session_handover_2026-02-08_0252.md`](file:///workspaces/__DEBUG__/training_failures/session_handover_2026-02-08_0252.md)

**Official PARSeq Integration - COMPLETE**
- **Achievement:** Integrated `baudm/parseq` permutation language modeling
- **Model:** 24.5M parameters, proven architecture
- **Files:**
  - Adapter: `ocr/domains/recognition/models/parseq_official_adapter.py`
  - Config: `configs/experiment/rec_baseline_official.yaml`
- **Status:** ✅ Ready for full training runs

---

## ✅ Previously Resolved Issues

1. **Segmentation Faults (Jan 2026)**
   - **Cause:** `pin_memory=true` with CUDA multiprocessing
   - **Archive:** [`2026-01-14_cuda_segfault_archived/`](file:///workspaces/__DEBUG__/2026-01-14_cuda_segfault_archived/)

2. **Zero Learning / 0% Accuracy (Feb 2026)**
   - **Cause:** Missing permutation language modeling logic
   - **Solution:** Official PARSeq integration (Feb 2026-02-08)

3. **Dimension Mismatch (Feb 2026)**
   - **Cause:** Encoder/decoder channel mismatch
   - **Solution:** Added `input_proj` layer

4. **Token Index Out of Bounds**
   - **Cause:** Predicted token IDs >= vocab_size
   - **Solution:** Token clamping in `architecture.py`

---

## 📁 Directory Structure

```
__DEBUG__/training_failures/
├── README.md                                    # This file
├── session_handover_2026-02-08_0252.md         # Latest session summary
├── configs/                                     # Debug configs
├── logs/                                        # Training logs
│   ├── run45_official_parseq.log               # CPU success
│   ├── run46_fixed_cuda.log                    # Failed attempts
│   ├── run47_no_pin_memory.log
│   └── run48_no_workers.log                    # ✅ GPU success
├── implementations/                             # Reference code
│   └── baudm_parseq/                           # Official PARSeq clone
├── findings/                                    # Analysis artifacts
├── scripts/                                     # Debug scripts
└── reports/                                     # Deferred investigations
```

---

## 🚀 Current Working Configuration

**Global defaults now safe** - no overrides needed!

### Quick Test
```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=15
```

### Full Training
```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=20 \
  data.batch_size=64
```

**Key Settings (in `configs/global/default.yaml`):**
- `num_workers: 0` - ✅ Required for Docker CUDA stability
- `pin_memory: false` - ✅ Required for Docker compatibility
- Precision: 16-mixed (automatic)
- Accelerator: GPU

---

## 📊 Test Results

| Run | Config | Result | Notes |
|-----|--------|--------|-------|
| 45 (CPU) | Official PARSeq | ✅ Success | Diverse predictions |
| 46-47 | Various fixes | ❌ Failed | Partial solutions |
| 48 (GPU) | num_workers=0 | ✅ **SUCCESS** | Stable GPU training |

---

## 🎓 Lessons Learned

1. **Check historical sessions first** - Jan 2026 had same fix
2. **Docker CUDA is different** - baudm/parseq config doesn't directly apply
3. **Multiprocessing + pin_memory + Docker GPU = crashes**
4. **Solution: Disable both** - Performance trade-off acceptable

---

## 📝 Next Steps

### Immediate
1. ✅ GPU training works - proceed with full runs
2. Run 20-epoch training to baseline accuracy
3. Monitor permutation LM performance

### Future (Optional)
1. Test multiprocessing on native Linux
2. Profile data loading bottleneck
3. Optimize batch size for GPU utilization

---

## 📚 Related Documentation

### Current Session
- **Handover:** [`session_handover_2026-02-08_0252.md`](file:///workspaces/__DEBUG__/training_failures/session_handover_2026-02-08_0252.md)
- **Walkthrough:** [Brain artifacts](file:///home/vscode/.gemini/antigravity/brain/5a68977d-fe39-47f2-bea0-95f735f9a519/)

### Historical
- **Jan 2026 Segfault:** [`2026-01-14_cuda_segfault_archived/`](file:///workspaces/__DEBUG__/2026-01-14_cuda_segfault_archived/)
- **Previous sessions:** [`session_handover_*.md`](file:///workspaces/__DEBUG__/training_failures/)

---

## ⚠️ Important Notes

### Don't Change
- ✅ `num_workers=0` in global config
- ✅ `pin_memory=false` in global config
- ✅ Official PARSeq adapter implementation

### Safe to Tune
- Batch size (adjust for GPU memory)
- Learning rate (experiment as needed)
- Max epochs (for convergence)

---

**Status:** 🎉 Ready for production training on GPU!
