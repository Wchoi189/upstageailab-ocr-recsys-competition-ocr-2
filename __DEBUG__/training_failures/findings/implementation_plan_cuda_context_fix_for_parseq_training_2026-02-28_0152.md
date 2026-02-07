# Strategic Plan: CUDA Context Fix for PARSeq Training

**Date:** 2026-02-08 01:52  
**Status:** Planning Phase  
**Objective:** Fix CUDA initialization error by adopting proven implementations

---

## Current Status

### ✅ What Works
- Official PARSeq model architecture (24.5M params)
- Permutation language modeling logic
- Training on CPU (verified with 5 batches, diverse predictions)
- Integration with our tokenizer (PAD=0, BOS=1, EOS=2)

### ❌ What Fails
- CUDA training: `CUDA error: initialization error` during tensor cleanup
- Root cause: Docker GPU context switching during tensor deallocation
- Occurs even with `num_workers=0` (single process)

### 📊 Research Findings

**From Perplexity:**
- Issue: CUDA context switching during cleanup in Docker containers
- Specific to PyTorch + Docker + GPU isolation
- Upgrading to **PyTorch Lightning 2.2+** resolves many CUDA issues ([baudm/parseq#129](https://github.com/baudm/parseq/issues/129))

**From GitHub:**
- Official `baudm/parseq` is the gold standard reference
- Updated for PyTorch 2.0 + Lightning 2.0 compatibility
- No public forks specifically solve Docker CUDA issues

---

## Strategic Options (Prioritized)

### Option A: Use Official baudm/parseq Training Script Directly ⭐ **RECOMMENDED**

**Approach:** Clone `baudm/parseq` and use their [train.py](file:///tmp/parseq-official/train.py) instead of ours

**Why:**
- Battle-tested on thousands of GPUs
- No custom integration issues
- Proven Docker/CUDA compatibility
- We can retrofit our data format to theirs

**Steps:**
1. Clone `baudm/parseq` repository
2. Study their [train.py](file:///tmp/parseq-official/train.py) - how they initialize CUDA, handle Lightning Trainer
3. Adapt our dataset to match their expected format
4. Copy their training configuration wholesale
5. Run training using their script
6. If successful, gradually integrate learnings back to our codebase

**Timeline:** 2-4 hours (mostly data format adaptation)

**Risk:** Low - using proven code

---

### Option B: Upgrade PyTorch Lightning to 2.2+

**Approach:** Upgrade Lightning version, test if CUDA issue resolves

**Why:**
- Official PARSeq issue #129 reports Lightning 2.2 fixes CUDA compatibility
- Minimal code changes required
- Addresses root cause at framework level

**Steps:**
1. Check current Lightning version: `pip show pytorch-lightning`
2. Upgrade: `uv pip install pytorch-lightning>=2.2`
3. Test compatibility with our codebase
4. Retry CUDA training

**Timeline:** 1-2 hours

**Risk:** Medium - may break other dependencies

---

### Option C: Implement Proven CUDA Context Patterns

**Approach:** Copy specific CUDA initialization patterns from baudm/parseq

**Why:**
- Surgical fix without full script replacement
- Learn the exact pattern that works

**Patterns to Copy:**
1. **CUDA device initialization** (from baudm/parseq [train.py](file:///tmp/parseq-official/train.py)):
   ```python
   # Set device early, before any model loading
   torch.cuda.set_device(args.gpu_id)
   device = torch.device(f'cuda:{args.gpu_id}')
   ```

2. **Lightning Trainer configuration**:
   ```python
   trainer = Trainer(
       accelerator='gpu',
       devices=1,
       strategy='auto',  # Let Lightning choose
       precision='16-mixed',
   )
   ```

3. **Multiprocessing spawn method** (if baudm uses it):
   ```python
   import torch.multiprocessing as mp
   mp.set_start_method('spawn', force=True)
   ```

**Steps:**
1. Clone baudm/parseq
2. Extract CUDA-related code patterns
3. Apply to our [train.py](file:///tmp/parseq-official/train.py)
4. Test on CUDA

**Timeline:** 2-3 hours

**Risk:** Medium - may not capture all nuances

---

## Recommended Approach: Hybrid Strategy

**Phase 1: Quick Win (Option A)**
1. Clone `baudm/parseq`
2. Run their training script with their synthetic data
3. Verify CUDA works in their environment
4. **Deliverable:** Proof that CUDA training works

**Phase 2: Data Adaptation**
1. Study their dataset format
2. Create adapter to convert our LMDB data to their format
3. Run their script with our data
4. **Deliverable:** PARSeq trained on our data using their script

**Phase 3: Integration (Option C)**
1. Compare their [train.py](file:///tmp/parseq-official/train.py) with ours line-by-line
2. Identify key differences in CUDA handling
3. Port patterns back to our codebase
4. **Deliverable:** Our pipeline working with CUDA

---

## Workspace Organization

```
__DEBUG__/training_failures/
├── findings/               # Analysis artifacts
│   ├── run42_analysis.md
│   ├── run43_analysis.md
│   ├── run45_analysis.md
│   ├── perplexity_findings.md
│   └── cross_attention_analysis.md
├── implementations/        # Reference implementations
│   └── baudm_parseq/      # Clone of official repo (TBD)
├── configs/               # Working configurations
│   └── official_lightning_config.yaml (TBD)
├── logs/                  # Training logs
│   ├── run42*.log
│   ├── run43*.log
│   └── run45*.log
└── scripts/               # Diagnostic scripts
    ├── inspect_tokens.py
    └── check_visual_padding.py
```

---

## Implementation Plan (Phase 1: Quick Win)

### 1. Clone and Setup Official PARSeq

```bash
cd __DEBUG__/training_failures/implementations/
git clone https://github.com/baudm/parseq.git baudm_parseq
cd baudm_parseq
pip install -r requirements/train.txt
```

### 2. Inspect Their Training Script

**Files to review:**
- [train.py](file:///tmp/parseq-official/train.py) - Main training logic, CUDA initialization
- [strhub/models/parseq/system.py](file:///tmp/parseq-official/strhub/models/parseq/system.py) - Lightning module
- `configs/` - Training configurations
- `requirements/` - Dependency versions

**Key questions:**
- How do they initialize CUDA?
- What Lightning Trainer settings do they use?
- How do they handle multiprocessing?
- What's their PyTorch/Lightning version?

### 3. Run Their Synthetic Data Test

```bash
# Use their example command (from README)
python train.py --data synthetic --max_epochs 1 --limit_train_batches 10
```

**Expected: ✅ Training completes without CUDA errors**

### 4. Compare Dependency Versions

```bash
# Create version comparison
pip freeze > /tmp/baudm_versions.txt
cd /workspaces
pip freeze > /tmp/our_versions.txt
diff /tmp/baudm_versions.txt /tmp/our_versions.txt
```

**Focus on:**
- `pytorch-lightning` version
- `torch` version
- `timm` version

### 5. Document Findings

Create `__DEBUG__/training_failures/findings/baudm_parseq_analysis.md`:
- CUDA initialization pattern
- Lightning Trainer configuration
- Dependency versions
- Any Docker-specific setup

### 6. Adapt Our Data (If Phase 1 Successful)

Study their data pipeline:
- `strhub/data/module.py` - DataModule structure
- Expected data format (LMDB structure)
- Tokenizer interface

Create adapter:
- `__DEBUG__/training_failures/scripts/adapt_data_to_baudm.py`

---

## Verification Plan

### Test 1: Official PARSeq Synthetic Data (Baseline)
**Command:**
```bash
cd __DEBUG__/training_failures/implementations/baudm_parseq
python train.py --data synthetic --max_epochs 1 --limit_train_batches 10 --accelerator gpu
```

**Expected:**
- ✅ Training starts without errors
- ✅ CUDA is utilized
- ✅ Completes 10 batches
- ✅ No tensor cleanup errors

**Pass Criteria:** Training completes, logs show GPU usage

### Test 2: Official PARSeq with Real Data
**Command:**  
```bash
python train.py --data <their_format> --max_epochs 1 --limit_train_batches 20
```

**Expected:**
- ✅ Loads real data successfully
- ✅ Trains without CUDA errors

### Test 3: Our Pipeline with Copied Patterns
**Command:**
```bash
cd /workspaces
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.accelerator=gpu \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  data.num_workers=0
```

**Expected:**
- ✅ Training works on CUDA
- ✅ No context switching errors

**Pass Criteria:** All 3 tests pass

---

## Rollback Plan

If Option A fails:
1. Fall back to CPU training for immediate results
2. Pursue Option B (Lightning upgrade)
3. If all fails: Use CPU for training, GPU for inference only

---

## Success Metrics

**Immediate (Phase 1):**
- [ ] Official PARSeq trains successfully on our GPU
- [ ] Identified exact CUDA initialization pattern
- [ ] Version differences documented

**Short-term (Phase 2):**
- [ ] PARSeq trains on our data using their script
- [ ] Verified permutation LM fixes 0% accuracy issue

**Long-term (Phase 3):**
- [ ] Our pipeline works with CUDA
- [ ] Clean integration of proven patterns
- [ ] Documentation for future reference

---

## Next Steps

1. **Immediate:** Clone baudm/parseq to `__DEBUG__/training_failures/implementations/`
2. **Run Test 1:** Verify their script works on our hardware
3. **Document:** Create findings artifact with their CUDA patterns
4. **Decision Point:** If Test 1 passes, proceed to Phase 2; if fails, pursue Option B

**Estimated Time:** 4-6 hours total for all phases
