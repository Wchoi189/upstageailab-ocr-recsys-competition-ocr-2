# Implementation Plan: Integrate Official PARSeq Architecture

**Date:** 2026-02-08 00:54
**Status:** 🔴 CRITICAL - Our Implementation is Fundamentally Incomplete

---

## Executive Summary

After comparing our implementation with the official `baudm/parseq` repository, I've identified that **our PARSeq implementation is missing CORE components** that are essential to the permutation language modeling training strategy. This explains the 0% accuracy completely.

### Critical Missing Components

1. **❌ Permutation Generation (`gen_tgt_perms`)** - NOT IMPLEMENTED
2. **❌ Attention Mask Generation (`generate_attn_masks`)** - NOT IMPLEMENTED
3. **❌ Multi-Permutation Training Loop** - NOT IMPLEMENTED

Our code appears to implement a standard transformer decoder with cross-attention, but PARSeq requires **permutation language modeling** which we completely lack.

---

## Official vs Our Implementation

### Official Implementation Structure

```
strhub/models/parseq/
├── system.py       # Lightning module with permutation logic
├── model.py        # Core PARSeq model
└── modules.py      # Decoder, Encoder components
```

**system.py (Lines 90-200):**
- `gen_tgt_perms()`: Generates K permutations per batch
- `generate_attn_masks()`: Creates permutation-specific attention masks
- `training_step()`: Iterates over permutations, aggregates loss

**model.py (Lines 86-103):**
- `decode()`: Accepts `tgt_query_mask` for permutation-specific masking
- Proper integration of positional queries + text embeddings
- Dropout applied to both streams

### Our Implementation

**`ocr/domains/recognition/models/architecture.py`:**
- Has encoder (ResNet) ✓
- Has decoder (transformer) ✓
- **MISSING:** Permutation logic ❌
- **MISSING:** Attention mask generation ❌

**`ocr/domains/recognition/module.py`:**
- Has standard training_step ✓
- **MISSING:** Multi-permutation training loop ❌
- **MISSING:** `gen_tgt_perms()` ❌

---

## Root Cause Analysis

### Why Training Failed (0% Accuracy)

**The problem:** We trained a standard autoregressive decoder, NOT a permutation language model.

1. **No permutation sampling** → Model only sees left-to-right order
2. **No permutation-specific masks** → Model gets standard causal mask
3. **No multi-permutation loss** → Model doesn't learn permutation-invariant representations

**Result:** Model learns a weak language model without proper visual grounding, collapses to unigram prior.

---

## Proposed Solution

### Option A: Adopt Official PARSeq Wholesale **(RECOMMENDED)**

**Rationale:**
- Official implementation is battle-tested
- ~200 lines of complex permutation logic we'd need to reimplement
- Our custom components (data pipeline, SVTR plans) can integrate with official model

**Changes:**
1. Add `strhub` directory from official repo to our codebase
2. Create adapter in `ocr/domains/recognition/models/` that wraps official PARSeq
3. Keep our data pipeline

, tokenizer, and training infrastructure
4. Replace our broken decoder logic with official implementation

**Pros:**
- ✅ Proven to work
- ✅ Maintains compatibility with future SVTR plans
- ✅ Minimal changes to our data pipeline

**Cons:**
- ⚠️ Adds external dependency (strhub package)
- ⚠️ Some code duplication

### Option B: Reimplement Missing Components

**NOT RECOMMENDED** - Too complex, error-prone

---

## Implementation Plan (Option A)

### Phase 1: Integrate Official PARSeq

#### 1.1 Add Official Code
- [ ] Copy `strhub/models/parseq/` directory to our codebase
- [ ] Copy required base classes (`strhub/models/base.py`, `strhub/data/utils.py`)
- [ ] Add as submodule OR vendored dependency

#### 1.2 Create Adapter Layer
**File:** `ocr/domains/recognition/models/parseq_official.py`

```python
from strhub.models.parseq.system import PARSeq as OfficialPARSeq
from strhub.models.parseq.model import PARSeq as OfficialModel

class PARSeqAdapter(nn.Module):
    \"\"\"Adapter to use official PARSeq with our pipeline\"\"\"

    def __init__(self, cfg):
        super().__init__()
        # Extract config params
        self.model = OfficialPARSeq(
            charset_train=cfg.charset,
            max_label_length=cfg.max_len,
            # ... map our config to official config
        )

    def forward(self, images, text_tokens=None, **kwargs):
        # Adapt our interface to official interface
        if text_tokens is not None:
            # Training mode
            return self.model.training_step((images, text_tokens), 0)
        else:
            # Inference mode
            return self.model(images)
```

#### 1.3 Update Configuration
**File:** `configs/model/architectures/parseq_official.yaml`

```yaml
_target_: ocr.domains.recognition.models.parseq_official.PARSeqAdapter
charset: ${data.charset}
max_len: 25
img_size: [32, 128]
patch_size: [4, 8]
embed_dim: 384
enc_num_heads: 6
enc_depth: 12
dec_num_heads: 12
dec_depth: 1
perm_num: 6  # Number of permutations to sample
dropout: 0.1
```

#### 1.4 Update Module
**File:** `ocr/domains/recognition/module.py`

Modify `training_step()` to work with official PARSeq's interface:
- Remove our broken permutation-less training
- Let official PARSeq handle permutations internally
- Just pass batch to `model.training_step()`

---

### Phase 2: Verification

#### 2.1 Micro-Training Test (Run 45)
```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_official \
  model=parseq_official \
  trainer.max_epochs=3 \
  trainer.limit_train_batches=20 \
  trainer.limit_val_batches=5 \
  data.batch_size=32 \
  +run_name="run45_official_parseq"
```

**Expected Results:**
- ✅ Non-zero validation accuracy (>5%)
- ✅ Gradients remain non-zero throughout training
- ✅ Diverse predictions (not just BOS+EOS or repeated chars)

#### 2.2 Gradient Verification
Add logging to confirm permutations are being sampled:
```python
# In training_step
print(f"Number of permutations: {len(tgt_perms)}")
print(f"First perm: {tgt_perms[0]}")
print(f"Loss components: {loss_numel}")
```

#### 2.3 Full Training Run (If Micro-Test Passes)
```bash
uv run python scripts/runners/train.py \
  experiment=rec_baseline_official \
  model=parseq_official \
  trainer.max_epochs=100 \
  data.batch_size=256 \
  +run_name="run46_official_full"
```

---

## Architecture Compatibility for SVTR

The official PARSeq uses a Vision Transformer encoder (`Encoder` class). When we implement SVTR:

1. Replace `self.encoder = Encoder(...)` with `self.encoder = SVTR(...)`
2. Keep the official decoder and permutation logic
3. SVTR outputs compatible features for the decoder

**No conflicts** - the encoder is modular.

---

## Files to Modify

### New Files
- `ocr/domains/recognition/models/parseq_official.py` - Adapter
- `configs/model/architectures/parseq_official.yaml` - Config
- `configs/experiment/rec_baseline_official.yaml` - Experiment config
- `strhub/` - Vendored official code (or git submodule)

### Modified Files
- `ocr/domains/recognition/module.py` - Update training_step interface
- `ocr/core/data/tokenizer.py` - Ensure compatibility with official tokenizer expectations

### Deprecated Files (keep for reference)
- `ocr/domains/recognition/models/architecture.py` - Our broken implementation
- `ocr/domains/recognition/models/decoder.py` - Our broken decoder

---

## Timeline

**Phase 1 (Integration):** 2-3 hours
- Copy official code
- Write adapter
- Update configs

**Phase 2 (Verification):** 1 hour
- Run micro-training test
- Verify gradients and metrics

**Total:** ~4 hours to working implementation

---

## Alternative: Minimal Fix (NOT RECOMMENDED)

If we want to keep our implementation and just add permutation support:

1. Port `gen_tgt_perms()` from official (60 lines)
2. Port `generate_attn_masks()` from official (15 lines)
3. Update `training_step()` to loop over permutations (30 lines)

**Risk:** High chance of subtle bugs in porting complex logic

---

## Recommendation

✅ **Proceed with Option A (Adopt Official PARSeq)**

**Justification:**
1. Our implementation is fundamentally incomplete
2. Official code is proven and battle-tested
3. Maintains flexibility for SVTR integration
4. Faster path to working model (~4 hours vs unknown debugging time)
5. Reduces maintenance burden (upstream fixes benefit us)

**Next Step:** Request approval to integrate official PARSeq implementation.
