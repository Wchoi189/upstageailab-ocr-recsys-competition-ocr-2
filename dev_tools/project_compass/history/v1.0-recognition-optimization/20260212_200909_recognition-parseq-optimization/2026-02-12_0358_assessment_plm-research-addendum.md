# Research Addendum: Additional Findings

**Date**: 2026-02-12
**Source**: Perplexity Web UI + Local Source Code Analysis
**Status**: High-Value Additional Context

---

## Critical Discovery: Official Source Code in Workspace

### Location
**Path**: `__DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/strhub/models/parseq/system.py`

This is the **authoritative reference** for PLM implementation.

### Exact Implementation Details

#### 1. gen_tgt_perms() (Lines 90-151)

**Key Insights from Official Code**:

```python
def gen_tgt_perms(self, tgt):
    """Generate shared permutations for the whole batch.
    This works because the same attention mask can be used for the shorter sequences
    because of the padding mask.
    """
    # Lines 95-99: Special case for 1-character sequences
    max_num_chars = tgt.shape[1] - 2
    if max_num_chars == 1:
        return torch.arange(3, device=self._device).unsqueeze(0)

    # Lines 100-130: Permutation generation logic
    perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []

    # Lines 108-125: Special handling for ≤4 characters
    if max_num_chars < 5:
        if max_num_chars == 4 and self.perm_mirrored:
            selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
        # ... pool sampling logic ...

    # Lines 131-135: Mirrored pairs
    if self.perm_mirrored:
        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

    # Lines 142-145: BOS/EOS indices
    bos_idx = perms.new_zeros((len(perms), 1))
    eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
    perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)

    # Lines 146-150: CRITICAL - Reverse direction special handling
    if len(perms) > 1:
        perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)

    return perms
```

**NEW INSIGHT - Lines 146-150**: Special reverse direction handling!
- Not just complementary flip
- Actually creates specific reverse pattern: `max_num_chars + 1 - torch.arange(...)`
- Required for NAR mode EOS prediction with null context

**CRITICAL**: Our parseq_official_adapter.py might be missing this special case!

#### 2. generate_attn_masks() (Lines 153-167)

```python
def generate_attn_masks(self, perm):
    """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
    :param perm: the permutation sequence. i = 0 is always the BOS
    :return: lookahead attention masks
    """
    sz = perm.shape[0]
    mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)
    for i in range(sz):
        query_idx = perm[i]
        masked_keys = perm[i + 1 :]  # NOTE: Python slice, not torch
        mask[query_idx, masked_keys] = True
    content_mask = mask[:-1, :-1].clone()
    mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True  # mask "self"
    query_mask = mask[1:, :-1]
    return content_mask, query_mask
```

**Implementation Note**: Uses Python slice `perm[i + 1:]` (not `perm[i + 1 :]` with space), but functionally identical.

#### 3. training_step() - Complete PLM Loop (Lines 169-200)

```python
def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
    images, labels = batch
    tgt = self.tokenizer.encode(labels, self._device)

    # Encode the source sequence (i.e. the image codes)
    memory = self.model.encode(images)

    # Prepare the target sequences (input and output)
    tgt_perms = self.gen_tgt_perms(tgt)
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    # The [EOS] token is not depended upon by any other token in any permutation ordering
    tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

    loss = 0
    loss_numel = 0
    n = (tgt_out != self.pad_id).sum().item()

    for i, perm in enumerate(tgt_perms):
        tgt_mask, query_mask = self.generate_attn_masks(perm)
        out = self.model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
        logits = self.model.head(out).flatten(end_dim=1)
        loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
        loss_numel += n

        # After the second iteration (i.e. done with canonical and reverse orderings),
        # remove the [EOS] tokens for the succeeding perms
        if i == 1:
            tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
            n = (tgt_out != self.pad_id).sum().item()

    loss /= loss_numel

    self.log('loss', loss)
    return loss
```

**KEY INSIGHT - Line 181**: EOS padding mask!
```python
tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
```

EOS is treated as padding in the input! This prevents the decoder from using EOS as context.

---

## Successful Reference Implementation: dilithjay/Sinhala-ParSeq

### Source
- **Repository**: https://github.com/dilithjay/Sinhala-ParSeq
- **Blog Guide**: https://dilithjay.com/blog/parseq-train-a-custom-model
- **Status**: Production-ready fork with custom language support

### Key Adaptations

1. **Custom Charset Configuration**:
   - Demonstrates how to adapt PARSeq for non-English languages
   - Handles complex character mappings (Sinhala characters don't always correspond 1:1 with visual glyphs)
   - Proves architecture is flexible for custom tokenizers

2. **Training Utilities**:
   - LMDB dataset creation tools
   - Dataset YAML configuration
   - Inference examples with NAR decoding

3. **Validation**:
   - Successfully trained on custom Sinhala handwriting dataset
   - Proves PLM logic works with non-standard character sets
   - Demonstrates end-to-end training pipeline

### Implications for Our Implementation
- **Architecture is proven flexible**: Custom charset works without modifying PLM core
- **Training pipeline is stable**: Successful real-world deployment
- **Our approach is validated**: Atomic refactoring is feasible

---

## Flash Attention: PyTorch Native Fallback

### Key Finding
**PyTorch automatically handles Flash Attention fallback** - no custom logic needed!

### Implementation Details

From PyTorch `TransformerDecoderLayer` documentation:

```python
# PyTorch 2.0+ automatically dispatches to optimal backend
out = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    dropout_p=dropout,
    is_causal=is_causal  # Optional causal mask hint
)
```

**Fallback Triggers** (automatic):
1. Hardware lacks FlashAttention support (non-Ampere, CPU)
2. Input exceeds kernel limits (sequence length, head dims)
3. Custom mask incompatible with Flash kernel
4. Dtype not fp16/bf16

**Fallback Targets**:
- **Memory-efficient attention** (if available)
- **Math kernel** (slower but always works)
- **Pure Python** (slowest, guaranteed fallback)

### Implications for Our Implementation
- ✅ **No fallback logic needed** - PyTorch handles it
- ✅ **Just use `F.scaled_dot_product_attention`** directly
- ✅ **Auto-selects FlashAttention on RTX 3090**
- ⚠️ **Must use fp16/bf16** for Flash kernel (use `torch.cuda.amp.autocast()`)

### Example Usage Pattern
```python
class FlashMultiheadAttention(nn.Module):
    def forward(self, q, k, v, attn_mask=None):
        # PyTorch automatically selects optimal backend
        # No manual fallback logic required
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
```

---

## Critical Gotchas Discovered

### 1. Reverse Direction Special Handling (NEW!)

**Location**: baudm_parseq system.py:146-150

```python
if len(perms) > 1:
    perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
```

**What it does**:
- Not just flipped permutation
- Special reverse pattern for null-context EOS learning
- Required for NAR (Non-AutoRegressive) mode

**Impact**: Our parseq_official_adapter.py may have this wrong!
- Need to verify line 139 in our adapter matches this exactly
- This is subtle and could cause silent training failure

### 2. EOS Padding Mask

**Location**: baudm_parseq system.py:181

```python
tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)
```

**Why**: EOS should NOT be used as context for other tokens.

**Impact**: Need to ensure our decoder supports this padding mask correctly.

### 3. Hardcoded Selector Rationale

**Location**: baudm_parseq system.py:112

```python
if max_num_chars == 4 and self.perm_mirrored:
    selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
```

**Explanation from comment (lines 106-107)**:
> "For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions"

**Why 4?**: 4! = 24 permutations total, 12 if mirrored. Small enough to enumerate all.

**Implementation Note**: This is deterministic - must match exactly for test validation.

---

## Updated Validation Strategy

### Phase 1: Compare Against Official Source

**Priority 1**: Verify our parseq_official_adapter.py matches baudm_parseq system.py

```bash
# Compare implementations
diff <(sed -n '92,156p' ocr/domains/recognition/models/parseq_official_adapter.py) \
     <(sed -n '90,167p' __DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/strhub/models/parseq/system.py)
```

**Critical Lines to Check**:
- Line 139 (our adapter): Reverse direction special handling
- Line 142-156 (our adapter): generate_attn_masks logic
- Line 232-248 (our adapter): training_step loop

### Phase 2: Reference Test Cases

Use baudm_parseq as oracle for test validation:

```python
def test_against_official_baudm():
    """Validate against official baudm implementation"""
    from __DEBUG__.training_failures_2026_02_07.implementations.baudm_parseq.strhub.models.parseq.system import PARSeq as BaudmPARSeq

    # Create both implementations
    baudm_model = BaudmPARSeq(...)  # Official
    our_plm = PermutationLanguageModeling(...)  # Ours

    # Test cases
    for seq_len in [1, 2, 4, 5, 10, 25]:
        tgt = create_dummy_tokens(seq_len)

        baudm_perms = baudm_model.gen_tgt_perms(tgt)
        our_perms = our_plm.gen_tgt_perms(tgt)

        # Must match exactly
        assert torch.equal(baudm_perms, our_perms), \
            f"Mismatch at seq_len={seq_len}"
```

---

## Revised Implementation Plan Updates

### Update to Phase 1

**Step 1.0: Verify Existing Adapter** (NEW STEP)

Before extracting to PLM module, verify our parseq_official_adapter.py is correct:

```bash
# Check reverse direction handling (line 139)
grep -A 2 "if len(perms) > 1:" ocr/domains/recognition/models/parseq_official_adapter.py

# Expected: perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
```

**Action**: If mismatch found, fix parseq_official_adapter.py FIRST before extraction.

### Update to Phase 3: Flash Attention

**Simplified Approach**:

No custom fallback logic needed! Just use `F.scaled_dot_product_attention` directly:

```python
class FlashMultiheadAttention(nn.Module):
    def forward(self, q, k, v, attn_mask=None):
        # PyTorch handles fallback automatically
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )
```

**Test**: Verify Flash is selected on RTX 3090:

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    output = model(images, targets)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
# Should show "flash_attention" kernel
```

---

## Additional Reference Resources

### Official baudm/parseq
- **Full implementation**: `__DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/`
- **System module**: `strhub/models/parseq/system.py` (lines 90-200)
- **Model module**: `strhub/models/parseq/model.py`
- **Training script**: `train.py` (PyTorch Lightning)

### dilithjay/Sinhala-ParSeq
- **Repository**: https://github.com/dilithjay/Sinhala-ParSeq
- **Training guide**: https://dilithjay.com/blog/parseq-train-a-custom-model
- **Colab notebook**: Available via GitHub gists

### PyTorch Flash Attention
- **API Docs**: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- **TransformerDecoder**: https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html
- **Tutorial**: https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html

---

## Risk Mitigation Updates

### NEW Risk: Reverse Direction Bug

**Risk**: Our parseq_official_adapter.py line 139 might not match official implementation
**Impact**: Silent training failure, loss divergence
**Mitigation**:
1. Verify line 139 matches baudm system.py:150 exactly
2. Add specific unit test for reverse direction permutation
3. Compare perms[1, :] output with official implementation

### NEW Risk: EOS Padding Mask Missing

**Risk**: Decoder might not handle EOS as padding correctly
**Impact**: EOS tokens used as context, incorrect PLM behavior
**Mitigation**:
1. Ensure decoder accepts and uses tgt_padding_mask
2. Verify mask includes both PAD and EOS: `(tgt_in == pad_id) | (tgt_in == eos_id)`
3. Test with sequences containing EOS tokens

---

## Success Criteria Updates

### Must Have (Blocking)

- ✅ Our adapter matches baudm system.py (lines 90-200) exactly
- ✅ Reverse direction handling correct (line 139/150)
- ✅ EOS padding mask implemented
- ✅ PLM logic matches official reference (ε ≤ 1e-5)
- ✅ Training converges (loss ↓)

### Should Have (High Priority)

- ✅ Flash Attention auto-selected on RTX 3090
- ✅ No custom fallback logic (rely on PyTorch)
- ✅ 2x speedup achieved
- ✅ CER within ±0.5% baseline

---

## Next Steps (Immediate)

### 1. Verify Existing Implementation

```bash
# Compare our adapter against official
cd /workspaces
diff -u \
  ocr/domains/recognition/models/parseq_official_adapter.py \
  __DEBUG__/training_failures_2026-02-07/implementations/baudm_parseq/strhub/models/parseq/system.py
```

### 2. Fix Any Discrepancies

If reverse direction handling differs, update parseq_official_adapter.py line 139.

### 3. Proceed with Phase 1

Once adapter is verified correct, extract to PLM module using exact logic.

---

**Status**: ✅ Additional research complete - critical insights discovered
**Action Required**: Verify parseq_official_adapter.py before Phase 1 extraction
**References Updated**: Official source, successful fork, Flash Attention details
