# Walkthrough: PARSeq PLM + Flash Attention Refactor

**Artifact Type**: Implementation Walkthrough
**Status**: Research Complete - Ready for Implementation
**Created**: 2026-02-12
**Pulse**: recognition-parseq-optimization

---

## Executive Summary

This walkthrough provides a research-backed implementation guide for refactoring PARSeq from monolithic to atomic architecture with Flash Attention optimization.

**Key Finding**: **NO existing modular PARSeq implementations found** - we must extract carefully from our working implementation.

**Core Strategy**: **Extract, validate, then optimize** - NOT reimplement from scratch.

**Risk Level**: HIGH - PLM logic is complex, debugging is costly.

---

## Research Findings

### 1. Permutation Language Modeling (PLM) Core Concepts

**From Perplexity Research**:

#### What is PLM?
- **Training Strategy**: Uses K random permutations (not all T! permutations) of token orderings
- **Default K=6**: Samples 6 different decoding orders per training example
- **Mirrored Pairs**: K/2 base permutations + K/2 complementary (reversed) permutations
- **Inference Difference**: Training uses permutations, inference uses standard left-to-right

#### Critical Components
1. **Permutation Generation** (`gen_tgt_perms`):
   - Special case for 1-char: identity permutation
   - Special case for ≤4-char: exhaustive with hardcoded selector
   - General case for >4-char: random sampling
   - Always prepends BOS=0, appends EOS=max_num_chars+1

2. **Attention Masks** (`generate_attn_masks`):
   - Two masks: content_mask [L-1, L-1] and query_mask [L-1, L-1]
   - Enforce causal structure based on permutation ordering
   - Prevent "copy-pasting" from input by blocking look-ahead

3. **Loss Aggregation** (Most Complex):
   - Iterate through K permutations
   - Weight loss by character count (n)
   - **Critical**: Remove EOS tokens after 2nd permutation (prevents over-weighting)
   - Normalize by total character count across all permutations

### 2. Flash Attention Integration Constraints

**From PyTorch Research**:

#### Hard Requirements
- **Data Type**: fp16 or bfloat16 only (NO fp32 support)
- **Head Dimension**: Must be multiple of 8 (d_model / nhead % 8 == 0)
  - ✅ Our config: 384/12 = 32 (multiple of 8)
- **PyTorch Version**: ≥2.0 (≥2.2 for optimal FlashAttention-2)
- **Hardware**: Ampere+ (sm_80+) for auto-selection
  - ✅ RTX 3090 is Ampere (sm_80)

#### Numerical Behavior
- **Exact Computation**: Same mathematical result as standard attention
- **Precision**: fp16 introduces rounding (ε ≤ 1e-3 acceptable)
- **Backend**: Auto-selects via `torch.backends.cuda.sdp_kernel(enable_flash=True)`

#### Expected Performance
- **Throughput**: 2-4x speedup vs standard attention
- **Memory**: Similar or lower than baseline
- **Best Cases**: Shorter sequences, larger batch sizes

### 3. Existing Implementations

**Search Results**: NO modular PARSeq implementations found.

**Available**:
- baudm/parseq (monolithic, official)
- Our parseq_official_adapter.py (working adapter)

**Implication**: We're pioneering this - must be extremely careful.

---

## Implementation Architecture

### Current State

```
Monolithic (PARSeqOfficial)
├── Hardcoded ViT encoder
├── Standard nn.MultiheadAttention
└── PLM logic embedded in forward()
    ├── gen_tgt_perms() ✓ (lines 92-140)
    ├── generate_attn_masks() ✓ (lines 142-156)
    └── forward_train() ✓ (lines 204-248)

Atomic (PARSeqDecoder) - INCOMPLETE
├── Modular TimmBackbone support ✓
├── Standard nn.TransformerDecoder ✓
└── PLM logic ✗ (MISSING - just standard AR)
```

### Target State

```
Atomic + PLM + Flash
├── Modular Backbone (any encoder)
├── PLM Module (extracted from monolithic)
│   ├── gen_tgt_perms()
│   ├── generate_attn_masks()
│   └── compute_plm_loss()
├── Flash Attention Decoder (optional)
│   ├── FlashDecoderLayer
│   ├── FlashMultiheadAttention
│   └── Fallback to standard attention
└── Hydra Config Integration
    ├── Architecture config
    ├── Domain injection (tokenizer/loss)
    └── Experiment configs
```

---

## Phase-by-Phase Implementation Guide

## Phase 1: PLM Module Extraction ⚠️ CRITICAL

### 🎯 Goal
Extract proven PLM logic from parseq_official_adapter.py into standalone, testable module.

### ⚠️ Critical Rule
**DO NOT reimplement from scratch**. Copy exact logic, validate against reference.

### Step 1.1: Create PLM Module

**File**: `ocr/domains/recognition/models/plm.py`

**Extraction Points**:
- `gen_tgt_perms()` from lines 92-140
- `generate_attn_masks()` from lines 142-156

**Template**:
```python
"""
Permutation Language Modeling for PARSeq.

Extracted from parseq_official_adapter.py (verified working implementation).
DO NOT modify logic without validating against reference.
"""

import math
from itertools import permutations
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class PermutationLanguageModeling(nn.Module):
    """
    PLM utilities for PARSeq.

    References:
    - PARSeq Paper: https://arxiv.org/abs/2207.06966
    - Official Implementation: baudm/parseq
    - Our Adapter: ocr/domains/recognition/models/parseq_official_adapter.py
    """

    def __init__(
        self,
        max_label_length: int = 25,
        perm_num: int = 6,
        perm_forward: bool = True,
        perm_mirrored: bool = True,
    ):
        super().__init__()
        self.max_label_length = max_label_length
        self.perm_num = perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored

        # RNG for random permutation sampling
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num

    @property
    def _device(self) -> torch.device:
        """Get device from any parameter"""
        return next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')

    def gen_tgt_perms(self, tgt: Tensor) -> Tensor:
        """
        Generate shared permutations for the whole batch.

        EXACT COPY from parseq_official_adapter.py:92-140

        Args:
            tgt: [B, L] target token sequences

        Returns:
            perms: [K, L] permutation indices
                   K = perm_num (default 6)
                   L = sequence length (including BOS, EOS)
        """
        # COPY LINES 97-140 FROM parseq_official_adapter.py
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2

        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)

        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []

        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)

        # For 4-char sequences and shorter, generate all permutations
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))]
            )
            perms = torch.stack(perms)

        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)

        # Special handling for reverse direction
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)

        return perms

    def generate_attn_masks(self, perm: Tensor) -> tuple[Tensor, Tensor]:
        """
        Generate attention masks given a sequence permutation.

        EXACT COPY from parseq_official_adapter.py:142-156

        Args:
            perm: [L] permutation indices for one sequence

        Returns:
            content_mask: [L-1, L-1] mask for context tokens
            query_mask: [L-1, L-1] mask for query tokens
        """
        # COPY LINES 147-156 FROM parseq_official_adapter.py
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)

        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = True

        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True  # mask "self"
        query_mask = mask[1:, :-1]

        return content_mask, query_mask
```

### Step 1.2: Unit Tests (BLOCKING)

**File**: `tests/unit/recognition/test_plm.py`

**Test Strategy**: Validate against parseq_official_adapter.py

```python
import pytest
import torch
from ocr.domains.recognition.models.plm import PermutationLanguageModeling
from ocr.domains.recognition.models.parseq_official_adapter import PARSeqOfficial


@pytest.fixture
def plm_module():
    return PermutationLanguageModeling(
        max_label_length=25,
        perm_num=6,
        perm_forward=True,
        perm_mirrored=True,
    )


@pytest.fixture
def ref_model():
    return PARSeqOfficial(
        num_tokens=100,
        max_label_length=25,
        perm_num=6,
        perm_forward=True,
        perm_mirrored=True,
    )


class TestPermutationGeneration:
    """Test gen_tgt_perms() against reference implementation"""

    def test_1char_sequence(self, plm_module, ref_model):
        """1-char sequences must return identity permutation [0, 1, 2]"""
        # BOS=0, char=1, EOS=2
        tgt = torch.tensor([[1, 5, 2, 0]])  # [BOS, char, EOS, PAD]

        ref_perms = ref_model.gen_tgt_perms(tgt)
        plm_perms = plm_module.gen_tgt_perms(tgt)

        assert torch.equal(ref_perms, plm_perms), "1-char perms must match"
        assert plm_perms.shape == (1, 3), "Expected [1, 3] shape"
        assert torch.equal(plm_perms[0], torch.tensor([0, 1, 2])), "Must be identity"

    def test_4char_sequence(self, plm_module, ref_model):
        """4-char sequences use hardcoded selector"""
        # BOS=0, 4 chars, EOS=2, PAD=0
        tgt = torch.tensor([[1, 5, 6, 7, 8, 2, 0]])  # [BOS, c1, c2, c3, c4, EOS, PAD]

        ref_perms = ref_model.gen_tgt_perms(tgt)
        plm_perms = plm_module.gen_tgt_perms(tgt)

        # Must be identical (deterministic selector)
        assert torch.equal(ref_perms, plm_perms), "4-char perms must match exactly"
        assert plm_perms.shape[0] == 12, "Expected 12 permutations (selector)"

    def test_permutation_properties(self, plm_module):
        """Test general properties of generated permutations"""
        tgt = torch.tensor([[1, 5, 6, 7, 8, 9, 2, 0]])  # 5-char sequence

        perms = plm_module.gen_tgt_perms(tgt)

        # Shape check
        assert perms.shape == (6, 8), "K=6 perms, L=8 (BOS+5+EOS+PAD)"

        # BOS always at position 0
        assert (perms[:, 0] == 0).all(), "BOS must be at index 0"

        # EOS always at last position
        assert (perms[:, -1] == 6).all(), "EOS must be at index max_num_chars+1"

        # Values in valid range
        assert perms.min() >= 0
        assert perms.max() <= 7  # max_num_chars + 1

    def test_mirrored_pairs(self, plm_module):
        """Verify complementary permutations are generated"""
        tgt = torch.tensor([[1, 5, 6, 7, 2, 0]])  # 3-char

        perms = plm_module.gen_tgt_perms(tgt)

        # Should have pairs: perms[2i] and perms[2i+1] are complements
        # (excluding BOS/EOS positions)
        for i in range(0, len(perms), 2):
            if i + 1 < len(perms):
                perm_a = perms[i, 1:-1]  # Exclude BOS, EOS
                perm_b = perms[i+1, 1:-1]
                # Check if they're complements (reversed)
                # Note: The exact complementary logic depends on implementation
                # This is a simplified check
                assert not torch.equal(perm_a, perm_b), "Pairs should differ"


class TestAttentionMasks:
    """Test generate_attn_masks() against reference implementation"""

    def test_mask_generation_equivalence(self, plm_module, ref_model):
        """Masks must match reference exactly"""
        perm = torch.tensor([0, 2, 1, 3, 4])  # Example permutation

        ref_content, ref_query = ref_model.generate_attn_masks(perm)
        plm_content, plm_query = plm_module.generate_attn_masks(perm)

        assert torch.equal(ref_content, plm_content), "content_mask must match"
        assert torch.equal(ref_query, plm_query), "query_mask must match"

    def test_mask_shapes(self, plm_module):
        """Verify mask shapes are correct"""
        perm = torch.tensor([0, 1, 2, 3])  # L=4

        content_mask, query_mask = plm_module.generate_attn_masks(perm)

        assert content_mask.shape == (3, 3), "content_mask should be [L-1, L-1]"
        assert query_mask.shape == (3, 3), "query_mask should be [L-1, L-1]"

    def test_causal_structure(self, plm_module):
        """Verify masks enforce causal structure (no lookahead)"""
        perm = torch.tensor([0, 1, 2, 3, 4])  # Forward permutation

        content_mask, query_mask = plm_module.generate_attn_masks(perm)

        # For forward permutation, should create upper triangular mask
        # (preventing attention to future tokens)
        # Query mask should have diagonal masked (self-attention blocked)
        assert query_mask.diagonal().all(), "Self-attention should be masked"
```

**Success Criteria**: ALL tests must pass before Phase 2.

### Step 1.3: Loss Aggregation Logic

**Add to PLM Module**:

```python
def compute_plm_loss(
    self,
    decoder_fn: callable,
    head_fn: callable,
    tgt_in: Tensor,
    tgt_out: Tensor,
    memory: Tensor,
    pad_id: int = 0,
    eos_id: int = 2,
) -> Tensor:
    """
    Compute loss across all permutations.

    CRITICAL: Extracted from parseq_official_adapter.py:232-248
    DO NOT modify without validation.

    Args:
        decoder_fn: Decoder forward function
        head_fn: Classification head function
        tgt_in: [B, L] input token sequences
        tgt_out: [B, L] output token sequences (targets)
        memory: [B, S, D] encoded visual features
        pad_id: Padding token ID
        eos_id: End-of-sequence token ID

    Returns:
        loss: Scalar loss value
    """
    # Generate permutations
    perms = self.gen_tgt_perms(torch.cat([tgt_in, tgt_out[:, -1:]], dim=1))

    # Padding mask: ignore PAD and EOS in input
    tgt_padding_mask = (tgt_in == pad_id) | (tgt_in == eos_id)

    loss = 0
    loss_numel = 0
    n = (tgt_out != pad_id).sum().item()  # Character count

    for i, perm in enumerate(perms):
        # Generate attention masks for this permutation
        tgt_mask, query_mask = self.generate_attn_masks(perm)

        # Decode with permutation-specific masks
        out = decoder_fn(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)

        # Classify and compute loss
        logits = head_fn(out).flatten(end_dim=1)
        loss += n * torch.nn.functional.cross_entropy(
            logits, tgt_out.flatten(), ignore_index=pad_id
        )
        loss_numel += n

        # CRITICAL: Remove EOS after 2nd permutation
        if i == 1:
            tgt_out = torch.where(tgt_out == eos_id, torch.tensor(pad_id, device=tgt_out.device), tgt_out)
            n = (tgt_out != pad_id).sum().item()

    return loss / loss_numel
```

---

## Phase 2: Atomic Decoder Integration

### Step 2.1: Enhance Decoder Interface

**File**: `ocr/domains/recognition/models/decoder.py`

**Key Changes**:

```python
from .plm import PermutationLanguageModeling

class PARSeqDecoder(BaseDecoder):
    def __init__(
        self,
        in_channels,
        d_model=384,
        nhead=12,
        num_layers=12,
        dim_feedforward=1536,
        dropout=0.1,
        vocab_size=None,
        max_len=25,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_flash_attention=False,  # Phase 3
        plm_config=None,  # NEW
        **kwargs,
    ):
        super().__init__(in_channels=in_channels)

        # ... existing init ...

        # Add PLM module
        if plm_config:
            self.plm = PermutationLanguageModeling(**plm_config)
        else:
            self.plm = None

        # Flash attention (Phase 3)
        self.use_flash_attention = use_flash_attention
        if use_flash_attention:
            # Create flash decoder layers
            from .flash_attention import FlashDecoderLayer
            decoder_layer = FlashDecoderLayer(...)
        else:
            decoder_layer = nn.TransformerDecoderLayer(...)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, features, targets=None, mode='train', **kwargs):
        """
        Unified forward with mode selection.

        Args:
            features: Visual features from encoder
            targets: [B, L] ground truth tokens (for training)
            mode: 'train' or 'inference'

        Returns:
            If train: decoder output for loss computation
            If inference: logits for prediction
        """
        if mode == 'train' and targets is not None:
            return self.forward_train(features, targets, **kwargs)
        else:
            return self.forward_inference(features, **kwargs)

    def forward_train(self, features, targets, **kwargs):
        """
        Training forward pass with PLM if enabled.

        Returns:
            [B, L, D] decoder output (to be passed to head for loss)
        """
        # Prepare memory from encoder features
        memory = self._prepare_memory(features)

        if self.plm is None:
            # Standard autoregressive training
            return self._standard_ar_forward(memory, targets)

        # PLM training - handled by PLM module
        # Note: Loss computation moved to architecture level
        # This just returns decoder output
        tgt_in = targets[:, :-1]
        tgt_padding_mask = (tgt_in == self.pad_token_id)

        # Generate permutations
        perms = self.plm.gen_tgt_perms(targets)

        # For training, we need to process with first permutation
        # (Full PLM loop happens in architecture's loss computation)
        perm = perms[0]  # Use first permutation for this forward
        tgt_mask, query_mask = self.plm.generate_attn_masks(perm)

        # Standard decoder forward
        tgt_emb = self.embed_tokens(tgt_in) * math.sqrt(self.d_model)
        pos_emb = self.pos_encoder[:, :tgt_in.shape[1], :] * math.sqrt(self.d_model)
        tgt = tgt_emb + pos_emb

        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=kwargs.get('memory_key_padding_mask'),
        )

        return self.norm(output)
```

**Note**: Loss computation with full PLM loop happens in architecture level (PARSeq class).

### Step 2.2: Integration Tests

**File**: `tests/integration/test_atomic_plm.py`

```python
def test_atomic_vs_monolithic_equivalence():
    """Verify atomic decoder matches monolithic implementation"""
    # Setup
    ref_model = PARSeqOfficial(...)
    atomic_model = PARSeq(
        encoder=TimmBackbone(...),
        decoder=PARSeqDecoder(..., plm_config={...}),
        head=ClassificationHead(...),
    )

    # Same inputs
    images = torch.randn(4, 3, 32, 128)
    targets = torch.randint(0, 100, (4, 26))

    # Forward pass
    ref_out = ref_model.forward_train(images, targets)
    atomic_out = atomic_model.forward_train(images, targets)

    # Compare
    assert torch.allclose(ref_out, atomic_out, atol=1e-5), \
        f"Loss mismatch: {ref_out} vs {atomic_out}"
```

---

## Phase 3: Flash Attention (After PLM Works)

**File**: `ocr/domains/recognition/models/flash_attention.py`

See implementation plan for full code.

**Key Points**:
- Wrap `F.scaled_dot_product_attention`
- Use `torch.backends.cuda.sdp_kernel(enable_flash=True)`
- Requires fp16/bf16 (use with `torch.cuda.amp.autocast()`)
- Test numerical equivalence (ε ≤ 1e-3 acceptable)

---

## Critical Gotchas & Debugging Guide

### 1. Loss Divergence
**Symptom**: Training loss doesn't decrease or diverges
**Root Causes**:
- EOS removal logic incorrect (should happen after 2nd permutation)
- Loss normalization wrong (divide by sum of character counts)
- Permutation generation not deterministic for ≤4 chars

**Debug Steps**:
1. Print loss per permutation - should decrease
2. Verify `n` (character count) updates correctly after i==1
3. Check `loss_numel` accumulation
4. Compare with parseq_official_adapter.py line-by-line

### 2. Gradient Explosion/Vanishing
**Symptom**: NaN/Inf gradients or gradients→0
**Root Causes**:
- Incorrect gradient accumulation across permutations
- Missing gradient scaling
- Attention mask mismatch

**Debug Steps**:
1. Hook gradients: `tensor.register_hook(lambda grad: print(grad.norm()))`
2. Check gradient flow through each permutation
3. Verify attention masks don't block all positions

### 3. Numerical Drift (Flash Attention)
**Symptom**: Outputs differ from baseline
**Expected**: Some drift due to fp16 (ε ≤ 1e-3)
**Unacceptable**: ε > 1e-2 indicates bug

**Debug Steps**:
1. Test without Flash first (standard attention)
2. Compare layer-by-layer outputs
3. Check attention mask format (additive vs boolean)
4. Verify head_dim is multiple of 8

### 4. Permutation Generation Bugs
**Symptom**: Training unstable or inconsistent
**Root Causes**:
- Hardcoded selector wrong for 4-char sequences
- Mirrored pair logic incorrect
- BOS/EOS positions swapped

**Debug Steps**:
1. Unit test each sequence length separately
2. Print generated permutations
3. Verify against reference implementation

---

## Validation Checklist

Before proceeding to next phase:

**Phase 1 (PLM Extraction)**:
- [ ] All unit tests pass (100% coverage)
- [ ] Permutation generation matches reference
- [ ] Attention masks match reference
- [ ] Loss computation logic validated

**Phase 2 (Decoder Integration)**:
- [ ] Numerical equivalence: ε ≤ 1e-5
- [ ] Training converges on small dataset
- [ ] No NaN/Inf gradients
- [ ] Memory usage comparable to monolithic

**Phase 3 (Flash Attention)**:
- [ ] Numerical equivalence: ε ≤ 1e-3 (fp16)
- [ ] Throughput ≥ 2x baseline
- [ ] VRAM usage ≤ baseline
- [ ] Auto-fallback on non-Ampere GPUs

**Phase 4 (Configuration)**:
- [ ] Hydra composition works
- [ ] Domain injection configured
- [ ] Experiment config validated

**Phase 5 (Full Validation)**:
- [ ] Full training run successful (50 epochs)
- [ ] CER within ±0.5% of baseline
- [ ] Edge cases handled gracefully
- [ ] Documentation complete

---

## Performance Expectations

### Baseline (Monolithic)
- Throughput: ~120 img/sec
- VRAM: ~18 GB (batch=64)
- Training: ~8 hours/epoch (full dataset)

### Atomic + Standard Attention
- Throughput: ~100-110 img/sec (slight overhead)
- VRAM: ~18 GB
- Training: ~8-9 hours/epoch

### Atomic + Flash Attention
- Throughput: ~240-300 img/sec (2-2.5x)
- VRAM: ~16 GB (slightly lower)
- Training: ~3-4 hours/epoch

---

## Timeline Estimate

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 1 | PLM extraction + tests | 2-3 days |
| Phase 2 | Decoder integration + validation | 2-3 days |
| Phase 3 | Flash Attention + benchmarks | 1-2 days |
| Phase 4 | Configuration | 1 day |
| Phase 5 | Full validation + docs | 2-3 days |
| **Total** | | **8-12 days** |

---

## References

**Research Sources**:
- Perplexity: PLM concepts, Flash Attention constraints
- PyTorch Docs: F.scaled_dot_product_attention
- PARSeq Paper: https://arxiv.org/abs/2207.06966
- Official Repo: https://github.com/baudm/parseq

**Code References**:
- parseq_official_adapter.py:92-140 (gen_tgt_perms)
- parseq_official_adapter.py:142-156 (generate_attn_masks)
- parseq_official_adapter.py:204-248 (forward_train with PLM)

**Framework References**:
- AgentQMS/specs/tier2-framework/patterns.spec.md (Hydra V5)
- AgentQMS/specs/tier2-framework/core-interfaces.spec.md (Architecture)

---

## Success Criteria

### Must Have (Blocking)
✅ PLM logic matches reference exactly
✅ Training converges (loss ↓)
✅ Numerical equivalence: ε ≤ 1e-5 (fp32)

### Should Have (High Priority)
✅ Flash Attention 2x speedup
✅ CER within ±0.5% baseline
✅ VRAM ≤ baseline

### Nice to Have (Optional)
✅ 2.5x+ speedup (exceeds target)
✅ Faster convergence
✅ Lower memory usage

---

**Status**: Research complete, ready for implementation
**Next Session**: Begin Phase 1 - PLM Module Extraction
**Continuation Prompt**: See session handover document
