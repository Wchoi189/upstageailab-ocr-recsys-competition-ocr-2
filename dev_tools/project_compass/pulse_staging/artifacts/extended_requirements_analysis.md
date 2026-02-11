# Extended Requirements Analysis: ParseQ Atomic Architecture Refactor

**Artifact Type**: Research & Requirements Analysis
**Status**: Draft - For Review
**Created**: 2026-02-12
**Pulse**: recognition-parseq-optimization

---

## Executive Summary

This document extends the existing implementation plan with:
1. **Data Contracts & Type Safety** requirements
2. **AST Analysis Tools** and productivity enhancers
3. **High-Risk Areas** with mitigation strategies
4. **Phase-Specific Context Bundles** for multi-session work
5. **Missing Documentation** gaps identified
6. **Feedback on Development Environment** pain points

---

## 1. Data Contracts & Type Safety Analysis

### 1.1 Current State Assessment

#### ✅ Existing Contracts (Good Foundation)

**Base Interfaces** (`ocr/core/interfaces/models.py`):
- `BaseEncoder`: Contract for feature extraction
  - Output: `list[torch.Tensor]` - multi-scale features
  - Properties: `out_channels`, `strides`
- `BaseDecoder`: Contract for feature decoding
  - Input: `list[torch.Tensor]` (features), Optional `torch.Tensor` (targets)
  - Output: `torch.Tensor` (decoded features)
  - Property: `out_channels`
- `BaseHead`: Contract for prediction heads
  - Input: `torch.Tensor`
  - Output: `dict[str, torch.Tensor]`

**Domain Schemas** (`ocr/core/interfaces/schemas.py`):
- `Box`, `DetectionResult`, `RecognitionResult`, `PageResult`
- Dataclass-based, serializable
- Clean domain boundaries

**Validation Models** (`ocr/core/interfaces/validation_models.py`):
- `DataItem`: Validated dataset samples (Pydantic)
- `MapData`: Cached probability/threshold maps
- `MetricConfig`: CLEval metric configuration

#### ❌ Missing Contracts (Critical Gaps)

**PLM Data Contracts** (NEW - Priority: HIGH):
```python
# File: ocr/core/interfaces/plm.py (TO BE CREATED)

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
import torch
from torch import Tensor

@dataclass
class PLMConfig:
    """Configuration for Permutation Language Modeling."""
    max_len: int = 25
    perm_num: int = 6
    perm_forward: bool = True
    perm_mirrored: bool = True

    def __post_init__(self):
        assert self.perm_num > 0, "perm_num must be positive"
        assert self.max_len > 0, "max_len must be positive"
        if self.perm_mirrored:
            assert self.perm_num % 2 == 0, "perm_num must be even when mirrored"

@dataclass
class AttentionMasks:
    """Attention masks for permutation-based decoding."""
    content_mask: Tensor  # [L-1, L-1] - for context tokens
    query_mask: Tensor    # [L-1, L-1] - for query tokens

    def __post_init__(self):
        assert self.content_mask.shape == self.query_mask.shape
        assert self.content_mask.dtype == torch.bool
        assert self.query_mask.dtype == torch.bool

@runtime_checkable
class PLMModule(Protocol):
    """Protocol for PLM components."""

    def gen_tgt_perms(self, tgt: Tensor) -> Tensor:
        """Generate K permutations for target sequence.

        Args:
            tgt: [B, L] Target token indices

        Returns:
            perms: [K, L] Permutation indices
        """
        ...

    def generate_attn_masks(self, perm: Tensor) -> AttentionMasks:
        """Generate attention masks for a permutation.

        Args:
            perm: [L] Single permutation indices

        Returns:
            AttentionMasks containing content and query masks
        """
        ...

@runtime_checkable
class FlashAttentionLayer(Protocol):
    """Protocol for Flash Attention compatible layers."""

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """Standard TransformerDecoderLayer signature with Flash support."""
        ...
```

**Decoder Mode Contracts** (NEW - Priority: HIGH):
```python
# File: ocr/core/interfaces/decoder.py (TO BE CREATED)

from enum import Enum
from dataclasses import dataclass
from typing import Protocol
import torch

class DecoderMode(Enum):
    """Decoder operating modes."""
    TRAIN = "train"           # Training with PLM permutations
    INFERENCE = "inference"   # Autoregressive inference
    VALIDATION = "validation" # Validation (may use reduced perms)

@dataclass
class DecoderOutput:
    """Standardized decoder output."""
    logits: torch.Tensor      # [B, L, vocab_size] or [B*K, L, vocab_size]
    loss: torch.Tensor | None = None
    metadata: dict | None = None

    def __post_init__(self):
        assert self.logits.ndim == 3
        if self.loss is not None:
            assert self.loss.ndim == 0  # Scalar

@runtime_checkable
class AutoregressiveDecoder(Protocol):
    """Protocol for AR decoders supporting multiple modes."""

    def forward(
        self,
        features: list[torch.Tensor] | torch.Tensor,
        targets: torch.Tensor | None = None,
        mode: DecoderMode = DecoderMode.TRAIN,
        **kwargs
    ) -> DecoderOutput:
        """Unified forward supporting train/inference modes."""
        ...
```

**Flash Attention Constraints** (NEW - Priority: MEDIUM):
```python
# File: ocr/core/interfaces/flash_constraints.py (TO BE CREATED)

from dataclasses import dataclass
import torch

@dataclass
class FlashAttentionConfig:
    """Configuration and constraints for Flash Attention."""
    d_model: int
    nhead: int
    enabled: bool = True
    force_fp16: bool = True

    def __post_init__(self):
        self.head_dim = self.d_model // self.nhead

        # Validate constraints
        assert self.d_model % self.nhead == 0, \
            f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"

        if self.enabled:
            assert self.head_dim % 8 == 0, \
                f"head_dim ({self.head_dim}) must be multiple of 8 for Flash Attention"

            # Check PyTorch version
            torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
            assert torch_version >= (2, 0), \
                f"Flash Attention requires PyTorch ≥2.0, got {torch.__version__}"

    @property
    def is_compatible(self) -> bool:
        """Check if Flash Attention is compatible with current setup."""
        if not self.enabled:
            return False
        if not torch.cuda.is_available():
            return False
        # Check for Ampere+ (sm_80+)
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 8  # sm_80 = Ampere
```

### 1.2 Type Safety Recommendations

#### Mypy Configuration (NEW)
```toml
# pyproject.toml (add this section)
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = false
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = "ocr.domains.recognition.*"
disallow_untyped_defs = true  # Enforce for recognition refactor

[[tool.mypy.overrides]]
module = "strhub.*"  # Vendor code
ignore_errors = true
```

#### Runtime Type Checking (NEW)
```python
# ocr/core/utils/type_guards.py (TO BE CREATED)

from typing import TypeGuard, runtime_checkable
import torch
from torch import Tensor

def is_valid_permutation(perm: Tensor) -> TypeGuard[Tensor]:
    """Validate permutation tensor structure."""
    if perm.ndim != 1:
        return False
    if perm.dtype not in (torch.long, torch.int32, torch.int64):
        return False
    # Check for valid permutation (0 to N-1, no duplicates)
    sorted_perm = perm.sort()[0]
    expected = torch.arange(len(perm), device=perm.device, dtype=perm.dtype)
    return torch.equal(sorted_perm, expected)

def is_attention_mask(mask: Tensor) -> TypeGuard[Tensor]:
    """Validate attention mask structure."""
    if mask.ndim != 2:
        return False
    if mask.dtype != torch.bool:
        return False
    if mask.shape[0] != mask.shape[1]:  # Must be square
        return False
    return True
```

### 1.3 Implementation Requirements

**Phase 1 Additions**:
- [ ] Create `ocr/core/interfaces/plm.py` with protocols
- [ ] Create `ocr/core/interfaces/decoder.py` with modes
- [ ] Create `ocr/core/interfaces/flash_constraints.py`
- [ ] Add runtime type guards in `ocr/core/utils/type_guards.py`
- [ ] Configure mypy for recognition domain

**Testing Requirements**:
- [ ] Type checking in CI/CD (mypy)
- [ ] Runtime contract validation tests
- [ ] Protocol compliance tests (isinstance checks)

---

## 2. AST Analysis Tools & Productivity Enhancers

### 2.1 Available Tools from Agent Debug Toolkit

**ADT Meta Query Tool** (`mcp__unified__adt_meta_query`):

#### High-Value Analysis Kinds for This Project:

1. **`dependency_graph`** - CRITICAL for Phase 1
   ```python
   # Usage: Map PLM dependencies before extraction
   mcp__unified__adt_meta_query(
       kind="dependency_graph",
       target="ocr/domains/recognition/models",
       options={"depth": 3, "output": "graph"}
   )
   ```
   **Use Case**: Identify all dependencies of parseq_official_adapter.py before extracting PLM logic to ensure no circular imports.

2. **`imports`** - HIGH for verification
   ```python
   # Usage: Verify clean imports after refactor
   mcp__unified__adt_meta_query(
       kind="imports",
       target="ocr/domains/recognition/models/plm.py",
       options={"output": "summary"}
   )
   ```
   **Use Case**: Ensure new PLM module only imports from core/interfaces, not from domains.

3. **`symbol_search`** - MEDIUM for discovery
   ```python
   # Usage: Find all usages of gen_tgt_perms
   mcp__unified__adt_meta_query(
       kind="symbol_search",
       target="gen_tgt_perms",
       options={"fuzzy": true, "scope": "ocr/"}
   )
   ```
   **Use Case**: Locate all references to PLM functions before refactor.

4. **`complexity`** - LOW (informational)
   ```python
   # Usage: Measure cyclomatic complexity
   mcp__unified__adt_meta_query(
       kind="complexity",
       target="ocr/domains/recognition/models/parseq_official_adapter.py",
       options={"threshold": 10}
   )
   ```
   **Use Case**: Identify complex functions that need extra testing.

5. **`ast_dump`** - HIGH for validation
   ```python
   # Usage: Compare AST structure before/after
   mcp__unified__adt_meta_query(
       kind="ast_dump",
       target="ocr/domains/recognition/models/plm.py",
       options={"mode": "structure"}
   )
   ```
   **Use Case**: Validate that extracted PLM logic maintains same structure as original.

**ADT Meta Edit Tool** (`mcp__unified__adt_meta_edit`):

1. **`apply_diff`** - CRITICAL for Phase 1
   ```python
   # Usage: Apply complex multi-file refactors
   mcp__unified__adt_meta_edit(
       kind="apply_diff",
       target="unified_diff_content",
       options={"dry_run": true, "fuzzy": true}
   )
   ```
   **Use Case**: Safe application of large refactors with fuzzy matching.

2. **`read_slice`** - MEDIUM for extraction
   ```python
   # Usage: Extract specific line ranges
   mcp__unified__adt_meta_edit(
       kind="read_slice",
       target="parseq_official_adapter.py",
       options={"start": 92, "end": 140}  # gen_tgt_perms lines
   )
   ```
   **Use Case**: Extract exact code blocks for PLM module.

### 2.2 Custom Scripts to Create

**Script 1: PLM Extraction Validator** (Priority: HIGH)
```python
# scripts/validation/validate_plm_extraction.py

"""Validate that PLM extraction matches original implementation."""

import torch
from ocr.domains.recognition.models.parseq_official_adapter import PARSeqOfficial
from ocr.domains.recognition.models.plm import PermutationLanguageModeling

def validate_gen_tgt_perms():
    """Compare gen_tgt_perms outputs."""
    # Create instances
    ref = PARSeqOfficial(num_tokens=100, max_label_length=25)
    plm = PermutationLanguageModeling(max_label_length=25)

    # Test cases
    test_cases = [
        torch.tensor([[1, 5, 10, 2]]),  # 1-char (after BOS/EOS removal)
        torch.tensor([[1, 5, 10, 7, 2]]),  # 2-char
        torch.tensor([[1, 5, 10, 7, 9, 3, 2]]),  # 4-char (hardcoded selector)
    ]

    for tgt in test_cases:
        ref_perms = ref.gen_tgt_perms(tgt)
        plm_perms = plm.gen_tgt_perms(tgt)

        # For deterministic cases (≤4 chars), must be identical
        if tgt.shape[1] <= 6:  # ≤4 chars after BOS/EOS
            assert torch.equal(ref_perms, plm_perms), \
                f"Mismatch for {tgt.shape[1]-2} chars: {ref_perms} vs {plm_perms}"
        else:
            # For random cases, check shape and range
            assert ref_perms.shape == plm_perms.shape
            assert plm_perms.min() >= 0
            assert plm_perms.max() <= tgt.shape[1]

    print("✅ gen_tgt_perms validation passed")

def validate_attn_masks():
    """Compare attention mask generation."""
    plm = PermutationLanguageModeling(max_label_length=25)
    perm = torch.tensor([0, 3, 1, 2, 5, 4, 6])  # Example permutation

    masks = plm.generate_attn_masks(perm)

    # Validate shapes
    assert masks.content_mask.shape[0] == masks.content_mask.shape[1]
    assert masks.query_mask.shape == masks.content_mask.shape

    # Validate dtype
    assert masks.content_mask.dtype == torch.bool
    assert masks.query_mask.dtype == torch.bool

    # Validate causal structure (no look-ahead)
    L = masks.content_mask.shape[0]
    for i in range(L):
        for j in range(i+1, L):
            # Position i cannot attend to position j (j > i)
            assert masks.content_mask[i, j] == True  # True = masked

    print("✅ generate_attn_masks validation passed")

if __name__ == "__main__":
    validate_gen_tgt_perms()
    validate_attn_masks()
```

**Script 2: Numerical Equivalence Checker** (Priority: CRITICAL)
```python
# scripts/validation/check_numerical_equivalence.py

"""Check numerical equivalence between implementations."""

import torch
import torch.nn.functional as F
from typing import Callable

def compare_outputs(
    ref_fn: Callable,
    new_fn: Callable,
    inputs: dict,
    atol: float = 1e-5,
    rtol: float = 1e-5
) -> bool:
    """Compare outputs of two functions with tolerance.

    Args:
        ref_fn: Reference implementation
        new_fn: New implementation
        inputs: Dictionary of input arguments
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if outputs match within tolerance
    """
    ref_out = ref_fn(**inputs)
    new_out = new_fn(**inputs)

    if isinstance(ref_out, torch.Tensor):
        match = torch.allclose(ref_out, new_out, atol=atol, rtol=rtol)
        if not match:
            diff = (ref_out - new_out).abs()
            print(f"❌ Mismatch detected:")
            print(f"   Max diff: {diff.max().item():.2e}")
            print(f"   Mean diff: {diff.mean().item():.2e}")
            print(f"   Tolerance: atol={atol:.2e}, rtol={rtol:.2e}")
        return match
    elif isinstance(ref_out, dict):
        for key in ref_out:
            if not compare_outputs(
                lambda: ref_out[key],
                lambda: new_out[key],
                {},
                atol=atol,
                rtol=rtol
            ):
                return False
        return True
    else:
        raise ValueError(f"Unsupported output type: {type(ref_out)}")
```

**Script 3: Flash Attention Benchmark** (Priority: MEDIUM)
```python
# scripts/benchmark/flash_attention_bench.py

"""Benchmark Flash Attention vs Standard Attention."""

import time
import torch
import torch.cuda.amp as amp
from contextlib import contextmanager

@contextmanager
def benchmark_context(name: str, warmup: int = 10, iters: int = 100):
    """Context manager for benchmarking."""
    # Warmup
    for _ in range(warmup):
        yield

    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iters):
        yield

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"{name}: {elapsed/iters*1000:.2f} ms/iter")

def benchmark_attention_variants():
    """Compare standard vs Flash Attention."""
    B, L, D = 64, 25, 384
    nhead = 12

    # Create models
    std_layer = torch.nn.TransformerDecoderLayer(
        d_model=D, nhead=nhead, dim_feedforward=1536, batch_first=True
    ).cuda()

    flash_layer = create_flash_decoder_layer(
        d_model=D, nhead=nhead, dim_feedforward=1536
    ).cuda()

    # Create inputs
    tgt = torch.randn(B, L, D, device='cuda')
    memory = torch.randn(B, 196, D, device='cuda')  # 14x14 patches

    # Benchmark standard
    with benchmark_context("Standard Attention"):
        with amp.autocast():
            _ = std_layer(tgt, memory)

    # Benchmark Flash
    with benchmark_context("Flash Attention"):
        with amp.autocast():
            _ = flash_layer(tgt, memory)

    # Memory profiling
    torch.cuda.reset_peak_memory_stats()
    with amp.autocast():
        _ = std_layer(tgt, memory)
    std_mem = torch.cuda.max_memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    with amp.autocast():
        _ = flash_layer(tgt, memory)
    flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\nMemory Usage:")
    print(f"  Standard: {std_mem:.2f} MB")
    print(f"  Flash: {flash_mem:.2f} MB")
    print(f"  Reduction: {(1 - flash_mem/std_mem)*100:.1f}%")
```

### 2.3 CI/CD Integration Scripts

**Pre-commit Hook for Type Safety**:
```bash
# .pre-commit-config.yaml additions

repos:
  - repo: local
    hooks:
      - id: mypy-recognition
        name: Type check recognition domain
        entry: mypy ocr/domains/recognition --config-file pyproject.toml
        language: system
        pass_filenames: false

      - id: validate-plm-extraction
        name: Validate PLM extraction (if exists)
        entry: python scripts/validation/validate_plm_extraction.py
        language: system
        pass_filenames: false
        files: 'ocr/domains/recognition/models/(plm|decoder)\.py'
```

---

## 3. High-Risk Areas & Additional Research Topics

### 3.1 Critical High-Risk Areas

#### Risk 1: EOS Token Handling (SEVERITY: CRITICAL)
**Problem**: EOS must be removed after 2nd permutation to prevent over-weighting.

**Current Knowledge Gap**:
- parseq_official_adapter.py:243-248 shows the logic
- But WHY after 2nd permutation specifically?
- What happens if removed earlier/later?

**Additional Research Needed**:
- [ ] Search PARSeq paper (arxiv.org/abs/2207.06966) for EOS removal rationale
- [ ] Check baudm/parseq GitHub issues for discussions
- [ ] Test loss convergence with different removal timings

**Mitigation**:
```python
# Add explicit validation in PLM module
def compute_plm_loss(...):
    """Compute loss with EOS removal tracking."""
    eos_removed = False
    for i, perm in enumerate(perms):
        ...
        if i == 1:
            if not eos_removed:
                # Remove EOS
                tgt_out = torch.where(tgt_out == eos_id, pad_id, tgt_out)
                eos_removed = True
                logging.debug(f"EOS removed after permutation {i}")

        # Validate no EOS in loss computation after removal
        if eos_removed:
            assert (tgt_out == eos_id).sum() == 0, \
                "EOS tokens still present after removal!"
```

**Test Case**:
```python
def test_eos_removal_timing():
    """Verify EOS removal happens at correct iteration."""
    # Track when EOS is present in targets
    eos_presence = []

    for i, perm in enumerate(perms):
        has_eos = (tgt_out == eos_id).any().item()
        eos_presence.append((i, has_eos))

    # Assertions
    assert eos_presence[0][1] == True, "EOS should be present in perm 0"
    assert eos_presence[1][1] == True, "EOS should be present in perm 1"
    assert eos_presence[2][1] == False, "EOS should be removed after perm 1"
```

#### Risk 2: Attention Mask Semantics (SEVERITY: HIGH)
**Problem**: PyTorch uses True=ignore, but some implementations use True=attend.

**Current Knowledge Gap**:
- parseq_official_adapter.py uses specific mask convention
- Flash Attention may have different convention
- nn.TransformerDecoderLayer convention unclear

**Additional Research Needed**:
- [ ] Document exact mask semantics in parseq_official_adapter.py
- [ ] Check F.scaled_dot_product_attention mask semantics
- [ ] Create test to validate mask convention

**Proposed Research Script**:
```python
# scripts/research/attention_mask_semantics.py

"""Research attention mask conventions."""

import torch
import torch.nn.functional as F

def test_mask_convention():
    """Determine PyTorch attention mask convention."""
    Q = torch.randn(1, 1, 4, 64)  # [B, nhead, L, head_dim]
    K = Q.clone()
    V = Q.clone()

    # Create mask: True for position 2
    mask = torch.zeros(4, 4, dtype=torch.bool)
    mask[:, 2] = True  # Mask out position 2

    # Standard attention
    attn_std = F.multi_head_attention_forward(...)

    # Flash attention
    attn_flash = F.scaled_dot_product_attention(
        Q, K, V, attn_mask=mask
    )

    # If outputs differ, conventions differ!
    print(f"Standard: {attn_std}")
    print(f"Flash: {attn_flash}")
    print(f"Convention: True = {'ignore' if torch.allclose(attn_std, attn_flash) else 'DIFFERENT!'}")
```

**Mitigation**:
- Create explicit mask convention documentation
- Add runtime assertion to validate mask behavior
- Use consistent mask creation utilities

#### Risk 3: Permutation Sampling Randomness (SEVERITY: MEDIUM)
**Problem**: Random permutation sampling for >4-char sequences may cause non-deterministic training.

**Current Knowledge Gap**:
- Does randomness affect reproducibility?
- Should we use seeded RNG?
- How does this interact with DistributedDataParallel?

**Additional Research Needed**:
- [ ] Check if baudm/parseq uses seeded RNG
- [ ] Test training with fixed vs random seeds
- [ ] Document best practices for distributed training

**Proposed Solution**:
```python
class PermutationLanguageModeling(nn.Module):
    def __init__(self, ..., seed: int | None = None):
        super().__init__()
        # Use numpy RNG for reproducibility
        self.rng = np.random.default_rng(seed)

    def gen_tgt_perms(self, tgt):
        """Generate permutations with optional seeding."""
        # Use self.rng instead of global random
        sampled = self.rng.choice(all_perms, size=k, replace=False)
        ...
```

### 3.2 Research Topics for GitHub

#### Topic 1: Alternative PLM Implementations
**Search Query**: "permutation language modeling OCR site:github.com"

**Expected Findings**:
- Custom adaptations of PARSeq
- Simplified PLM implementations
- Performance comparisons

**Action**:
```python
# Use GitHub API to search
from github import Github

g = Github()
repos = g.search_repositories(
    query="parseq permutation language modeling",
    sort="stars"
)

for repo in repos[:10]:
    print(f"{repo.full_name}: {repo.description}")
    # Clone and analyze implementation
```

#### Topic 2: Flash Attention Production Usage
**Search Query**: "F.scaled_dot_product_attention transformer site:github.com"

**Expected Findings**:
- Real-world Flash Attention implementations
- Common pitfalls and solutions
- Fallback strategies

**Specific Repos to Check**:
1. `facebookresearch/llama` - LLaMA uses Flash Attention
2. `Dao-AILab/flash-attention` - Official Flash Attention repo
3. `pytorch/torchtune` - PyTorch official fine-tuning library

#### Topic 3: Autoregressive Decoder Patterns
**Search Query**: "autoregressive decoder transformer training inference site:github.com"

**Expected Findings**:
- Mode switching patterns (train vs inference)
- Efficient inference implementations
- Beam search with permutations

### 3.3 Perplexity Deep Research Topics

**Topic 1: PARSeq PLM Theoretical Foundation**
```python
mcp__perplexity__deep_research(
    query="""
    Explain the theoretical foundation of Permutation Language Modeling in PARSeq:
    1. Why use permutations instead of left-to-right only?
    2. Why remove EOS after 2nd permutation specifically?
    3. What is the mathematical justification for loss weighting?
    4. How does PLM improve over autoregressive-only training?

    Include citations to original PARSeq paper and related work.
    """,
    focus_areas=["theoretical justification", "EOS removal rationale", "loss weighting"]
)
```

**Topic 2: Flash Attention Memory Hierarchy**
```python
mcp__perplexity__deep_research(
    query="""
    Explain Flash Attention's memory optimization strategy:
    1. How does it reduce memory from O(N²) to O(N)?
    2. What are the hardware-specific optimizations for Ampere GPUs?
    3. Why is fp16 required vs fp32?
    4. What are the failure modes and fallback strategies?

    Include performance benchmarks on RTX 3090 specifically.
    """,
    focus_areas=["memory optimization", "Ampere specifics", "fp16 requirements"]
)
```

---

## 4. Phase-Specific Context Bundles

### 4.1 Context Bundle Strategy

**Problem**: Multi-session implementation requires consistent context across phases.

**Solution**: Pre-defined context bundles for each phase that can be loaded on-demand.

### 4.2 Phase 1: PLM Extraction Context Bundle

**Bundle Name**: `plm-extraction-phase1`

**Critical Files** (Auto-load):
```yaml
tier1_critical:
  - ocr/domains/recognition/models/parseq_official_adapter.py
    lines: [92-156, 204-248]  # PLM logic
    priority: CRITICAL
    mode: full_content

  - ocr/core/interfaces/models.py
    priority: HIGH
    mode: full_content
    reason: BaseDecoder contract

  - configs/domain/recognition.yaml
    priority: HIGH
    mode: full_content
    reason: Domain configuration

tier2_reference:
  - ocr/domains/recognition/models/decoder.py
    lines: [1-100]  # Current decoder implementation
    priority: MEDIUM
    mode: structure_only

  - ocr/domains/recognition/models/architecture.py
    priority: MEDIUM
    mode: structure_only
    reason: Model composition

tier3_validation:
  - tests/unit/test_parseq.py
    priority: LOW
    mode: reference_only
    reason: Existing test patterns
```

**Research Documents**:
- walkthrough_parseq_plm_flash_refactor.md (lines 1-150)
- specification.md (Section: FR1, AR1)
- implementation_plan.md (Phase 1 only)

**Tools Available**:
- `adt_meta_query kind=ast_dump` - Compare AST structure
- `adt_meta_query kind=imports` - Verify dependencies
- `validate_plm_extraction.py` - Numerical equivalence

**Exit Criteria**:
- [ ] PLM module created with 100% test coverage
- [ ] Numerical equivalence validated (ε ≤ 1e-5)
- [ ] Zero circular imports
- [ ] Type checking passes (mypy)

### 4.3 Phase 2: Decoder Integration Context Bundle

**Bundle Name**: `decoder-integration-phase2`

**Critical Files**:
```yaml
tier1_critical:
  - ocr/domains/recognition/models/plm.py
    priority: CRITICAL
    mode: full_content
    reason: PLM module from Phase 1

  - ocr/domains/recognition/models/decoder.py
    priority: CRITICAL
    mode: full_content
    reason: Target for integration

  - ocr/core/interfaces/decoder.py
    priority: HIGH
    mode: full_content
    reason: Decoder contracts (NEW)

tier2_reference:
  - ocr/domains/recognition/models/parseq_official_adapter.py
    lines: [204-285]  # forward_train implementation
    priority: HIGH
    mode: reference_only
    reason: Loss computation reference

  - ocr/domains/recognition/models/head.py
    priority: MEDIUM
    mode: structure_only
    reason: Head interface

tier3_config:
  - configs/model/architectures/parseq_atomic.yaml
    priority: MEDIUM
    mode: full_content
    reason: New architecture config (to be created)
```

**Research Documents**:
- implementation_plan.md (Phase 2 only)
- specification.md (Section: FR2, AR2)

**Tools Available**:
- `check_numerical_equivalence.py` - Compare atomic vs monolithic
- `adt_meta_query kind=dependency_graph` - Verify no circular deps
- Integration test template

**Exit Criteria**:
- [ ] Decoder accepts PLM module
- [ ] forward_train() matches reference (ε ≤ 1e-5)
- [ ] Training convergence validated (10 epochs)
- [ ] Gradient flow verified

### 4.4 Phase 3: Flash Attention Context Bundle

**Bundle Name**: `flash-attention-phase3`

**Critical Files**:
```yaml
tier1_critical:
  - ocr/domains/recognition/models/decoder.py
    priority: CRITICAL
    mode: full_content
    reason: Target for Flash integration

  - ocr/core/interfaces/flash_constraints.py
    priority: HIGH
    mode: full_content
    reason: Flash constraints (NEW)

  - ocr/domains/recognition/models/flash_attention.py
    priority: CRITICAL
    mode: full_content
    reason: Flash layer implementation (NEW)

tier2_reference:
  - PyTorch docs: F.scaled_dot_product_attention
    url: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    priority: HIGH
    mode: external_reference
```

**Research Documents**:
- implementation_plan.md (Phase 3 only)
- specification.md (Section: PF1, AR3)
- research_addendum_additional_findings.md (Flash Attention section)

**Tools Available**:
- `flash_attention_bench.py` - Performance benchmarking
- `check_numerical_equivalence.py` - Flash vs standard comparison

**Exit Criteria**:
- [ ] Flash layer implemented
- [ ] Numerical equivalence (ε ≤ 1e-3 fp16)
- [ ] 2x throughput improvement verified
- [ ] Memory usage ≤ baseline

### 4.5 Phase 4: Configuration Context Bundle

**Bundle Name**: `hydra-config-phase4`

**Critical Files**:
```yaml
tier1_critical:
  - configs/domain/recognition.yaml
    priority: CRITICAL
    mode: full_content

  - configs/model/architectures/parseq_atomic_flash.yaml
    priority: CRITICAL
    mode: full_content
    reason: New architecture config

  - AgentQMS/specs/tier2-framework/patterns.spec.md
    priority: HIGH
    mode: full_content
    reason: Hydra v5 patterns

tier2_validation:
  - scripts/utils/show_config.py
    priority: MEDIUM
    mode: reference_only
    reason: Config debugging
```

**Research Documents**:
- implementation_plan.md (Phase 4 only)
- specification.md (Section: AR4)

**Tools Available**:
- `show_config.py` - Hydra composition debugging
- `adt_meta_query kind=hydra_usage` - Config analysis

**Exit Criteria**:
- [ ] Architecture config created
- [ ] Domain injection configured
- [ ] Hydra composition validated
- [ ] No namespace collisions

### 4.6 Phase 5: Validation Context Bundle

**Bundle Name**: `full-validation-phase5`

**Critical Files**:
```yaml
tier1_critical:
  - Full codebase (recognition domain)
    priority: CRITICAL
    mode: full_access

  - Performance benchmarks
    priority: HIGH
    mode: results_only

tier2_documentation:
  - docs/design/parseq_atomic_flash.md
    priority: HIGH
    mode: full_content
    reason: Design document (to be created)
```

**Research Documents**:
- implementation_plan.md (Phase 5 only)
- specification.md (All sections)
- All validation test results

**Exit Criteria**:
- [ ] Full training run completed
- [ ] Edge cases tested
- [ ] Documentation complete
- [ ] Production ready

---

## 5. Missing Documentation Gaps

### 5.1 Critical Missing Documentation

#### Gap 1: PLM Theoretical Documentation
**Missing File**: `docs/theory/permutation_language_modeling.md`

**Required Content**:
- Mathematical foundation of PLM
- Why K=6 permutations (not 4 or 8)
- EOS removal rationale
- Loss weighting justification
- Comparison with standard AR training

**Priority**: HIGH - Needed for Phase 1

#### Gap 2: Flash Attention Integration Guide
**Missing File**: `docs/guides/flash_attention_integration.md`

**Required Content**:
- Hardware requirements (Ampere+, CUDA capability)
- PyTorch version compatibility matrix
- fp16 vs fp32 tradeoffs
- Fallback strategies for incompatible hardware
- Performance tuning guide

**Priority**: MEDIUM - Needed for Phase 3

#### Gap 3: Decoder Mode Switching Pattern
**Missing File**: `docs/patterns/decoder_mode_switching.md`

**Required Content**:
- Training vs inference mode differences
- When to use each mode
- Performance implications
- Example usage patterns

**Priority**: HIGH - Needed for Phase 2

#### Gap 4: Type Safety Guidelines
**Missing File**: `docs/contributing/type_safety.md`

**Required Content**:
- Mypy configuration
- Protocol usage patterns
- Runtime type checking guidelines
- When to use TypeGuard

**Priority**: MEDIUM - Needed before Phase 1

### 5.2 AI-Facing Documentation Needs

**Location**: `ocr/domains/recognition/.ai-instructions/`

**Missing Files**:
1. `plm_gotchas.md` - Common PLM pitfalls
2. `flash_attention_constraints.md` - Flash requirements
3. `decoder_testing.md` - How to test decoders
4. `numerical_validation.md` - Equivalence testing guide

**Priority**: HIGH - Create during Phase 1

---

## 6. Development Environment Feedback

### 6.1 Pain Points Identified

#### Pain Point 1: No Recognition Unit Tests
**Observation**: `tests/unit/recognition/` directory does not exist.

**Impact**:
- No existing test patterns to follow
- Higher risk of regression
- Harder to validate PLM extraction

**Recommendation**:
```bash
# Create test structure
mkdir -p tests/unit/recognition
touch tests/unit/recognition/__init__.py
touch tests/unit/recognition/test_plm.py
touch tests/unit/recognition/test_decoder.py
touch tests/unit/recognition/test_flash_attention.py
```

**Priority**: CRITICAL - Create before Phase 1

#### Pain Point 2: Ambiguous BaseDecoder Contract
**Observation**: BaseDecoder.forward() signature is vague about targets parameter.

**Current**:
```python
def forward(self, features: list[torch.Tensor], targets: torch.Tensor = None) -> torch.Tensor:
```

**Problem**:
- "Optional" suggests inference mode, but how to switch modes?
- What does output tensor represent? Logits? Features?
- No mode parameter

**Recommendation**:
```python
# Proposed new signature
def forward(
    self,
    features: list[torch.Tensor],
    targets: torch.Tensor | None = None,
    mode: str = "train",
    **kwargs
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Args:
        features: Encoder outputs
        targets: Target tokens (required for mode='train')
        mode: 'train' or 'inference'

    Returns:
        Training: dict with 'logits' and 'loss'
        Inference: logits tensor
    """
```

**Priority**: HIGH - Address in Phase 2

#### Pain Point 3: No AST Validation in CI
**Observation**: No automated AST structure validation after refactors.

**Impact**:
- Hard to catch structural regressions
- Manual verification required
- Risky refactors

**Recommendation**:
```yaml
# .github/workflows/structure_validation.yml

name: Structure Validation
on: [pull_request]

jobs:
  ast-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate PLM structure
        run: |
          python scripts/validation/validate_ast_structure.py \
            --reference ocr/domains/recognition/models/parseq_official_adapter.py:gen_tgt_perms \
            --target ocr/domains/recognition/models/plm.py:gen_tgt_perms
```

**Priority**: MEDIUM - Add during Phase 1

#### Pain Point 4: Verbose Hydra Debugging
**Observation**: Debugging Hydra config composition requires manual inspection.

**Impact**:
- Time-consuming config debugging
- Hard to trace namespace issues
- Frustrating for newcomers

**Recommendation**:
```python
# scripts/utils/hydra_diff.py

"""Compare two Hydra configs and show differences."""

import hydra
from omegaconf import OmegaConf

def diff_configs(config1_path: str, config2_path: str):
    """Show differences between two configs."""
    cfg1 = hydra.compose(config_name=config1_path)
    cfg2 = hydra.compose(config_name=config2_path)

    # Flatten to dict
    dict1 = OmegaConf.to_container(cfg1, resolve=True)
    dict2 = OmegaConf.to_container(cfg2, resolve=True)

    # Deep diff
    from deepdiff import DeepDiff
    diff = DeepDiff(dict1, dict2, view='tree')

    print(diff.pretty())
```

**Priority**: LOW - Nice to have

### 6.2 Tooling Gaps

#### Gap 1: No Profiling Utilities
**Missing**: GPU memory profiling, throughput measurement

**Recommendation**: Create `scripts/profiling/` directory with:
- `profile_memory.py` - CUDA memory profiling
- `profile_throughput.py` - Images/second measurement
- `profile_convergence.py` - Training speed tracking

**Priority**: MEDIUM - Needed for Phase 3

#### Gap 2: No Automated Benchmark Comparison
**Missing**: Tool to compare benchmark results over time

**Recommendation**: Create `scripts/benchmark/compare_runs.py`

**Priority**: LOW - Nice to have

---

## 7. Productivity Recommendations

### 7.1 Session Handover Template

Create structured handover documents for each session:

**Template**:
```markdown
# Session Handover: {DATE} - {PULSE_ID}

## Context
- **Phase**: {Phase Number}
- **Objective**: {What was attempted}
- **Duration**: {Hours worked}

## Completed
- [x] Task 1
- [x] Task 2

## Blocked/In Progress
- [ ] Task 3 - BLOCKED: Reason
- [ ] Task 4 - IN PROGRESS: Status

## Next Session
- [ ] Task 5 - Start here
- [ ] Task 6 - Then this

## Key Files Modified
- file1.py:120-150 - Change description
- file2.yaml - Config update

## Tests Added/Modified
- test_plm.py::test_gen_tgt_perms - Added
- test_decoder.py::test_forward - Modified

## Research Findings
- Finding 1: Details
- Finding 2: Details

## Questions for Next Session
1. Question 1?
2. Question 2?

## Continuation Prompt
```
Load Phase {N} context bundle.
Previous session completed {tasks}.
Start with {next_task}.
Reference: {handover_doc}
```
```

### 7.2 Cognitive Complexity Management

**Strategy**: Break phases into smaller checkpoints with clear validation.

**Example for Phase 1**:
```markdown
## Phase 1 Checkpoints

### Checkpoint 1.1: PLM Module Structure (30 min)
- [ ] Create plm.py file
- [ ] Define PLMConfig dataclass
- [ ] Define AttentionMasks dataclass
- [ ] Validate: File imports, no errors

### Checkpoint 1.2: gen_tgt_perms Extraction (2 hours)
- [ ] Copy gen_tgt_perms from adapter
- [ ] Adapt device handling
- [ ] Add type annotations
- [ ] Validate: AST structure matches

### Checkpoint 1.3: Unit Tests (3 hours)
- [ ] Test 1-char case
- [ ] Test 4-char case (hardcoded selector)
- [ ] Test random sampling
- [ ] Validate: All tests pass

### Checkpoint 1.4: Numerical Equivalence (1 hour)
- [ ] Run validation script
- [ ] Compare outputs (ε ≤ 1e-5)
- [ ] Document any discrepancies
- [ ] Validate: 100% equivalence
```

**Benefit**: Clear stopping/starting points for sessions.

### 7.3 Automated Session Context Loading

**Proposal**: Create a context loader script that agents can invoke.

**Script**:
```python
# scripts/session/load_context.py

"""Load phase-specific context for agent sessions."""

import sys
from pathlib import Path

def load_phase_context(phase: int) -> dict:
    """Load context bundle for specific phase."""
    bundles = {
        1: "plm-extraction-phase1",
        2: "decoder-integration-phase2",
        3: "flash-attention-phase3",
        4: "hydra-config-phase4",
        5: "full-validation-phase5",
    }

    bundle_name = bundles.get(phase)
    if not bundle_name:
        raise ValueError(f"Invalid phase: {phase}")

    # Load context bundle via AgentQMS
    context = get_context_bundle(
        task_description=f"Load {bundle_name} context",
        task_type=bundle_name
    )

    return context

def print_session_prompt(phase: int, prev_handover: Path):
    """Generate continuation prompt for agents."""
    context = load_phase_context(phase)

    print(f"""
# Session Continuation Prompt

**Phase**: {phase}
**Previous Handover**: {prev_handover}
**Context Bundle**: {context['detected']['context_bundle']}

## Critical Files Loaded
{format_files(context['files'])}

## Tools Available
- AST analyzers (adt_meta_query)
- Validation scripts
- Benchmark utilities

## Next Steps
1. Read previous handover: {prev_handover}
2. Review Phase {phase} implementation plan
3. Start with next checkpoint
4. Validate incrementally

## Exit Criteria for Phase {phase}
{get_exit_criteria(phase)}
    """)

if __name__ == "__main__":
    phase = int(sys.argv[1])
    handover = Path(sys.argv[2])
    print_session_prompt(phase, handover)
```

**Usage**:
```bash
python scripts/session/load_context.py 2 pulse_staging/artifacts/session_handover_2026-02-12.md
```

---

## 8. Summary & Action Items

### 8.1 Immediate Actions (Before Phase 1)

**Critical (Block Phase 1)**:
- [ ] Create data contract files:
  - `ocr/core/interfaces/plm.py`
  - `ocr/core/interfaces/decoder.py`
  - `ocr/core/interfaces/flash_constraints.py`
- [ ] Create test directory: `tests/unit/recognition/`
- [ ] Create validation scripts:
  - `scripts/validation/validate_plm_extraction.py`
  - `scripts/validation/check_numerical_equivalence.py`
- [ ] Configure mypy for recognition domain
- [ ] Create Phase 1 context bundle documentation

**High Priority (Parallel to Phase 1)**:
- [ ] Research EOS removal rationale (Perplexity Deep Research)
- [ ] Document attention mask semantics
- [ ] Create missing documentation:
  - `docs/theory/permutation_language_modeling.md`
  - `docs/patterns/decoder_mode_switching.md`

### 8.2 Ongoing Throughout Implementation

**Per Phase**:
- [ ] Load phase-specific context bundle
- [ ] Use AST analyzers for structural validation
- [ ] Run numerical equivalence checks
- [ ] Document findings in phase-specific handover
- [ ] Update exit criteria tracking

**Per Session**:
- [ ] Create session handover document
- [ ] Update Project Compass pulse state
- [ ] Archive artifacts if session ends
- [ ] Provide continuation prompt

### 8.3 Post-Implementation (Phase 5)

**Documentation**:
- [ ] Complete all missing docs
- [ ] Create AI-facing instructions
- [ ] Write migration guide

**Validation**:
- [ ] Full benchmark suite
- [ ] Edge case testing
- [ ] Production readiness checklist

---

## 9. Success Metrics

### 9.1 Technical Metrics
- **Numerical Equivalence**: ε ≤ 1e-5 (fp32), ε ≤ 1e-3 (fp16)
- **Performance**: ≥2x throughput improvement
- **Memory**: ≤ baseline VRAM usage
- **Convergence**: Loss curves match ±1%

### 9.2 Process Metrics
- **Test Coverage**: 100% for PLM module
- **Type Safety**: 0 mypy errors in recognition domain
- **Session Efficiency**: Clear handover documents
- **Documentation**: All gaps filled

### 9.3 Quality Metrics
- **Code Review**: No architectural violations
- **AST Validation**: Structural equivalence confirmed
- **Integration**: Smooth Hydra composition

---

## Appendix A: Tool Reference Matrix

| Phase | AST Tools | Validation Scripts | Research Tools |
|-------|-----------|-------------------|----------------|
| 1 | dependency_graph, ast_dump, imports | validate_plm_extraction.py | Perplexity (EOS) |
| 2 | imports, symbol_search | check_numerical_equivalence.py | GitHub (decoders) |
| 3 | complexity | flash_attention_bench.py | Perplexity (Flash) |
| 4 | hydra_usage | show_config.py | - |
| 5 | (all) | (all) | - |

---

## Appendix B: Context Bundle File Inventory

**Phase 1 Files** (15 files, ~8000 tokens):
- parseq_official_adapter.py (critical sections)
- BaseDecoder interface
- PLM contracts (new)
- Type guards (new)
- Unit test templates

**Phase 2 Files** (20 files, ~12000 tokens):
- Phase 1 outputs
- Decoder implementation
- Loss computation reference
- Integration test templates

**Phase 3 Files** (18 files, ~10000 tokens):
- Phase 2 outputs
- Flash constraints
- PyTorch Flash docs
- Benchmark scripts

**Phase 4 Files** (12 files, ~6000 tokens):
- Config files
- Hydra patterns spec
- Domain injection examples

**Phase 5 Files** (all recognized domain):
- Full codebase access
- All validation results
- Documentation drafts

---

**Status**: Ready for Review
**Next**: Get approval, then begin Phase 1 with data contracts creation.
