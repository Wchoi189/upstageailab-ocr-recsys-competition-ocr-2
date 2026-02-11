# Session Handover: PARSeq Recognition Pipeline Optimization
**Date**: 2026-02-12
**Pulse ID**: recognition-parseq-optimization
**Milestone**: v1.0-recognition-optimization
**Phase**: Recognition

---

## Context
The current recognition pipeline has two implementations:
1. **PARSeqOfficial** (Monolithic) - Functionally correct but rigid (hardcoded ViT, legacy attention)
2. **PARSeqDecoder** (Atomic) - Architecturally modular but **incomplete** (missing PLM logic)

**Goal**: Create a **True Atomic PARSeq** that combines modularity with full PLM functionality and Flash Attention for 2-4x performance improvement.

## Audit Directives Summary
Key findings from [parseq_audit_directives.md.resolved](file:///home/vscode/.gemini/antigravity/brain/e31aa158-3485-430b-a432-c554745646eb/parseq_audit_directives.md.resolved):

### Issues with Monolithic Implementation
- **Backbone**: Hardcoded ViT prevents using project's `TimmBackbone` wrapper
- **Attention**: Standard `nn.MultiheadAttention` is a performance bottleneck vs Flash Attention
- **Logic**: PLM logic tightly coupled in forward method

### Issues with Atomic Implementation
- **Status**: Incomplete/Misleading - implements standard autoregressive decoder only
- **Missing**:
  - Permutation generation (`gen_tgt_perms`)
  - Context/Query padding mask generation (`generate_attn_masks`)
  - Two-stream attention support

## Requirements Gathered

### Functional Requirements
- Port `gen_tgt_perms()` and `generate_attn_masks()` from [PARSeqOfficial](file:///workspaces/ocr/domains/recognition/models/parseq_official_adapter.py) to [PARSeqDecoder](file:///workspaces/ocr/domains/recognition/models/decoder.py)
- Implement `FlashDecoderLayer` using `torch.nn.functional.scaled_dot_product_attention`
- Support both training (permutation loop) and inference (autoregressive) modes
- Maintain compatibility with existing tokenizer and loss configurations
- Accept memory input `[B, S, D]` from any encoder backbone

### Performance Requirements
- Achieve 2-4x throughput improvement vs baseline (nn.MultiheadAttention)
- Support batch sizes up to 64 on RTX 3090 (24GB VRAM)
- Maintain training convergence comparable to monolithic implementation

### Architecture Requirements
- Follow [Core/Interfaces pattern](file:///workspaces/AgentQMS/specs/tier2-framework/core-interfaces.spec.md) for shared validation models
- Use [Hydra V5 self-mounting pattern](file:///workspaces/AgentQMS/specs/tier2-framework/patterns.spec.md) with @package directives
- Implement Domain Injection pattern (tokenizer/loss in domain config)
- Ensure atomic components are independently testable

### Validation Requirements
- Unit tests for permutation generation correctness
- Unit tests for attention mask generation
- Integration tests comparing atomic vs monolithic loss curves
- Benchmark throughput on standardized dataset (img/sec)

## Project Principles Established

### Architecture
- **Atomic Design**: Components are modular and composable via Hydra config
- **Core/Interfaces Pattern**: Shared validation models live in `ocr/core/interfaces/`
- **Domain Isolation**: Recognition logic independent of detection/KIE domains
- **Backbone Agnostic**: Decoder accepts `[B, S, D]` memory from any encoder

### Performance
- **Flash Attention**: Use `torch.nn.functional.scaled_dot_product_attention` for 2-4x speedup
- **Memory Efficiency**: Minimize redundant tensor copies and allocations
- **GPU Optimization**: Leverage CUDA-optimized operations where available

### Configuration
- **Hydra V5**: Follow self-mounting pattern with @package directives
- **Domain Injection**: Data-dependent components (tokenizer, loss) in domain configs
- **No Presets**: Architecture configs contain ONLY neural network structure

## Implementation Strategy

### Phase 1: PLM Logic Port (Critical Path)
**Goal**: Make PARSeqDecoder functionally equivalent to PARSeqOfficial

**Tasks**:
1. Extract permutation generation logic from [parseq_official_adapter.py:92-141](file:///workspaces/ocr/domains/recognition/models/parseq_official_adapter.py#L92-L141)
2. Create atomic PLM module in `recognition/models/plm.py` with:
   - `gen_tgt_perms()` - generates shared permutations for batch
   - `generate_attn_masks()` - creates content/query attention masks
3. Update [PARSeqDecoder.forward()](file:///workspaces/ocr/domains/recognition/models/decoder.py#L81-L152) to support mode parameter (train/inference)
4. Implement `forward_train()` with permutation loop
5. Unit test permutation generation against official implementation

**Risk Mitigation**:
- Validate permutation outputs match official implementation exactly
- Test with edge cases (1-char sequences, max length sequences)
- Verify training loss curves match monolithic baseline

### Phase 2: Flash Attention Integration (Performance)
**Goal**: Replace `nn.TransformerDecoderLayer` with Flash Attention

**Tasks**:
1. Create `FlashDecoderLayer` class using `F.scaled_dot_product_attention`
2. Implement custom `TransformerDecoder` stack with Flash layers
3. Add fallback to standard attention for compatibility
4. Benchmark throughput improvement (img/sec)

**Configuration**:
- Add `use_flash_attention` flag in architecture config
- Default to True on CUDA devices with sm_80+ (Ampere/Ada)
- Auto-fallback on CPU or older GPUs

**Risk Mitigation**:
- Verify attention outputs are numerically equivalent
- Test with different sequence lengths and batch sizes
- Profile memory usage to ensure no regressions

### Phase 3: Configuration and Integration
**Goal**: Enable atomic PARSeq via Hydra config

**Tasks**:
1. Create `configs/model/architectures/parseq_atomic.yaml`
2. Move tokenizer/loss to `configs/domain/recognition.yaml` (Domain Injection)
3. Update PARSeq architecture class to use atomic decoder
4. Add gradient flow validation

**Example Configuration**:
```yaml
# configs/model/architectures/parseq_atomic.yaml
# @package model.architectures
_target_: ocr.domains.recognition.models.PARSeq
backbone:
  _target_: ocr.core.models.encoder.TimmBackbone
  model_name: resnet18
  pretrained: true
decoder:
  _target_: ocr.domains.recognition.models.decoder.PARSeqDecoder
  d_model: 384
  nhead: 12
  num_layers: 12
  use_flash_attention: true
head:
  _target_: ocr.domains.recognition.models.head.ClassificationHead
max_len: 25
```

### Phase 4: Validation and Benchmarking
**Goal**: Verify correctness and measure performance gains

**Validation Tests**:
1. Permutation generation correctness
2. Attention mask correctness
3. Training convergence (loss curves)
4. Inference accuracy (CER/WER metrics)

**Performance Benchmarks**:
1. Throughput (img/sec) - baseline vs atomic vs atomic+flash
2. Memory usage (VRAM peak)
3. Training time per epoch

**Success Criteria**:
- Training loss matches monolithic ±1%
- Inference accuracy matches monolithic ±0.5% CER
- Throughput improvement ≥2x with Flash Attention
- VRAM usage ≤ monolithic baseline

### Phase 5: Documentation and Cleanup
**Goal**: Prepare for production deployment

**Tasks**:
1. Update `.ai-instructions/` for recognition domain
2. Document PLM logic in docstrings
3. Add performance comparison table to docs/
4. Archive legacy PARSeqOfficial (move to `ocr/vendor/legacy/`)
5. Update training configs to use atomic implementation

## Implementation Order
1. PLM logic extraction and testing (blocking)
2. Flash Attention layer implementation (parallel with #3)
3. Configuration setup and Hydra integration (parallel with #2)
4. Full integration and gradient flow validation (requires #1-3)
5. Benchmarking and validation (requires #4)
6. Documentation and cleanup (requires #5)

## Key Files for Implementation

### Source Files (Read)
- [ocr/domains/recognition/models/parseq_official_adapter.py](file:///workspaces/ocr/domains/recognition/models/parseq_official_adapter.py) - Extract PLM logic
- [ocr/domains/recognition/models/decoder.py](file:///workspaces/ocr/domains/recognition/models/decoder.py) - Atomic decoder to upgrade
- [ocr/domains/recognition/models/architecture.py](file:///workspaces/ocr/domains/recognition/models/architecture.py) - PARSeq orchestration

### Configuration Files (Read/Modify)
- [configs/domain/recognition.yaml](file:///workspaces/configs/domain/recognition.yaml) - Domain controller (tokenizer/loss injection)
- [configs/model/architectures/](file:///workspaces/configs/model/architectures/) - Architecture configs

### Framework References
- [AgentQMS/specs/tier2-framework/patterns.spec.md](file:///workspaces/AgentQMS/specs/tier2-framework/patterns.spec.md) - Hydra V5 patterns
- [AgentQMS/specs/tier2-framework/core-interfaces.spec.md](file:///workspaces/AgentQMS/specs/tier2-framework/core-interfaces.spec.md) - Core/Interfaces pattern

## Context Bundles
The context bundle system identified these critical files:
- `configs/domain/recognition.yaml` (priority: critical)
- `ocr/core/models/__init__.py` (priority: high)
- `configs/data/` (priority: medium)
- `data/data_catalog.yaml` (priority: low)

Access via: `uv run python AgentQMS/tools/utilities/suggest_context.py "recognition optimization"`

## Tools Available

### AgentQMS Framework
- **AQMS CLI**: `aqms artifact validate`, `aqms registry resolve`
- **Context Bundling**: `uv run python AgentQMS/tools/utilities/suggest_context.py "<task>"`
- **MCP Unified Server**: `mcp__unified__get_context_bundle`

### Project Compass
- **Pulse Management**: `compass pulse-status`, `compass pulse-sync`, `compass pulse-export`
- **Spec Kit**: `compass_meta_spec` (constitution, specify, plan, tasks)

### Agent Debug Toolkit
- **Config Analysis**: `uv run adt analyze-config <path>`
- **Merge Tracing**: `uv run adt trace-merges <file> --output markdown`

## Next Steps

### Immediate (Start here)
1. **Extract PLM Logic**:
   ```bash
   # Create PLM module
   touch ocr/domains/recognition/models/plm.py

   # Extract gen_tgt_perms from parseq_official_adapter.py:92-141
   # Extract generate_attn_masks from parseq_official_adapter.py:142-157
   ```

2. **Create Unit Tests**:
   ```bash
   # Create test file
   touch tests/unit/recognition/test_plm.py

   # Test permutation generation against official implementation
   ```

3. **Update PARSeqDecoder**:
   - Add mode parameter to `forward()`
   - Implement `forward_train()` with permutation loop
   - Implement `forward_inference()` with autoregressive decoding

### Follow-up (After Phase 1)
4. **Implement Flash Attention** (Phase 2)
5. **Configure Hydra** (Phase 3)
6. **Validate and Benchmark** (Phase 4)
7. **Document and Cleanup** (Phase 5)

## Continuation Prompt

```
Continue PARSeq optimization from Phase 1. Focus on extracting PLM logic from
parseq_official_adapter.py and implementing it in atomic PARSeqDecoder.

Key tasks:
1. Create ocr/domains/recognition/models/plm.py with gen_tgt_perms() and generate_attn_masks()
2. Update PARSeqDecoder to support training mode with permutation loop
3. Add unit tests verifying permutation generation matches official implementation

Reference audit report at: /home/vscode/.gemini/antigravity/brain/e31aa158-3485-430b-a432-c554745646eb/parseq_audit_directives.md.resolved
```

## Artifacts Generated
- Project Compass pulse initialized: `recognition-parseq-optimization`
- Session handover document: This file
- Principles established (constitution)
- Requirements specification
- Implementation plan with 5 phases
- Detailed task breakdown

## Token Burden
**Current**: Low
**Estimated Peak**: Medium (during Phase 4 validation)

---

**Status**: Ready for implementation
**Blocking Issues**: None
**Required Context**: All gathered and documented above
