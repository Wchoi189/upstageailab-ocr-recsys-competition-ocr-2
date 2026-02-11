# Project Constitution

## Principles
## PARSeq Optimization: Research-Backed Principles

### Critical Understanding (From Research)
**DO NOT implement from scratch** - PARSeq's Permutation Language Modeling (PLM) is complex and costly to debug. We have working implementations to reference.

### PLM Core Insights
1. **Permutation Sampling**: Uses K random permutations (default K=6), not all T! permutations
2. **Loss Aggregation**: Weights by character count, removes EOS tokens after 2nd permutation
3. **Dual Attention Masks**: content_mask and query_mask enforce permutation ordering
4. **Inference Switch**: Training uses permutations, inference uses standard left-to-right

### Architecture Constraints
- **Atomic Components**: Extract PLM logic to standalone module, not inline in decoder
- **Reference Implementation**: Port from working parseq_official_adapter.py (lines 92-156, 204-248)
- **Numerical Equivalence**: Must match monolithic implementation exactly
- **Testability**: PLM module must be unit-testable in isolation

### Flash Attention Integration Constraints (From Research)
1. **Data Type**: Requires fp16 or bfloat16 (no fp32 support)
2. **Head Dimension**: Must be multiple of 8
3. **Mask Format**: Additive masks with -inf for masked positions
4. **Backend Selection**: Auto-selects on Ampere+ (sm_80), requires PyTorch 2.0+
5. **Numerical Equivalence**: Produces exact same output as standard attention

### Risk Mitigation Strategy
1. **Phase 1 Validation**: Test PLM logic against monolithic before Flash Attention
2. **Incremental Integration**: Add Flash Attention only after PLM works correctly
3. **Numerical Tests**: Compare outputs at each layer (epsilon=1e-5 for fp32, 1e-3 for fp16)
4. **Edge Case Coverage**: Test 1-char sequences, max-length sequences, varied K values
5. **Loss Parity**: Training curves must match monolithic ±1% over 10 epochs

### Configuration Hygiene
- **Hydra V5 Compliance**: Self-mounting @package directives
- **Domain Injection**: Tokenizer/loss in domain config, not architecture
- **No Hardcoding**: All hyperparameters (K, perm_forward, perm_mirrored) in config
- **Fallback Support**: Flash Attention disabled automatically on incompatible hardware

### Documentation Requirements
- **Decision Log**: Document why each approach was chosen
- **Failure Modes**: Document known edge cases and error patterns
- **Debugging Guide**: Step-by-step validation procedures
- **Performance Metrics**: Baseline vs optimized throughput data

## Established
Date: 2026-02-12T03:43:10.762696
Tool: Project Compass v2
