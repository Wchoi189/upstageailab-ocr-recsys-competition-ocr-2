# Project Specification

## Scope
Port PARSeq Permutation Language Modeling (PLM) from monolithic implementation to atomic, composable architecture with optional Flash Attention optimization. Maintain exact numerical equivalence while enabling backbone flexibility and 2-4x throughput improvement.

## Requirements
## Functional Requirements

### FR1: PLM Logic Extraction
**Source**: parseq_official_adapter.py:92-156, 204-248
**Components**:
- `gen_tgt_perms(tgt, perm_num=6, perm_forward=True, perm_mirrored=True)` → [K, L] permutation tensor
  - Handle 1-char sequences (return identity)
  - Handle ≤4-char sequences (use all permutations)
  - Handle >4-char sequences (random sampling)
  - Support mirrored pairs (complementary permutations)
  - Prepend BOS=0, append EOS=max_num_chars+1
- `generate_attn_masks(perm)` → (content_mask, query_mask) [L, L] boolean tensors
  - content_mask: [:-1, :-1] - for context tokens
  - query_mask: [1:, :-1] - for query tokens
  - Mask "self" attention (diagonal = True)

### FR2: Training Loop Integration
**Loss Aggregation Pattern** (Critical):
```python
for i, perm in enumerate(perms):
    tgt_mask, query_mask = generate_attn_masks(perm)
    out = decoder(tgt_in, memory, tgt_mask, query_mask)
    logits = head(out)
    loss += n * F.cross_entropy(logits, tgt_out, ignore_index=pad_id)
    loss_numel += n
    # Critical: Remove EOS after 2nd permutation
    if i == 1:
        tgt_out = torch.where(tgt_out == eos_id, pad_id, tgt_out)
        n = (tgt_out != pad_id).sum().item()
loss /= loss_numel
```
**Requirements**:
- Iterate over K permutations
- Weight loss by character count (n)
- Remove EOS tokens after 2nd iteration (prevents over-weighting)
- Normalize by total character count across all permutations

### FR3: Inference Mode
- Standard left-to-right autoregressive decoding
- No permutation generation needed
- Use causal mask: `torch.triu(torch.ones((L, L), bool), 1)`

### FR4: Decoder Interface
**Current**: `PARSeqDecoder.forward(features, targets, memory_key_padding_mask)`
**Required**: Support both training and inference modes
```python
def forward(self, features, targets=None, mode='train', 
            perm_config=None, memory_key_padding_mask=None):
    if mode == 'train':
        return self.forward_train(features, targets, perm_config)
    else:
        return self.forward_inference(features, memory_key_padding_mask)
```

## Performance Requirements

### PF1: Flash Attention Integration
**Target**: 2-4x throughput improvement on RTX 3090 (Ampere, sm_80)
**Implementation**: Replace `nn.TransformerDecoderLayer` with custom layer using `F.scaled_dot_product_attention`
**Constraints**:
- Data type: fp16 or bfloat16 (no fp32 support)
- Head dimension: Must be multiple of 8 (d_model % nhead == 0, d_model/nhead % 8 == 0)
- PyTorch version: ≥2.0 (≥2.2 for optimal FlashAttention-2)
- Backend selection: Auto via `torch.backends.cuda.sdp_kernel(enable_flash=True)`

### PF2: Memory Efficiency
- Batch size support: ≥64 on RTX 3090 (24GB VRAM)
- Memory usage: ≤ monolithic baseline
- No redundant tensor copies in permutation loop

### PF3: Training Convergence
- Loss curves must match monolithic ±1% over 10 epochs
- Character Error Rate (CER): ±0.5% of baseline
- Word Error Rate (WER): ±0.5% of baseline

## Architecture Requirements

### AR1: Modular PLM Component
**Location**: `ocr/domains/recognition/models/plm.py`
**Interface**:
```python
class PermutationLanguageModeling:
    def __init__(self, max_len, perm_num=6, perm_forward=True, perm_mirrored=True):
        ...
    def gen_tgt_perms(self, tgt: Tensor) -> Tensor:
        # Returns [K, L] permutation indices
    def generate_attn_masks(self, perm: Tensor) -> tuple[Tensor, Tensor]:
        # Returns (content_mask, query_mask)
```

### AR2: Atomic Decoder Enhancement
**Location**: `ocr/domains/recognition/models/decoder.py`
**Changes**:
- Add mode parameter to `forward()`
- Implement `forward_train()` with PLM loop
- Implement `forward_inference()` with AR decoding
- Support custom attention masks from PLM

### AR3: Flash Attention Layer
**Location**: `ocr/domains/recognition/models/flash_attention.py`
**Interface**:
```python
class FlashDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        ...
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, 
                memory_key_padding_mask=None):
        # Uses F.scaled_dot_product_attention internally
```

### AR4: Configuration Structure
**New File**: `configs/model/architectures/parseq_atomic_flash.yaml`
```yaml
# @package model.architectures
_target_: ocr.domains.recognition.models.PARSeq
backbone:
  _target_: ocr.core.models.encoder.TimmBackbone
  model_name: resnet18
  pretrained: true
decoder:
  _target_: ocr.domains.recognition.models.decoder.PARSeqDecoder
  d_model: 384
  nhead: 12  # Ensures 384/12=32, which is % 8 == 0
  num_layers: 12
  use_flash_attention: true
  plm_config:
    perm_num: 6
    perm_forward: true
    perm_mirrored: true
head:
  _target_: ocr.domains.recognition.models.head.ClassificationHead
max_len: 25
```

## Validation Requirements

### VR1: Unit Tests
**Critical Test Cases**:
1. **Permutation Generation**:
   - 1-char sequence → identity permutation
   - 2-char sequence → [forward, reverse] if mirrored
   - 4-char sequence → 12 specific permutations (hardcoded selector)
   - 5-char sequence → 6 random permutations
   - Verify BOS prepend, EOS append
2. **Attention Mask Generation**:
   - Verify content_mask shape [L-1, L-1]
   - Verify query_mask shape [L-1, L-1]
   - Verify causal structure (no lookahead)
   - Verify self-masking on diagonal
3. **Loss Aggregation**:
   - Verify EOS removal after 2nd iteration
   - Verify character count weighting
   - Verify normalization

### VR2: Integration Tests
1. **Numerical Equivalence**:
   - Compare atomic PLM vs monolithic output (ε=1e-5 fp32)
   - Compare atomic PLM+Flash vs monolithic (ε=1e-3 fp16)
   - Test on batch sizes: 1, 8, 32, 64
2. **Gradient Flow**:
   - Verify gradients propagate through all permutations
   - Check for NaN/Inf gradients
   - Verify gradient magnitudes match baseline

### VR3: Performance Benchmarks
**Test Dataset**: ICDAR 2015 validation set (500 images)
**Metrics**:
- Throughput (img/sec): Baseline, Atomic, Atomic+Flash
- VRAM peak (GB): Monitor via `torch.cuda.max_memory_allocated()`
- Training time per epoch: On full training set
- Convergence: Loss curves over 10 epochs

### VR4: Edge Case Validation
- Empty sequences (should error gracefully)
- Single character recognition
- Maximum length sequences (L=25)
- Mixed length batches
- GPU fallback on CPU
- Non-Ampere GPU fallback (disable Flash Attention)

## Risk Assessment

### Critical Risks
1. **Loss Divergence**: If EOS removal logic incorrect, loss will not converge
2. **Gradient Explosion**: Improper loss normalization causes training instability
3. **Numerical Drift**: Flash Attention fp16 introduces rounding errors vs fp32 baseline
4. **Mask Mismatch**: Incorrect attention masks break permutation constraints
5. **Memory OOM**: Flash Attention may have different memory profile than standard

### Mitigation Strategies
1. **Incremental Validation**: Test each component before integration
2. **Reference Comparison**: Always validate against parseq_official_adapter.py
3. **Automated Testing**: CI/CD runs full test suite on each commit
4. **Logging**: Detailed loss/gradient logging during training
5. **Checkpointing**: Save checkpoints every epoch for rollback

## Status
- Created: 2026-02-12T03:43:59.030226
- Tool: Project Compass v2
- Status: Draft
