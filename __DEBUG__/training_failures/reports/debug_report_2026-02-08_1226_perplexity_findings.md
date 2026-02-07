# PARSeq Research Findings - Critical Issues Matching Our Symptoms

**Source:** Perplexity Deep Research on PARSeq training failures  
**Date:** 2026-02-08 00:38

---

## Exact Symptom Matches

### 1. **Vanishing Gradients (Confirmed) - Lines 35-41**
> "The vanishing gradient problem becomes particularly acute when training on character sets with variable lengths... Networks initialized with weights drawn from distributions with inappropriate variance can have gradients that begin vanishing immediately during the first training steps."

**Our Symptoms:** ✓ Gradients vanish by step 10-20  
**Cause:** Permutation language modeling creates variable context per character, inducing instability

### 2. **Attention Masking Issues (CRITICAL) - Lines 69-76**
> "The attention masking strategy in the PARSeq decoder differs fundamentally from standard transformer decoders. Rather than applying a simple causal mask to enforce left-to-right information flow, PARSeq generalizes the masking pattern to enforce arbitrary permutation-specific orderings."

**CRITICAL ERROR POSSIBILITY:**
> "If the mask incorrectly allows the model to attend to position πₜ when computing predictions for that position, the model can bypass the learning objective by simply predicting the context token..."

**Action:** Check our decoder implementation for permutation masking bugs

### 3. **BOS/EOS Token Handling (Lines 53-60)**
> "The correct implementation in PARSeq includes a BOS token that is prepended to the ground truth sequence during training... The sequence structure becomes: [BOS] + [ground truth characters] + [EOS padding]."

**CRITICAL DETAIL:**
> "when sampling permutations, the BOS token position must either be fixed at the beginning of each permutation or appropriately handled..."

**Our Status:** Unknown - we haven't verified this yet

### 4. **Positional Encoding Problems (Lines 61-68)**
> "PARSeq employs fully trainable positional embeddings that are initialized randomly... The interaction between positional embeddings and permuted training creates potential for instability."

**Our Findings:** ✓ We identified 8:1 scale disparity  
**Research Confirms:** Positional embeddings + permutations = instability if not tuned

---

## Root Causes Identified in Research

### Issue #143: Digit-Only Vocabulary Failure (Line 103-108)
> "when training PARSeq on a simple vocabulary consisting only of digits (0-9), the training loss fails to converge and the model fails to learn meaningful representations"

**Explanation:**
> "With only 10 possible character outputs, the softmax distribution behaves differently than with larger vocabularies. The model may converge to a degenerate solution..."

**Our Case:** We have 1027 characters, so NOT this issue

### Charset Mismatch (Lines 91-96)
> "The PARSeq embedding layer is initialized with a fixed vocabulary size based on the training character set. When fine-tuning with a different or larger character set, dimension mismatches occur..."

**Action:** Verify our charset is consistent between tokenizer, model, and data

### Validation Sanity Check Failures (Lines 109-114)
> "models that progress through initialization and begin training normally but then fail during the first validation sanity check... typically indicate problems with loss computation on validation data"

**Our Case:** Not experiencing this - we pass validation, just get 0% accuracy

---

## Working Configuration Details

### Batch Size (Lines 119-126)
> "successful trainings typically employ batch sizes in the range of 256-512 when training on multi-GPU configurations"

**Our Config:** 32 (micro-training) - VALID for debugging
**For Production:** Should use 256-512

### Learning Rate (Lines 127-133)
> "Learning rates typically in the range of 7e-4 to 1e-3 for transformer models"  
> "Warmup phase gradually increases learning rate from near-zero over 500-2000 steps"

**Action:** Check our LR and warmup configuration

### Data Requirements (Lines 254-258)
> "Models trained from scratch on 14 million real images achieve 20-30% accuracy improvements... practitioners should prioritize collecting and cleaning large real-world datasets"

**Insight:** Synthetic data alone may be insufficient

---

## Critical Implementation Details

### Loss Calculation (Lines 77-86, 320-324)
```
L = (1/K) * Σ L_CE(y_k, ŷ)
```

> "This averaging is crucial—failing to properly normalize across permutations causes imbalanced gradient signals."

**Action:** Verify our loss calculation properly averages across permutations

### Token Sequence Structure (Line 57)
```
[BOS] + [ground truth characters] + [EOS padding]
```

**Action:** Verify our data pipeline creates this exact structure

### Permutation Masking (Lines 71-73)
> "For each training batch element, the permutation sequence must be tracked, converted to an appropriate attention mask, and applied consistently across all heads"

**CRITICAL:** Off-by-one errors in mask computation cause information leakage

---

## Diagnostic Steps from Research (Lines 219-247)

### For "Training loss remains constant" (our case):
1. **Verify gradient computation** - print gradient statistics
2. **Check learning rate** - if <1e-7, too small
3. **Inspect attention masks** - verify correct computation
4. **Check for NaN/Inf** - use `torch.isnan()` assertions

### For "Model collapse":
1. **Monitor gradient magnitudes** - explosion or vanishing?
2. **Implement gradient clipping** - max norm 1-5
3. **Check loss for NaN/Inf**

---

## Next Actions (Priority Order)

### 1. **Compare Our Decoder with Official Implementation**
**Source:** baudm/parseq repository (line 1 reference)
- Check attention masking logic
- Verify permutation handling
- Compare loss calculation

### 2. **Verify BOS/EOS Token Structure**
- Manually inspect batch: `print(batch["text_tokens"][:5])`
- Confirm structure: `[1, ...chars..., 2, 0, 0, ...]` (BOS=1, EOS=2, PAD=0)

### 3. **Inspect Permutation Masking**
- Add debug prints to show attention masks
- Verify masks change per permutation
- Check for off-by-one errors

### 4. **Check Learning Rate & Warmup**
- Verify LR in range 7e-4 to 1e-3
- Confirm warmup is configured (500-2000 steps)

### 5. **Gradient Diagnostics**
```python
# Add to training step
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

---

## Code Reference from Working Implementation (Lines 287-316)

The research provides working inference code from official repo:
- Proper image transform: `img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)`
- Softmax on outputs: `p = model(image).softmax(-1)`
- Tokenizer decode: `pred, p = model.tokenizer.decode(p)`

**Action:** Compare our inference/training pipeline with this reference

---

## Alternative Approaches if Debugging Fails

### Option 1: Use Official Repository Directly
Clone `baudm/parseq` and adapt our data to their format

### Option 2: Switch to TrOCR (Lines 151-156)
> "TrOCR... may train more stably in some configurations due to the absence of cross-attention mechanism complexity"

### Option 3: Use Pretrained Weights (Lines 254-261)
> "pretrained models can be fine-tuned with relatively modest labeled data"

---

## Key Insight

**The research confirms our scale disparity finding BUT reveals deeper issues:**

> "proper configuration of learnable positional encodings and their interaction with permutation sequences" (line 329)

**Our fixes addressed scale, but NOT the permutation-positional interaction.**

The permutation masking and BOS/EOS handling are MORE CRITICAL than we thought.
