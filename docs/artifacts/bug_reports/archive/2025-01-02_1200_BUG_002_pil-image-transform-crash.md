---
title: "002 Pil Image Transform Crash"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



-----------|------------|----------------|----------------------|--------|
| `prenormalize_images=True` | numpy float32 | numpy array | numpy array | ✅ Works |
| `prenormalize_images=False` | numpy uint8 | **PIL Image** | numpy array | ❌ **Crashes** |
| `preload_images=False` | None | PIL Image | numpy array | ❌ **Should crash but doesn't?** |

**Why line 232 exists:** The code assumes that when images are NOT pre-normalized, they should be converted back to PIL Images to match the behavior of the non-cached path (line 258: `normalized_image` is a PIL Image). However, this assumption is **wrong** because:

1. **Albumentations requires numpy arrays**: The transform library expects numpy arrays, not PIL Images
2. **DBTransforms doesn't handle PIL**: Line 42 directly accesses `.shape[:2]` without type checking
3. **Inconsistent with documentation**: Phase 6B/6E documentation assumes numpy arrays are passed to transforms

### Impact Analysis

**Affected Configurations:**
- ✅ Works: `preload_images=False` (loads PIL, passes PIL... but should fail?)
- ❌ **FAILS**: `preload_images=True, prenormalize_images=False` ← **Current issue**
- ✅ Works: `preload_images=True, prenormalize_images=True`

**User Impact:**
- Training crashes immediately during validation sanity check
- Unable to use RAM caching without pre-normalization
- No workaround except enabling `prenormalize_images=True`

### Resolution Strategy

**Option 1: Fix DBTransforms to handle both types** (Defensive)
```python
# transforms.py:42
def __call__(self, image, polygons):
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width = image.shape[:2]
    # ... rest of code
```

**Pros:**
- Defensive programming
- Handles both PIL and numpy inputs
- No changes to dataset loading logic

**Cons:**
- Adds overhead of type checking
- Doesn't address the root inconsistency

**Option 2: Always pass numpy arrays from dataset** (Preferred)
```python
# base.py:232
# Remove PIL conversion - keep as numpy array
if is_normalized:
    image = image_array  # normalized numpy array
else:
    image = image_array  # uint8 numpy array (don't convert to PIL!)
```

**Pros:**
- Consistent with Albumentations requirements
- No runtime type checking overhead
- Aligns with Phase 6B/6E design intent
- Simpler, cleaner code

**Cons:**
- Need to verify all transforms handle uint8 numpy arrays correctly

**Recommendation:** **Option 2** - Always pass numpy arrays to transforms, as this is what Albumentations expects and what the performance optimizations assume.

### Testing Requirements

**Pre-Fix Verification:**
- [x] Minimal reproduction test created: `test_bug_reproduction.py`
- [x] Confirmed PIL Image crashes in DBTransforms
- [x] Confirmed numpy array works in DBTransforms

**Post-Fix Verification:**
- [ ] Unit test: DBTransforms with uint8 numpy array
- [ ] Unit test: DBTransforms with float32 numpy array
- [ ] Integration test: Full training run with `preload_images=True, prenormalize_images=False`
- [ ] Integration test: Full training run with `preload_images=True, prenormalize_images=True`
- [ ] Regression test: Training without caching still works

### Prevention Measures

1. **Add type checking in DBTransforms** (defensive layer even after fix)
2. **Add unit tests** for transform input types
3. **Document data type contracts**:
   - Dataset `__getitem__` returns: numpy arrays (uint8 or float32)
   - Transforms expect: numpy arrays
   - Collate function expects: tensors
4. **Add assertions** in debug mode to catch type mismatches early
5. **Update Phase 6 documentation** to clarify numpy array contract

### Related Issues

- Phase 6B: RAM caching implementation
- Phase 6C: Pre-normalization feature
- Phase 6E: Tensor caching (exposes this bug due to increased validation runs)

### Next Steps

1. ✅ Create bug report (this document)
2. ⏳ Implement Option 2 fix (remove PIL conversion at line 232)
3. ⏳ Add type assertion at transforms.py:42 (defensive check)
4. ⏳ Run full test suite to verify fix
5. ⏳ Update documentation with type contracts
6. ⏳ Commit with reference to BUG-2025-002

---

**Notes:**
- This bug was introduced during Phase 6B refactoring when adding RAM caching
- The bug was dormant until Phase 6E increased validation frequency
- The root cause is architectural: PIL vs numpy type ambiguity in the pipeline
- Fix is simple but requires careful testing to avoid regressions

---

## Reproduction Test Results

**Test Script:** `test_bug_reproduction.py`

```
🐛 BUG REPRODUCTION TEST SUITE
Testing: AttributeError in transforms.py line 42

======================================================================
TEST 1: DBTransforms with PIL Image (SHOULD FAIL)
======================================================================
✅ EXPECTED FAILURE: 'Image' object has no attribute 'shape'
   Error message: 'Image' object has no attribute 'shape'

======================================================================
TEST 2: DBTransforms with numpy array (SHOULD WORK)
======================================================================
✅ SUCCESS: Transform completed
   Input shape: (600, 800, 3)
   Output shape: torch.Size([3, 640, 640])

======================================================================
ROOT CAUSE ANALYSIS
======================================================================
When preload_images=True but prenormalize_images=False:
1. Images are loaded and cached as numpy arrays (base.py:169)
2. In __getitem__, cached numpy arrays are converted back to PIL Images (base.py:232)
3. PIL Images are passed to DBTransforms.__call__ (base.py:307)
4. DBTransforms tries to access image.shape[:2] (transforms.py:42)
5. PIL Images don't have .shape attribute → AttributeError!
```

**Conclusion:** Bug successfully reproduced. The issue is confirmed as a type mismatch between PIL Images and numpy arrays in the transform pipeline.

---

*Template reference: docs/bug_reports/BUG_REPORT_TEMPLATE.md*
