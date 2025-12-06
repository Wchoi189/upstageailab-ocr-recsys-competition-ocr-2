---
title: "003 Albumentations Contract Violation"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





**Notes:**
- This bug was dormant because preprocessing was recently added
- BUG-2025-002 fix exposed this by making the pipeline progress further
- Root cause is misunderstanding of Albumentations transform contract
- Fix is straightforward but requires careful testing

---

## Technical Deep Dive

### How Albumentations Transforms Work

Albumentations has three main transform types:

1. **ImageOnlyTransform**: Affects only the image
   ```python
   class MyTransform(A.ImageOnlyTransform):
       def apply(self, img, **params):
           return modified_img  # Albumentations wraps in dict
   ```

2. **DualTransform**: Affects image and targets (masks, keypoints, etc.)
   ```python
   class MyDualTransform(A.DualTransform):
       def apply(self, img, **params):
           return modified_img

       def apply_to_keypoint(self, keypoint, **params):
           return modified_keypoint
   ```

3. **BasicTransform**: Base class for custom behavior

### Why Direct __call__ Doesn't Work

When you implement `__call__` directly without inheriting from `BasicTransform`:
- Albumentations can't recognize it as a valid transform
- Type checking fails (see lint error in test)
- Data flow breaks because Albumentations expects dict → dict
- Internal validation fails with IndexError or AttributeError

### The Correct Pattern

```python
# Good: Inherit from ImageOnlyTransform
class GoodTransform(A.ImageOnlyTransform):
    def __init__(self, param, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.param = param

    def apply(self, img, **params):
        # Process image
        return processed_img  # Framework handles dict wrapping

    def get_transform_init_args_names(self):
        return ("param",)
```

---

*Commit Message:*
```
Fix BUG-2025-003: LensStylePreprocessorAlbumentations violates Albumentations contract

- Inherit from A.ImageOnlyTransform instead of plain class
- Implement apply() method instead of __call__()
- Properly handle Albumentations transform lifecycle
- Add unit tests for transform contract compliance

Refs: BUG-2025-003
```
