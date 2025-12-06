---
title: "004 Polygon Shape Dimension Error"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---



-----|------------|-----------|-------------|
| test/hmean | 0.00011 | 0.28502 | **2591x** |
| test/precision | 0.00166 | 0.2621 | **158x** |
| test/recall | 0.00007 | 0.3125 | **4464x** |

### Testing

- [x] Unit tests pass (87/89) - **Note: Need better polygon shape tests**
- [x] Integration test completes
- [x] Training produces non-zero metrics
- [ ] Full 3-epoch training (in progress)
- [ ] Performance benchmark vs commit 8252600 (pending)

### Prevention Strategies

#### 1. Type Checking & Documentation
**Problem:** No type hints or shape documentation for polygons

**Solution:**
```python
# Add type hints and shape documentation
def __call__(
    self,
    image: np.ndarray,
    polygons: list[np.ndarray] | None
) -> OrderedDict:
    """
    Transform image and polygons.

    Args:
        image: RGB image array with shape (H, W, 3)
        polygons: List of polygon arrays with shape (N, 2) where:
                  - N is number of points (>= 3)
                  - 2 represents (x, y) coordinates
    """
```

#### 2. Shape Validation
**Problem:** No runtime validation of polygon shapes

**Solution:** Add validation in transform pipeline (see BUG-2025-004 fixes)

#### 3. Integration Tests with Real Data
**Problem:** Unit tests use mocked data that doesn't catch dimension issues

**Solution:**
- Create integration tests with real dataset samples
- Test full transform pipeline end-to-end
- Validate polygon shapes at each stage
- Add metric threshold checks (hmean > 0.1 minimum)

#### 4. Shape Contract Documentation
**Problem:** Inconsistent polygon shapes across codebase

**Solution:**
```markdown
# POLYGON SHAPE CONTRACTS

## Storage Format (JSON → Dataset)
- Shape: (N, 2) where N >= 3
- Type: np.float32
- Example: np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])

## Transform Pipeline
- Input: List[np.ndarray] with shape (N, 2)
- Output: List[np.ndarray] with shape (1, N, 2) or (N, 2)
- Conversion: Keypoints → List → Polygons

## Collate Function
- Expected: List[np.ndarray] with shape (N, 2) or (1, N, 2)
- Normalization: Convert all to (N, 2) before processing
```

#### 5. Git History Awareness
**Problem:** Shape format changed during refactoring without updating dependent code

**Solution:**
- Document breaking changes in commit messages
- Add deprecation warnings for shape format changes
- Maintain CHANGELOG.md with API contract changes
- Review all dependent code when changing data formats

#### 6. Performance Regression Tests
**Problem:** Model performance degradation not caught by CI

**Solution:**
```python
# Add to CI pipeline
def test_training_sanity_check():
    """Quick training run to catch catastrophic regressions."""
    trainer = train(
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=5
    )

    # Assert minimum viable metrics
    assert trainer.callback_metrics['val/hmean'] > 0.1, \
        "Training produced near-zero metrics - pipeline likely broken"
```

### Related Issues

- **BUG-2025-002**: PIL Image vs numpy array type confusion
- **BUG-2025-003**: Albumentations contract violation
- **CRITICAL-2025-001**: Collate function polygon shape mismatch

All three bugs related to inconsistent data type contracts and shape handling throughout the pipeline.

### Lessons Learned

1. **Shape Contracts Are Critical**: Mixing `(N, 2)` and `(1, N, 2)` shapes is dangerous
2. **Unit Tests Insufficient**: Mocked tests don't catch real-world dimension issues
3. **Silent Failures Are Deadly**: Training "succeeded" but model was broken
4. **Git History Matters**: Understanding when shapes changed is key to debugging
5. **End-to-End Metrics Essential**: Only way to catch this class of bug

### References

- Working Commit: `8252600e22929802ead538672d2f137e2da0781d`
- Session Log: `logs/2025-10-09_transforms_caching_debug/00_ROLLING_LOG.md`
- Continuation Summary: `SESSION_SUMMARY_2025-10-10_CONTINUATION.md`
- Fix: ocr/datasets/transforms.py:74-80

---
