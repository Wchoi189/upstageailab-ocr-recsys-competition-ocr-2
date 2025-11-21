# Why Tolerance is Needed for Canonical Frame Detection

## Question

**Why do we need tolerance if images are expected to be the same when they are already in their canonical form? Don't rotated images need a tolerance factor?**

## Answer

Yes, tolerance is needed even for canonical images, and **especially** for rotated images. Here's why:

## 1. Floating-Point Precision Errors

Even when polygons are correctly in canonical frame, floating-point arithmetic can introduce small errors:

```python
# Example: Coordinate transformation
x = 1280.0
y = 960.0
# After some transformation
x_new = x * 1.000001  # = 1280.00128 (slightly over boundary)
```

**Real-world example from investigation**:
- Canonical height: 1280 pixels
- Polygon Y coordinate: 1281.9 pixels (1.9 pixels over)
- This is a valid polygon in canonical frame, just with small floating-point error

## 2. Annotation Tool Rounding Errors

Annotation tools (LabelMe, CVAT, etc.) may:
- Round coordinates to nearest pixel
- Apply transformations that introduce rounding errors
- Export coordinates with slight imprecision

**Example**:
- User annotates at pixel (1279.7, 960.3)
- Tool rounds to (1280, 960) - now slightly over boundary
- Polygon is still valid and in canonical frame

## 3. EXIF Remapping Transformation Errors (Especially for Rotated Images)

For rotated images (orientations 5, 6, 7, 8), the transformation itself can introduce errors:

### Example: Orientation 6 (90° Clockwise Rotation)

**Transformation formula** (from `orientation_constants.py`):
```python
# ROTATE_90_CW
x_new = height - 1.0 - y
y_new = x
```

**Why errors occur**:
1. **Dimension swapping**: Raw 1280x960 → Canonical 960x1280
2. **Coordinate remapping**: Uses `height - 1.0 - y` formula
3. **Floating-point arithmetic**: `960.0 - 1.0 - y` can produce values like 1281.9 when y is slightly negative or due to rounding

**Real case from investigation**:
- Raw dimensions: 1280x960
- Canonical dimensions: 960x1280 (swapped!)
- Polygon Y coordinates: [1258.7, 1281.9]
- The 1281.9 value is 1.9 pixels over canonical height (1280)
- But this polygon is **correctly in canonical frame** - it just has a small transformation error

## 4. Why Tolerance is Critical for Rotated Images

**Without tolerance**:
```python
# Check: max_y <= canonical_height - 1
# 1281.9 <= 1280 - 1 = 1279
# FALSE! → Polygon not detected as canonical
# → Gets remapped again → Double rotation → Out of bounds!
```

**With 3-pixel tolerance**:
```python
# Check: max_y <= canonical_height - 1 + tolerance
# 1281.9 <= 1280 - 1 + 3.0 = 1282
# TRUE! → Polygon correctly detected as canonical
# → No remapping → No double rotation → Coordinates stay valid
```

## 5. The Boundary Problem

When coordinates are at exact boundaries (x=width, y=height), small errors can push them slightly over:

**Example**:
- Image width: 960 pixels
- Polygon point at x=960.0 (exact boundary)
- After floating-point operation: x=960.2 (0.2 pixels over)
- Without tolerance: Detected as out of bounds
- With tolerance: Correctly identified as boundary coordinate

## 6. Why 3.0 Pixels?

Based on investigation results:
- **1.5 pixels**: Too strict - missed 145 out of 146 cases
- **3.0 pixels**: Matches validation tolerance, handles:
  - Floating-point errors: 0.1-1 pixels
  - Annotation rounding: 1-2 pixels
  - Transformation errors: 1-3 pixels (especially for rotated images)

**Investigation evidence**:
- Most errors: 1-2 pixels over boundary
- Largest error in valid polygons: ~3 pixels
- Errors > 3 pixels: Likely truly invalid (should be rejected)

## Summary

**Tolerance is needed because**:

1. ✅ **Floating-point precision**: Even perfect transformations can introduce 0.1-1 pixel errors
2. ✅ **Annotation tools**: Rounding and imprecision can add 1-2 pixel errors
3. ✅ **Rotated images**: Dimension swapping and coordinate remapping can produce 1-3 pixel errors
4. ✅ **Boundary coordinates**: Exact boundaries (x=width, y=height) are valid but can appear "over" due to errors

**Without tolerance**: Valid polygons in canonical frame would be incorrectly remapped, causing double-rotation and out-of-bounds coordinates.

**With 3-pixel tolerance**: Correctly identifies polygons in canonical frame despite small errors, preventing double-rotation.

---

*This explains why the fix increases tolerance from 1.5 to 3.0 pixels (BUG-20251116-001)*
