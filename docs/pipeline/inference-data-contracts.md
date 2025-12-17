---
type: data_reference
component: inference
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Inference Data Contracts

**Purpose**: This file redirects to the canonical inference data contracts documentation.

**Canonical Reference**: [../reference/inference-data-contracts.md](../reference/inference-data-contracts.md)

---

## Quick Reference

For complete inference data contracts including:
- InferenceMetadata
- PreprocessingResult
- PostprocessingResult
- LoadedImage
- Coordinate transformation rules
- Padding strategies

See the canonical documentation: [../reference/inference-data-contracts.md](../reference/inference-data-contracts.md)

---

## References

- [Canonical Inference Data Contracts](../reference/inference-data-contracts.md)
- [Training Data Contracts](data-contracts.md)
- [Preprocessing Data Contracts](preprocessing-data-contracts.md)

- Use `meta.padding_position` to determine coordinate handling
- Apply display centering offsets (dx, dy) only
- For centered padding: coordinates already include padding offset

## Validation Rules

- `padding_position` must be present (no default assumption)
- `content_area` must be present and valid
- `padding` values must match `padding_position` (top-left: top=0, left=0)
- Coordinates must be within content_area bounds

## Implementation Notes

### OCR Inference Console

The OCR Inference Console uses `preview_image_base64` to display annotations aligned with the coordinate system:

- **Preview Image**: Preprocessed image in `processed_size` space (matches coordinates)
- **Padding Trim**: Uses SVG `viewBox` to crop black padding using `content_area` calculated from padding metadata
- **No Coordinate Transformation**: Since image and coordinates are in the same space, direct mapping is used

See [`apps/ocr-inference-console/docs/annotation-rendering.md`](../../apps/ocr-inference-console/docs/annotation-rendering.md) for implementation details.

## Related Contracts

- [Pipeline Data Contracts](data_contracts.md#inference-engine-contract)
- [Coordinate Transformation](data_contracts.md#critical-areas---do-not-modify-without-tests)
- [OCR Inference Console Annotation Rendering](../../apps/ocr-inference-console/docs/annotation-rendering.md)
