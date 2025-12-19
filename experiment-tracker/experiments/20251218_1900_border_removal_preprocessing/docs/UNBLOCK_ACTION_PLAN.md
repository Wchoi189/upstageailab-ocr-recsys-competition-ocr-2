# Border Removal Experiment - Unblocking Action Plan (REVISED)

## Experiment Scope Clarification

**This is a STANDALONE EXPERIMENT** - not immediate pipeline integration.

**Goals**:
1. Develop and validate 3 border removal methods (Canny, Morph, Hough)
2. Test on 000732 and border-affected images
3. Generate comprehensive VLM analysis reports
4. Document findings for future integration (Options A/B will handle pipeline work)

**NOT in scope**:
- Albumentations integration (deferred to Options A/B)
- Real-time latency constraints (no 50ms requirement)
- Production deployment decisions

## Critical Path to Resume

### IMMEDIATE: Run Dataset Collection (5 minutes)

```bash
# Collect border cases from existing test data
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

uv run python experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/collect_border_cases.py \
  --source-dirs data/zero_prediction_worst_performers \
  --output experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/outputs/border_cases_manifest.json \
  --skew-threshold 20.0 \
  --max-samples 50
```

**Expected output**:
- Manifest with 10-20 border cases (including 000732)
- Skew angles and border detection metrics
- Ready for Phase 2 implementation

### NEXT: Implement Border Removal Methods (2 hours)

Create standalone script: `experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/border_removal.py`

**Note**: This is a standalone experiment script, NOT pipeline integration code.
methods for experimentation (standalone)."""

import cv2
import numpy as np
from typing import Tuple, Dict


class BorderRemover:
    """
    Remove black borders from scanned documents.

    This is an EXPERIMENT-ONLY implementation for testing and validation.
    Pipeline integration will be handled separately in Options A/B.

    Methods:
        - canny: Canny edge + largest contour
        - morph: Morphological closing + connected components
        - hough: Hough lines + intersection detection
            - morph: Morphological closing + connected components
        - hough: Hough lines + intersection detection

    Args:
        method: Detection method ['canny', 'morph', 'hough']
        min_area_ratio: Minimum crop area vs original (safety)
        confidence_threshold: Minimum detection confidence
        canny_low: Canny lower threshold
        canny_high: Canny upper threshold
        morph_kernel_size: Morphological kernel size
    """

    def __init__(self,
                 method='canny',
                 min_area_ratio=0.75,
                 confidence_threshold=0.8,
                 canny_low=50,
                 canny_high=150,
                 morph_kernel_size=5,
                 always_apply=False,
                 p=1.0):
        super().__init__(always_apply, p)
        self.method = method
        self.min_area_ratio = min_area_ratio
        self.confidence_threshold = confidence_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel_size = morph_kernel_size

        # Store crop box for keypoint adjustment
        self._crop_box = None

    def apply(self, img, **params):
        """Apply border removal to image."""
        # Detect border and get crop box
        crop_box, confidence = self._detect_border(img)

        # Safety checks
        if confidence < self.confidence_threshold:
            self._crop_box = None  # No crop
            return img

        x1, y1, x2, y2 = crop_box
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img.shape[0] * img.shape[1])

        if area_ratio < self.min_area_ratio:
            self._crop_box = None  # Crop too aggressive
            return img

        # Store for keypoint adjustment
        self._crop_box = crop_box

        # Crop image
        return img[y1:y2, x1:x2]

    def apply_to_keypoint(self, keypoint, **params):
        """Adjust keypoint coordinates after crop."""
        if self._crop_box is None:
            return keypoint

        x, y, angle, scale = keypoint
        x1, y1, x2, y2 = self._crop_box

        # Adjust coordinates
        new_x = x - x1
        new_y = y - y1

        return (new_x, new_y, angle, scale)

    def _detect_border(self, img: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
        """
        Detect border and return crop box.

        Returns:
            Tuple of ((x1, y1, x2, y2), confidence)
        """
        if self.method == 'canny':
            return self._detect_canny(img)
        elif self.method == 'morph':
            return self._detect_morph(img)
        elif self.method == 'hough':
            return self._detect_hough(img)
        else:
            # Fallback: no crop
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

    def _detect_canny(self, img: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
        """Canny edge detection + largest contour."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)

        # Calculate confidence (area ratio)
        img_area = img.shape[0] * img.shape[1]
        contour_area = w * h
        confidence = min(1.0, contour_area / img_area)

        return (x, y, x + w, y + h), confidence

    def _detect_morph(self, img: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
        """Morphological operations + connected components."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

        if num_labels <= 1:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

        # Find largest component (skip background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        x = stats[largest_label, cv2.CC_STAT_LEFT]
        y = stats[largest_label, cv2.CC_STAT_TOP]
        w = stats[largest_label, cv2.CC_STAT_WIDTH]
        h = stats[largest_label, cv2.CC_STAT_HEIGHT]

        # Calculate confidence
        img_area = img.shape[0] * img.shape[1]
        component_area = w * h
        confidence = min(1.0, component_area / img_area)

        return (x, y, x + w, y + h), confidence

    def _detect_hough(self, img: np.ndarray) -> Tuple[Tuple[int, int, int, int], float]:
        """Hough line detection + intersection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None or len(lines) < 4:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.0

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            if angle < 45 or angle > 135:  # Horizontal-ish
                h_lines.append((x1, y1, x2, y2))
            else:  # Vertical-ish
                v_lines.append((x1, y1, x2, y2))

        if len(h_lines) < 2 or len(v_lines) < 2:
            h, w = img.shape[:2]
            return (0, 0, w, h), 0.5  # Medium confidence

        # Find bounding box from lines
        x_coords = [x for line in v_lines for x in [line[0], line[2]]]
        y_coords = [y for line in h_lines for y in [line[1], line[3]]]

        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        # Calculate confidence
        img_area = img.shape[0] * img.shape[1]
        detected_area = (x2 - x1) * (y2 - y1)
        confidence = min(1.0, detected_area / img_area)

        return (x1, y1, x2, y2), confidence

    def get_transform_init_args_names(self):
        """For serialization."""
        return ('method', 'min_area_ratio', 'confidence_threshold',
                'canny_low', 'canny_high', 'morph_kernel_size')
```

**Add to `ocr/datasets/transforms.py`**:
```python
from ocr.datasets.transforms.border_removal import BorderRemoval

# In ValidatedDBTransforms.__init__():
if config.get('border_removal', {}).get('enabled', False):
    border_removal = BorderRemoval(
        method=config.border_removal.get('method', 'canny'),
        min_area_ratio=config.border_removal.get('min_area_ratio', 0.75),
        confidence_threshold=config.border_removal.get('confidence_threshold', 0.8),
    )
else:
    border_removal = A.NoOp()

self.transform = A.Compose([
    border_removal,  # FIRST - before resizing
    A.Resize(height=config.img_h, width=config.img_w),
    # ... rest of pipeline
])
```

### THEN: Test on 000732 (30 minutes)

```bash
# Create test script
cat > experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/test_000732.py << 'EOF'
"""Test border removal on image 000732."""

import cv2
import numpy as np
from ocr.datasets.transforms.border_removal import BorderRemoval

# Load image
img_path = "data/zero_prediction_worst_performers/drp.en_ko.in_house.selectstar_000732.jpg"
img = cv2.imread(img_path)

# Test all 3 methods
for method in ['canny', 'morph', 'hough']:
    remover = BorderRemoval(method=method)
    cropped = remover.apply(img)

    print(f"{method.upper()}: {img.shape} -> {cropped.shape}")
    print(f"  Crop box: {remover._crop_box}")
    print(f"  Area ratio: {(cropped.shape[0] * cropped.shape[1]) / (img.shape[0] * img.shape[1]):.3f}")

    # Save result
    output_path = f"outputs/000732_{method}.jpg"
    cv2.imwrite(output_path, cropped)
    print(f"  Saved: {output_path}\n")
EOF

# Run test
uv run python experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/test_000732.py
```

### FINALLY: Validate Skew Improvement (15 minutes)

```bash
# Use existing deskewing script from Week 2
uv run python experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts/text_deskewing.py \
  --input outputs/000732_canny.jpg \
  --output outputs/000732_canny_deskewed.jpg \
  --method hough

# Compare skew before/after
# Before (baseline): -83° (from previous experiment)
# After (with border removal): Should be <15°
```

## Resolved Blockers Summary
 (REVISED)

| Blocker | Resolution | Status |
|---------|-----------|--------|
| **VLM Integration** | Use VLM extensively for quality analysis & reports | ✅ |
| **Preprocessing Stack** | DEFERRED to Options A/B (experiment is standalone) | ✅ |
| **Metrics Schema** | Defined in `BorderRemover._detect_border()` return | ✅ |
| **Failure Dataset** | Script `collect_border_cases.py` created | ✅ |
| **Hardware Spec** | NOT APPLICABLE (no latency requirement for experiment) | ✅ |
| **Integration Tests** | DEFERRED to Options A/B (focus on method validation)
## Updated Timeline (STANDALONE EXPERIMENT)

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| **Phase 1** | Dataset collection + VLM baseline | 30 min | READY TO RUN |
| **Phase 2** | Implement 3 methods (standalone) | 3 hours | CODE TEMPLATE BELOW |
| **Phase 3** | Test on 000732 + VLM analysis | 1 hour | PENDING |
| **Phase 4** | Method comparison + VLM reports | 2 hours | PENDING |
| **Phase 5** | Full validation + documentation | 3 hours | PENDING |

**Total**: ~9 hours to complete experiment (integration deferred to Options A/B)

## Critical Dependencies

✅ **Resolved**:
- Experiment scope clarified (standalone, not pipeline integration)
- Metrics schema (confidence, area_ratio, crop_box)
- Test data source (zero_prediction_worst_performers)
- VLM usage approach (extensive analysis encouraged)
- Latency constraints (removed - not applicable)

⏳ **Remaining**:
- Method implementation (3 border removal algorithms)
- VLM analysis workflows (baseline + validation)
- Comparison assessments (which method works best?)

## Next Command to Run

```bash
# Start with dataset collection
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

uv run python experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/scripts/collect_border_cases.py \
  --source-dirs data/zero_prediction_worst_performers \
  --output experiment-tracker/experiments/20251218_1900_border_removal_preprocessing/outputs/border_cases_manifest.json
```

This will generate the manifest and unblock Phase 2 implementation.
