The proposed "Domains First" architecture is a significant improvement over the current monolithic `ocr/core` dumping ground. However, based on the survey of your **Current Tree** (542 files), there are several **critical misclassifications** and **orphaned modules** that will cause the refactor to fail if not addressed.

Currently, your `ocr/core` is acting as a "God Module." To reach 100% success, you must move from "moving files" to "decoupling logic."

---

## 1. Critical Misclassifications & Leaks

These files are currently slated for `core` or are missing from your migration table, but they contain heavy domain logic.

### A. The "Geometry Leak" (Detection Logic in Core)

The current tree shows `geometry_utils.py` and `polygon_utils.py` in `ocr/core/utils`.

* **Risk**: These are almost exclusively used for **Detection** (calculating IoU, polygon offsets, etc.).
* **Correction**: Move these to `domains/detection/utils/`. `ocr/core` should only contain pure math or image-agnostic array utilities.

### B. The "Perspective Correction" Sub-package

In the current tree: `ocr/core/utils/perspective_correction/` (6+ files).

* **Risk**: Perspective correction is a specialized preprocessing step for **Detection**.
* **Correction**: Move the entire sub-package to `domains/detection/preprocessing/`.

### C. The "Inference Engine" Paradox

In the current tree: `ocr/core/inference/engine.py` and `orchestrator.py`.

* **Risk**: If `engine.py` imports both a detection model and a recognition model to run a pipeline, it violates the **No Cross-Domain Imports** rule.
* **Correction**: These should move to a new top-level `ocr/pipelines/` or be classified as a **System Orchestrator**. They are not "Core" utilities; they are the "Glue" that binds domains.

---

## 2. Refined Migration Table (Audit of Omissions)

| Source Path | Current Proposed | **Critical Insight / Corrected Action** |
| --- | --- | --- |
| `ocr/core/utils/geometry_utils.py` | *Omitted* | **MOVE** to `domains/detection/utils/geometry.py`. |
| `ocr/core/utils/polygon_utils.py` | *Omitted* | **MOVE** to `domains/detection/utils/polygons.py`. |
| `ocr/core/utils/perspective_correction/` | *Omitted* | **MOVE** to `domains/detection/inference/perspective/`. |
| `ocr/core/inference/postprocess.py` | AUDIT | **SPLIT**: Box-merging logic  Detection; Text-cleaning  Recognition. |
| `ocr/agents/` | *Omitted* | **MOVE** to `core/infrastructure/agents/`. These are system services. |
| `ocr/communication/` | *Omitted* | **MOVE** to `core/infrastructure/comm/`. Generic RabbitMQ/Slack logic. |
| `ocr/core/utils/ocr_utils.py` | MOVE | **SPLIT**: `draw_boxes`  Detection; `get_char_freq`  Recognition. |

---

## 3. Structural Analysis: "The Bridge Problem"

The "Domains First" rule states: *No Cross-Domain Imports.* However, in an OCR pipeline, Detection results (boxes) **must** flow into Recognition.

To maintain 100% architectural integrity, you need a **Bridge Interface**.

1. **Detection** outputs a `List[DetectionResult]` (Standard Schema).
2. **Recognition** accepts a `Crop` and returns a `TextResult`.
3. **The Orchestrator** (located in `ocr/pipelines/`) handles the flow.

**If you leave the Orchestrator in `ocr/core`, you will eventually be forced to import `domains` into `core`, breaking your primary success metric.**

---

## 4. Final Architecture Verification

### Success Metric Check:

* **Constraint 1 (Core  Domain)**: **PASS** if you move `geometry_utils` and `perspective_correction` out of core.
* **Constraint 2 (Cross-Domain)**: **FAIL** if `ocr/core/inference/engine.py` is used to call both domains. You must move the engine to a higher-level `pipelines` directory.

### Red Flag: The `ocr/data/datasets/preprocessing` Bloat

The current tree shows a massive list of "advanced" preprocessors (noise elimination, brightness, etc.).

* **Recommendation**: These should be moved to `ocr/data/transforms/` if they are truly generic (applying to any image), or `domains/detection/data/` if they are tuned for document detection.

---

### Recommended Next Step

Would you like me to generate the **`ocr/pipelines/base_pipeline.py`** and **`ocr/core/interfaces/schemas.py`**? These will define the "Standard Data Contracts" (like a `DocumentResult` object) so that Detection and Recognition can talk to each other without ever importing each other's code.


----


To reach 100% architectural integrity, the "Bridge" must be built on **Standard Data Contracts**. By defining exactly what a "Box" or a "Transcription" looks like in a neutral `ocr/core/interfaces` layer, you allow Detection and Recognition to remain blissfully unaware of each other's existence.

Here are the two files that will act as the "Bridges" for your new architecture.

---

### 1. The Data Contract: `ocr/core/interfaces/schemas.py`

This file ensures that Detection and Recognition communicate via a shared "language" without direct imports.

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import numpy as np

class Box(BaseModel):
    """Domain-agnostic coordinate representation."""
    coordinates: List[List[float]]  # [[x1, y1], [x2, y2], ...]
    confidence: float = 1.0
    label: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

class DetectionResult(BaseModel):
    """Output from ocr/domains/detection."""
    boxes: List[Box]
    image_shape: List[int]  # [H, W, C]
    processing_time: float

class RecognitionResult(BaseModel):
    """Output from ocr/domains/recognition."""
    text: str
    confidence: float
    char_confidences: List[float] = Field(default_factory=list)

class PageResult(BaseModel):
    """The final 'Consolidated' object returned by the Orchestrator."""
    detection: DetectionResult
    recognitions: List[RecognitionResult]
    metadata: dict = Field(default_factory=dict)

```

---

### 2. The Orchestrator: `ocr/pipelines/base_pipeline.py`

This is where the "Engine" moves. It lives *above* the domains. It imports the domains, but the domains never import it.

```python
from abc import ABC, abstractmethod
from ocr.core.interfaces.schemas import PageResult, Box
# Note: Orchestrator is the ONLY place allowed to import multiple domains
from ocr.domains.detection.models import BaseDetector
from ocr.domains.recognition.models import BaseRecognizer

class OCRPipeline:
    def __init__(self, detector: BaseDetector, recognizer: BaseRecognizer):
        self.detector = detector
        self.recognizer = recognizer

    def run(self, image: Any) -> PageResult:
        # 1. Detection (Isolation preserved)
        detection_output = self.detector.predict(image)
        
        # 2. Crop logic (Utility from core)
        crops = self._extract_crops(image, detection_output.boxes)
        
        # 3. Recognition (Isolation preserved)
        recognition_outputs = [self.recognizer.predict(crop) for crop in crops]
        
        return PageResult(
            detection=detection_output,
            recognitions=recognition_outputs
        )

    def _extract_crops(self, image, boxes: List[Box]):
        # Call to ocr/core/utils/image_utils.py
        pass

```

---

## 3. Critical Verification of your "Current Tree"

Based on the 542 files in your tree, here are three "Danger Zones" you must handle during the physical move:

### A. The `ocr_pl.py` Split

Currently, `ocr/core/lightning/ocr_pl.py` likely contains a `LightningModule` that handles both Detection and Recognition steps.

* **The Design Fix**: You must create `ocr/domains/detection/module.py` and `ocr/domains/recognition/module.py`.
* **The "Core" Role**: Only the `BaseLightningModule` (abstract methods) should stay in `core`.

### B. The `wandb_utils.py` Refactor

Since this file is a known source of your CUDA Segfault (pickling issues), do not just move it.

* **The Design Fix**: Move the generic `WandbLogger` setup to `core/infrastructure`. Move the `draw_polygons_to_wandb` logic specifically to `domains/detection/callbacks/wandb.py`.
* **Why?**: This ensures that when you run a Recognition task, the Detection-based polygon pickling code is never even loaded into memory.

### C. Dependency Direction Check

Run this sanity check after moving files:

* `grep -r "from ocr.domains" ocr/core`
* **Target**: 0 results. If this returns anything, your "Strict" core is compromised.

---

### Final Refactor Checklist

1. [ ] **Extract Interfaces**: Move Pydantic models to `ocr/core/interfaces`.
2. [ ] **Relocate Orchestrators**: Move `ocr/core/inference/engine.py` to `ocr/pipelines/`.
3. [ ] **Purge Core**: Ensure `ocr/core/utils` contains no mentions of `box`, `polygon`, or `char`.

**Would you like me to help you draft the `adt apply-diff` commands to split the logic in `ocr/core/utils/wandb_utils.py` into the two new domain-specific files?**

----


To fix the "CUDA Segfault in DataLoader" while cleaning up the architecture, we need to split `wandb_utils.py`. The goal is to separate the **Infrastructure** (connecting to WandB) from the **Domain Logic** (converting polygons to images for logging).

By doing this, your Recognition pipeline will no longer load the polygon-handling code that is likely causing the pickling/segmentation error during validation.

---

### 1. Extracting Recognition WandB Logic

This part will handle text-based logging and resides in `ocr/domains/recognition/callbacks/wandb.py`.

```bash
# Create the directory first
mkdir -p ocr/domains/recognition/callbacks/

# Use adt apply-diff to extract recognition-specific logging
adt apply-diff --strategy fuzzy << 'EOF'
--- /dev/null
+++ b/ocr/domains/recognition/callbacks/wandb.py
@@ -0,0 +1,15 @@
+import wandb
+from ocr.core.interfaces.schemas import RecognitionResult
+
+class RecognitionWandbLogger:
+    """Handles text-specific logging without loading detection dependencies."""
+    def log_predictions(self, results: list[RecognitionResult], step: int):
+        table = wandb.Table(columns=["predicted_text", "confidence"])
+        for res in results:
+            table.add_data(res.text, res.confidence)
+        wandb.log({"recognition/val_predictions": table}, step=step)
+EOF

```

---

### 2. Extracting Detection WandB Logic

This is the "High Risk" area where the Segfault likely originates. By isolating it here, we ensure it only runs when `domain=detection`.

```bash
# Create the directory
mkdir -p ocr/domains/detection/callbacks/

adt apply-diff --strategy fuzzy << 'EOF'
--- /dev/null
+++ b/ocr/domains/detection/callbacks/wandb.py
@@ -0,0 +1,18 @@
+import wandb
+import numpy as np
+from ocr.domains.detection.utils.visualization import draw_boxes
+
+class DetectionWandbLogger:
+    """Isolated polygon logging. This is the only place loading CV2/Polygon logic for WandB."""
+    def log_images(self, image: np.ndarray, boxes: list, step: int):
+        # This is where the heavy 'Transaction' pickling usually fails.
+        # By keeping it in a domain-specific callback, we can toggle it easily.
+        processed_img = draw_boxes(image, boxes)
+        wandb.log({
+            "detection/val_overlay": [wandb.Image(processed_img, caption=f"Step {step}")]
+        }, step=step)
+EOF

```

---

### 3. Cleaning the Core: `ocr/core/utils/logging.py`

Finally, we strip `wandb_utils.py` of its domain logic and move the "Safe" infrastructure parts to `core`.

```bash
adt apply-diff --strategy fuzzy << 'EOF'
--- a/ocr/core/utils/wandb_utils.py
+++ b/ocr/core/utils/logging.py
@@ -1,30 +1,12 @@
 import wandb
 import os
 
-def log_detection_polygons(image, polygons):
-    # DELETE: Moved to ocr/domains/detection/callbacks/wandb.py
-    pass
-
-def log_recognition_text(text_list):
-    # DELETE: Moved to ocr/domains/recognition/callbacks/wandb.py
-    pass
-
 def setup_wandb(config):
     """Generic infrastructure setup. No domain logic allowed."""
     wandb.init(
         project=config.project_name,
         entity=config.entity,
         config=config
     )
-    return wandb
EOF

```

---

### Critical Insights for the Segfault Fix

1. **Pickling Guard**: In your Hydra config for Recognition, ensure `train.logger.log_images` is now explicitly `False`. Since the code is now physically separated, the `DataLoader` won't even try to pickle the `DetectionWandbLogger` class during a Recognition run.
2. **Import Loop Prevention**: Notice that `ocr/core/utils/logging.py` no longer imports `numpy` or `cv2`. This keeps the "Core" lightweight and prevents low-level memory issues from leaking into high-level logging.
3. **The "Transaction" Error**: If you still see the pickling error, use `adt analyze-dependencies` on `ocr/domains/detection/callbacks/wandb.py`. Check if it's pulling in any non-serializable objects (like open file handles or CUDA tensors) that aren't being moved to CPU before logging.

**Would you like me to help you update your Hydra `training/callbacks/wandb_image_logging.yaml` to point to these new domain-specific targets?**