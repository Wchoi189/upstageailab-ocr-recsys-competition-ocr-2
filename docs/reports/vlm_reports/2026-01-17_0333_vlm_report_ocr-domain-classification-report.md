# OCR Core Domain Classification Report

> [!WARNING]
> This report identifies "Feature First" legacy code in `ocr/core` AND `ocr/data` that violates the "Domains First" architecture.

## 1. Detection Domain Leakage
**Keywords**: `polygon`, `bbox`, `quad`, `box_thresh`
**Target**: `ocr/domains/detection/`

| File                                                   | Leakage Details                        | Recommendation                                                      |
| :----------------------------------------------------- | :------------------------------------- | :------------------------------------------------------------------ |
| `ocr/core/metrics/box_types.py`                        | Defines `Polygon`, `Quad` types        | **Move** to `ocr/domains/detection/utils/box_types.py`              |
| `ocr/core/metrics/eval_functions.py`                   | Box IoU, Precision/Recall logic        | **Move** to `ocr/domains/detection/metrics/functional.py`           |
| `ocr/core/validation.py`                               | Validates `polygon` inputs             | **Split** or **Move** validation logic to `detection/validation.py` |
| `ocr/core/analysis/validation/analyze_worst_images.py` | Draws bounding boxes (`cv2.polylines`) | **Move** to `detection/analysis/`                                   |
| `ocr/core/utils/ocr_utils.py`                          | `draw_boxes` function                  | **Move** to `detection/utils/visualization.py`                      |
| `ocr/core/inference/coordinate_manager.py`             | Transforms polygons                    | **Move** to `detection/inference/`                                  |
| `ocr/data/datasets/db_collate_fn.py`                   | DBNet specific batching                | **Move** to `detection/data/collate_fn.py`                          |
| `ocr/data/datasets/craft_collate_fn.py`                | CRAFT specific batching                | **Move** to `detection/data/collate_fn_craft.py`                    |

## 2. Recognition Domain Leakage
**Keywords**: `tokenizer`, `text_render`, `put_text`
**Target**: `ocr/domains/recognition/`

| File                                          | Leakage Details                   | Recommendation                                                            |
| :-------------------------------------------- | :-------------------------------- | :------------------------------------------------------------------------ |
| `ocr/core/lightning/ocr_pl.py`                | Uses `self.tokenizer`             | **Abstract** tokenizer access or move logic to `RecognitionTask` subclass |
| `ocr/core/utils/text_rendering.py`            | Renders text strings (`put_text`) | **Move** to `recognition/utils/visualization.py`                          |
| `ocr/core/utils/wandb_utils.py`               | `log_recognition_images`          | **Move** to `recognition/callbacks/wandb_utils.py`                        |
| `ocr/data/datasets/recognition_collate_fn.py` | Text sequence batching            | **Move** to `recognition/data/collate_fn.py`                              |

## 3. Mixed / Common Utilities
These files contain both generic and domain-specific code.

| File                               | Strategy                                                                                                      |
| :--------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| `ocr/core/evaluation/evaluator.py` | **Split**: Contains polygon reshaping. Extract `DetectionEvaluator` and `RecognitionEvaluator`.               |
| `ocr/core/utils/wandb_utils.py`    | **Split**: Extract domain loggers to `ocr/domains/{domain}/callbacks`. Keep generic config parsing in `core`. |
| `ocr/core/image_utils.py`          | **Keep**: Ensure `_to_u8_bgr` and basic loading stays here.                                                   |

## 4. Action Items for Next Session
1. **Initialize Directory**: `mkdir -p ocr/domains/{detection,recognition,kie,layout}`
2. **Bulk Move**: Execute `git mv` for the "The Mover" files (Section 1 & 2).
3. **Refactor**: Split `wandb_utils.py`, `evaluator.py`, and `ocr_pl.py`.
