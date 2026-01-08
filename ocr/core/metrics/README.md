# OCR Core Metrics

## Purpose
This package contains **shared evaluation metrics** used across multiple OCR features (detection, recognition, KIE).

## Included Metrics

### CLEval (Character-Level Evaluation)
- **Paper**: [CLEval: Character-Level Evaluation for Text Detection and Recognition Tasks](https://arxiv.org/pdf/2006.06244.pdf)
- **Source**: [Clova AI CLEval](https://github.com/clovaai/CLEval)
- **Purpose**: Character-level precision/recall/F1 for text detection and recognition
- **Usage**: Used by `OCRPLModule` for validation/test metric computation across all features

### Architecture Justification
CLEval is classified as **core infrastructure** (not feature-specific) because:
1. Used by core training module (`ocr/core/lightning/ocr_pl.py`) that serves all features
2. Provides domain-agnostic evaluation mechanism (boxes + text → metrics)
3. Paper explicitly covers both detection AND recognition tasks
4. No feature-specific business logic

## Inclusion Criteria
Metrics belong in `ocr/core/metrics/` if they:
1. Are used by 2+ features OR core infrastructure
2. Provide evaluation mechanisms, not domain policies
3. Contain no feature-specific business logic

## Files
- `cleval_metric.py` - Main CLEval metric implementation
- `box_types.py` - Polygon and box type definitions for evaluation
- `data.py` - Data structures for evaluation results
- `eval_functions.py` - Core evaluation algorithms
- `utils.py` - Utility functions for metric computation
