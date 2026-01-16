# OCR Domain Separation Refactor Roadmap

> [!IMPORTANT]
> **Mission**: Implement a strict **"Domains First"** architecture.
> **Key Change**: Move domain-specific logic to `ocr/domains/`. `ocr/core` must contain ONLY truly common, domain-agnostic utilities.

## 1. Tooling Status
- **sg_search**: ✅ Fixed and operational. Used for audit.
- **adt CLI**: ✅ Conditional access (with `source .venv/bin/activate`).

## 2. Audit Findings (See `ocr-domain-classification-report`)
We have performed a keyword audit (`polygon`, `tokenizer`, `box`) on `ocr/core` and identified the following migration targets:

| File Path                          | Original Scope   | Detected Leakage              | Target Domain       |
| :--------------------------------- | :--------------- | :---------------------------- | :------------------ |
| `ocr/core/metrics/box_types.py`    | Core Metrics     | `polygon`, `quad`             | **Detection**       |
| `ocr/core/validation.py`           | Core Validation  | `polygon` validation          | **Detection**       |
| `ocr/core/utils/wandb_utils.py`    | Common Utils     | `polylines`, `text rendering` | **Split** (Det/Rec) |
| `ocr/core/lightning/ocr_pl.py`     | Lightning Module | `tokenizer`                   | **Recognition**     |
| `ocr/core/utils/text_rendering.py` | Common Utils     | `font`, `draw_text`           | **Recognition**     |

## 3. Task Distribution Strategy
For the next session (Execution), we will distribute tasks as follows:

### A. The "Splitter" (Agent + Human Loop)
**Target**: Files that need decomposing (e.g., `wandb_utils.py`, `ocr_pl.py`).
- **Strategy**:
    1. Create empty destination files in `ocr/domains/{domain}/...`.
    2. Move domain-specific functions.
    3. Update imports in the original file to point to new locations (if backward compatibility is needed) or remove them.

### B. The "Mover" (Automated)
**Target**: Files that move 1:1 (e.g., `box_types.py`).
- **Strategy**:
    1. `git mv` (or file move) to `ocr/domains/detection/utils/box_types.py`.
    2. Search & Replace imports across the codebase using `sed` or `ripgrep`.

### C. The "Verifier" (Test Loop)
**Target**: Ensuring no broken paths.
- **Strategy**:
    1. Run `source .venv/bin/activate && pytest tests/unit/core`.
    2. Use `adt analyze-dependencies` to check for circular imports.

## 4. Next Session Goals
1. **Initialize `ocr/domains`**: Create the directory structure.
2. **Execute Moves**: Start with "The Mover" tasks (easier).
3. **Execute Splits**: Tackle `wandb_utils.py` and `ocr_pl.py`.
