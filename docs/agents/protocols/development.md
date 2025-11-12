# Development Protocols

**Purpose:** Concise instructions for development tasks. For detailed context, see `docs/maintainers/protocols/`.

## Coding Standards

**Formatting:**
- Use Ruff: `uv run ruff check . --fix && uv run ruff format .`
- Pre-commit hooks handle formatting automatically

**Naming:**
- Modules/Packages: `snake_case` (e.g., `data_loader.py`)
- Classes: `PascalCase` (e.g., `OCRLightningModule`)
- Functions/Methods: `snake_case` (e.g., `validate_polygons`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_IMAGE_SIZE`)

**Type Hints:**
- All public functions/methods must include type hints
- Use specific types: `Dict`, `List`, `Optional`, `Tuple`, `Union`

## Command Registry

**Validation:**
```bash
uv run python scripts/agent_tools/validate_config.py --config-name <name>
uv run python runners/train.py --config-name train trainer.fast_dev_run=true
```

**Data Tools:**
```bash
uv run python scripts/agent_tools/generate_samples.py --num-samples 5
uv run python scripts/agent_tools/list_checkpoints.py
uv run python tests/debug/data_analyzer.py --mode orientation|polygons|both
```

**UI Tools:**
```bash
uv run python ui/visualize_predictions.py --image_dir <path> --checkpoint <path>
```

## Debugging Workflow

**Triage:**
- Config errors → Check Hydra configs first
- Data errors → Inspect data pipeline (shape mismatches)
- Model errors → Check architecture/hyperparameters (CUDA OOM, NaN loss)

**Tools:**
```bash
# Smoke test
uv run python runners/train.py --config-name <name> trainer.fast_dev_run=True

# Data analysis
uv run python tests/debug/data_analyzer.py --mode both
```

**Common Fixes:**
- CUDA OOM: Reduce `batch_size`, enable `precision=16-mixed`, use `accumulate_grad_batches=2`
- NaN Loss: Lower `learning_rate`, enable `gradient_clip_val=1.0`, check data corruption
- Shape Mismatch: Verify `out_channels`/`in_channels`, use `ic(tensor.shape)` for debugging

**Debugging:**
```python
from icecream import ic
ic(features.shape)  # Print tensor shapes
```

## Bug Fix Protocol

**Bug ID Format:** `BUG-YYYYMMDD-###`
```bash
uv run python scripts/bug_tools/next_bug_id.py  # Get next ID
```

**Bug Report Location:** `docs/bug_reports/BUG-YYYYMMDD-###_name.md`
**Template:** `docs/bug_reports/BUG_REPORT_TEMPLATE.md`

**Required Sections:**
- Summary, Environment, Steps to Reproduce
- Expected vs Actual, Root Cause, Resolution
- Testing, Files Changed, Impact

**⚠️ CRITICAL: Code Indexing with Bug ID**
When making changes to core project files, **ALWAYS embed the bug ID in the code**:

1. **Function/Method Level:**
   - Add bug ID to function docstring
   - Add bug ID comments at change locations
   - Link to code changes document

2. **Example:**
   ```python
   def forward(self, pred_logits, gt, mask=None):
       """
       Forward pass for BCE loss computation.

       BUG-20251109-002: Fixed CUDA illegal memory access by:
       - Adding input validation (shape/device checks)
       - Adding CUDA synchronization before operations
       - Moving operations to CPU to avoid corrupted memory access

       See: docs/bug_reports/BUG-20251109-002-code-changes.md
       """
       # BUG-20251109-002: Validate inputs to prevent CUDA illegal memory access
       if pred_logits.shape != gt.shape:
           raise ValueError(...)
   ```

3. **Why Function-Level Indexing:**
   - Functions can be found even if files are moved/renamed/refactored
   - Bug ID is embedded in the code where changes were made
   - Searching for `BUG-YYYYMMDD-###` finds all related functions
   - Makes changes traceable and maintainable

4. **Code Changes Document:**
   - Create `docs/bug_reports/BUG-YYYYMMDD-###-code-changes.md`
   - Document all functions changed with bug ID
   - Reference specific functions, not just files
   - Include before/after code references

**Update Changelog:**
```markdown
#### Bug Fixes
- **BUG-YYYYMMDD-###**: Description (link)
```

## Utility Adoption

**Process:**
1. Identify utility need
2. Check existing utilities in `ocr/utils/`
3. Review utility adoption protocol
4. Add to registry if approved

## Refactoring

**Modular Refactor:**
- Extract common patterns into utilities
- Maintain backward compatibility
- Update tests after refactoring

**Hydra Config Refactoring:**
- Use relative paths from project root
- Validate with `validate_config.py`
- Test with `fast_dev_run=True`

## Context Logging

**Start Log:**
```bash
make context-log-start LABEL="<task>"
```

**Summarize Log:**
```bash
make context-log-summarize LOG=logs/agent_runs/<file>.jsonl
```

## Feature Implementation

**Process:**
1. Create implementation plan (use QMF toolbelt)
2. Follow coding standards
3. Add tests
4. Update documentation
5. Validate with smoke tests

