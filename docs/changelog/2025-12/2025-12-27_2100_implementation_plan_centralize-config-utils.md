# Configuration Hardening Strategy

## Goal Description
Eliminate pipeline fragility caused by ambiguous configuration objects (e.g., `DictConfig` vs [dict](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/model_loader.py#59-100)) and silent failures. Establish strict standards for configuration handling that "industry experts" rely on: explicit schemas, boundary enforcement, and centralized utilities.

## User Review Required
> [!IMPORTANT]
> This plan proposes strictly banning ad-hoc `isinstance(x, dict)` checks on configuration objects. All configuration handling must go through the centralized [ocr/utils/config_utils.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/config_utils.py).

## Proposed Changes

### 1. Codebase Audit & Refactor
Identify and fix risky patterns where `OmegaConf` objects are treated as standard dictionaries without verification.

#### [MODIFY] Existing Python Files
- Audit all `isinstance(..., dict)` checks (grep results).
- Replace safe guards with `ocr.utils.config_utils.to_dict()` or `isinstance(..., (dict, DictConfig))` where appropriate.

### 2. Centralize Configuration Utilities
Strengthen [ocr/utils/config_utils.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/config_utils.py) to be the single source of truth.

#### [MODIFY] ocr/utils/config_utils.py
- Add `ensure_dict(cfg) -> dict` helper that safely converts `DictConfig` to [dict](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/inference/model_loader.py#59-100) (recursive).
- Add `ensure_config(cfg) -> DictConfig` helper.
- Add Strict Type Guards: `def is_config(obj) -> TypeGuard[DictConfig | dict]: ...`

### 3. AI Developer Standards
Create a specific guide for AI agents to prevent regression.

#### [NEW] .ai-instructions/standards/configuration.md
- **Rule 1:** Never check `isinstance(x, dict)` on a config variable.
- **Rule 2:** Always resolve configs to native python types (list, dict, int) before passing to 3rd party libraries (like OpenCV/Numpy) unless they explicitly support `Omegaconf`.
- **Rule 3:** Use `ocr.utils.config_utils` for all config I/O.

## Verification Plan
### Automated Tests
- Create a unit test suite `tests/test_config_safety.py` that specifically passes `DictConfig` objects to all utility functions and asserts they do not fail silently.
- Run `make test` (or equivalent) to ensure no regressions.
