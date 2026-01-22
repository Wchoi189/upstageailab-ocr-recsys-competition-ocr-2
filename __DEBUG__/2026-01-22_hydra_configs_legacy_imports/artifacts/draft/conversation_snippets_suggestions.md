The discovery of that duplicate file is a classic symptom of the **"Ghost Code"** phenomenon you reported earlier. Having `ocr/core/lightning/utils/config_utils.py` exist twice (or having the logic duplicated in `ocr/core/utils/config_utils.py`) is exactly how your AI agent gets trapped in a cycle of "fixing" code that isn't actually being executed.

Here is the breakdown of the problematic script, the "Doc Sync" architecture, and your automation roadmap.

---

### 1. Analysis: Is your `config_utils` problematic?

**Yes, but for structural reasons rather than just the recursive logic.**

* **The Duplicate Trap:** You have `ocr.core.lightning.utils.config_utils` importing from `ocr.core.utils.config_utils`. This circular-sounding dependency suggests that the "Utility" layer and the "Domain" layer are blurred.
* **The Recursive Search Issue:** In `extract_normalize_stats`, the script manually iterates through `train`, `val`, `test`, and `predict` transforms to find `albumentations.Normalize`.
* **The Risk:** If your Hydra config structure changes (e.g., adding a `test_v2_transform`), this utility will **silently fail** to find the stats.
* **Better Approach:** Instead of hardcoded attribute names, this should use a visitor pattern or a generic search tool that looks for the `_target_` regardless of which transform block it's in.


* **The `isinstance(cfg_dict, dict)` violation:** Your own **AgentQMS** standard says "NEVER use `isinstance(obj, dict)` on configs." Yet, your utility script does exactly that on line 29. Even if `ensure_dict` was called, if it returned a `DictConfig` that failed to resolve, `isinstance` would return `False` and silently return `{}`.

---

### 2. The Doc-Sync Audit Tool for AgentQMS

The goal of **Doc-Sync** is to ensure that the **Agent Standards** (YAML) and the **Python Implementation** (Docstrings) are in perfect alignment.

**How it works:**

1. **Extract Patterns:** It parses `configuration-standards.yaml` to find "bad_patterns" (e.g., `isinstance(cfg, dict)`).
2. **Structural Search:** It uses your **ADT (ast-grep)** to scan the codebase for those exact patterns.
3. **Cross-Reference:** It checks the `AgentQMS` registry to see if the file being scanned is marked as "Tier 2 Compliant."
4. **Reporting:** If it finds a "bad_pattern" in a "Tier 2" file, it fails the build.

**Example execution command for your agent:**

```bash
# Verify if the duplicate config_utils violates the new standards
adt sg-search pattern="isinstance($CFG, dict)" --path ocr/core/lightning/utils/config_utils.py

```

---

---

### 4. Implementation: The "Kill-Duplicate" Plan

Before you let the agent start debugging, you must resolve the "Config Utils" split.

**Phase 1: Consolidation**

1. Move the generic `is_config` and `ensure_dict` logic into `ocr/core/utils/config_utils.py`.
2. Move the Lightning-specific `extract_metric_kwargs` into a specialized factory, e.g., `ocr/core/lightning/factories/metrics.py`.
3. **Delete** the duplicate `ocr/core/lightning/utils/config_utils.py` entirely.

**Phase 2: The Agent Instruction**
Tell your agent:

> "I have identified a duplicate file conflict between `core/utils` and `core/lightning/utils`.
> 1. Your first task is to **RECONCILE** these into a single source of truth in `core/utils`.
> 2. Ensure the code adheres to `AgentQMS/standards/tier2-framework/configuration-standards.yaml`.
> 3. Use `is_config(cfg)` instead of `isinstance(cfg, dict)`."
>
>

---

### Elite Pro Tip: The "Shadow Import" Test

To prove the duplicate is causing issues, run this in your terminal:

```bash
python -c "import ocr.core.lightning.utils.config_utils as c1; import ocr.core.utils.config_utils as c2; print(f'C1: {c1.__file__}\nC2: {c2.__file__}'); print(f'Same object? {c1 is c2}')"

```

If "Same object?" is `False`, you have a **split-brain scenario** where updating one file does absolutely nothing to the behavior of the other, even though they look identical.

---

The confirmation that `Same object? False` is exactly the "Split-Brain" scenario we feared. You have two physically distinct files with nearly identical code, and Python is treating them as entirely different modules. Any fix applied to `ocr.core.utils.config_utils` will **not** affect the behavior of `ocr.core.lightning.utils.config_utils`, which is likely where the actual training pipeline is importing from.

Also, your `adt sg-search` failed to find the match because of a subtle syntax mismatch in the CLI call (it interpreted the pattern slightly differently), but the manual inspection confirms the violation is there.

---

### 1. The Consolidation Diff (The "Healing" Edit)

This unified diff will consolidate the logic into the `core/utils` "Source of Truth" and update the code to comply with your **AgentQMS Tier 2** standards (replacing `isinstance` with `is_config`).

```diff
--- a/ocr/core/utils/config_utils.py
+++ b/ocr/core/utils/config_utils.py
@@ -1,11 +1,38 @@
-"""Core utility helpers for configuration."""
+"""Core utility helpers for configuration and extraction."""
 from __future__ import annotations
-from typing import Any
+from typing import Any, TypeGuard
 from omegaconf import DictConfig, ListConfig, OmegaConf
+import numpy as np

-def ensure_dict(cfg: Any, resolve: bool = True) -> dict | list | Any:
+def is_config(obj: Any) -> TypeGuard[dict | DictConfig]:
+    """Returns True if object is a dict or an OmegaConf DictConfig."""
+    return isinstance(obj, (dict, DictConfig))
+
+def ensure_dict(cfg: Any, resolve: bool = True) -> dict | list | Any:
     """Recursively converts OmegaConf objects to native Python types."""
     if isinstance(cfg, (DictConfig, ListConfig)):
         return OmegaConf.to_container(cfg, resolve=resolve)
     return cfg
+
+def extract_normalize_stats(config: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
+    """Locate normalize transform statistics using structural search."""
+    transforms_cfg = getattr(config, "transforms", None)
+    if not transforms_cfg:
+        return None, None
+
+    # Search through all standard transform slots
+    for attr in ("train_transform", "val_transform", "test_transform", "predict_transform"):
+        section = getattr(transforms_cfg, attr, None)
+        transforms = getattr(section, "transforms", None) if section else None
+
+        if isinstance(transforms, ListConfig):
+            for transform in transforms:
+                t_dict = ensure_dict(transform, resolve=True)
+                if is_config(t_dict) and t_dict.get("_target_") == "albumentations.Normalize":
+                    mean = t_dict.get("mean")
+                    std = t_dict.get("std")
+                    if mean is not None and std is not None:
+                        return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)
+
+    return None, None

```

**Next Step for you/Agent:**
Delete `ocr/core/lightning/utils/config_utils.py` and update imports in `orchestrator.py` to point to `ocr.core.utils.config_utils`.

---
### Pro Tip

1. **The "Atomic Delete" Rule:** When you find a duplicate, do not leave it as a "bak" file. AI agents will often read `.py.bak` files and get confused. **Delete it immediately.**



