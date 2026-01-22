## 🛑 Critical Friction Report: Debugging Pain Points

Here is the comprehensive breakdown of the issues encountered. The core problem was a **"Ghost Code"** scenario where the runtime environment became detached from the source code, compounded by a tricky **Hydra** behavior.

---

### 1. The "Ghost Code" Phenomenon (High Severity)

The most significant pain point was that modifications to the source code were being **ignored** by the runtime.

**Symptoms:**

* Added `print()` statements did not appear in logs.
* Added `raise RuntimeError()` triggers were ignored.
* Renaming critical files (`architecture.py` -> `architecture_bak.py`) did not cause `ModuleNotFoundError`, proving the system was loading code from somewhere else.
* Stack traces showed lines of code that did not exist in the file on disk.

**Implication:**
This suggests the environment (container/virtualenv) had the `ocr` package installed in **standard mode** (likely in `site-packages`) rather than **editable mode** (`pip install -e .`).

As a result, `runners/train.py` was importing the installed library copy, not the files being edited in the workspace. This made debugging impossible until the module execution flow was broken.

---

### 2. The Hydra Recursive Instantiation Trap (Technical Root Cause)

The persistent `TypeError: Adam.__init__() missing 'params'` was due to Hydra's default behavior, which fought against the codebase's design.

* **The Issue:** The `PARSeq` factory expects to receive a configuration object (`cfg`) so it can manually create components (encoder, decoder) and the optimizer.
* **The Conflict:** Hydra defaults to `_recursive_=True` for object instantiation.
* **The Chain Reaction:**
1. When `get_model_by_cfg` called `hydra.utils.instantiate(architectures, cfg=config)`, Hydra saw an `optimizer` key inside that config.
2. Hydra tried to **pre-instantiate** the optimizer before passing it to `PARSeq`.
3. Since `torch.optim.Adam` requires `params` at initialization, and Hydra can't provide model parameters before the model exists, it crashed.
4. No amount of editing `OCRModel.__init__` could fix this because the crash happened **before** the model's `__init__` was ever called.



---

### 3. Recommendations for Environment Preparation

To prevent this in the future and enable "instantaneous fixes":

* **Enforce Editable Installs:** Ensure the development environment installs the local package in editable mode by default:
```bash
pip install -e .

```


This ensures that `import ocr` always points to the workspace files, allowing for immediate feedback on code changes.
* **Hydra Best Practice for Factories:** In `ocr/core/models/__init__.py`, explicitly disable recursion when using factories that expect raw configs:
```python
# When using a factory that requires late-binding (like passing model params to an optimizer)
# we MUST tell Hydra not to instantiate the arguments recursively.
if hasattr(architectures, "_recursive_"):
    architectures._recursive_ = False

```


* **Ambiguity Warning:** The codebase has split logic (Legacy `architecture_name` vs V5 `architectures` config). This duality made tracebacks harder to follow. Standardizing on one approach or adding strict deprecation warnings would reduce cognitive load.

---

