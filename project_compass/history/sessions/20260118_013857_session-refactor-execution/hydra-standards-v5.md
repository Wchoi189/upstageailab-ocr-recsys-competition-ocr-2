# 🛡️ Hydra Configuration Standards v5.0

**Status**: Production Ready | **Architecture**: Domains-First | **Hydra Version**: 1.3.2

## 1. The Flattening Rule

To prevent "Double Namespacing" (e.g., `train.logger.wandb.wandb`), all YAML files in component groups MUST be flattened.

* **Rule**: If a file is located in a group-namespaced folder (e.g., `train/callbacks/`), the YAML must NOT contain a top-level key matching the filename or folder.
* **Implementation**: All keys, including `_target_`, must start at column 0.
* **Standard Template**:
```yaml
# configs/train/callbacks/early_stopping.yaml
# @package _group_
_target_: lightning.pytorch.callbacks.EarlyStopping
monitor: "val/hmean"
patience: 5

```



## 2. Absolute Root Anchoring

To prevent "Interpolation Key Not Found" errors, all cross-references must use absolute paths anchored to the root namespace.

* **Rule**: Never use relative interpolations like `${train_transform}` if the key exists in a different package.
* **Standard Patterns**:
* **Global Paths**: Always use `${global.paths.data_dir}`.
* **Component Logic**: Always use `${data.transforms.train_transform}`.
* **Local Variables**: Use `${.local_key}` ONLY for variables defined within the same file.



## 3. Aliasing & Namespacing Logic

Use aliasing in `defaults` lists to support multiple components of the same type (e.g., multiple loggers) without key collisions.

* **Syntax**: `[filename] @ [package_directive] . [custom_alias]`.
* **Example**:
```yaml
# configs/train/logger/default.yaml
defaults:
  - wandb@_group_.wandb_logger  # Resolved as: train.logger.wandb_logger
  - csv@_group_.csv_logger      # Resolved as: train.logger.csv_logger

```



## 4. Domain Isolation Protocol

Domain Controllers (`configs/domain/*.yaml`) are the orchestrators. They must explicitly nullify irrelevant keys to ensure system stability and prevent CUDA segfaults.

* **Rule**: If a variable belongs to the "Detection" domain, it must be set to `null` in the "Recognition" domain controller.
* **Example**:
```yaml
# configs/domain/recognition.yaml
# @package _group_
detection: null
max_polygons: null

```



## 5. Directory Responsibilities

| Tier | Directory              | Package Directive     | Responsibility                                   |
| ---- | ---------------------- | --------------------- | ------------------------------------------------ |
| 1    | `global/`              | `# @package _global_` | Path constants, seeds, and trainer defaults.     |
| 2    | `hardware/`            | `# @package _global_` | GPU/CPU limits and batch size overrides.         |
| 3    | `domain/`              | `# @package _group_`  | Task-specific logic (Detection vs. Recognition). |
| 4    | `model/architectures/` | `# @package _group_`  | Pure neural network layer definitions (Atomic).  |
| 5    | `data/datasets/`       | `# @package _group_`  | Source paths and metadata (No logic).            |

---

### 🚀 Implementation Strategy

Whenever you or your agent modifies a file:

1. **Audit** against the **Flattening Rule**.
2. **Anchor** all interpolations to the **Absolute Root**.
3. **Verify** using the validation command:
`python scripts/utils/show_config.py main domain=[task]`.
