### 🛡️ Hydra Refactor Policy: "Domains First" v5.0


**Directive**: Refactor the `configs/` directory to maximize **Separation of Concerns** and **Atomic Composition**.
**I. The Namespace Protocol**
* **Package *global***: Strictly limited to `hardware/`, `experiment/`, and `global/`. These files must be able to override any key in the tree.
* **Package *group***: Mandatory for all sub-folders (`model/`, `data/`, `train/`).
* **Schema Enforcement**: Every YAML in a group MUST have a top-level key matching its group name (e.g., `model:` or `data:`).


**II. The Domain Isolation Rule**
* The `domain/` file is the **Logic Controller**.
* **Cross-Contamination Prevention**: If a key from a foreign domain (e.g., `max_polygons` in a Recognition run) is present in the merged result, the refactor has failed.
* **Nullification**: Use `null` or `~` in domain controllers to "zero out" irrelevant global variables.


**III. The State vs. Logic Separation**
* **Infrastructure/UI Logic**: Remove all files related to frontend dashboards or preprocessing profiles from the `train/` path. These are application-state, not training-logic.
* **Performance Tiers**: Move all caching/preset logic to `runtime/performance/`. They are "Execution Strategies," not "Model Definitions."


**IV. Path Resolution**
* **No Magic Paths**: All dynamic variables `${...}` must be rooted in `global/paths.yaml`.
* **Validation**: Use `python train.py --print-config` as the ONLY source of truth for resolving interpolations.

---


---

## 1. Prototype: The "North Star" Directory Tree

Give this to the agent to define the "Shape" of the final product.

```text
configs/
├── main.yaml                # Primary entry point (Architecture & Defaults)
├── global/                  # Tier 1: System-wide (Paths, Seeds)
│   └── default.yaml
├── hardware/                # Tier 2: Resource Constraints (RTX 3060 vs 3090)
│   ├── rtx3060.yaml         # Set batch_size, workers, pin_memory here
│   └── rtx3090.yaml
├── domain/                  # Tier 3: The Controllers (The Brain)
│   ├── detection.yaml       # Links to Model Presets + Data Transforms
│   ├── recognition.yaml
│   └── kie.yaml
├── model/                   # Tier 4: The Architecture
│   └── presets/             # craft.yaml, parseq.yaml (Pre-linked fragments)
├── data/                    # Tier 5: The Source
│   ├── datasets/            # Pruned list (e.g., aihub.yaml, camscanner.yaml)
│   └── runtime/             # Transforms & Dataloader settings
└── train/                   # Tier 6: The Optimization
    ├── optimizer/
    └── logger/              # Clean: wandb.yaml, csv.yaml (No UI logic)

```

---

## 🧭 The "North Star" Directory Responsibilities (Updated)

This table serves as the definitive reference for file placement and package directives.

| Tier  | Directory              | Package Directive     | Responsibility & Constraints                                                                    |
| ----- | ---------------------- | --------------------- | ----------------------------------------------------------------------------------------------- |
| **1** | `global/`              | `# @package _global_` | **System Foundations**: Global paths, seeds, and universal trainer defaults.                    |
| **2** | `hardware/`            | `# @package _global_` | **Resource Constraints**: GPU/CPU limits, precision, and hardware-specific batch sizes.         |
| **3** | `domain/`              | `# @package _group_`  | **The Orchestrator**: Bridges model, data, and train logic. Must nullify other domains.         |
| **4** | `model/architectures/` | `# @package _group_`  | **Atomic Layers**: Pure neural network structure. No optimizers, loss, or tokenizers.           |
| **5** | `data/datasets/`       | `# @package _group_`  | **Source Identity**: Image/Annotation paths and metadata. No transforms or logic.               |
| **6** | `data/transforms/`     | `# @package _group_`  | **Data Logic**: Atomic augmentation and preprocessing components (e.g., geometry, enhancement). |
| **7** | `train/`               | `# @package _group_`  | **Optimization**: Schedulers, optimizers, callbacks, and loggers.                               |
| **8** | `experiment/`          | `# @package _global_` | **Specific Runs**: Temporary overrides for a specific training session or research hypothesis.  |


---
## 2. Prototype: The "Controller" Logic

Provide this example of `domain/recognition.yaml` to show how to enforce **Domain Separation** and fix the **CUDA Segfault** (by overriding hardware settings).

```yaml
# configs/domain/recognition.yaml
# @package _group_
defaults:
  - /model/presets: parseq
  - /data/datasets: recognition_canonical
  - /train/optimizer: adamw
  - /train/logger: wandb
  - _self_

# --- Domain Isolation ---
# Explicitly nullify detection-specific leakage
detection: null
max_polygons: null

# --- Logic Overrides ---
# Recognition needs specific transforms and logging behavior
data:
  transforms: recognition_base

```

---

These are sharp observations. The transition from a "flat" or "bloat-heavy" structure to a "Domain-First" architecture often creates confusion around execution and the technical implementation of Hydra's namespacing.

Here is the strategy to handle execution variants, package directives, and ground rules for your configuration strut.

---

## 1. Execution Variants: The `experiment/` Folder

You are right—storing dozens of variants in the `domain/` folder leads back to "overgrown lawn" territory.

* **The Rule:** The `domain/` folder should contain the **stable base** for that task (e.g., `detection.yaml`, `recognition.yaml`).
* **The Execution:** Use an `experiment/` (or `run/`) directory for specific execution variants. These are the files you actually call from the CLI.

```text
configs/
├── experiment/                  # Execution-ready configs
│   ├── detection_rtx3090_v1.yaml
│   ├── recognition_finetune_aihub.yaml
│   └── kie_debug_segfault.yaml

```

**Example: `configs/experiment/recognition_finetune_aihub.yaml**`

```yaml
# @package _global_
defaults:
  - /main                      # Pull the primary strut
  - override /domain: recognition
  - override /hardware: rtx3090
  - _self_

# Experiment-specific overrides
train:
  max_epochs: 50
  optimizer:
    lr: 1e-4

```


## 3. Package Directives: The "Hydra Namespace"

This is the most confusing part of Hydra. Think of the `@package` directive as a way to tell Hydra **"Where in the final merged dictionary should these keys live?"**

| Directive                 | Use Case                                  | Result                                                    |
| ------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| **`# @package _global_`** | Hardware, Experiments, Global constants.  | Keys are placed at the **root** of the config.            |
| **`# @package _group_`**  | Folders like `model/`, `data/`, `train/`. | Keys are placed under a key **matching the folder name**. |
| **`# @package model`**    | Explicit naming.                          | Forces keys into the `model:` key regardless of folder.   |

**The Strategy:** Use `# @package _global_` for your `hardware/` and `experiment/` files so they can easily override anything. Use `# @package _group_` for your library of components (optimizers, datasets) so they stay organized under their respective keys.

---

## 4. Ground Rules: The Strut vs. Overrides

To maintain 100% success, you must define what lives in the "Skeleton" and what is injected.

### The "Must-Be-In-Strut" Keys:

These must exist in your primary `config.yaml` (even if null) so the AI and collaborators know they exist:

* `domain: ???` (Must be provided)
* `hardware: ???` (Must be provided)
* `paths: ${...}` (Interpolated)
* `seed: 42`

### The "Override Only" Keys:

* `train.optimizer.lr`
* `data.loader.batch_size`
* `model.pretrained_path`

---
