To ensure a successful project-wide migration, follow this **Refactor Checklist**. It maps your current "fragmented" files to the **Absolute Root** pattern to prevent the interpolation errors found in your debug logs.

---

### 📋 Hydra "Absolute Root" Refactor Checklist

#### 1. Tier 1: Global Foundation (The Anchors)

| File                          | Action                       | Rationale                                                                     |
| ----------------------------- | ---------------------------- | ----------------------------------------------------------------------------- |
| `configs/global/paths.yaml`   | Add `# @package _global_`    | Ensures `${global.paths}` is the absolute source of truth for all components. |
| `configs/global/default.yaml` | Add `# @package _global_`    | Makes `trainer` and `dataloader` defaults reachable at the root level.        |
| `configs/main.yaml`           | Ensure `# @package _global_` | Acts as the primary orchestrator and top-level namespace.                     |

#### 2. Tier 2: Component Libraries (The Groups)

| File                      | Action                   | Rationale                                                                            |
| ------------------------- | ------------------------ | ------------------------------------------------------------------------------------ |
| `configs/data/**/*.yaml`  | Set `# @package _group_` | Places all data keys under the `data:` root key for consistent absolute referencing. |
| `configs/model/**/*.yaml` | Set `# @package _group_` | Places all model keys under the `model:` root key.                                   |
| `configs/train/**/*.yaml` | Set `# @package _group_` | Organizes optimization logic under the `train:` root key.                            |

#### 3. Tier 3: Interpolation Repair (The Stability Fix)

* **Use Full Paths**: Replace relative keys like `${train_transform}` with `${data.transforms.train_transform}` in all dataset definitions.
* **Global References**: Always reference paths using `${global.paths.root_dir}` instead of relative `./` paths to avoid "Magic Path" errors.
* **Dot-Notation for Local Keys**: Use `${.key_name}` *only* for variables defined within the exact same YAML file to minimize scope confusion.

---

### 🧪 Immediate Validation Test

After applying the POC to your `data/canonical.yaml` and `data/transforms/base.yaml`, run the following command to verify the namespace structure without triggering lexer errors:

```bash
# Verify that 'data' and 'global' keys appear at the top level
python scripts/utils/show_config.py main domain=detection

```

### 💡 Pro-Tip for Agent Stability

When working with your AI thought partner, enforce the **"Law of Absolute Resolution"**: always provide the output of `python train.py --print-config`. This allows the agent to see exactly how Hydra has resolved the nested namespaces, eliminating guesswork during the refactor.
