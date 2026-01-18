# Last Session Context and Pain Points + Research Follow-up

- We experieced a lot of trouble debugging the Hydra Configuration path and default value interpolation resolutions and had to terminate the session.
- I have provided you with the requested research questions and rules needed to proceed.

## Contents
1. Conversation Thread (Context and Pain Points)
2. Research Response

## 1. Conversation Thread

Agent
```
You are absolutely right. We are spinning our wheels on Hydra scope resolution mechanics which should be deterministic, but the "Domains First" architecture combined with aggressive pruning has left some ambiguity on the exact merging behavior of nested configs with package directives.

Here are the specific pain points and rules where clarification/research would turn this from a debugging slog into a "slam dunk":

🛑 Pain Points
Package Directive Interaction: It is unclear how a config file with @package _group_ (like
base.yaml
) interacts when included by another config that also defines a package directive (e.g.,
canonical.yaml
 with @package data). Does the includer's package override the included one, or do they nest (data.transforms vs just data)?
Interpolation Timing vs. Merging: We are seeing Interpolation key not found errors for paths that should exist in the final resolved config. This suggests interpolation might be attempting to resolve before the defaults are fully merged into the target namespace.
Relative vs. Absolute Paths: We've failed with both relative (${train_transform}) and absolute (${data.train_transform}) paths. We lack a clear rule for which referencing style is required when a variable is brought in via defaults.
🔬 Research Requests / Rules Needed (// Hydra (1.3.2))
To proceed swiftly, I need definitive rules or "standard patterns" for the following scenarios in this V5 architecture:

The "Include Pattern" Rule: When
canonical.yaml
 includes
base.yaml
, what is the strictly correct way to reference base's variables?
Option A:
base.yaml
 uses @package _global_ (or no directive) to merge directly into canonical's scope.
Option B:
base.yaml
 uses a strict name, and we must reference that exact structure.
The "Data Scope" Rule: Since main.yaml loads
detection.yaml
 which loads
canonical.yaml
, at what point does the data namespace actually become the root for interpolation context?
Does
canonical.yaml
 see itself as data (absolute) or root (relative)?
Recommendations
Stop debugging code. Instead, I recommend we:

Pause the verification task to avoid further "flailing."
Consult documentation/examples (if available) specifically for nested defaults interpolation in Hydra.
Use a minimal reproduction (which I can create if you wish, or you can provide) to verify the behavior of just the package merging logic, isolated from the complex OCR config.
If you can provide the correct standard/rule for how defaults included files should expose variables to their parent, I can apply that pattern immediately to
canonical.yaml
 and
base.yaml
 and we will likely be done in one step.
 ```

---
## 2. Research Response

# Hydra 1.3.2 Configuration Architecture Patterns

Hydra 1.3.2 uses clear, fixed rules for packages, merging, and interpolation that you can lean on to make your "Domains First" layout deterministic.

## 1. Nested `@package` rules

For any given included config, Hydra computes **one** effective package; they do not nest multiple `@package` directives from different files.

- The final package for a config option is chosen by this precedence:
  1) package specified in the **defaults list** entry that selects it (relative to the includer's package),
  2) else the `@package` directive inside that config (absolute),
  3) else the default package (config group → `_group_`, or `_global_` when explicitly requested).
- Because the package is computed per **config option** from its selector and its own header, a parent's `@package` does **not** wrap or compose with the child's `@package`; the child's package is determined independently.

Applied to your scenario:

- `datatransforms/base.yaml` has `@package group` (i.e. `_group_`). In the `data` config group, that gives it package `data.transforms` when selected as `- datatransformsbase`.
- `datacanonical.yaml` has `@package group`, so it is packaged as `data` when selected as `- datacanonical` in `data`.
- If `canonical.yaml` uses `defaults: - datatransformsbase`, the **transform keys live under `canonical`'s package** (`data`) by default, giving you `data.transforms.*` in the merged config, not a separate `transforms.*` root.

Actionable rule:

- To get `data.transforms.*` from any child, either:
  - keep the child as `@package group` inside a `data/transforms` group, or
  - select with a package override in the defaults list, e.g. `- datatransformsbase@data.transforms`.
- Do **not** expect `@package data` in the includer plus `@package _group_` in the included to form a nested package; the included config's package is fully determined by its own group and `@package` or by a package override in the defaults list.

## 2. Interpolation timing vs merging

Hydra composes configs in a few distinct phases; the key point is that **defaults lists are resolved and merged before normal value interpolation is fully evaluated**.

- Composition order:
  - Resolve and expand the **defaults lists** (including overrides), merging selected configs by order ("last wins" for scalars, dictionary merge for mappings).
  - This yields the **Final Config Object** structure.
  - Value interpolations like `${foo.bar}` are then resolved when accessing or when `OmegaConf.to_yaml(resolve=True)` is called, but they see the **fully merged** tree.
- Exception: **interpolations inside defaults lists themselves** are evaluated earlier and have strict limits:
  - They can only refer to other config groups (not arbitrary final values).
  - They are always absolute.
  - The sub-tree of a config selected by such an interpolated defaults item may not contain further defaults overrides.

Your "Interpolation key not found" errors (e.g. `traintransform`, `defaultinterpolation`, `datasetbasepath`) in the log show failures like:

- `Interpolation key transforms.traintransform not found fullkey group.datasets.traindataset.transform`
- `Interpolation key defaultinterpolation not found fullkey group.transforms.traintransform.transforms0.interpolation`
- These originate from configs like `datatransforms/base.yaml` where `interpolation: defaultinterpolation` is used but `defaultinterpolation` is defined in a **different config file** (`datatransforms/backgroundremoval.yaml`) or under a different package (`data.transforms`).

Interpreting these with the timing rules:

- The defaults chain `main.yaml → domaindetection.yaml → datacanonical.yaml → datatransformsbase.yaml` is merged first, giving you a tree rooted at `global`, `hardware`, `domain`, `data`, etc.
- When OmegaConf then evaluates `interpolation: defaultinterpolation` inside `transforms.traintransform`, it looks **from the final merged root**, using the package of that config; if `defaultinterpolation` is not in the same package or referenced via a fully qualified path, the interpolation fails.

Actionable rules:

- Avoid **unqualified** scalars like `interpolation: defaultinterpolation` that depend on another config file, unless the defining key is in the **same package** and guaranteed to be merged first.
- When a value is defined in a different config (e.g. `datatransformsbackgroundremoval.yaml` with `defaultinterpolation: 1` under `data.transforms`), reference it with an absolute path from the final root, e.g. `interpolation: ${data.transforms.defaultinterpolation}` inside `data.transforms.base`.
- Keep **all cross-file interpolations out of defaults lists** unless they are config-group interpolations obeying Hydra's restrictions.

## 3. Variable references in nested hierarchies

OmegaConf (and thus Hydra 1.3) supports both **absolute** and **relative** value interpolation:

- Absolute: `${a.b.c}` from the global root.
- Relative: `${.foo}`, `${..bar}`, etc., relative to the current node, available with OmegaConf ≥ 2.1 and Hydra ≥ 1.1.

In multi-level Hydra defaults chains like:

```yaml
# main.yaml
defaults:
  - _self_
  - global: default
  - hardware: rtx3060
  - domain: detection
  - runtime: performancenone
  - experiment: null
```

```yaml
# domaindetection.yaml
# @package group
defaults:
  - override modelarchitectures: dbnet
  - override data: canonical
  - _self_
```

```yaml
# datacanonical.yaml
# @package group
defaults:
  - datatransformsbase
  - datadataloadersdefault
  - performancepreset: none
  - _self_
```

```yaml
# datatransformsbase.yaml
# @package group
# defines transforms.traintransform, defaultinterpolation, etc.
```

the **final names** of nodes that can be referenced by interpolation are:

- Anything in `data/...` lives under `data` because of `@package group` in the `data` group.
- `traintransform` under `datatransformsbase` is merged as `data.transforms.traintransform`, not as a root `traintransform` or a bare `transforms.traintransform`.

Your failed attempts:

- `${train_transform}` fails because `train_transform` is not at the root; it lives under `data.transforms.traintransform` (and your key is `traintransform` without underscore).
- `${data.train_transform}` fails because that path does not exist; the actual path in your configs is `data.transforms.traintransform`.

Actionable patterns:

- Within any `data/*` config with `@package group`, reference `traintransform` from another data file as:
  - Absolute: `${data.transforms.traintransform}`.
  - Relative (inside another `data.*` node): `${..transforms.traintransform}` or `${data.transforms.traintransform}`, depending on how deep you are.
- To use tokenizer parameters from `datarecognition.yaml` where tokenizer is defined as `data.tokenizer`, reference them anywhere as `${data.tokenizer.maxlen}`.
- To reuse `datasetbasepath` from `datacanonical.yaml` inside `datadefault.yaml`, reference `${data.datasetbasepath}` rather than `${datasetbasepath}` or `${data.dataset_base_path}`.

**Cross-package** rule:

- When crossing package boundaries (e.g. from `data` to `global.paths` or `hardware`), **always use absolute paths** rooted at the final merged namespace, e.g. `${global.paths.datasets.train}`.
- Keep "domain globals" (paths, seeds, etc.) in `package global` configs and only refer to them as `${global.*}` from elsewhere, as your `globalpaths.yaml` already does.

## 4. Namespace context across the chain

Each config's **effective package** is determined once, based on its config group and its own `@package` (or a package override in the defaults selector); the config does **not** see itself as root in the final tree.

- `main.yaml` has `@package global`, so it contributes to the root keys `global.*`, `hardware.*`, `domain.*`, etc., not to bare `trainer` or `dataloaders` at the root.
- `domaindetection.yaml` has `@package group`, so it contributes under `domain.detection.*` in the final config; within that file, unqualified keys like `task`, `batchsize` end up as `domain.detection.task`, `domain.detection.batchsize`.
- `datacanonical.yaml` has `@package group`, so unqualified keys become `data.*`, and defaults entries like `- datatransformsbase` bring in `data.transforms.*`.
- `datatransformsbase.yaml` also uses `@package group`, so all its content maps into `data.transforms.*` when selected via the `data/transforms` group.

Effective "view" from each file:

- **Inside `canonical.yaml`**:
  - The (package, relative path) of a key `trainnumsamples` is (`data`, `trainnumsamples`).
  - Interpolation `${trainnumsamples}` is resolved as `${data.trainnumsamples}` only if Hydra supports relative interpolation and you are under `data`; otherwise you must use `${data.trainnumsamples}` explicitly.
- **Inside `base.yaml` (transforms)**:
  - Package is `data.transforms`, so unqualified scalar `defaultinterpolation` becomes `data.transforms.defaultinterpolation` if defined there, and `interpolation: ${defaultinterpolation}` resolves to `data.transforms.defaultinterpolation`.
  - In your setup, `defaultinterpolation` is actually defined in a different file (`datatransformsbase.yaml` vs `datatransformsbackgroundremoval.yaml`), so unqualified interpolation fails unless that key is merged into the same package before evaluation.
- The namespace **does not change** as configs are included; inclusion only determines which configs are merged and in what order, not the package from which they interpret relative paths.

Actionable mental model:

- Treat each config file as owning a **fixed, absolute package** such as `global`, `domain.detection`, `data`, or `data.transforms`; defaults composition just decides whether its contents are merged into the final tree, not how they are nested.
- When thinking "what does this config see itself as?", answer: **"It always sees itself as its package root"** (`data`, `data.transforms`, `domain.recognition`, etc.). Interpolation paths are evaluated from the true root, but the package determines which unqualified names are legal (when using relative interpolation).

## 5. Best-practice patterns for your Domains First layout

Drawing from Hydra's official docs and from the concrete configs and errors in your repo, these patterns will remove the interpolation ambiguity:

1. **Make every config explicitly packaged**
   - Use `@package global` for `main.yaml`, `globalpaths.yaml`, `hardwarertx3060.yaml` so they can override anything at the root.
   - Use `@package group` for config groups like `data`, `model`, `train`, and domain controllers so their contents live under predictable keys (`data.*`, `model.*`, `train.*`, `domain.*`).

2. **Keep all cross-domain or cross-package values referenced with absolute interpolation**
   - Paths: `datasetbasepath: ${global.paths.datasets.train}` rather than a relative interpolation to `hydra.runtime.cwd` in multiple places.
   - Domain-specific wiring: from a `domain.recognition` controller, reference datasets as `${data.datasets.traindataset}` and transforms as `${data.transforms.traintransform}`.

3. **Never rely on "sibling magic" for constants like `defaultinterpolation`**
   - Put such constants into a **single, authoritative config** (e.g. `datatransformsbase.yaml` or `datatransformsconstants.yaml`, with `@package data.transforms`).
   - Refer to them everywhere as `${data.transforms.defaultinterpolation}`. That removes ambiguity about which file needs to be merged first.

4. **Include pattern for exposing variables to parents**
   - Child config should:
     - Use `@package group` to map under the right subtree (`data.transforms`, `data.datasets`, etc.).
     - Define strongly named nodes (`traintransform`, `valtransform`, `defaultinterpolation`, `datasettemplate`, etc.).
   - Parent config should:
     - Select the child in its defaults normally, *without* trying to repackage it.
     - Reference the child's exported nodes by fully qualified paths (`${data.transforms.traintransform}`, `${data.datasets.datasettemplate}`, `${data.transforms.defaultinterpolation}`).

5. **Domain controller pattern**
   - Domain controllers (`domainrecognition.yaml`, `domaindetection.yaml`) stay `@package group` so they own `domain.recognition.*`, `domain.detection.*` and contain only:
     - explicit domain-specific keys (e.g. `recognition.charset`, `detection.maxpolygons`),
     - `null`-overrides to kill foreign-domain leakage,
     - selections of model/data/train presets via the defaults list.
   - They should **not** redefine `data.*` nodes themselves; instead, they drive selection of the appropriate `data` presets and read domain-specific settings from `domain.*`. This keeps config pruning predictable and respects the Domain Isolation rules in your policy doc.

With these rules, you can:

- Make `data.canonical` the canonical detection dataset, always referencing its `datasetbasepath` as `${data.datasetbasepath}` and its transforms as `${data.transforms.traintransform}`.
- Make `data.recognition` own LMDB paths and tokenizer, referenced as `${data.tokenizer}` and `${data.datasettemplate}`, and keep recognition-only transforms under a dedicated `data.transformsrecognition` config group.
- Ensure that any interpolation you write either resolves within the same package (relative) or from the root (absolute), regardless of how many levels of defaults indirection you have between `main.yaml` and the final fragment that defines the value.
