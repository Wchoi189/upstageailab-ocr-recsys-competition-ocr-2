---
ads_version: "1.0"
title: "Documentation Conventions"
date: "2025-12-16 22:11 (KST)"
type: "reference"
category: "reference"
status: "active"
version: "1.0"
tags: ['reference', 'reference']
---



# Documentation Conventions

**Purpose**: Enforce consistency and conciseness across inference module documentation.

**Scope**: All documents in `docs/api/`, `docs/reference/`, `docs/architecture/` covering inference module.

---

## File Organization

### Tier 1: API Reference (`docs/api/inference/`)
Component contracts and public API specifications.

**Files**:
- `orchestrator.md` - InferenceOrchestrator API
- `model_manager.md` - ModelManager API
- `preprocessing_pipeline.md` - PreprocessingPipeline API
- `postprocessing_pipeline.md` - PostprocessingPipeline API
- `preview_generator.md` - PreviewGenerator API
- `image_loader.md` - ImageLoader API
- `coordinate_manager.md` - CoordinateManager API
- `preprocessing_metadata.md` - PreprocessingMetadata API
- `contracts.md` - Data contracts (InferenceMetadata, PreprocessingResult, etc.)

### Tier 2: Architecture (`docs/architecture/`)
System-level design and compatibility statements.

**Files**:
- `inference-overview.md` - Architecture summary
- `backward-compatibility.md` - Compatibility statement

### Tier 3: Reference (`docs/reference/`)
Data structures, contracts, and system relationships.

**Files**:
- `inference-data-contracts.md` - Data flow specifications
- `module-structure.md` - Component dependency graph

### Supporting (`docs/_templates/`)
Schema definitions for consistent documentation generation.

**Files**:
- `component-spec.yaml` - Template for component specs
- `api-signature.yaml` - Template for method signatures
- `data-contract.yaml` - Template for data contracts

---

## Filename Rules

**Format**: lowercase, hyphens only
- ✅ `orchestrator.md`, `model_manager.md`, `backward-compatibility.md`
- ❌ `Orchestrator.md`, `model-manager.md`, `BackwardCompatibility.md`

**Descriptive**: Use full terms, no abbreviations
- ✅ `backward-compatibility.md`, `preprocessing-pipeline.md`
- ❌ `compat.md`, `preproc-pipe.md`

**Patterns**:
- Components: `{component-name}.md`
- Data: `{data-entity}-contracts.md`
- System: `{system}-overview.md`

---

## Frontmatter (Required All Documents)

```yaml
---
type: api_reference | architecture | data_reference | changelog
component: orchestrator | model_manager | ... | (null for multi-component)
status: current | deprecated
version: "X.Y"
last_updated: "YYYY-MM-DD"
---
```

**Required Fields**:
- `type`: Document category
- `status`: current | deprecated (if deprecated, include migration path)
- `version`: Document version (independent of code version)

**Optional**:
- `component`: Single component if applicable; null for cross-component docs

---

## Content Style (STRICT)

### Mandatory Constraints
- **Concise**: Max 3 sentences per logical section
- **Technical Only**: Objective facts, no tutorials
- **No Explanations**: Say "what", not "why"
- **No Narrative**: No "The component helps you..." or "First, you need to..."
- **Imperative**: "Accepts ImageData → Returns ProcessedData"

### Forbidden Patterns
❌ "This component is useful for..."
❌ "How to use..."
❌ "The reason we..."
❌ "In summary, the..."
❌ Explanatory prose
❌ Rationale or justification

### Approved Patterns
✅ "Accepts: ImageData. Returns: ProcessedData."
✅ "Thread-safe: No. Stateful: Yes."
✅ "Raises: ValueError if image format unsupported."
✅ "Dependencies: ModelManager, PreprocessingPipeline"
✅ "Breaking change: Method signature changed in v2.0"

---

## Section Structure

Every component document must have these sections in order:

### 1. Purpose (1 line)
```markdown
## Purpose
Coordinates inference workflow between components.
```

### 2. Interface (table)
```markdown
## Interface

| Method | Signature | Returns | Raises |
|--------|-----------|---------|--------|
| predict | `predict(image, settings)` | InferenceResult | FileNotFoundError, ValueError |
| shutdown | `shutdown()` | None | RuntimeError |
```

### 3. Dependencies (list)
```markdown
## Dependencies
- **Imports**: numpy, torch, PIL
- **Components**: ModelManager, PreprocessingPipeline, PostprocessingPipeline
- **External**: perspective_correction (module)
```

### 4. State Management (declarative)
```markdown
## State
- **Stateful**: Yes (maintains component references)
- **Thread-safe**: No
- **Lifecycle**: initialized → loaded → ready → inferring
```

### 5. Constraints (list)
```markdown
## Constraints
- Requires model loaded before inference
- GPU/CPU device determined at initialization
- Single concurrent inference per instance
```

### 6. Backward Compatibility (explicit yes/no)
```markdown
## Backward Compatibility
✅ **Status**: Maintained
- Public method signatures unchanged
- Return types unchanged
- Exception behavior unchanged
```

**If breaking changes exist**:
```markdown
## Backward Compatibility
❌ **Status**: Breaking Changes in v2.0

| What | Was | Now | Migration |
|------|-----|-----|-----------|
| Method | `predict(img, cfg)` | `predict(img, settings)` | Rename parameter |
```

---

## Data Contracts Section Format

For data structure documentation:

```markdown
## InferenceMetadata

**Source**: PreprocessingMetadata
**Targets**: PreprocessingPipeline, PostprocessingPipeline, PreviewGenerator
**Version**: 1.0

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| original_size | Tuple[int, int] | Yes | (width, height) |
| processed_size | Tuple[int, int] | Yes | typically 640x640 |
| padding | Dict[str, int] | Yes | {top, bottom, left, right} |
| padding_position | Literal | Yes | 'top_left' or 'center' |

**Invariants**:
- If padding_position='top_left': top=0, left=0
- processed_size >= original_size
- content_area within processed_size

**Backward Compatible**: ✅ Yes
```

---

## Code Examples (When Required)

Use only to define behavior, NOT teach usage:

```markdown
## Example: Coordinate Transformation

Input: {x: 100, y: 200, padding_position: 'center', padding: {left: 50, top: 50}}
Output: {x: 50, y: 150} (adjusted for padding offset)
```

❌ Don't show "how to import" or "how to call"
✅ Show input → output to define behavior

---

## Formatting Rules

### Tables (Preferred for data)
Use tables instead of prose for:
- Method signatures
- Parameter specifications
- Component relationships
- Constraint lists

**Example**:
```markdown
| Component | Purpose | State | Thread-safe |
|-----------|---------|-------|-------------|
| ModelManager | Lifecycle | Stateful | No |
```

### Lists (For sequences)
Use for:
- Dependencies
- Constraints
- Lifecycle states
- Breaking changes

```markdown
- Component A
- Component B
- Component C
```

### Prose (Minimal)
Use only for:
- 1-sentence purpose statement
- Non-obvious notes
- Migration instructions

**Max length**: 3 sentences

---

## Backward Compatibility Section

Every document must explicitly state compatibility status:

### If Maintained ✅
```markdown
## Backward Compatibility
✅ **Maintained**: No breaking changes
- Public API unchanged
- Method signatures identical
- Return types identical
- Exception behavior identical
```

### If Deprecated ⚠️
```markdown
## Backward Compatibility
⚠️ **Deprecated**: Use [New Component](../new-component.md) instead
- Last supported version: 1.9
- Removed in version: 2.0
- Migration: [See guide](../migration.md)
```

### If New ✨
```markdown
## Backward Compatibility
✨ **New Component**: Introduced in v2.0
- No prior version exists
- No migration needed
```

---

## Link Format

### Internal Links
```markdown
[Component Name](../api/inference/orchestrator.md)
[Data Contract](../reference/inference-data-contracts.md)
[Architecture](../architecture/inference-overview.md)
```

**Never**:
- Use backticks for file names: `orchestrator.md` ❌
- Use absolute paths
- Use relative paths beyond `../`

### Code References
```markdown
`ocr/inference/orchestrator.py`
`InferenceOrchestrator.predict()`
`InferenceMetadata`
```

---

## Validation Checklist

Before submission, verify:

- [ ] Filename lowercase with hyphens
- [ ] Frontmatter complete (type, status, version, last_updated)
- [ ] Purpose is 1 sentence
- [ ] No prose explanations (only facts)
- [ ] All constraints listed
- [ ] Backward compatibility explicitly stated
- [ ] No "how to" or tutorial content
- [ ] Tables used for data (not prose)
- [ ] Internal links relative paths only
- [ ] No backticks for filenames/paths
- [ ] Section order: Purpose → Interface → Dependencies → State → Constraints → Compatibility

---

## Examples of Compliant Sections

### ✅ Correct: Concise, Objective
```markdown
## Constraints
- Requires model loaded before inference
- Single concurrent inference per instance (not thread-safe)
- GPU/CPU device set at initialization
```

### ❌ Incorrect: Explanatory
```markdown
## Constraints
The orchestrator requires that you load the model before calling predict
because it doesn't check if the model is loaded internally. It also can't
handle multiple concurrent inferences, which is why we made it stateful
to maintain component references across calls.
```

### ✅ Correct: Tabular Data
```markdown
| Method | Parameters | Returns |
|--------|-----------|---------|
| predict | image, settings | InferenceResult |
```

### ❌ Incorrect: Prose Data
```markdown
The predict method takes an image and optional settings, and returns
an InferenceResult object containing predictions and preview.
```

---

## File Status Tracking

**Use in frontmatter**:
- `status: current` - Active, maintained
- `status: deprecated` - Replace with X, removal date Y

**Update frequency**: When code changes or version bumps

---

*Conventions effective: 2025-12-15*
*All new documentation must comply with these rules*
*Exceptions require documentation lead approval*
