---
type: index
component: pipeline
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# Pipeline Data Contracts Index

**Purpose**: Centralized index for OCR pipeline data contracts; navigation to dataset, transform, model, inference, preprocessing contracts.

---

## Organization

```
docs/pipeline/
├── README.md                       # This file
├── data-contracts.md               # Core pipeline (dataset, transforms, model, loss)
├── data-contract-organization.md   # Organization standards
├── inference-data-contracts.md     # Redirect to canonical reference
├── preprocessing-data-contracts.md # Preprocessing module contracts
└── training-refactoring-summary.md # Training refactoring summary
```

---

## Contract Index

### Core Training Pipeline
- [Dataset Output Contract](data-contracts.md#dataset-sample-contract) - `DatasetSample` (Pydantic V2)
- [Transform Output Contract](data-contracts.md#transform-output-contract) - `TransformOutput` (Pydantic V2)
- [Batch Output Contract](data-contracts.md#batch-output-contract-collate) - `CollateOutput` (Pydantic V2)
- [Model Input/Output Contract](data-contracts.md#model-inputoutput-contract) - Tensor dict
- [Loss Function Contract](data-contracts.md#loss-function-contract) - Scalar tensor

### Inference Pipeline
- [Inference Data Contracts](inference-data-contracts.md) - Redirect to [canonical reference](../reference/inference-data-contracts.md)
- [InferenceMetadata](../reference/inference-data-contracts.md#inferencemetadata) - Coordinate transformation, padding
- [PreprocessingResult](../reference/inference-data-contracts.md#preprocessingresult) - Preprocessing output
- [PostprocessingResult](../reference/inference-data-contracts.md#postprocessingresult) - Postprocessing output

### Preprocessing
- [Preprocessing Data Contracts](preprocessing-data-contracts.md) - Preprocessing module contracts

### Organization
- [Data Contract Organization](data-contract-organization.md) - Organization standards, templates, update process

### Summaries
- [Training Refactoring Summary](training-refactoring-summary.md) - Lazy import optimization (26x speedup)

---

## Cross-Reference Format

**Within pipeline directory**:
```markdown
[Dataset Output Contract](data-contracts.md#dataset-sample-contract)
```

**To other directories**:
```markdown
[InferenceMetadata](../reference/inference-data-contracts.md#inferencemetadata)
```

---

## Contract Update Process

| Step | Action |
|------|--------|
| 1 | Update contract definition file |
| 2 | Update this README if structure changes |
| 3 | Update cross-references in related files |
| 4 | Update implementation code |
| 5 | Update tests |

---

## Validation Models (Pydantic V2)

| Model | Purpose | File |
|-------|---------|------|
| `DatasetSample` | Dataset output validation | [data-contracts.md](data-contracts.md#dataset-sample-contract) |
| `TransformOutput` | Transform pipeline validation | [data-contracts.md](data-contracts.md#transform-output-contract) |
| `CollateOutput` | Batch output validation | [data-contracts.md](data-contracts.md#batch-output-contract-collate) |
| `ValidatedPolygonData` | Polygon bounds validation | [data-contracts.md](data-contracts.md#polygon-validation) |
| `ValidatedTensorData` | Tensor shape/value validation | [data-contracts.md](data-contracts.md#tensor-validation) |
| `PreprocessingResultContract` | Preprocessing output validation | [preprocessing-data-contracts.md](preprocessing-data-contracts.md) |
| `InferenceMetadata` | Inference metadata validation | [../reference/inference-data-contracts.md](../reference/inference-data-contracts.md) |

---

## References

- [Data Contracts](data-contracts.md)
- [Data Contract Organization](data-contract-organization.md)
- [Inference Data Contracts](../reference/inference-data-contracts.md) (canonical)
- [Preprocessing Data Contracts](preprocessing-data-contracts.md)
- [Training Refactoring Summary](training-refactoring-summary.md)
