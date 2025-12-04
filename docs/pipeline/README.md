# Pipeline Data Contracts

**Purpose**: Centralized data contract definitions for OCR pipeline.

## Organization

```
docs/pipeline/
├── README.md                    # This file - index and cross-references
├── data_contracts.md            # Core pipeline contracts (dataset, transforms, model)
├── inference-data-contracts.md  # Inference-specific contracts (NEW)
└── preprocessing-data-contracts.md # Preprocessing module contracts
```

## Contract Index

### Core Pipeline
- **Dataset Output**: `data_contracts.md#dataset-output-contract`
- **Transform Pipeline**: `data_contracts.md#transform-pipeline-contract`
- **Model Input/Output**: `data_contracts.md#model-inputoutput-contract`
- **Loss Function**: `data_contracts.md#loss-function-contract`

### Inference Pipeline
- **InferenceMetadata**: `inference-data-contracts.md#inferencemetadata`
- **Coordinate Transformation**: `inference-data-contracts.md#coordinate-transformation`
- **Polygon Coordinates**: `inference-data-contracts.md#polygon-coordinate-space`

### Preprocessing
- **ImageInputContract**: `preprocessing-data-contracts.md#imageinputcontract`
- **PreprocessingResultContract**: `preprocessing-data-contracts.md#preprocessingresultcontract`

## Cross-References

Use explicit file paths for cross-references:
- `[InferenceMetadata](inference-data-contracts.md#inferencemetadata)`
- `[Coordinate Transformation](data_contracts.md#critical-areas---do-not-modify-without-tests)`

## Contract Updates

When updating contracts:
1. Update contract definition file
2. Update this README if structure changes
3. Update cross-references in related files
4. Update implementation code
5. Update tests
