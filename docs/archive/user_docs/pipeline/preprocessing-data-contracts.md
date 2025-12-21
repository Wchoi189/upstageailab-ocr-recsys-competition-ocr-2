# Preprocessing Module Data Contracts

**Purpose**: Data contracts for document preprocessing pipeline.

## Core Contracts

### ImageInputContract
```python
image: np.ndarray  # Shape: (H, W, C), dtype: uint8, C ∈ [1,2,3,4]
```
**Validation**: Numpy array, 2-3 dimensions, 1-4 channels, non-empty

### PreprocessingResultContract
```python
image: np.ndarray
metadata: dict[str, Any]
```
**Validation**: Image numpy array, metadata dictionary

### DetectionResultContract
```python
corners: np.ndarray | None      # Shape: (4, 2)
confidence: float | None         # Range: [0.0, 1.0]
method: str | None
```
**Validation**: Corners valid array if provided, confidence in range if provided

## Validation Decorators

- `@validate_image_input_with_fallback` - Validates inputs, fallback on failure
- `@validate_preprocessing_result_with_fallback` - Validates results

## Contract Enforcement

**ContractEnforcer** utilities:
- `validate_image_input_contract()`
- `validate_preprocessing_result_contract()`
- `validate_detection_result_contract()`

## Testing

See `tests/unit/test_preprocessing_contracts.py`

## Related Contracts

- [Pipeline Data Contracts](data_contracts.md#preprocessing-pipeline-contract)
- [Inference Data Contracts](inference-data-contracts.md)
