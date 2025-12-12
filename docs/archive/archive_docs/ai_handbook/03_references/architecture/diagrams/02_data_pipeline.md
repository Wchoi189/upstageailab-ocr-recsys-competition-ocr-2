# Data Processing Pipeline Diagrams

<!-- ai_cue:diagram=data_pipeline -->
<!-- ai_cue:priority=high -->
<!-- ai_cue:use_when=preprocessing,data-loading,collate-functions,transforms -->

## Geometric Preprocessing Pipeline

This diagram shows the complete geometric preprocessing pipeline that transforms raw input images into properly oriented, flattened documents ready for OCR processing.

```mermaid
graph LR
    A[Raw Image] --> B[Load Image]
    B --> C[Geometric Preprocessing]

    subgraph "Transform Pipeline"
        C --> D[2 Transforms]
        D --> E[Warping]
        E --> F[LensStylePreprocessorAlbumentations]
        D --> F[LensStylePreprocessorAlbumentations]
    end

    Z --> F[DB Collate Function]
    F --> G[Batch Assembly]
    G --> H[DataLoader]
    H --> I[Lightning Module]

    subgraph "Data Contracts"
        J[DatasetSample] --> K[(H,W,3) uint8]
        J --> L[List of polygons]
        J --> M[Metadata dict]
    end

    I --> N[Training Loop]
```

## Transform Operations Chain

Detailed view of individual transform operations and their data flow validation.

```mermaid
graph LR
    A[Raw Image] --> B[Load Image]
    B --> C[Geometric Preprocessing]

    subgraph "Transform Pipeline"
        C --> D[2 Transforms]
        D --> E[Warping]
        E --> F[LensStylePreprocessorAlbumentations]
        D --> F[LensStylePreprocessorAlbumentations]
    end

    Z --> F[DB Collate Function]
    F --> G[Batch Assembly]
    G --> H[DataLoader]
    H --> I[Lightning Module]

    subgraph "Data Contracts"
        J[DatasetSample] --> K[(H,W,3) uint8]
        J --> L[List of polygons]
        J --> M[Metadata dict]
    end

    I --> N[Training Loop]
```

## DB Collate Function Flow

This diagram illustrates how the DB (Differentiable Binarization) collate function processes polygon annotations and generates training targets for the detection model.

```mermaid
graph LR
    A[Raw Image] --> B[Load Image]
    B --> C[Geometric Preprocessing]

    subgraph "Transform Pipeline"
        C --> D[2 Transforms]
        D --> E[Warping]
        E --> F[LensStylePreprocessorAlbumentations]
        D --> F[LensStylePreprocessorAlbumentations]
    end

    Z --> F[DB Collate Function]
    F --> G[Batch Assembly]
    G --> H[DataLoader]
    H --> I[Lightning Module]

    subgraph "Data Contracts"
        J[DatasetSample] --> K[(H,W,3) uint8]
        J --> L[List of polygons]
        J --> M[Metadata dict]
    end

    I --> N[Training Loop]
```

## Data Loading Pipeline

Complete data flow from disk to GPU training batch, showing the full pipeline integration.

```mermaid
graph LR
    A[Raw Image] --> B[Load Image]
    B --> C[Geometric Preprocessing]

    subgraph "Transform Pipeline"
        C --> D[2 Transforms]
        D --> E[Warping]
        E --> F[LensStylePreprocessorAlbumentations]
        D --> F[LensStylePreprocessorAlbumentations]
    end

    Z --> F[DB Collate Function]
    F --> G[Batch Assembly]
    G --> H[DataLoader]
    H --> I[Lightning Module]

    subgraph "Data Contracts"
        J[DatasetSample] --> K[(H,W,3) uint8]
        J --> L[List of polygons]
        J --> M[Metadata dict]
    end

    I --> N[Training Loop]
```

## Key Data Contracts

### **Input Contracts**
- **Images**: `(H, W, 3)` numpy arrays, uint8, RGB
- **Polygons**: List of `(N, 2)` numpy arrays (x, y coordinates)
- **Annotations**: ICDAR format with text content and polygon boundaries

### **Output Contracts**
- **Probability Maps**: `(batch_size, 1, H, W)` float32 tensors ∈ [0, 1]
- **Threshold Maps**: `(batch_size, 1, H, W)` float32 tensors ∈ [0, 1]
- **Text Polygons**: List of validated `(N, 2)` polygons per batch item

### **Validation Rules**
- Polygon area > 0 (no degenerate polygons)
- Polygon coordinates within image bounds
- Consistent shrink ratio application
- Proper tensor shapes and dtypes

## Performance Characteristics

### **Geometric Pipeline**
- **Orientation Detection**: ~50-100ms per image
- **Perspective Correction**: ~20-50ms per image
- **Document Flattening**: ~100-200ms per image (complex cases)

### **DB Collate Function**
- **Polygon Processing**: ~5-10ms per batch
- **Map Generation**: ~10-20ms per batch
- **GPU Transfer**: ~1-5ms per batch

### **Bottlenecks**
- Document flattening for severely curved documents
- Large polygon counts (>100 polygons per image)
- High-resolution images requiring downsampling

## Data Contracts Visualization

This diagram shows the Pydantic data contracts that enforce type safety and shape validation throughout the data pipeline.

```mermaid
graph LR
    A[Raw Image] --> B[Load Image]
    B --> C[Geometric Preprocessing]

    subgraph "Transform Pipeline"
        C --> D[2 Transforms]
        D --> E[Warping]
        E --> F[LensStylePreprocessorAlbumentations]
        D --> F[LensStylePreprocessorAlbumentations]
    end

    Z --> F[DB Collate Function]
    F --> G[Batch Assembly]
    G --> H[DataLoader]
    H --> I[Lightning Module]

    subgraph "Data Contracts"
        J[DatasetSample] --> K[(H,W,3) uint8]
        J --> L[List of polygons]
        J --> M[Metadata dict]
    end

    I --> N[Training Loop]
```

### **Contract Enforcement Points**
- **Image Loading**: Input validation before preprocessing
- **Transform Chain**: Result validation after each step
- **Dataset Creation**: Sample validation before collation
- **Batch Formation**: Final validation before model input

### **Error Handling**
- **Fallback Mechanisms**: Graceful degradation on validation failures
- **Logging**: Detailed error reporting for debugging
- **Recovery**: Alternative processing paths for edge cases

## Usage Examples

### **Geometric Pipeline Configuration**
```python
# In preprocessing config
geometric_pipeline:
  enable_orientation: true
  enable_perspective: true
  enable_flattening: true
  confidence_threshold: 0.8
  max_correction_angle: 45
```

### **DB Collate Configuration**
```python
# In dataloader config
collate_fn:
  _target_: ocr.datasets.db_collate_fn.DBCollateFN
  shrink_ratio: 0.4
  thresh_min: 0.3
  thresh_max: 0.7
```

## Related References

- **Code**: `ocr/datasets/preprocessing/` - Geometric operations implementation
- **Code**: `ocr/datasets/db_collate_fn.py` - DB collate function
- **Docs**: `docs/pipeline/data_contracts.md` - Data format specifications
- **Config**: `configs/data/` - Data loading configurations

---

*Generated: 2025-10-19 | Auto-update when: Data pipeline changes, new preprocessing steps*
