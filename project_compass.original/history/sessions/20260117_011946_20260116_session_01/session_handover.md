# Session Handover: Recognition Strategy Pivot (SVTR)

## Context
We strictly pivoted away from **PARSeq (ViT + 224x224)** because it is computationally wasteful and architecturally ill-suited for the dataset (High-Definition Word Strips).
The new direction is **SVTR (Spatial Visual Transformer) + Rectangular Inputs (32x128)**.

## Current State
-   **Architecture**: Currently `resnet18` is configured as a temporary placeholder in [parseq.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/model/architectures/parseq.yaml).
-   **SVTR Components**:
    -   [ocr/features/recognition/models/merging.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/merging.py): **Implemented** (2D -> 1D Merging).
    -   `SVTRBackbone`: **Missing**.
-   **Performance**: Training is blocked by CPU resizing bottleneck (0.4it/s).

## Priority Action Items for Next Agent

### 1. Offline Data Preprocessing (Smart Resize)
**Goal**: Eliminate CPU bottleneck and square image waste.
**Task**: Create `scripts/preprocess/resize_dataset.py`.
**Specs**:
-   **Input**: Raw images from source LMDB.
-   **Transformation**:
    1.  Resize Height to `32` (preserving aspect ratio).
    2.  If Width > 128: Resize width to `128` (Squash) OR Center Crop (Decide based on crop statistics).
    3.  If Width < 128: Pad Right with 0 to width `128`.
-   **Output**: New LMDB `aihub_lmdb_validation_32_128`.

### 2. Implement SVTR Backbone
**Task**: Create `ocr/features/recognition/models/backbone/svtr.py`.
**Specs**:
-   3-Stage Pyramid (Downsample 1/2, 1/4, 1/8 or similar).
-   Mixing Blocks: Local (Win 7x7) + Global Attention.
-   Output: Feature Map `[B, 256, 1, 32]` (Height 1).

### 3. Training Command (Proposed)
```bash
uv run runners/train.py \
    experiment.name=svtr_pilot \
    model.architecture_name=svtr \
    data.transforms.train.resize=[32, 128] \
    batch_size=1024  # Small inputs allow huge batches
    trainer.precision=16-mixed
    num_workers=8
```

## Continuation Prompt
> **Role**: AI Optimization Engineer (SVTR Specialist).
> **Goal**: Execute the SVTR Migration.
> 1.  **Data**: Write the Offline Smart-Resize script (32x128) and run it. This is the prerequisite for speed.
> 2.  **Model**: Implement the `SVTRBackbone` class to pair with the existing [SVTRMerging](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/features/recognition/models/merging.py#3-35) layer.
> 3.  **Train**: Launch the first SVTR run targeting >1000 iter/s.
>
> **Constraint**: Do not go back to 224x224. Stick to 32x128 Rectangular inputs.
