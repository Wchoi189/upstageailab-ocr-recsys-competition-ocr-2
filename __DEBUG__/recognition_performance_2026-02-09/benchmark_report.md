# Recognition Pipeline Validation Report

## 1. Performance Benchmarks (RTX 3090)

Testing conducted using `rec_baseline_official` experiment with varying dataloader configurations.

| Configuration | Workers | Pin Memory | Batch Size | Speed (it/s) | Est. Samples/Sec | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0 | False | 16 | 7.46 | ~119 | Stable |
| **MP (Spawn)** | 4 | True | 16 | 8.41 | ~135 | Stable (+13%) |
| **MP (Spawn)** | 8 | True | 16 | 8.61 | ~138 | Saturated |
| **Optimal** | **4** | **True** | **64** | **8.39** | **~537** | **Best (4.5x)** |
| **High Load** | 4 | True | 128 | N/A | N/A | **Unstable/Hung** |

**Recommendation:**
- Use `num_workers=4` with `pin_memory=True` (requires `spawn` method, already enabled).
- Use `batch_size=64` for optimal throughput on RTX 3090.

## 2. Validation Accuracy Issue (`val/acc: 0.000`)

**Findings:**
- Confirmed Tokenizer IDs mismatch is **NOT** the cause (BOS=1, EOS=2, PAD=0 are consistent).
- The low accuracy is expected behavior for an **untrained** model initialized with random weights.
- The model predicts EOS or random tokens efficiently, leading to `val/cer: 1.000` (100% error).
- Training loop performance (it/s) is **valid** because it processes fixed-length sequences regardless of prediction quality.

## 3. Next Steps
- Adopt recommended settings in `configs/experimental/rec_baseline_official.yaml` or as overrides.
- Proceed with full training run (recommend 100 epochs for convergence).
