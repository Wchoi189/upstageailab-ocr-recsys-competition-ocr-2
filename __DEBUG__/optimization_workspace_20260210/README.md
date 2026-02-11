# Optimization Workspace (Feb 10, 2026)

## 🚨 Critical Context for Next Agent
1.  **Status:** The model is at 82.6% accuracy. Training is stable but slow (~6 it/s).
2.  **The "Slow" Problem:** The previous runs **failed to load** the optimized `rtx3090.yaml` config. The system defaulted to `num_workers=2`.
3.  **The "Plateau" Problem:** No Learning Rate Scheduler was active.
4.  **Constraint:** **Do NOT run >10 epochs at a time.** The user requires iterative updates. 100 epochs is the *cumulative* goal, not the single-run command.

## 🛠️ Directives

### 1. Enable the Optimizations
Use the provided `rec_optimized_rtx3090.yaml` config. It explicitly loads:
- `hardware/rtx3090` (Workers=12, Batch=160)
- `train/scheduler/cosine` (LR Decay)

### 2. Run Iteratively (Split Training)
Execute 10 epochs, check stability/speed, then resume.
**Goal Speed:** >10 it/s (vs current 6 it/s).

### 3. Usage
**Start New Optimization Run (Epoch 0-10):**
```bash
./run_optimization.sh
```

**Resume (Epoch 10-20):**
Edit `run_optimization.sh` to set `CKPT_PATH` to the new `best-acc` checkpoint.

## 📂 Files
- `rec_optimized_rtx3090.yaml`: The corrected configuration.
- `run_optimization.sh`: Script to launch the 10-epoch chunk.
