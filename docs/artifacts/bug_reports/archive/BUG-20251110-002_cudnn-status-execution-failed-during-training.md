---
title: "Bug 20251110 002 Cudnn Status Execution Failed During Training"
date: "2025-12-06 18:08 (KST)"
type: "bug_report"
category: "troubleshooting"
status: "active"
version: "1.0"
tags: ['bug_report', 'troubleshooting']
---





## Summary
Training crashes with `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED`, followed by `torch.AcceleratorError: CUDA error: an illegal instruction was encountered` when the trainer attempts to tear down optimizers.

## Environment
- GPU: NVIDIA GeForce RTX 3060 (driver 581.57, CUDA 13.0 via `nvidia-smi`)
- PyTorch: 2.8.0+cu128 (`torch.version.cuda=12.8`)
- Command:
  - `python runners/train.py model.encoder.model_name=resnet18 dataloaders.train_dataloader.batch_size=2 trainer.devices=1 trainer.strategy=auto trainer.max_steps=10`
- Configs:
  - `paths: default`
  - `callbacks: metadata`
  - W&B logging disabled

## Steps to Reproduce
1. Ensure repository environment is activated (same as prior bug reproduction).
2. Execute:
   - `python runners/train.py model.encoder.model_name=resnet18 dataloaders.train_dataloader.batch_size=2 trainer.devices=1 trainer.strategy=auto trainer.max_steps=10`
3. Observe failure around step 10 in epoch 0. A `.FAILURE` sentinel is written to `outputs/ocr_training_b/checkpoints/.FAILURE`.

## Expected Behavior
Training should complete the configured steps, writing checkpoints and metadata without GPU runtime failures.

## Actual Behavior
- Training progresses for a few iterations then fails with cuDNN execution error inside the FPN decoder (`torch.nn.functional.conv2d`).
- Lightning teardown triggers a secondary `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`.
- Run terminates and writes a failure sentinel.

## Logs / Stack Trace
```
RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
  File ".../ocr/models/decoder/fpn_decoder.py", line 68, in forward
    return self.fusion(fused)
  ...
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
  File ".../lightning/fabric/utilities/optimizer.py", line 104, in batch_to
    data_output = data.to(device, **kwargs)
```

## Root Cause Analysis
- cuDNN failure suggests either invalid tensors (e.g., NaNs/Infs) or unsupported convolution parameters sent to `conv2d` during training. The follow-up illegal instruction indicates GPU context corruption.
- Likely triggered by unstable activations/weights after Dice loss clamp change; needs validation of tensor contents before convolution, or safeguarding decoder inputs.

## Resolution
- Add runtime checks around `FPNDecoder.fusion` inputs (verify finite tensors before `conv2d`).
- Optionally enable anomaly detection (`torch.autograd.set_detect_anomaly(True)`) for debugging.
- Investigate data pipeline and loss outputs to ensure no NaNs/Infs propagate.
- Consider upgrading/downgrading cuDNN or forcing deterministic algorithms as temporary workaround (`torch.backends.cudnn.deterministic = True`).

## Testing
1. Apply proposed checks/fixes.
2. Re-run reproduction command to confirm training completes 10 steps without cuDNN errors.
3. Monitor GPU logs for repeated failures.

## Prevention
- Add GPU runtime health checks in training loop (Fail fast with informative message if tensors contain NaNs).
- Include automated test harness that runs a short training smoke test on GPU hardware to catch cuDNN regressions early.
