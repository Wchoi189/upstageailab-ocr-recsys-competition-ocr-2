# Miscellaneous Specification

**Tier**: 2 (Framework)
**Scope**: Debugging Protocols and ML Framework Choices.

## 1. Debugging Protocol
1.  **Isolate**: Reproduce with minimal script (`scripts/repro_X.py`).
2.  **Log**: Enable `debug=True` in Hydra.
3.  **Trace**: Use `pdb` or VSCode Debugger. **Start simple**.

### Common Failures
*   **CUDA OOM**: Check batch size, look for zombie tensors.
*   **Hydra Key Error**: Check `@package` directives (Flattening Rule).

## 2. ML Frameworks
*   **Core**: PyTorch 2.x
*   **Training**: PyTorch Lightning
*   **Config**: Hydra 1.3
*   **Serving**: FastAPI
*   **Vision**: OpenCV (Pre/Post), Albumentations (Augmentation)
*   **Dev**: standard `black`, `isort`, `mypy`.
