# Hydra Configuration Architecture

**AI-Optimized Documentation**: [`AgentQMS/specs/tier2-framework/configuration.spec.md`](../AgentQMS/specs/tier2-framework/configuration.spec.md)

**Legacy Human-Readable Guide**: [`__LEGACY__/README_20260108_deprecated.md`](__LEGACY__/README_20260108_deprecated.md)

---

## Quick Reference (2026-02-14)
### WandB Recognition Image Logging
- Source of truth: Module-based logger in `ocr.domains.recognition.module.RecognitionPLModule._log_validation_images`.
- Required batch keys: `images` and `text_tokens` (preferred) or `label`.
- Required inference outputs: `tokens` (decoded via the dataset tokenizer).
- Legacy callback `RecognitionWandbImageLogger` is removed; do not add `recognition_wandb` to callbacks.
- Logged panel name: `val_recognition_samples`.
