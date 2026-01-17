# Hydra Configuration Architecture

**AI-Optimized Documentation**: [`AgentQMS/standards/tier2-framework/hydra-configuration-architecture.yaml`](../AgentQMS/standards/tier2-framework/hydra-configuration-architecture.yaml)

**Legacy Human-Readable Guide**: [`__LEGACY__/README_20260108_deprecated.md`](__LEGACY__/README_20260108_deprecated.md)

---

## Quick Reference

### Entry Points
- `train.yaml` - Universal training (domain switchable)
- `eval.yaml` - Testing/evaluation
- `predict.yaml` - Inference
- `synthetic.yaml` - Synthetic data generation

### Domain Switching
```bash
python runners/train.py domain=detection    # Default
python runners/train.py domain=recognition
python runners/train.py domain=kie
python runners/train.py domain=layout
```

### Structure
```
configs/
├── {train,eval,predict,synthetic}.yaml  # Entry points
├── _foundation/                          # Core composition fragments
├── domain/                               # Multi-domain configs
├── model/                                # Architecture components
├── data/                                 # Dataset configs
├── training/                             # Training infrastructure
├── __EXTENDED__/                         # Archived/experimental
└── __LEGACY__/                           # Deprecated (read-only)
```

### Override Rules
- **In base.yaml defaults**: `model=craft` (no `+`)
- **Not in defaults**: `+debug=default` (with `+`)
- **Direct overrides**: `++model.optimizer.lr=0.0001`

### Common Issues
**ConfigCompositionException**: Move all `override X: Y` to end of defaults list.

---

**Migration Date**: 2026-01-08
**Baseline**: 107 YAML files / 37 directories
**Current**: 112 YAML files / 41 directories (+5 consolidated domain/foundation configs)
