---
type: standard
category: framework
tier: 2
version: "1.0"
ads_version: "1.0"
status: active
created: 2026-01-25 21:00 (KST)
updated: 2026-01-25 21:00 (KST)
---

# Anti-Patterns Catalog

## Purpose

Documents prohibited patterns identified through audits and refactoring efforts. Use this as a reference during code reviews and when writing new code.

## Critical Anti-Patterns (Block PR)

### AP-001: Model-Level Configuration

**Description:** Models should not create optimizers, schedulers, or handle configuration logic.

**Why It's Bad:**
- Violates single responsibility principle
- Makes testing difficult
- Duplicates configuration logic
- Prevents centralized optimizer management

**Bad Example:**
```python
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def get_optimizers(self):
        """❌ ANTI-PATTERN: Model creating optimizer"""
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return [optimizer], []
    
    def _get_optimizers_impl(self):
        """❌ ANTI-PATTERN: Config detection in model"""
        if "optimizer" in self.config:
            opt_cfg = self.config.optimizer
        elif "train" in self.config:
            opt_cfg = self.config.train.optimizer
        # ... more detection logic
        return optimizer, scheduler
```

**Good Example:**
```python
class MyModel(nn.Module):
    """✅ Model is optimizer-agnostic"""
    def __init__(self, config):
        super().__init__()
        # Only model architecture
        self.encoder = build_encoder(config.encoder)
        self.decoder = build_decoder(config.decoder)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Lightning module handles optimizer
class MyLightningModule(pl.LightningModule):
    def configure_optimizers(self):
        """✅ Centralized optimizer configuration"""
        return instantiate(
            self.config.train.optimizer,
            params=self.model.parameters()
        )
```

**Enforcement:**
- Pre-commit hook: Detect `get_optimizers` in models
- Code review: Flag optimizer creation in `__init__`
- Linter rule: `model-optimizer-separation`

**Related:**
- [V5 Architecture Patterns](./v5-architecture-patterns.yaml)
- [Legacy Purge Audit](../../docs/artifacts/audits/2026-01-25_2100_audit_legacy-purge.md)

---

### AP-002: Silent Fallback Chains

**Description:** Exception handlers that silently return defaults or fall back through multiple alternatives.

**Why It's Bad:**
- Masks real errors
- Makes debugging impossible
- Hides configuration issues
- Users don't know what's actually being used

**Bad Example:**
```python
def configure_optimizers(self):
    """❌ ANTI-PATTERN: Silent fallbacks"""
    try:
        # Try path 1
        if hasattr(self.config, "train"):
            opt_cfg = self.config.train.optimizer
    except:
        pass
    
    try:
        # Try path 2
        if hasattr(self.config, "model"):
            opt_cfg = self.config.model.optimizer
    except:
        pass
    
    try:
        # Try legacy method
        return self.model.get_optimizers()
    except:
        pass
    
    # Silent fallback - user has no idea this happened!
    return torch.optim.Adam(self.model.parameters(), lr=0.001)
```

**Good Example:**
```python
def configure_optimizers(self):
    """✅ Fail-fast with helpful message"""
    if not hasattr(self.config, "train") or not hasattr(self.config.train, "optimizer"):
        raise ValueError(
            "V5 Hydra config missing: config.train.optimizer is required.\n"
            "Legacy model.get_optimizers() is no longer supported.\n"
            "See configs/train/optimizer/adam.yaml for template."
        )
    
    return instantiate(
        self.config.train.optimizer,
        params=self.model.parameters()
    )
```

**Enforcement:**
- Linter rule: `no-bare-except`
- Code review: Flag `except: pass` and `except Exception:`
- Max fallback levels: 2 (direct + 1 alternative)

**Exceptions:**
- `except KeyboardInterrupt`: Always allowed
- Logging cleanup: `except: log.debug("cleanup failed")`
- Context managers with explicit logging

---

### AP-003: Multiple Configuration Paths

**Description:** Supporting multiple locations for the same configuration value.

**Why It's Bad:**
- Increases cognitive load
- Makes debugging confusing ("which path is being used?")
- Prevents standardization
- Leads to inconsistent behavior

**Bad Example:**
```python
# ❌ ANTI-PATTERN: Multiple paths for optimizer
opt_cfg = None
if hasattr(config, "train") and hasattr(config.train, "optimizer"):
    opt_cfg = config.train.optimizer
elif hasattr(config, "model") and hasattr(config.model, "optimizer"):
    opt_cfg = config.model.optimizer
elif hasattr(config, "optimizer"):
    opt_cfg = config.optimizer
```

**Good Example:**
```python
# ✅ Single standard path
if not hasattr(config, "train") or not hasattr(config.train, "optimizer"):
    raise ValueError("config.train.optimizer is required")

opt_cfg = config.train.optimizer
```

**Standard Paths (V5):**
- Optimizer: `config.train.optimizer` ONLY
- Scheduler: `config.train.scheduler` ONLY
- Model: `config.model.architectures` ONLY
- Data: `config.data.datasets` ONLY

**Enforcement:**
- Document ONE standard path per config type
- Reject PRs adding alternative paths
- Migration guide for deprecated paths

---

### AP-004: Overly Permissive Checkpoint Loading

**Description:** Using `strict=False` or excessive fallback chains when loading checkpoints.

**Why It's Bad:**
- Silently ignores missing parameters
- Hides architecture mismatches
- Makes version tracking impossible
- Can lead to incorrect model behavior

**Bad Example:**
```python
# ❌ ANTI-PATTERN: Always permissive
model.load_state_dict(checkpoint, strict=False)

# ❌ ANTI-PATTERN: Too many fallbacks
try:
    model.load_state_dict(checkpoint, strict=True)
except:
    try:
        # Remove "model." prefix
        new_checkpoint = remove_prefix(checkpoint, "model.")
        model.load_state_dict(new_checkpoint, strict=True)
    except:
        try:
            # Remove "_orig_mod." prefix
            new_checkpoint = remove_orig_mod(checkpoint)
            model.load_state_dict(new_checkpoint, strict=True)
        except:
            # Give up, load permissively
            model.load_state_dict(checkpoint, strict=False)
```

**Good Example:**
```python
# ✅ Fail-fast with 2-level max
try:
    return model.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    # Only handle torch.compile prefix
    if "_orig_mod" not in str(e):
        raise RuntimeError(
            f"Checkpoint incompatible with model architecture.\n"
            f"Original error: {e}\n"
            f"For legacy checkpoints, use: scripts/checkpoints/convert.py"
        ) from e
    
    # Single fallback for torch.compile
    modified = {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}
    return model.load_state_dict(modified, strict=True)
```

**Rules:**
- Max 2 fallback levels (direct + 1 alternative)
- Always use `strict=True` in production
- `strict=False` only in migration scripts
- Explicit error messages with resolution path

---

## High Severity Anti-Patterns (Should Fix)

### AP-005: Magic Numbers

**Description:** Hardcoded numeric values without named constants.

**Bad Example:**
```python
# ❌ What do these numbers mean?
image = resize(image, 640, 640)
if confidence > 0.7:
    predictions = filter_boxes(boxes, 1024)
```

**Good Example:**
```python
# ✅ Named constants
from ocr.core.constants.detection import (
    DETECTION_IMAGE_SIZE,
    CONFIDENCE_THRESHOLD,
    MAX_BOXES
)

image = resize(image, DETECTION_IMAGE_SIZE, DETECTION_IMAGE_SIZE)
if confidence > CONFIDENCE_THRESHOLD:
    predictions = filter_boxes(boxes, MAX_BOXES)
```

**Exceptions:**
- Mathematical constants: `0, 1, 2, -1`
- Array indices: `arr[0]`, `shape[1]`
- Common fractions: `0.5`, `0.25` (but consider named)

---

### AP-006: Domain-Specific Code in Shared Utils

**Description:** Utilities that contain domain-specific logic in shared locations.

**Bad Example:**
```python
# ❌ ocr/core/utils/text_processing.py
def process_text(text, domain="recognition"):
    if domain == "recognition":
        return tokenize_text(text)
    elif domain == "kie":
        return extract_entities(text)
    # Domain-specific branches in shared code!
```

**Good Example:**
```python
# ✅ ocr/domains/recognition/utils/text_processing.py
def tokenize_recognition_text(text):
    """Recognition-specific tokenization"""
    return tokens

# ✅ ocr/domains/kie/utils/text_processing.py
def extract_kie_entities(text):
    """KIE-specific entity extraction"""
    return entities
```

---

### AP-007: God Classes/Functions

**Description:** Single class or function doing too much.

**Thresholds:**
- Function > 50 lines (should be < 30)
- File > 500 lines (should be < 300)
- Cyclomatic complexity > 10
- Parameters > 5

**Solution:**
- Extract methods
- Create helper classes
- Use composition
- Apply single responsibility principle

---

## Medium Severity Anti-Patterns (Nice to Fix)

### AP-008: Commented-Out Code

**Description:** Large blocks of commented code left in repository.

**Why It's Bad:**
- Clutters codebase
- Confuses readers ("should I use this?")
- Git history already preserves old code

**Rule:** Delete commented code. Use git history if needed.

**Exceptions:**
- Temporary debugging (max 1 week)
- Alternative implementations being evaluated (document why)

---

### AP-009: Duplicate Code

**Description:** Copy-pasted code instead of extraction.

**Threshold:** > 10 lines similar across 2+ locations

**Tools:**
```bash
# Detect duplication
uv run pylint ocr/ --disable=all --enable=duplicate-code
```

---

### AP-010: Unclear Naming

**Description:** Variable/function names that don't convey intent.

**Bad:**
```python
def proc(d):  # What is this processing?
    tmp = []  # Temporary what?
    for x in d:  # What is x?
        tmp.append(x * 2)
    return tmp
```

**Good:**
```python
def scale_confidence_scores(predictions: list[Prediction]) -> list[float]:
    scaled_scores = []
    for prediction in predictions:
        scaled_scores.append(prediction.confidence * 2)
    return scaled_scores
```

---

## Prevention Measures

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: check-anti-patterns
        name: Check for anti-patterns
        entry: python AgentQMS/tools/check_anti_patterns.py
        language: python
        pass_filenames: true
        types: [python]
```

### Linter Rules

```toml
# pyproject.toml
[tool.pylint.messages_control]
enable = [
    "bare-except",              # AP-002
    "broad-except",             # AP-002
    "duplicate-code",           # AP-009
    "too-many-lines",           # AP-007
    "too-many-arguments",       # AP-007
]

[tool.pylint.design]
max-line-length = 120
max-args = 5                    # AP-007
max-locals = 15
max-branches = 12
max-statements = 50
```

### Code Review Checklist

- [ ] No model-level configuration (AP-001)
- [ ] No silent fallbacks (AP-002)
- [ ] Single configuration path (AP-003)
- [ ] Strict checkpoint loading (AP-004)
- [ ] No magic numbers (AP-005)
- [ ] Domain separation maintained (AP-006)
- [ ] Functions < 50 lines (AP-007)
- [ ] No commented code (AP-008)
- [ ] No duplication (AP-009)
- [ ] Clear naming (AP-010)

---

## Related Standards

- [V5 Architecture Patterns](./v5-architecture-patterns.yaml)
- [Naming Conventions](../tier1-sst/naming-conventions.yaml)
- [File Placement Rules](../tier1-sst/file-placement-rules.yaml)
- [Bloat Detection Rules](../tier2-framework/coding/bloat-detection-rules.yaml)

---

**Audit History:**
- 2026-01-25: Initial catalog from [Legacy Purge Audit](../../docs/artifacts/audits/2026-01-25_2100_audit_legacy-purge.md)

**Maintenance:**
- Review quarterly
- Update after major refactors
- Add patterns from code reviews
- Remove patterns no longer relevant
