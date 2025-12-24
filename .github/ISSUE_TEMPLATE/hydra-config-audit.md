---
name: Hydra Configuration Audit
about: Audit the Hydra configuration system to identify legacy configs and review for removal
title: '[AUDIT] Hydra Configuration System - Legacy Config Identification'
labels: ['audit', 'config', 'technical-debt']
assignees: ''
---

## Audit Request

Please audit the Hydra configuration system to identify legacy configurations and review them for removal.

## Context

The project has undergone Hydra configuration refactoring and now uses `configs/_base/` as the foundation. However, legacy configurations still exist alongside the new architecture, creating confusion about:
- Which configurations are active
- Which override patterns require `+` prefix
- Which configs can be safely removed

## Audit Objectives

1. **Identify Legacy vs New Architecture**
   - Classify each config as new/legacy/hybrid
   - Map dependencies and relationships

2. **Analyze Override Patterns**
   - Document which configs require `+` prefix
   - Document which can be overridden without `+`
   - Identify override rules

3. **Assess Removal Impact**
   - Find all references to each config
   - Assess impact of removal
   - Document lost configuration options

4. **Propose Migration Strategy**
   - Recommend removal vs migration
   - Suggest `__LEGACY__/` folder structure
   - Provide migration path

## Considerations

- ⚠️ Purging configs could lose configuration options/usage references
- ⚠️ Legacy and new architectures need clear separation
- 💡 Consider moving legacy configs to `__LEGACY__/` folder for containerization

## Reference

See `HYDRA_CONFIG_AUDIT_PROMPT.md` for complete audit instructions.

## Expected Deliverables

1. Configuration inventory (new/legacy classification)
2. Override pattern guide
3. Legacy config report with recommendations
4. Migration plan with phases

---

**Tag**: `@claude` or `@github-copilot` to begin the audit.
