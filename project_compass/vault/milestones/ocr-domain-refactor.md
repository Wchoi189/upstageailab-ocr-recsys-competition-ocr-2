# Milestone: OCR Domain Refactor

**ID**: `ocr-domain-refactor`
**Status**: `active`
**Architecture**: `Domains First`

## Objectives

1. **Eliminate Domain Leakage**: Strictly separate Detection, Recognition, KIE, and Layout domains.
2. **Architecture Decoupling**: Ensure `ocr/core` contains NO domain-specific keywords (polygon, bbox, tokenizer).
3. **Registry Purge**: Remove `ocr.core.registry` and use Hydra `_target_` instantiation.

## Tasks

### Phase 5: Surgical Audit 2 (The Data Purge)
- [ ] **Audit `ocr/data/datasets`**: Identify domain leakage (advanced_detector.py, etc).
- [ ] **Relocate Logic**: Move domain-specific logic to `ocr/domains/{domain}/data/`.
- [ ] **Refactor Init**: Modify `ocr/data/datasets/__init__.py` to prevent eager loading.
- [ ] **Lazy Validation**: Decouple validation from dataset initialization.

### Phase 6: The Registry Purge
- [ ] **Deprecate Registry**: Mark `ocr.core.registry` as deprecated.
- [ ] **Hydra Migration**: Convert all YAML configs to use `_target_`.
- [ ] **Cleanup**: Remove factory patterns and eager imports.
- [ ] **Verification**: Ensure 0 cross-domain imports at startup.

## Success Criteria

- `ocr/core` has zero imports of `ocr/domains`.
- Startup time < 2 seconds.
- All experiments run via `uv run compass ...`.
