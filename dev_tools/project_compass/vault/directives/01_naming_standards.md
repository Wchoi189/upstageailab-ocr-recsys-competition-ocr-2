# [Compass:Lock] Core Directive 01: Naming Standards

## Pulse ID Format

**Pattern**: `{domain}-{action}-{target}`

**Regex**: `^[a-z][a-z0-9]*(-[a-z][a-z0-9]*){1,4}$`

**Examples**:
- ✅ `recognition-optimize-vocab`
- ✅ `detection-fix-cuda-mismatch`
- ✅ `hydra-refactor-domains`
- ❌ `new_session_01`
- ❌ `test-123`
- ❌ `session`

## Banned Terms

These words are BLOCKED in pulse IDs:
- `new`, `session`, `test`, `tmp`, `untitled`, `default`

## Artifact Naming

**Pattern**: `{milestone_id}-{type}-{description}.md`

**Examples**:
- `rec-opt-research-vocab-size.md`
- `det-scale-design-batch-tuning.md`
- `hydra-v5-walkthrough-migration.md`

## Milestone ID Format

**Pattern**: `{domain_abbrev}-{short_name}`

**Examples**:
- `rec-opt` (recognition optimization)
- `det-scale` (detection scaling)
- `kie-align` (KIE alignment)
