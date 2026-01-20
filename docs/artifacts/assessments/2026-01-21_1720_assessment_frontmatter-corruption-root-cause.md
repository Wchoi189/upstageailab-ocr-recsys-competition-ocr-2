---
ads_version: "1.0"
type: assessment
title: "Frontmatter Corruption Root Cause Analysis"
date: "2026-01-21 17:20 (KST)"
status: active
category: evaluation
tags: [bug-analysis, frontmatter, validation, schema]
---

# Frontmatter Corruption Root Cause Analysis

## Executive Summary

**Issue**: Artifacts contain non-standard frontmatter fields (`kst`, `scope`, `impact`, `confidence`, `priority`, `related_artifacts`) that are NOT defined in plugin schemas.

**Root Cause**: `create_frontmatter()` accepts arbitrary `**kwargs` and merges them into frontmatter without validating against plugin schema.

**Impact**: 6+ artifacts corrupted with custom fields, violating plugin-only architecture.

---

## The Problem

### Corrupt Frontmatter Example

```yaml
ads_version: "1.0"
type: assessment
title: "generate_ide_configs.py Analysis & Recommendation"
date: "2026-01-21 17:15 (KST)"
kst: "2026-01-21T17:15:00+09:00"      # ← NOT in plugin
status: active
priority: medium                       # ← NOT in plugin
category: architecture
tags: [bloat-audit, utilities, ide-configs, documentation]
scope: utility                         # ← NOT in plugin
impact: low                            # ← NOT in plugin
confidence: high                       # ← NOT in plugin
related_artifacts:                     # ← NOT in plugin (except audit.yaml)
  - 2026-01-21_1645_assessment_agentqms-nuclear-bloat-audit.md
```

### Plugin Schema (assessment.yaml)

**Defined fields ONLY**:
```yaml
metadata:
  frontmatter:
    ads_version: "1.0"
    type: assessment
    category: evaluation
    status: active
    tags: [assessment, evaluation, analysis]
```

**System-added fields**:
- `title` (user-provided)
- `date` (system-generated)
- `branch` (git branch)

**Total allowed fields**: 8 fields only

---

## Root Cause Code

### Location: `artifact_templates.py` lines 473-503

```python
def create_frontmatter(self, template_type: str, title: str, **kwargs) -> str:
    """Create frontmatter for an artifact."""
    template = self.get_template(template_type)
    if not template:
        raise ValueError(f"Unknown template type: {template_type}")

    frontmatter = template["frontmatter"].copy()
    frontmatter["title"] = title
    frontmatter["date"] = self._get_kst_timestamp_str()
    frontmatter.setdefault("ads_version", "1.0")

    if "branch" not in kwargs:
        frontmatter["branch"] = self._get_branch_name()

    # ❌ THE PROBLEM: Merges ANY kwargs without validation
    config = self._get_config()
    denylist = set(config["frontmatter_denylist"])
    for key, value in kwargs.items():
        if key not in denylist:
            frontmatter[key] = value  # ← Allows pollution!

    return self._format_frontmatter_yaml(frontmatter)
```

**The denylist only blocks**:
- `output_dir`
- `interactive`
- `steps_to_reproduce`
- `quiet`

**Everything else passes through**, including:
- `kst`
- `scope`
- `impact`
- `confidence`
- `priority`
- `related_artifacts`
- Any other random field

---

## How Corruption Happened

### Agent Actions (Me, Today)

When creating artifacts manually via `create_file`, I passed custom kwargs:

```python
create_file(
    filePath="docs/artifacts/assessments/2026-01-21_1715_assessment_generate-ide-configs-analysis.md",
    content="""---
ads_version: "1.0"
type: assessment
title: "generate_ide_configs.py Analysis & Recommendation"
date: "2026-01-21 17:15 (KST)"
kst: "2026-01-21T17:15:00+09:00"      # I added this!
status: active
priority: medium                       # I added this!
category: architecture
tags: [bloat-audit, utilities, ide-configs, documentation]
scope: utility                         # I added this!
impact: low                            # I added this!
confidence: high                       # I added this!
related_artifacts:                     # I added this!
```

**Why?** I was trying to add "metadata" fields for richer context, not realizing they violate the plugin schema.

---

## Affected Artifacts (6 files)

All created today (2026-01-21) with corrupt frontmatter:

1. `2026-01-21_1510_assessment_agentqms-comprehensive-audit-handover.md`
2. `2026-01-21_1550_assessment_agentqms-architectural-inventory.md`
3. `2026-01-21_1605_assessment_agentqms-phase2-documentation-audit.md`
4. `2026-01-21_1610_assessment_agentqms-comprehensive-audit-final-report.md`
5. `2026-01-21_1645_assessment_agentqms-nuclear-bloat-audit.md`
6. `2026-01-21_1715_assessment_generate-ide-configs-analysis.md`

**Common corrupt fields**:
- `kst` (redundant ISO timestamp)
- `priority` (high/medium/low/critical)
- `scope` (framework/utility/project)
- `impact` (high/medium/low/critical)
- `confidence` (high/medium/low)
- `related_artifacts` (list of related files)

---

## Why This Is Bad

### 1. Violates Plugin-Only Architecture

We just completed nuclear cleanup to enforce "plugin YAML files as single source of truth." Custom kwargs bypass this entirely.

### 2. No Validation

The system accepts literally ANY field name:
```python
frontmatter = create_frontmatter(
    "assessment",
    "My Title",
    my_random_field="foo",        # ← Accepted!
    another_random_field="bar",   # ← Accepted!
    hello_world=123,              # ← Accepted!
)
```

### 3. Schema Drift

Different artifacts have different fields, making parsing/indexing impossible:
- Some have `priority`, some don't
- Some have `related_artifacts`, some don't
- No consistency = no machine-readable metadata

### 4. Redundant Fields

- `kst` duplicates `date` (both are timestamps)
- `related_artifacts` exists in audit.yaml but NOT assessment.yaml

---

## The Fix

### Phase 1: Validate kwargs Against Plugin Schema

**Location**: `artifact_templates.py` `create_frontmatter()`

```python
def create_frontmatter(self, template_type: str, title: str, **kwargs) -> str:
    """Create frontmatter with strict plugin schema validation."""
    template = self.get_template(template_type)
    if not template:
        raise ValueError(f"Unknown template type: {template_type}")

    # Start with plugin-defined frontmatter
    frontmatter = template["frontmatter"].copy()

    # System-generated fields (always allowed)
    frontmatter["title"] = title
    frontmatter["date"] = self._get_kst_timestamp_str()
    frontmatter.setdefault("ads_version", "1.0")

    if "branch" not in kwargs:
        frontmatter["branch"] = self._get_branch_name()

    # Get allowed fields from plugin schema
    allowed_fields = set(template["frontmatter"].keys())
    allowed_fields.update(["title", "date", "branch", "ads_version"])  # System fields

    # Validate kwargs against schema
    config = self._get_config()
    denylist = set(config["frontmatter_denylist"])

    for key, value in kwargs.items():
        if key in denylist:
            continue  # Skip denylist

        if key not in allowed_fields:
            # FAIL LOUDLY instead of silently accepting
            raise ValueError(
                f"Field '{key}' not allowed in {template_type} frontmatter. "
                f"Allowed fields: {sorted(allowed_fields)}. "
                f"Add to plugin schema at .agentqms/plugins/artifact_types/{template_type}.yaml"
            )

        frontmatter[key] = value

    return self._format_frontmatter_yaml(frontmatter)
```

**Behavior**:
- ✅ Plugin fields: Allowed
- ✅ System fields (`title`, `date`, `branch`, `ads_version`): Allowed
- ✅ Denylist fields: Silently skipped
- ❌ Unknown fields: **RAISES ERROR** with helpful message

---

### Phase 2: Clean Up Corrupt Artifacts

**Affected files**: 6 artifacts from today

**Fields to remove**:
- `kst` → redundant with `date`
- `priority` → not in assessment.yaml schema
- `scope` → not in assessment.yaml schema
- `impact` → not in assessment.yaml schema
- `confidence` → not in assessment.yaml schema
- `related_artifacts` → only in audit.yaml, not assessment.yaml

**Keep**:
- `ads_version`
- `type`
- `title`
- `date`
- `status`
- `category`
- `tags`
- `branch` (if present)

---

### Phase 3: Update audit.yaml If Needed

If `related_artifacts` is genuinely useful across artifact types, add it to assessment.yaml:

```yaml
metadata:
  frontmatter:
    ads_version: "1.0"
    type: assessment
    category: evaluation
    status: active
    tags: [assessment, evaluation, analysis]
    related_artifacts: []  # ← ADD if needed
```

**But**: Only add if there's a clear use case. Don't pollute schema with rarely-used fields.

---

## Questions

### Q: What's the difference between `kst` and `date`?

**A**: They're the SAME data in different formats:
- `date`: `"2026-01-21 17:15 (KST)"` (human-readable)
- `kst`: `"2026-01-21T17:15:00+09:00"` (ISO 8601)

**Solution**: Remove `kst`, keep only `date`. If ISO format needed, parse `date`.

### Q: Are `priority`, `scope`, `impact`, `confidence` useful?

**A**: Maybe, but they're NOT standardized. If we want them:
1. Define in plugin schema
2. Document allowed values
3. Add to ALL artifact types (not just assessment)
4. Update validation to enforce enum values

**Current state**: Just noise without validation.

### Q: Why does audit.yaml have `related_artifacts` but assessment.yaml doesn't?

**A**: Each plugin defines its own schema. `related_artifacts` makes sense for audits (linking to implementation plans). May or may not make sense for assessments.

**Action**: Review each plugin and decide which fields are truly needed.

---

## Execution Plan

### Immediate (Now)
1. ✅ Document root cause (this file)
2. ✅ Fix `create_frontmatter()` to validate kwargs
3. ✅ Clean 6 corrupt artifacts (remove non-schema fields)
4. ✅ Test artifact creation with invalid kwargs (should fail)

### Short-term (Today)
1. Update validation tool to check frontmatter against plugin schema
2. Run full validation on all artifacts
3. Update documentation: "How to add custom frontmatter fields"

### Mid-term (This Week)
1. Audit all 9 plugin schemas
2. Decide on common fields (priority? impact?)
3. Standardize schemas if needed
4. Add JSON schema validation

---

## Success Criteria

**After fix**:
- ✅ All artifacts have ONLY plugin-defined + system fields
- ✅ Attempting to add custom field raises explicit error
- ✅ Plugin schema is the single source of truth (enforced)
- ✅ No more silent pollution

**Test**:
```python
# Should work
frontmatter = templates.create_frontmatter("assessment", "My Title", status="draft")

# Should FAIL with helpful error
frontmatter = templates.create_frontmatter("assessment", "My Title", priority="high")
# ValueError: Field 'priority' not allowed in assessment frontmatter...
```

---

## Related

- Nuclear Bloat Audit: [2026-01-21_1645_assessment_agentqms-nuclear-bloat-audit.md](2026-01-21_1645_assessment_agentqms-nuclear-bloat-audit.md)
- Affected artifacts: 6 files in docs/artifacts/assessments/ from 2026-01-21

---

## Lessons Learned

1. **Fail-fast > Silent acceptance**: Accept invalid input = corrupt data
2. **Schema validation matters**: Without validation, "standards" are just suggestions
3. **Test boundary cases**: We tested happy path, not "what if I pass garbage?"
4. **Nuclear cleanup incomplete**: Removed hardcoded defaults but didn't add validation

This is why the nuclear cleanup felt incomplete—we removed one path to corruption (hardcoded defaults) but left another wide open (unvalidated kwargs).
