---
name: audit-run
description: Execute audit checklist systematically with artifact generation. Use when user requests code audit or validation.
disable-model-invocation: false
context: fork
agent: Explore
---

# Audit Runner - Systematic Code Validation

## Audit Workflow

Execute audit for: $ARGUMENTS

Default: "all" (run full audit)

---

## Audit Categories

Available categories:
- `plm-correctness` - Permutation logic, loss computation, sequence handling
- `flash-attention` - Numerical equivalence, kernel validation
- `device-placement` - CUDA context, tensor device verification
- `gradient-flow` - Backprop validation through custom layers
- `configuration` - Hydra config consistency
- `performance` - Throughput benchmarks, optimization opportunities
- `all` - Run complete audit suite

---

## Execution Strategy

For each category:

### 1. Load Audit Checklist
Read from:
- Exported pulse audit prompts (if exists)
- Category-specific templates (if exists)
- Generate checklist from code analysis

### 2. Execute Checks
- Read relevant source files
- Trace execution paths
- Identify potential issues
- Document findings with line references

### 3. Generate Artifact
Create in `pulse_staging/artifacts/audit/`:
- `[category]_audit.md` with findings
- Structure:
  ```markdown
  # [Category] Audit

  ## Summary
  [2-3 bullet points]

  ## Critical Issues
  - [issue] ([file:line])

  ## Warnings
  - [warning] ([file:line])

  ## Pass
  - [validation] ([file:line])

  ## Recommendations
  - [recommendation]
  ```

### 4. Auto-Register
Register audit artifact automatically via CLI:
```bash
uv run compass pulse-sync \
  --path "audit/[category]_audit.md" \
  --type "audit"
```

---

## Output Format

```
🔍 Executing [category] Audit...

[Progress indicators]

✅ Audit Complete: audit/[category]_audit.md

📊 Findings:
   🔴 Critical: [count]
   🟡 Warnings: [count]
   🟢 Passed: [count]

📝 Artifact registered automatically
```

---

## Example Usage

```
/audit-run plm-correctness
```

Analyzes:
- `ocr/domains/recognition/models/plm.py`
- `ocr/domains/recognition/models/decoder.py`
- `ocr/domains/recognition/models/architecture.py`

Generates: `pulse_staging/artifacts/audit/plm-correctness_audit.md`

---

## Integration with Exported Pulses

If previous pulse has audit prompt:
1. Read from history: `history/[milestone]/[pulse]/artifacts/*audit*.md`
2. Use as checklist template
3. Reference previous findings
4. Track regression

---

## Fail Fast Philosophy

- Critical issues = Immediate stop and report
- Document exact file:line references
- No speculation, only code evidence
- Provide runnable test code for validation

---

## Post-Audit Actions

After generating audit:
1. Summarize critical issues (max 3 bullet points)
2. Ask user: "Create test suite for findings? (yes/no)"
3. If yes, generate `tests/test_[category].py`
4. Register test file as implementation_plan artifact
