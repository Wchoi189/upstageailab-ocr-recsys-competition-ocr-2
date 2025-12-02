---
title: "Bug Fix Protocol"
audience: agent
status: active
domain: governance
id: PROTO-GOV-004
---

## When This Applies

- You are fixing a functional bug, regression, performance issue, or data processing error.
- You are preparing or updating a bug-report artifact under the artifacts tree.

## Required Artifacts

- Bug report in `artifacts/bug_reports/`.
- Changelog entry in `docs/CHANGELOG.md` for non-trivial fixes.

## Agent Steps

1. **Reproduce** the issue in a controlled environment.
2. **Identify root cause** via code and log analysis (do not patch blindly).
3. **Implement minimal fix** that addresses the root cause without side effects.
4. **Add tests** (or extend existing ones) to prevent regression.
5. **Run validation**:
   - `python -m AgentQMS.agent_tools.compliance.validate_artifacts --all`
6. **Document fix**:
   - Create/update a bug report artifact summarizing root cause, fix, and tests.
7. **Update changelog** under the appropriate section.

## Artifact Rules

- Bug reports must follow the **Artifact Rules** protocol (`artifact_rules.md`).
- Recommended filename template: `YYYY-MM-DD_BUG_<short-description>_V1.md`.
- Location: `artifacts/bug_reports/`.

## Checklist

- [ ] Root cause identified and documented.
- [ ] Fix addresses root cause and passes tests.
- [ ] Code follows project standards.
- [ ] Bug report updated/created.
- [ ] Changelog entry added if appropriate.

## Related

- `artifact_rules.md` – canonical artifact naming and placement rules.


