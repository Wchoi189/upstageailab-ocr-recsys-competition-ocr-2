# Validation & Runbooks Specification

**Tier**: 4 (Workflows)
**Scope**: Execution logic for Validation and Compliance.

## 1. Validation Runbook
**Command**: `aqms validate` (or `make qms-validate`)

### Triggers
*   **Pre-Commit**: Fast check (names, placement).
*   **CI/CD**: Full check (schema, content, imports).

### Failure Handling
1.  **Stop**: Do not bypass validation.
2.  **Read**: Error messages contain specific error codes (e.g., `E001`).
3.  **Fix**: Address the root cause.
4.  **Retry**: Verify fix.

## 2. Compliance Reporting
**Command**: `aqms artifact check-compliance`
*   **Output**: JSON report of all artifacts.
*   **Metric**: % of artifacts passing schema validation.
*   **Goal**: Maintain > 95% compliance health.
