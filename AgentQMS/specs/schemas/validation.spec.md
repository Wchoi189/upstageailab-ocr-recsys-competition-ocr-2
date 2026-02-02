# Validation Logic Specification

**Tier**: Schemas
**Scope**: Logic for Compliance Checker and Rule Sets.

## 1. Compliance Checker
**Script**: `validate_artifacts.py`
**Feature**: Dual-Mode Validation (V1 vs V2).

### Logic
1.  **Parse Frontmatter**: Detect `ads_version`.
2.  **Load Rules**:
    *   If V2: Load `tier1-contracts/compliance.spec.md` rules.
    *   If V1: Load legacy fallback rules.
3.  **Check**:
    *   Required Fields present?
    *   Values in Enum list?
    *   Filename matches Pattern?

## 2. Rule Sets
*   **Source**: Now derived from Markdown Tables in `specs/`.
*   **Parser**: Tools must parse MD tables to extract valid enums (e.g., `Status`, `Category`).
