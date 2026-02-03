---
title: Context Bundle Integrity Assessment
date: 2026-02-02 20:45 (KST)
type: assessment
category: compliance
status: active
version: 1.0
ads_version: 2.0
---

# Context Bundle Integrity Assessment

## Executive Summary
**Status:** 🔴 **CRITICAL FAILURE**
The recent refactor has structurally compromised the AgentQMS context system. 78% of context bundles are broken. Critical technical rules required for autonomous debugging have been moved to an archival state or lost entirely, replaced by high-level summaries that are insufficient for deep technical tasks.

## 1. Audit Findings
| Metric | Value | Status |
| :--- | :--- | :--- |
| **Broken Bundles** | 14 / 18 | 🔴 78% Failure |
| **Missing References** | 104 Files | 🔴 Critical |
| **DB Sync Status** | 56/104 in DB | 🟡 Partial Recovery Possible |
| **Data Loss** | `tool-catalog.yaml` | 🔴 Irrecoverable (Requires Reconstruction) |
| **Toolchain Breakage** | `aqms registry sync` | 🔴 Fails (Missing `sync_registry.py`) |

## 2. Criticality Verification
**User Question:** *"Verify that the missing informations are critical and should not be removed."*

**Verdict:** **YES, CRITICAL & SYSTEMIC.**
The "pruning" was catastrophic. It removed:
1.  **Operational Knowledge**: Technical rules required for debugging.
2.  **Toolchain Logic**: The `sync_registry.py` script required to operate the new "Spec-driven" registry is **GONE**.
    *   *Effect:* `aqms registry sync` crashes. The system cannot update its own brain.

### Comparison: Hydra Configuration
*   **Old Standard (Archived/DB)**: `FW-019` (Hydra V5 Rules) contains specific error signatures ("ConfigCompositionException"), detailed anti-patterns ("Active Refactor Cycle"), and precise syntax rules ("Absolute Root Anchoring").
    *   *Usage:* Essential for an agent to diagnose *why* a training run crashed.
*   **New Spec (Active)**: `configuration.spec.md` contains high-level philosophy ("Domain-First Architecture") and a brief checklist.
    *   *Usage:* Good for a human architect, useless for an agent debugging a specific `OmegaConf` error.

**Conclusion:** The missing information is **Operational Knowledge** required for execution. The new information is **Architectural Knowledge** useful for planning. **Both are needed.**

## 3. Missing Component Analysis
*   **`tool-catalog.yaml`**: Referenced by `ocr-debugging`. **MISSING** from filesystem, archive, and DB.
    *   *Impact:* Agents cannot look up correct arguments for `adt` or `bloat_detector` tools.
    *   *Fix:* Must be reconstructed from `utility-scripts-manifest.yaml` and code analysis.
*   **`hydra-configuration-architecture.yaml`**: **MISSING** from filesystem, but **PRESENT** in `standards_db.json`.
    *   *Fix:* Can be restored 1:1 from database.

## 4. Recommendations

### Option A: Restore Legacy Standards & Toolchain (Recommended)
1.  **Resurrect Standards**: Restore detailed standards from `standards_db.json` to `AgentQMS/standards/`.
2.  **Restore Toolchain**: Locate or reconstruct `sync_registry.py` to fix `aqms registry sync`.
3.  **Reconstruct Catalog**: Recreate `tool-catalog.yaml`.

*   **Pros:** Restores full functionality and debugging capability.
*   **Cons:** Reverses the "cleanup" effort (which was flawed).

### Option B: Forward Fix (High Risk)
Rewrite `aqms` logic to work without `sync_registry.py` and update all bundles to use Markdown specs.
*   **Pros:** "Pure" V2 state.
*   **Cons:** **High effort, high risk.** Requires rewriting the core registry logic and accepting lower-fidelity context for agents.

## 5. Execution Plan
1.  **Extract & Restore**: Script a restoration of all `tier2` and `tier3` standards from `standards_db.json`.
2.  **Reconstruct Toolchain**: Find or rewrite `sync_registry.py` (check `AgentQMS/tools/core/plugins/registry.py` for logic reuse).
3.  **Reconstruct Catalog**: Manually recreate `tool-catalog.yaml`.
4.  **Validate**: Verify `aqms registry sync` runs and `audit_bundles.py` passes.
