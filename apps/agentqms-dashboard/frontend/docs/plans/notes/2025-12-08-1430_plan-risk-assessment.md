---
title: "Risk Assessment Report: AgentQMS Dashboard"
type: plan
status: complete
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 1
priority: high
tags: [plan, risk-assessment, mitigation, project-planning]
---

# Risk Assessment Report: AgentQMS Dashboard
**Date:** 2025-12-08 (created) → 2025-12-08 (updated)
**Status:** Initial Assessment

## 1. Executive Summary
The AgentQMS Dashboard is currently a **Client-Side Single Page Application (SPA)** acting as a manager for a **Local File System Framework**. This architecture presents a critical "Air Gap" risk where the dashboard cannot directly execute the Python maintenance scripts it recommends without a backend bridge.

## 2. Critical Risk Analysis

### A. The "Browser Sandbox" Constraint (High Risk)
*   **Issue:** The React app running in the browser cannot execute `subprocess.run(['python', 'audit.py'])` on the host machine.
*   **Impact:** The "Auditor" and "Tracking DB" features are strictly *visualizers* or *generators*. They cannot enforce rules or run database migrations directly.
*   **Mitigation:**
    1.  UI must clearly distinguish between "Simulation (AI)" and "Execution (CLI Commands)".
    2.  Future Phase: Implement a lightweight local server (FastAPI/Flask) to bridge the UI and the file system.

### B. State Persistence & Data Loss (Medium Risk)
*   **Issue:** Application relies on `localStorage` for Settings and API Keys.
*   **Impact:** Clearing browser cache wipes configuration. No centralized database for team collaboration.
*   **Mitigation:** Implement an "Export/Import Config" feature JSON to allow developers to save their setup.

### C. AI Hallucination in Auditing (Medium Risk)
*   **Issue:** Using LLMs (Gemini/GPT) to audit code/docs vs. using deterministic Python scripts.
*   **Impact:** AI might flag false positives or miss syntax errors that a linter would catch.
*   **Mitigation:**
    1.  Prioritize deterministic tools (`agent_tools/audit/`) over AI where possible.
    2.  Use AI only for qualitative analysis (clarity, tone, completeness).

## 3. Maintenance Bottlenecks
*   **Spaghetti Code Potential:** `App.tsx` and `IntegrationHub.tsx` are growing large.
    *   *Action:* Refactor into `features/` folders immediately.
*   **Hardcoded Paths:** References to `AgentQMS/modules/` are hardcoded strings.
    *   *Action:* Move to a `config` object or Context provider.

## 4. Performance Concerns
*   **Large Contexts:** Sending entire file contents to LLMs for auditing can hit token limits or incur high costs.
    *   *Action:* Implement token counting estimation in the UI before submission.

## 5. Strategic Recommendations
1.  **Backend Bridge:** Develop a small `server.py` that the React app talks to. This enables real-time file system reads/writes and script execution.
2.  **Strict Schema:** Continue enforcing JSON/YAML schemas for all artifacts to ensure the "dumb" UI doesn't break when reading files.
