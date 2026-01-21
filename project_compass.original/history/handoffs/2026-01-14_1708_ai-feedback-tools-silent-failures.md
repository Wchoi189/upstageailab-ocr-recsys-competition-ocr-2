# Project Compass & Tooling Feedback

**Date:** 2026-01-14
**Topic:** Usability, Visibility, and Tooling Reliability Analysis

## 1. Pain Points & Visibility
### The "Session" Concept
*   **Confusion**: The concept of "Session" vs "Context Bundles" vs "History" is abstract. It's unclear if a session is a strict git branch, a mental box, or just a folder.
*   **Recommendation**: Rename "Session" to **"Work Cycle"** or **"Sprint Context"**. Make it explicit that it is a *temporary* workspace state that gets "committed" (exported) to history.
*   **Current Friction**: Users have to manually "excavate" documentation to understand that they need to update [current_session.yml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/project_compass/active_context/current_session.yml) manually before exporting. This should be automated or guided by the tool.

### Context Bundling
*   **Triggering**: The system mentions "Context Suggestion: The 'debugging' bundle may be relevant" but the user feels blindly reliant on keywords.
*   **Suggestion**: Trigger bundles based on **Task State** (e.g., if [task.md](file:///home/vscode/.gemini/antigravity/brain/6348116d-c3b4-4c35-8f2d-b151d96e4468/task.md) has "Fixing", auto-load debugging bundle) rather than just keyword matching in chat.
*   **Documentation**: Bundle documentation shouldn't just be "read this file". It should be "Here are the 3 actions you likely need to do now".

## 2. Tooling Failures (Silent & Runtime)
### Silent Failures observed
1.  **Grep & Search**: `grep_search` timed out silently or provided no useful feedback when the codebase was too large ("context deadline exceeded").
    *   *Fix*: Tool should fail fast with "Too many results" or "Time limit" and suggest narrowing scope immediately.
2.  **Debug Run Output**: The `run_command` output for the debug run was empty or misleading initially in [debug_run.log](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/debug_run.log) due to shell redirection issues or buffering.
    *   *Fix*: Ensure `run_command` captures both stdout/stderr reliably without requiring the user to manually redirect to a file to seeing it.
3.  **Config Overrides**: Hydra config overrides failed silently or with obscure errors (`Key 'trainer' is not in struct`) because the user/agent didn't know `+` was required for new keys vs existing keys.
    *   *Fix*: The `mcp_unified_project` should probably expose a `validate_config` tool that checks Hydra syntax before running.

## 3. Conflicting Information
*   **Roadmap vs Session**: `roadmap/*.yml` and [current_session.yml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/project_compass/active_context/current_session.yml) can easily drift apart. I found myself manually updating both.
*   **Config Defaults**: It was unclear if `wandb_image_logging` was on by default. I had to read 4 different files ([train.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/train.yaml), [base.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/base.yaml), [callbacks/default.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/training/callbacks/default.yaml), [callbacks/wandb_image_logging.yaml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/training/callbacks/wandb_image_logging.yaml)) to trace the inheritance.
    *   *Fix*: A `view_effective_config` tool would be invaluable to see the *final* merged YAML without running python.

## 4. Documentation Strategy
*   **Problem**: "You should not have to excavate the project's vast documentation".
*   **Solution**:
    *   **Schema-First**: The [current_session.yml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/project_compass/active_context/current_session.yml) has a schema comment. The MCP tool should technically be able to *show* the schema or a form-like interface.
    *   **In-Context Help**: When an error occurs (like the Hydra struct error), the system should inject a one-line "Did you mean to use `+key=value`?" hint, derived from a "common errors" bundle.

## 5. Immediate Action Plan
1.  **Fix Tooling**: Investigate `grep_search` timeouts.
2.  **Automate Context**: Create a "Session Wizard" that asks 3 questions and populates [current_session.yml](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/project_compass/active_context/current_session.yml) automatically.
3.  **Simplify Config Visibility**: Add a script `scripts/show_config.py` that dumps the fully resolved Hydra config for inspection.
