# **Project Compass: AI Entrypoints & Protocols**

## **1. Interaction Protocol**

Every session **MUST** start with the AI executing the following sequence:

1. **Read project_compass/compass.json**: Identify the current phase and health.
2. **Read project_compass/active_context/current_session.yml**: Understand immediate goals and blockers.
3. **Validate Environment**: Compare system state against project_compass/environments/uv_lock_state.yml.
   * *Failure to match uv_path or torch_version results in immediate halt and recovery request.*

## **2. Chat Dialogue Nomenclature**

When discussing the project in chat, use these specific references to minimize ambiguity:

| Reference | Scope | Target File |
| [Compass:State] | Global Health | compass.json |
| [Compass:Session] | Sprint Context (Active) | active_context/current_session.yml |
| [Compass:Lock] | Environment Constraints | environments/uv_lock_state.yml |
| [Compass:Roadmap] | Pipeline Milestones | roadmap/*.yml |

## **3. Atomic Operations**

Agents are prohibited from manual YAML editing. Use the following internal CLI patterns (Run from project root):

* `uv run python -m project_compass.cli session-init --objective "..."` : Initializes a new Sprint Context.
* `uv run python scripts/utils/show_config.py [config_name] [overrides...]` : Inspects effective Hydra configuration.
* `uv run python -m project_compass.cli check-env` : Compares current shell uv and torch against [Compass:Lock].

## **4. Hard Constraints**

* **NO PIP/PYTHON**: All execution must be prefixed with uv run.
* **NO PROSE**: Status updates must be numeric or enum-based indicators in YAML shards.
*   **NO PIP/PYTHON**: All execution must be prefixed with uv run.
*   **NO PROSE**: Status updates must be numeric or enum-based indicators in YAML shards.
*   **NO DATA MIXING**: KIE and Layout Analysis data paths must remain strictly isolated.

## **5. Session Lifecycle Protocol**

Adhere to this cycle to ensure correct history tracking and prevent stale session artifacts.

1.  **INIT**: `uv run python -m project_compass.cli session-init --objective "{objective}"`
2.  **WORK**: Execute tasks. Update `task.md` and `implementation_plan.md`.
3.  **UPDATE CONTEXT (CRITICAL)**:
    *   **Target**: `active_context/current_session.yml` (Sprint Context)
    *   **Action**: Change `session_id`, set `status="completed"`, update `completed_date`, and add `notes`.
    *   **Constraint**: *Must* be done **BEFORE** export.
4.  **EXPORT SESSION**: `uv run python -m project_compass.cli session-export --note "Session completion note"`
