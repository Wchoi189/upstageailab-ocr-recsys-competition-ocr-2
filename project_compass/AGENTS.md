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
| [Compass:Session] | Active Work-in-progress | active_context/current_session.yml |
| [Compass:Lock] | Environment Constraints | environments/uv_lock_state.yml |
| [Compass:Roadmap] | Pipeline Milestones | roadmap/*.yml |

## **3. Atomic Operations**

Agents are prohibited from manual YAML editing. Use the following internal CLI patterns (to be implemented in etk.py):

* etk compass-sync: Updates compass.json timestamp and active handoff.
* etk session-start --goal "...": Initializes a new current_session.yml.
* etk check-env: Compares current shell uv and torch against [Compass:Lock].

## **4. Hard Constraints**

* **NO PIP/PYTHON**: All execution must be prefixed with uv run.
* **NO PROSE**: Status updates must be numeric or enum-based indicators in YAML shards.
* **NO DATA MIXING**: KIE and Layout Analysis data paths must remain strictly isolated.

## **5. Session Lifecycle Protocol**

Adhere to this cycle to ensure correct history tracking and prevent stale session artifacts.

1.  **INIT**: `uv run python -m etk.factory session-init --objective "{objective}"`
2.  **WORK**: Execute tasks. Update `task.md` and `implementation_plan.md`.
3.  **UPDATE CONTEXT (CRITICAL)**:
    *   **Target**: `active_context/current_session.yml`
    *   **Action**: Change `session_id`, set `status="completed"`, update `completed_date`, and add `notes`.
    *   **Constraint**: *Must* be done **BEFORE** export.
4.  **EXPORT**: `uv run python -m etk.factory reconcile`

