This is a comprehensive architectural analysis and a strategy for a "hard-reboot" of **Project Compass**.

Your current system suffers from **"Architectural Drift"**—where the AI interprets abstract concepts (like "Session" or "Context") through its own internal training data rather than your specific rules. The overlap of terms like "Project" or "Session" triggers generic AI behaviors instead of your strict protocols.

---

## 1. The Diagnosis: Why the AI is Failing

* **Term Ambiguity:** To an LLM, a "Session" is a chat history. A "Roadmap" is a high-level plan. By using these words, you are fighting against the AI's default token associations.
* **State Schizophrenia:** Your state is split between `compass.json`, `current_session.yml`, and `session_handover.md`. The AI doesn't know which one is the "Source of Truth" (SoT), so it tries to sync them all and fails.
* **Implicit vs. Explicit Boundaries:** AI agents are "path of least resistance" entities. If a directory isn't physically restricted or logically locked by a validator, they will put files wherever they "feel" appropriate.

---

## 2. Structural Redesign: "The Anchor & The Vault"

To fix the inconsistent artifact placement and context loss, move from a "Folder-based" mindset to a **"Manifest-based"** mindset.

### **The "Vault" Architecture**

Instead of `active_context/`, use a dedicated **`work_cycle/`** directory.

1. **`work_cycle/manifest.yml`**: This is the ONLY file the AI is allowed to update. It contains:
* `pointer_to_roadmap_item`: (e.g., `02_recognition.yml:rec-optimization`)
* `locked_instructions`: The specific constraints for this cycle.
* `artifact_manifest`: A list of file paths the agent *must* use.


2. **`work_cycle/artifacts/`**: A strictly enforced staging area.

### **The Requirements for the Workflow**

* **Zero-Narrative Policy:** Force the AI to use "Indicator-Only" notes (Max 20 words). If it wants to explain, it must do so in a temporary `scratchpad.md` that is *not* exported.
* **ID-Linkage:** Every session must be explicitly linked to a **Milestone ID** from the Roadmap. If the ID doesn't exist, the `session-init` tool should fail.

---

## 3. Hard-Enforcing Rules & Lifecycle

You cannot rely on the AI to "remember" rules in a long conversation. You must use **Pre-Execution Validation**.

### **The Lifecycle (The "Pulse" Pattern)**

1. **`pulse-init`**: Agent selects a Roadmap ID. The tool populates the `work_cycle/` with the specific rules for that milestone.
2. **`pulse-sync`**: A tool that the AI *must* call after creating any artifact. This tool checks: "Is this file inside `work_cycle/artifacts/`?" and "Is it in the manifest?". If not, it moves the file and updates the manifest.
3. **`pulse-checkpoint`**: Instead of a "Handover" (narrative), this generates a **Context-Delta**. It records: *What was changed, what is the next file to edit, and what is the current sub-goal.*

### **How to Stop "New_Session_01" Naming**

Modify your `validate_session_name` in `src/validation.py` to be a **Blocking Gate**.

* **Constraint:** The CLI must reject any name that doesn't follow a `[DOMAIN]-[VERB]-[TARGET]` regex.
* **Result:** The AI will receive an error message: *"ERROR: 'new_session_01' is non-descriptive. You must provide a name reflecting the technical intent (e.g., recognition-fix-cuda-mismatch)."*

---

## 4. Efficient Synchronization & Interface

The AI should interact with **one single abstraction**, not three different JSON/YAML files.

### **Optimal Interface: The "Command Center"**

Combine your tools into a single MCP tool called `sync_state`.

* **Input:** Current objective, updated metrics, list of new artifacts.
* **Logic:** The Python backend handles updating `compass.json`, `roadmap.yml`, and the `manifest.yml` simultaneously. The AI never sees the individual files; it only sees the "State Object."

---

## 5. What You Are Missing: The "Rule Injection"

The biggest gap is the loss of rules between conversations.

* **The Fix:** Create a `rules_vault/` directory.
* **The Workflow:** When a session is exported, the `session-export` tool should bundle the **specific instructions** that were used into the history folder. When `session-import` is called, it re-injects those rules into the prompt.

---

## 6. Public Alternatives & Open Source

You are currently building a custom **"Agentic Memory Layer."** While Python is brittle, it allows for the strict validation you need.

**Similar Projects:**

1. **Aider (Architect Mode):** Uses a `.aider.conf.yml` and a specific "chat-with-code" protocol. It handles context well but lacks your roadmap-driven milestones.
2. **MemGPT:** Manages "long-term memory" by swapping context in and out of the window. You could adapt their "archival memory" concept for your session history.
3. **Devin-style "Playbooks":** Some open-source clones use a `playbook.yml` that defines the exact steps an agent can take.

**The "Low-Boilerplate" Alternative:**
Instead of heavy Python logic, use **JSON-Schema + Pydantic**.

* Define your entire state in a single Pydantic model.
* The AI only interacts with a tool that accepts this model.
* Pydantic handles the validation, error messaging, and "logic" automatically, reducing your Python maintenance burden.

---

## Next Steps for You

Would you like me to:

1. **Draft a Pydantic-based schema** that consolidates `compass.json`, `session`, and `roadmap` into a single, unified "State Object" to reduce context burden?
2. **Create a "Rule-Injection" script** that ensures original instructions are never lost during session transitions?
3. **Refactor the `manage_session` tool** to enforce the "Staging Area" (artifacts must be in a specific folder) before allowing an export?



This "nuclear overhaul" moves **Project Compass** from a collection of loosely related files into a **Single Source of Truth (SSoT)** architecture. By consolidating state and enforcing strict "Staging Areas," we eliminate the "Architectural Drift" that causes AI agents to hallucinate file locations and formats.

---

## 1. The Unified State Schema (Pydantic)

To reduce context burden, we replace multiple files with a single `state.json`. This schema uses Pydantic to enforce strict enums and relationships, preventing agents from using non-descriptive names like `new_session_01`.

```python
from pydantic import BaseModel, Field, validator
from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

class ProjectHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BLOCKED = "blocked"

class PipelinePhase(str, Enum):
    DETECTION = "detection"
    RECOGNITION = "recognition"
    KIE = "kie"
    INTEGRATION = "integration"

class Artifact(BaseModel):
    path: str = Field(..., description="Must be within work_cycle/artifacts/")
    type: str = Field(..., regex="^(design|research|walkthrough|implementation_plan)$")
    milestone_id: str

class WorkCycle(BaseModel):
    cycle_id: str = Field(..., regex="^[a-z]+-[a-z]+-[a-z]+$") # domain-action-target
    objective: str = Field(..., min_length=20, max_length=200)
    instructions: List[str] = Field(..., description="Rule-injected guidelines for this cycle")
    artifacts: List[Artifact] = []

class CompassState(BaseModel):
    version: str = "2.0.0"
    last_updated: datetime = Field(default_factory=datetime.now)
    health: ProjectHealth
    current_phase: PipelinePhase
    active_cycle: Optional[WorkCycle]
    roadmap: Dict[str, List[str]] # Milestone ID -List of Task IDs

```

**Policy Change:** The AI agent is strictly forbidden from editing this file directly; it must use a `sync-state` tool that validates these constraints before writing to disk.

---

## 2. Refactored Lifecycle: The "Staging Area" Protocol

To solve the "scattered artifacts" problem, we implement a **Physical Lockdown**. The AI can no longer generate files in the project root or `docs/` arbitrarily.

### **Architecture: The Work Cycle Vault**

1. **Staging Required:** All new files *must* be created in `work_cycle/artifacts/`.
2. **Manifest Enforcement:** The `manage_session` tool will now scan this folder during export. If a file exists in the folder but is *not* in the `state.json` artifact list, the export is **blocked**.
3. **Naming Convention:** Files must follow the format `{milestone_id}_{artifact_type}.md` (e.g., `rec-opt_research.md`).

### **Refactored `manage_session` Logic**

```python
def validate_staging_area(cycle_artifacts):
    actual_files = list(Path("work_cycle/artifacts/").glob("*"))
    for file in actual_files:
        if str(file) not in [a.path for a in cycle_artifacts]:
            raise ValidationError(f"Unregistered artifact found: {file}. Update state first.")

```

---

## 3. The Rule-Injection Script

To prevent the loss of core guidelines between conversations, we implement a "Persistent Prompt" mechanism.

### **Design Document: Instruction Persistence**

* **Source:** `rules_vault/core_directives.md` and `rules_vault/milestone_rules/{milestone_id}.md`.
* **Trigger:** When `session-init` is called, the script assembles these rules and injects them into the `WorkCycle.instructions` field of the state object.
* **The AI Prompt:** "You are currently in a Work Cycle. Your behavior is governed by the specific instructions in your state object. You must not deviate from the formats defined in `instructions`.".

---

## 4. Usage Instructions for the Implementing Agent

### **Step 1: The Hard Reboot**

Run the following command to migrate the legacy `compass.json` and `current_session.yml` into the new unified state:
`uv run python -m project_compass.cli migrate-to-v2`

### **Step 2: Starting a Cycle**

Initialize a cycle by linking it to a Roadmap Milestone. This will automatically pull the necessary rules from the vault:
`uv run python -m project_compass.cli cycle-init --milestone rec-optimization --name recognition-fix-cuda`

### **Step 3: Post-Execution Validation**

Before you finish your turn, you MUST run the sync tool. This will catch non-compliant artifact names or locations:
`uv run python -m project_compass.cli sync-state`

---

## 5. Passivity Check & Architectural Summary

**Am I providing passive suggestions?** No. I have replaced "vague sessions" with "Locked Work Cycles" and "narrative handovers" with "Validated State Objects.".

| Feature       | Legacy Weakness               | Nuclear Overhaul Solution              |
| ------------- | ----------------------------- | -------------------------------------- |
| **State**     | Fragmented (JSON + YAML)      | Unified Pydantic State Object          |
| **Naming**    | Generic/Non-descriptive       | Regex-enforced Domain-Action-Target    |
| **Artifacts** | Scattered (docs/, root, etc.) | Strict `work_cycle/artifacts/` Staging |
| **Context**   | Lost between conversations    | Rule-Injection from the Rules Vault    |
| **Logic**     | Manual updates needed         | Automated Tool-based Synchronization   |

**Would you like me to generate the full `rules_vault/` directory structure and the specific "Hard Constraints" markdown file for the `rec-optimization` milestone?**.

---

To ensure your AI agent operates with high intentionality rather than rote execution, we are implementing a **"Thinking-First" Architecture**. The `rules_vault/` serves as the long-term memory that the agent must "check out" at the start of every cycle.

---

## 1. The `rules_vault/` Directory Structure

This structure ensures that the agent's behavior is anchored in static, high-level policies that do not drift between conversation turns.

```text
rules_vault/
├── core_directives/
│   ├── 00_meta_cognition.md       # "Thinking-First" protocols (The AI's logic core)
│   ├── 01_naming_standards.md      # Regex rules for cycles and artifacts
│   └── 02_artifact_purity.md       # Formatting and conciseness standards
├── milestone_rules/
│   ├── det-scaling.md              # Domain-specific constraints for Detection
│   ├── rec-optimization.md         # (Generated below) Hardware & accuracy rules
│   └── kie-alignment.md            # Data-source specific rules
└── lifecycle_scripts/
    └── inject_rules.py             # Script to populate WorkCycle.instructions

```

---

## 2. Hard Constraints: `rec-optimization.md`

This file is designed to be injected into the `WorkCycle.instructions` during any optimization sprint for text recognition.

```markdown
# [Compass:Lock] Milestone: rec-optimization

## 1. Hardware Constraints (RTX 3090)
* **Batch Size**: Minimum 128 (FP16/Mixed Precision) to maximize VRAM utilization.
* **Num Workers**: Fixed at 12 to 16 to eliminate CPU dataloading bottlenecks.
* **Memory**: Do not exceed 22GB VRAM usage to maintain system stability for logging.

## 2. Model Purity Rules
* **No Legacy Imports**: Strictly avoid `ocr.features`. Use `ocr.domains.recognition`.
* **Atomic Config**: Optimizer and Loss settings MUST live in `train/` or `domain/`, never inside `model/architectures/`.
* **Vocab Invariance**: Any change to character sets requires a mandatory vocab_size check in the Orchestrator.

## 3. Data Protocols
* **Staging Location**: New LMDBs must be registered in `dataset_registry.yml` before training launch.
* **Offline Preprocessing**: Images must be resized to 32x128 (Rectangular) to avoid SVTR coordinate drift.

```

---

## 3. Enhancing AI Participation: The Meta-Cognition Protocol

To stop the agent from "executing without thinking," the following instructions are injected into the `00_meta_cognition.md` file. The agent is forced to provide a **"Reasoning Trace"** before any tool call.

### **The "Pause-and-Reflect" Guardrail**

Every time the agent initializes a Work Cycle, it must generate a **State Assessment** following this template:

**[Compass:Reflection]**
* **Current Goal**: (Restate the goal in your own words)
* **Implicit Blockers**: (Identify what rules might be violated if I act too quickly)
* **Architectural Sanity**: (Does this move the project closer to 'Domains First' V5, or is it a shortcut?)
* **Proposed Path**: (Briefly outline the tool calls)



### **Behavioral Policy for the Agent**

1. **Challenge the Instructions**: If a roadmap task is vague, you are REQUIRED to ask for clarification instead of guessing.
2. **Proactive Pruning**: If you see legacy code (e.g., `wandb_utils.py`) that hasn't been cleaned up, you must add it to the `active_blockers` in the state object.
3. **Evidence-Based Success**: You cannot mark a phase as "Completed" unless you provide the output of a validation command (e.g., `uv run python -m project_compass.cli check-env`).

---

## 4. Architectural Summary

By moving the "State" to a Pydantic object and the "Rules" to a Vault, we create a **closed-loop system**.

* **The Vault** dictates the "How".
* **The State Object** tracks the "What".
* **The CLI** enforces the "Where" (Staging Area).

**Would you like me to generate the `migrate-to-v2.py` script to transform your current fragmented YAML files into this unified Pydantic `state.json`?**


---

To systematically organize artifacts and prevent the rapid cluttering of your project, you must shift from a **filesystem-first** approach to a **manifest-driven** approach. In this model, the directory is merely a temporary staging area, while the `state.json` acts as the authoritative ledger.

---

## 1. The Staging Area Architecture

The problem of "scattered" artifacts is solved by enforcing a strict **Physical Boundary**. The AI agent should be restricted to a single, ephemeral directory for all output during a work cycle.

* **Temporary Staging**: All artifacts (plans, logs, reports) are generated in `work_cycle/artifacts/`.
* **The "Audit Gate"**: The `session-export` tool must perform a **Staging Audit**. It compares the physical files in the artifacts folder against the `artifacts` list in your `state.json`.
* **Failure State**: If a file exists on disk but is not registered in the manifest, the tool blocks the export and forces the agent to categorize it or delete it.


* **Automated Sorting**: Upon export, the system uses the `type` field in the manifest to move files to their final destinations (e.g., `history/sessions/` for logs or `docs/canonical/` for permanent design docs).

---

## 2. Artifact Lifecycle & Promotion

To prevent artifacts from becoming outdated and cluttered, implement a **Tiered Promotion System**.

| Tier             | Status                    | Location                 | Persistence                                    |
| ---------------- | ------------------------- | ------------------------ | ---------------------------------------------- |
| **Transitional** | Draft / Working log       | `work_cycle/artifacts/`  | Deleted or moved on cycle completion.          |
| **Archived**     | Historic record           | `history/sessions/{ID}/` | Permanent, but out of the active path.         |
| **Canonical**    | Validated Source of Truth | `docs/canonical/`        | Manually "promoted" high-value plans or rules. |

### **The "Auto-Purge" Policy**

Instruct the agent to treat `work_cycle/artifacts/` as a "volatile" space. Any file not explicitly marked for "Promotion" in the `sync-state` call will be **deleted** when the session ends. This forces the agent to be intentional about what it keeps.

---

## 3. Standardized Naming & Metadata

Use a strict regex-enforced naming convention for all artifacts to ensure they remain recognizable even when buried in history.

**Format**: `[Milestone-ID]_[Artifact-Type]_[Short-ID].md`

* *Example*: `rec-opt_research_01.md`
* *Metadata*: Every artifact must include a YAML frontmatter block that links it to the specific `cycle_id` and the `milestone_id`.

---

## 4. Enhanced AI Participation: The [Compass:Reflection] Protocol

To stop the agent from "executing without thinking," you must mandate a **Reasoning Step** before any file is created. This should be part of the agent's core instructions in `rules_vault/core_directives/00_meta_cognition.md`.

### **The Artifact Creation Protocol**

Before the agent creates a file, it must output a brief reflection:

**[Compass:Reflection]**
* **Type**: (e.g., research, implementation_plan)
* **Justification**: Why is this file necessary for this specific cycle?
* **Redundancy Check**: Is there an existing artifact in `history/` or `docs/` that already covers this? If so, why am I not just updating it?
* **Lifecycle**: Is this file *Transitional* (delete after) or *Archived* (keep in history)?



---

## 5. Implementation Tooling

You need a `sync-state` tool that does more than just update JSON. It should act as an **Aggregator**.

```python
# Pseudo-logic for the sync-state tool
def sync_state(state_update):
    # 1. Update the Pydantic State Object
    # 2. Audit the staging folder (work_cycle/artifacts/)
    # 3. For each file found:
    #    a. Check if it's in the state_update metadata
    #    b. Check if the naming convention is met
    #    c. If valid, 'lock' the file for export
    # 4. Warn if 'stale' artifacts from previous attempts are still in the folder

```

This structural consideration transforms your directory from a "messy drawer" into a **Controlled Environment** where every file has a clear origin, purpose, and expiration date.

Would you like me to generate the **Pydantic migration script** (`migrate-to-v2.py`) to transition your current project to this manifest-driven state?


---

This nuclear overhaul replaces the ambiguous "Session" and "Project" terminology with a high-precision lexicon to prevent AI model drift. The system is now referred to as the **Vessel** (Project), and individual work units are called **Pulses** (Sessions).

### 1. The "Pulse-Feedback" Loop (Append vs. Export)

To prevent the agent from blindly appending data to a bloated context, you must implement a **Mandatory Checkpoint Tool**. The agent is forbidden from ending its turn without calling `pulse_checkpoint`.

**Tool Definition: `pulse_checkpoint**`

* **Purpose:** Forces the agent to evaluate state maturity.
* **Required Inputs:**
* `token_burden`: [Low/Medium/High]
* `objective_status`: [Partial/Complete/Blocked]
* `next_action`: [Continue/Export-and-Reset]


* **Logic:** If `token_burden` is "High" or `objective_status` is "Complete," the tool triggers a mandatory prompt: *"Current Pulse has reached maturity. Should I export the artifacts to History and initialize a fresh Pulse, or is there a critical dependency requiring an append?"*

---

### 2. The Clean Sweep: Pruning & Directory Overhaul

Delete the following folders and files to remove the "Legacy Narrative" burden:

* **PRUNE:** `active_session/`, `active_context/`, `docs/roadmap/` (too many places for the AI to hide files).
* **DELETE:** `session_handover.md` (Narrative is replaced by the `vessel_state.json` delta).
* **DELETE:** Any manual YAML update scripts in `src/`.

**NEW Directory Structure:**

```text
vessel_root/
├── .vessel/                 # Hidden system state
│   └── vessel_state.json     # The ONLY Source of Truth (Pydantic Model)
├── vault/                    # Rule Injection Source
│   ├── directives/           # Global AI behavioral rules
│   └── milestones/           # Specific technical constraints
├── pulse_staging/            # THE ONLY FOLDER the AI can write to
│   └── artifacts/            # Transient files awaiting validation
└── history/                  # Frozen Pulse exports

```

---

### 3. Vessel V2: Design & Architecture

This document serves as the "Primary Entry Point" for any agent. It must be small, indented, and factual.

**File: `.vessel/architecture.md**`

**Core Concept:** Single State Manifest.
1. **State Isolation:** The AI operates on a temporary `Pulse`. It cannot see `History` unless explicitly requested.
2. **Staging Constraint:** Any file created outside of `pulse_staging/` is a violation of protocol and will be ignored by the `sync` tool.
3. **The SSoT:** `vessel_state.json` contains the `Star-Chart` (Roadmap) and the `Current_Delta`. All changes must flow through this object.



---

### 4. Protocol: The "Pulse" Lifecycle

Add this to your context bundling system. It is the hard-enforced logic for the AI.

**File: `vault/directives/pulse_protocol.md**`

1. **Pulse Init:** On startup, read `vessel_state.json`. If no active Pulse exists, you MUST call `pulse_init` to pull rules from the `vault/`.
2. **Staging Rule:** Generate ALL artifacts in `pulse_staging/artifacts/`. No exceptions.
3. **Naming Convention:** Artifacts must use the format `[MilestoneID]-[Type]-[ShortDescription].md`.
4. **No Narrative:** Do not write summaries. Update the `vessel_state.json` with a 10-word status change instead.
5. **Export Trigger:** When a milestone task is marked `Complete`, the Pulse is automatically exported to `history/` and `pulse_staging/` is wiped.

---

### 5. Systematic Referencing (No Reserved Keywords)

To stop the AI from using its "standard session" training, use these terms in your prompts and tool names:

* **Vessel** (Project)
* **Pulse** (Session/Conversation)
* **Star-Chart** (Roadmap)
* **Manifest** (State Object/Compass)
* **Staging** (Active folder)

By using "Pulse" instead of "Session," the AI will stop trying to "handover" or "summarize" like a standard chatbot and will instead follow the "Vessel" lifecycle protocols.


----


The Rule-Injection script functions as a **State Pre-loader** that executes during the transition between the end of one Pulse (session) and the initialization of the next. It prevents context drift by transforming static markdown rules into active, immutable constraints within the system manifest.

### 1. The Core Mechanism: Aggregation and Embedding

Instead of relying on the AI to "remember" what was said three conversations ago, the script physically assembles the context before the AI even sees the prompt.

1. **Directives Fetching:** The script scans `vault/directives/` for global behavior rules (e.g., "Always use Pydantic," "Never write narratives").
2. **Milestone Mapping:** It checks the `vessel_state.json` to see which `milestone_id` is currently active (e.g., `rec-optimization`). It then fetches the specific constraints from `vault/milestones/rec-optimization.md`.
3. **The Injection:** It merges these rules and writes them into the `active_pulse.instructions` array inside the **Manifest**.

### 2. How it Handles Transitions

When you "Export" a Pulse, the AI loses its short-term memory. The Rule-Injection script bridges this gap by ensuring the **Manifest** is the "DNA" that survives the transition.

* **At Pulse-Export:** The script clears the "Staging" folder but *retains* the `active_milestone` and `pending_instructions` in the `vessel_state.json`.
* **At Pulse-Init:** The script runs automatically. It sees the `active_milestone`, pulls the fresh rules from the `vault/` (in case you updated them manually), and injects them into the new Pulse's starting context.

### 3. Conceptual Script Logic

This is a minimalist Python implementation that would live in `src/vessel/core.py`.

```python
import json
from pathlib import Path

def inject_rules(manifest_path: Path, vault_path: Path):
    # 1. Load the current Manifest (The DNA)
    with open(manifest_path, "r") as f:
        state = json.load(f)

    active_milestone = state["active_cycle"]["milestone_id"]

    # 2. Gather Core Directives
    rules = []
    for directive in (vault_path / "directives").glob("*.md"):
        rules.append(directive.read_text())

    # 3. Gather Milestone-Specific Rules
    milestone_file = vault_path / f"milestones/{active_milestone}.md"
    if milestone_file.exists():
        rules.append(milestone_file.read_text())

    # 4. Inject into the State Object
    state["active_cycle"]["instructions"] = rules

    # 5. Commit back to Disk
    with open(manifest_path, "w") as f:
        json.dump(state, f, indent=2)

```

### 4. Why this Stops Context Loss

* **No Narrative Burden:** The AI doesn't have to read a "Handover" document and interpret it. It simply sees a list of "Hard Constraints" that are part of its current operating state.
* **Immutable Source:** Even if the AI tries to hallucinate a new rule, the `sync-state` tool will compare its actions against the `instructions` array in the manifest. If it violates a rule injected from the vault, the tool will reject the change.
* **Versioned Authority:** If you change a rule in the `vault/`, every new Pulse created after that moment will immediately adopt the new rule, ensuring the entire implementation stays synchronized without manual document updates.


----


To eliminate the "scattered files" problem, we are replacing the vague `manage_session` logic with a strict **`PulseExporter`**. This tool acts as a physical gatekeeper: it will refuse to archive a session if the workspace is "dirty" or if artifacts are not properly registered in the **Manifest**.

### 1. The Staging Area Logic

The tool enforces a "clean desk" policy. Before an export is allowed, the following must be true:

1. **Exclusivity**: Every file created by the AI must reside in `pulse_staging/artifacts/`.
2. **Registration**: Every file in that directory must have a corresponding entry in the `vessel_state.json` artifact list.
3. **Naming**: Every file must match the `[Milestone]-[Type]-[Description]` regex.

### 2. Implementation: `pulse_exporter.py`

This script replaces your previous session management logic. It is designed to be called by the AI agent via a tool interface.

```python
import os
import shutil
import json
from pathlib import Path
from pydantic import ValidationError

# Configuration
STAGING_DIR = Path("pulse_staging/artifacts")
MANIFEST_PATH = Path(".vessel/vessel_state.json")
HISTORY_DIR = Path("history")

def export_pulse():
    """
    Refactored 'manage_session' tool with strict staging enforcement.
    """
    # 1. Load Manifest (The Source of Truth)
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    pulse_id = manifest["active_pulse"]["pulse_id"]
    registered_paths = {a["path"] for a in manifest["active_pulse"]["artifacts"]}

    # 2. Audit the Physical Staging Area
    actual_files = {str(p) for p in STAGING_DIR.glob("**/*") if p.is_file()}

    # 3. Check for "Stray" (Unregistered) Files
    strays = actual_files - registered_paths
    if strays:
        return {
            "status": "BLOCKED",
            "error": "Unregistered artifacts found in staging.",
            "action_required": f"Register these files in the manifest or delete them: {strays}"
        }

    # 4. Check for "Ghost" (Missing) Files
    ghosts = registered_paths - actual_files
    if ghosts:
        return {
            "status": "BLOCKED",
            "error": "Manifest references files that do not exist on disk.",
            "action_required": f"Create or remove references for: {ghosts}"
        }

    # 5. Execute Export (Promotion to History)
    pulse_history_path = HISTORY_DIR / pulse_id
    pulse_history_path.mkdir(parents=True, exist_ok=True)

    # Move artifacts and clear staging
    for file_path in actual_files:
        dest = pulse_history_path / Path(file_path).name
        shutil.move(file_path, dest)

    # Archive the manifest state into the history folder
    with open(pulse_history_path / "pulse_manifest_snapshot.json", "w") as f:
        json.dump(manifest["active_pulse"], f, indent=2)

    # 6. Reset the Vessel for the next Pulse
    manifest["active_pulse"] = None
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"status": "SUCCESS", "export_path": str(pulse_history_path)}

```

---

### 3. Usage Instructions for the AI Agent

The following rules are added to the **`vault/directives/02_artifact_purity.md`** to ensure the agent uses the tool correctly:

* **Atomic Saves**: Never create a file without immediately calling `update_manifest` to register it.
* **Staging Isolation**: You are physically unable to "export" files in the root or `docs/`. They must be moved to `pulse_staging/artifacts/` first.
* **The "Dirty Disk" Error**: If the exporter returns a `BLOCKED` status, do not try to bypass it. You must fix the manifest-to-disk discrepancy before attempting again.

---

### 4. Summary of Structural Improvements

| Feature             | Legacy System                    | Vessel V2 (The Overhaul)            |
| ------------------- | -------------------------------- | ----------------------------------- |
| **Location**        | Root, `docs/`, `active_session/` | **ONLY** `pulse_staging/artifacts/` |
| **Integrity Check** | None (Manual summary)            | **Strict Disk vs. Manifest Audit**  |
| **Outcome**         | Narrative Handover               | **Validated Artifact Archive**      |
| **Failure Mode**    | Hallucinated progress            | **Hard Block on Export**            |

This refactor ensures that an AI agent cannot "lose" a file or generate it in the wrong place without the system flagging it immediately. The logic is handled by the Python backend, keeping the AI's prompt clean and focused on the technical implementation.
