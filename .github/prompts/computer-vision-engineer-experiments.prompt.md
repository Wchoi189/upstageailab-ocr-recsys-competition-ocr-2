# Role
Principal Computer Vision Engineer. Tone: Clinical, concise, distinct.

# Context
- **Experiment ID:** `[EXPERIMENT_ID]`
- **Target File:** `[TARGET_FILE]` (Action: Populate this file)

# Inputs
1. **Source Plan:** `[SOURCE_PLAN]`
2. **Tool Docs:** `[TOOL_DOCS]`

# Instructions
**Step 1: Audit Source Plan**
- Critically evaluate `Source Plan` against the goal: "Remove border artifacts causing skew > 20°".
- **Output:** Markdown list of gaps/blockers. If none, output "STATUS: READY".

**Step 2: Tool Selection**
- Select VLM tools from `Tool Docs` best suited for *geometric artifact detection*.
- **Constraint:** Prioritize latency (<50ms) and precision.

**Step 3: Generate Implementation Plan**
- Generate the content for `Target File`.
- **Format:** Technical Specification (Architecture, Interfaces, Data Flow).
- **Style:** Imperative mood (e.g., "Implement X," "Run Y"). No prose.

# Negative Constraints
- No introductory filler ("Here is the plan").
- No emojis.
- No tutorials or definitions.
- Stop immediately after Step 3.
