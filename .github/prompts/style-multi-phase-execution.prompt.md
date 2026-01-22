name: Multi-Phase Execution Style
description: Enforces structured, multi-phase execution for complex tasks.

---

**Step 1: The Request Template**

**Goal:** [Specific technical objective]
**Constraint:** [Specific constraints, e.g., "Do not guess. Only use the provided `MAPPING_LIST`."]
**Reference Roadmap:** [Insert your #1 Friction Report here]
**Input Data:** [Insert the JSON output of your ADT `analyze-dependencies` tool]

**Step 2: Choosing the Style**

* **Audit Type:** Use **Pure Log Output** for discovery (to see the scale).
* **Instruction Style:** Use **Direct Execution Oriented** (e.g., "Apply these 5 diffs").
* **Presentation:** Present the **Roadmap** as a set of **Guardrails**.
* *Example:* "Guardrail 1: If an import starts with `[OLD_PATH]`, it MUST be changed to `[NEW_PATH]`. Any deviation is a failure."
