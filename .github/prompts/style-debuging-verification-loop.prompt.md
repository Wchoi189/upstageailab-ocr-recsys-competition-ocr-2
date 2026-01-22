name: The Verification Loop
description: When presenting this to the AI agent, use a **"Machine-Parse-able List"**. Instead of asking to "run the script," give it the **Verification Loop** so it can self-correct.
---

#### **Instruction Style: "The Verification Loop"**

"The refactor has caused module drift. Use the following **Healing Protocol**:
1. **Run** `python scripts/audit/auto_align_hydra.py`.
2. **Review** the output for any `❌ Could not locate` errors.
3. **For failures:** Manually use `adt intelligent-search` to find the missing symbol.
4. **Validate:** Run `python scripts/audit/master_audit.py` again. The 'Broken Hydra Targets' count must be 0 before proceeding to Phase 2."
