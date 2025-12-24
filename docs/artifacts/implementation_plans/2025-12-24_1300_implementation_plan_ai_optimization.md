---
ads_version: "1.0"
type: implementation_plan
category: development
status: active
version: 1.1
tags:
  - environment
  - ai-optimization
title: AI Environment Optimization and Pyenv Fix
date: 2025-12-24 13:00 (KST)
branch: main
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **AI Environment Optimization and Pyenv Fix**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically.

---

# Living Implementation Blueprint: AI Environment Optimization

## Progress Tracker
- **STATUS:** In Progress
- **CURRENT STEP:** Phase 1, Task 1.1 - Permanent Pyenv Fix
- **LAST COMPLETED TASK:** Analysis
- **NEXT TASK:** Search and clean `.bashrc` / configuration for `PYENV_VERSION`

### Implementation Outline (Checklist)

#### **Phase 1: Permanent Pyenv Fix (Immediate)**
1. [ ] **Task 1.1: Locate and Remove Variable**
   - [ ] Inspect `.bashrc`, `.bash_profile`, `.profile`, and workspace settings for `export PYENV_VERSION=doc-pyenv`.
   - [ ] Remove the offending line or add an explicit unset to `.bashrc` if inherited from container.
   - [ ] Verification: `echo $PYENV_VERSION` should be empty or valid in a new shell.

#### **Phase 2: AI Environment Optimization**
2. [ ] **Task 2.1: Establish AI Entry Point (`AGENTS.md`)**
   - [ ] Create `AGENTS.md` in root.
   - [ ] Content: Concise pointer to `.ai-instructions/INDEX.yaml` and brief "Context Map" explanation. No verbose prose.

3. [ ] **Task 2.2: Update README**
   - [ ] Add a discrete link in `README.md` pointing to `AGENTS.md` (e.g., "AI Agents: See [AGENTS.md](AGENTS.md)").

---

## 🎯 **Success Criteria Validation**

- [ ] `doc-pyenv` error is gone permanently (verified by checking config files).
- [ ] `AGENTS.md` exists and points clearly to `.ai-instructions/`.

---

## 🚀 **Immediate Next Action**

**TASK:** Task 1.1: Locate and Remove Variable

**OBJECTIVE:** Find where `PYENV_VERSION` is being set and remove it.

**APPROACH:**
1. Grep for `PYENV_VERSION` and `doc-pyenv` in home directory and workspace.
2. Edit the file to remove/fix it.

---
