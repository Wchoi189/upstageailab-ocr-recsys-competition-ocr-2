---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Complete AgentQMS Toolkit Deprecation and Migration (Phase 2)"
date: "2025-12-06 20:09 (KST)"
branch: "feature/outputs-reorg"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Complete AgentQMS Toolkit Deprecation and Migration (Phase 2)**. Your primary responsibility is to systematically migrate the remaining 15 wrapper modules and archive the deprecated toolkit. Execute the Living Implementation Blueprint step-by-step, handle outcomes, and track progress.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear goal will be provided to achieve
2. **Execute:** Start working on the NEXT TASK
3. **Handle Outcome & Update:** Summarize results and update progress

---

# Living Implementation Blueprint: Complete AgentQMS Toolkit Deprecation and Migration (Phase 2)

## Progress Tracker
- **STATUS:** Phase 1 Complete ✅ | Phase 2 In Planning
- **PHASE 1 COMPLETION DATE:** 2025-12-06
- **CURRENT STEP:** Phase 2 - Migrate remaining wrapper modules
- **LAST COMPLETED TASK:** Migrated FrontmatterGenerator and updated primary imports
- **NEXT TASK:** Migrate remaining 15 wrapper modules from toolkit imports

### Implementation Outline (Checklist)

#### **Phase 1: Foundation (COMPLETED ✅)**
1. [x] **Task 1.1: Migrate FrontmatterGenerator**
   - [x] Copy FrontmatterGenerator to AgentQMS/agent_tools/maintenance/add_frontmatter.py
   - [x] Create proper __init__.py for module
   - [x] Update artifact_audit.py to import from agent_tools

2. [x] **Task 1.2: Update Primary Imports**
   - [x] artifact_audit.py: toolkit.maintenance → agent_tools.maintenance
   - [x] CLI tools (quality.py, feedback.py, ast_analysis.py, audio/agent_audio_mcp.py): toolkit.utils → agent_tools.utils
   - [x] test_branch_metadata.py: toolkit.core → agent_tools.core
   - [x] Verified no DeprecationWarning in user code

#### **Phase 2: Wrapper Module Migration (CURRENT)**
3. [ ] **Task 2.1: Migrate audit wrapper modules**
   - [ ] checklist_tool.py: Move implementation from toolkit, update imports
   - [ ] audit_validator.py: Move implementation from toolkit, update imports
   - [ ] audit_generator.py: Move implementation from toolkit, update imports

4. [ ] **Task 2.2: Migrate core wrapper modules**
   - [ ] artifact_templates.py: Move implementation from toolkit, update imports

5. [ ] **Task 2.3: Migrate documentation wrapper modules**
   - [ ] auto_generate_index.py: Move implementation from toolkit, update imports
   - [ ] validate_links.py: Move implementation from toolkit, update imports
   - [ ] validate_manifest.py: Move implementation from toolkit, update imports

6. [ ] **Task 2.4: Migrate utilities wrapper modules**
   - [ ] adapt_project.py: Move implementation from toolkit, update imports
   - [ ] tracking_integration.py: Move implementation from toolkit, update imports
   - [ ] tracking/cli.py: Move implementation from toolkit, update imports

#### **Phase 3: Archive & Cleanup (PENDING)**
7. [ ] **Task 3.1: Verify all imports migrated**
   - [ ] Run grep to ensure no remaining toolkit imports
   - [ ] Verify all migration mappings documented

8. [ ] **Task 3.2: Archive toolkit directory**
   - [ ] Move AgentQMS/toolkit → AgentQMS/.old_toolkit
   - [ ] Update .gitignore if needed

9. [ ] **Task 3.3: Final validation & documentation**
   - [ ] Run full compliance check: `make validate && make compliance`
   - [ ] Update CHANGELOG.md with toolkit removal notice
   - [ ] Document migration path for external users

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Modular design: Each module in agent_tools has clear responsibility
- [x] Full feature parity: All migrated code maintains 100% functionality
- [x] Backward compatibility: Wrapper modules still work during transition
- [x] Import path consistency: All new code uses AgentQMS.agent_tools.X pattern

### **Integration Points**
- [x] No breaking changes to public APIs during Phase 1
- [x] All internal tools use agent_tools imports (artifact_audit.py verified)
- [x] CLI tools properly structured with ensure_project_root_on_sys_path

### **Quality Assurance**
- [x] No DeprecationWarning in primary user code
- [ ] 100% of code migrated before toolkit removal
- [ ] Full compliance check passes before archival
- [ ] All tests pass with new import structure

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [x] FrontmatterGenerator works with new imports
- [x] artifact_audit.py functions correctly without deprecation warning
- [ ] All 15 wrapper modules migrated and functional
- [ ] No toolkit imports remain in production code

### **Technical Requirements**
- [x] Code quality maintained (no regression)
- [x] Type hints preserved in all migrated code
- [x] Import paths follow agent_tools convention
- [ ] Full compliance check score ≥ 90%

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. **Incremental Migration:** Complete Phase 1 before Phase 2 (reduces risk)
2. **Backward Compatibility:** Toolkit stays in place during wrapper migration
3. **Comprehensive Testing:** Verify each migration with test runs
4. **Documentation:** Clear migration guide for future reference

### **Fallback Options**:
1. If wrapper migration fails: Keep toolkit in place, mark as legacy indefinitely
2. If compliance check fails: Identify and fix remaining issues before archival
3. If external code breaks: Maintain compatibility wrapper module

---

## 🔄 **Migration Status by Module**

### **Completed (Phase 1) - 1/16 Modules**
- [x] add_frontmatter.py ✅ (FrontmatterGenerator)

### **Pending (Phase 2) - 15/16 Modules**
- [ ] checklist_tool.py (audit)
- [ ] audit_validator.py (audit)
- [ ] audit_generator.py (audit)
- [ ] artifact_templates.py (core)
- [ ] auto_generate_index.py (documentation)
- [ ] validate_links.py (documentation)
- [ ] validate_manifest.py (documentation)
- [ ] adapt_project.py (utilities)
- [ ] tracking_integration.py (utilities)
- [ ] tracking/cli.py (utilities/tracking)
- [ ] (And 5 more internal toolkit modules with cross-dependencies)

---

## 🚀 **Immediate Next Action**

**TASK:** Migrate Phase 2 wrapper modules

**OBJECTIVE:** Migrate all 15 remaining wrapper modules from toolkit to agent_tools, preserving functionality

**APPROACH:**
1. For each wrapper module in agent_tools:
   - Copy the underlying implementation from toolkit
   - Update internal imports to use agent_tools paths
   - Create proper __init__.py if needed
   - Test with quick verification run
2. Run full grep to find any remaining toolkit imports
3. Run compliance check to verify migration success
4. Document any issues or special cases

**SUCCESS CRITERIA:**
- All 15 wrapper modules successfully migrated
- No "ModuleNotFoundError: No module named 'AgentQMS.toolkit'" errors
- Full compliance check passes
- Zero toolkit imports in production code

---

## 📝 **Implementation Notes**

**Decision Log:**
- Phase 1 chose to migrate FrontmatterGenerator first (highest priority)
- Using full migration approach (not wrapper approach) for cleaner long-term solution
- Toolkit directory retained during Phase 2 to support wrapper module dependencies

**Known Blockers:**
- 15 wrapper modules have internal dependencies on toolkit (cascading imports)
- Solution: Migrate in dependency order (lowest dependencies first)

**Testing Strategy:**
- Run `artifact_audit.py --all --report` after each batch of migrations
- Run `make compliance` before and after toolkit archival
- Verify no regressions in existing functionality

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*