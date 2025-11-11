---
title: "PLAN-005: Legacy Cleanup & Config Consolidation"
author: "ai-agent"
date: "2025-11-11"
type: "implementation_plan"
category: "architecture"
status: "draft"
version: "0.1"
tags: ["implementation_plan", "architecture", "cleanup", "consolidation", "low-risk"]
timestamp: "2025-11-11 02:00 KST"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **PLAN-005: Legacy Cleanup & Config Consolidation**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: PLAN-005: Legacy Cleanup & Config Consolidation

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Identify Backup Directories
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Locate backup directories for archiving

### Implementation Outline (Checklist)

#### **Phase 1: Archive Backup Directories**
1. [ ] **Task 1.1: Identify Backup Directories**
   - [ ] List `.backup/` directory contents: `ls -la .backup/`
   - [ ] Identify `scripts-backup-*` directories
   - [ ] Document backup directory structure
   - [ ] Verify backup directories are not actively used

2. [ ] **Task 1.2: Archive Backup Directories**
   - [ ] Create archive directory: `mkdir -p .archive/backups`
   - [ ] Move backup directories to archive: `mv .backup/scripts-backup-* .archive/backups/`
   - [ ] Verify archive was created successfully
   - [ ] Document archived directories

3. [ ] **Task 1.3: Validate Backup Archive**
   - [ ] Verify backup directories are in archive: `ls -la .archive/backups/`
   - [ ] Verify original directories are removed: `ls -la .backup/`
   - [ ] Check git status to confirm changes
   - [ ] Document archive location for future reference

#### **Phase 2: Remove Duplicate Scripts**
4. [ ] **Task 2.1: Identify Duplicate Scripts**
   - [ ] List `scripts/` directory: `ls -la scripts/`
   - [ ] Locate `scripts/debug_cuda.sh` if it exists
   - [ ] Search for Python version: `find scripts/ -name "*debug_cuda*.py"`
   - [ ] Document duplicate script locations

5. [ ] **Task 2.2: Verify Python Version Exists**
   - [ ] Check if Python debug_cuda script exists
   - [ ] Verify Python version is functional
   - [ ] Document which version to keep (Python preferred)
   - [ ] Verify no active references to shell script

6. [ ] **Task 2.3: Remove Duplicate Shell Script**
   - [ ] Remove `scripts/debug_cuda.sh` if Python version exists
   - [ ] Verify removal: `ls -la scripts/debug_cuda.sh` (should fail)
   - [ ] Check git status to confirm removal
   - [ ] Document removed script

#### **Phase 3: Consolidate Config Presets**
7. [ ] **Task 3.1: Identify Duplicate Config Files**
   - [ ] List config directories: `ls -la configs/`
   - [ ] Search for duplicate config patterns: `find configs/ -name "*.yaml" | sort`
   - [ ] Identify configs with similar names or purposes
   - [ ] Document duplicate config locations

8. [ ] **Task 3.2: Analyze Config File Differences**
   - [ ] Compare duplicate config files using diff
   - [ ] Identify which configs are truly duplicates
   - [ ] Document which configs to keep vs remove
   - [ ] Verify no active references to duplicate configs

9. [ ] **Task 3.3: Remove Duplicate Configs**
   - [ ] Remove duplicate config files
   - [ ] Verify removal: `ls -la configs/<removed_file>` (should fail)
   - [ ] Check git status to confirm removal
   - [ ] Document removed configs

#### **Phase 4: Update Documentation References**
10. [ ] **Task 4.1: Search for References to Removed Files**
    - [ ] Search for references to removed scripts: `grep -rn "debug_cuda.sh" docs/ scripts/`
    - [ ] Search for references to removed configs: `grep -rn "<removed_config>" docs/ configs/`
    - [ ] Search for references to backup directories: `grep -rn "scripts-backup" docs/`
    - [ ] Document all references found

11. [ ] **Task 4.2: Update Documentation**
    - [ ] Update documentation to remove references to deleted files
    - [ ] Update config references to use consolidated configs
    - [ ] Update script references to use Python versions
    - [ ] Verify documentation is accurate

12. [ ] **Task 4.3: Validate Documentation Updates**
    - [ ] Verify no broken references: `grep -rn "debug_cuda.sh\|scripts-backup" docs/`
    - [ ] Check that all config references are valid
    - [ ] Verify documentation links work correctly
    - [ ] Document updated references

#### **Phase 5: Final Validation and Cleanup**
13. [ ] **Task 5.1: Verify No Active References**
    - [ ] Search for references to removed files: `grep -rn "<removed_file>" .`
    - [ ] Verify no active code references deleted files
    - [ ] Check that git history preserves deleted files
    - [ ] Document any remaining references

14. [ ] **Task 5.2: Verify Git History**
    - [ ] Check git log for deleted files: `git log --all --full-history -- <deleted_file>`
    - [ ] Verify deleted files are in git history
    - [ ] Confirm files can be restored if needed
    - [ ] Document git history status

15. [ ] **Task 5.3: Run Final Validation**
    - [ ] Verify all removed files are documented
    - [ ] Check that documentation is updated
    - [ ] Confirm no broken references remain
    - [ ] Verify git status is clean

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Backup directories archived
- [ ] Duplicate scripts removed
- [ ] Duplicate configs consolidated
- [ ] Documentation updated

### **Integration Points**
- [ ] No active references to deleted files
- [ ] Config references updated
- [ ] Script references updated
- [ ] Documentation links work correctly

### **Quality Assurance**
- [ ] All removed files documented
- [ ] Git history preserves deleted files
- [ ] No broken references remain
- [ ] Documentation is accurate

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Backup directories archived successfully
- [ ] Duplicate scripts removed
- [ ] Duplicate configs consolidated
- [ ] Documentation references updated

### **Technical Requirements**
- [ ] No active references to deleted files
- [ ] Git history preserves deleted files
- [ ] Documentation is accurate
- [ ] All changes committed to git

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW ⚠️
### **Active Mitigation Strategies**:
1. **Archive Before Deletion**: Move files to archive before removing
2. **Git History**: Verify deleted files are in git history
3. **Reference Checking**: Verify no active references before deletion
4. **Documentation**: Update documentation to reflect changes

### **Fallback Options**:
1. **If deletion breaks something**: Restore from git history or archive
2. **If references break**: Restore deleted files, update references first
3. **If documentation is wrong**: Update documentation, verify accuracy
4. **If archive is needed**: Restore from archive directory

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Identify Backup Directories

**OBJECTIVE:** Locate backup directories that need to be archived before removal

**APPROACH:**
1. List `.backup/` directory contents: `ls -la .backup/`
2. Identify `scripts-backup-*` directories
3. Document backup directory structure
4. Verify backup directories are not actively used

**SUCCESS CRITERIA:**
- All backup directories identified
- Backup directory structure documented
- Ready to archive directories

---

## 📝 **Context Management for Web Workers**

### **Recommended Context Scope**
**Focus on ONE directory/file at a time** to avoid context overflow:

1. **Phase 1 (Backup Archive)**:
   - Context: Use terminal commands to list and move directories
   - Max files: 0 (terminal commands only)
   - Estimated tokens: ~500

2. **Phase 2 (Script Removal)**:
   - Context: One script file at a time
   - Max files: 1 per task
   - Estimated tokens: ~1000 per file

3. **Phase 3 (Config Consolidation)**:
   - Context: One config file at a time
   - Max files: 1 per task
   - Estimated tokens: ~1000 per file

4. **Phase 4 (Documentation)**:
   - Context: Use grep for pattern searches, read only if needed
   - Max files: 0 (grep only)
   - Estimated tokens: ~500

5. **Phase 5 (Validation)**:
   - Context: Use grep and git commands
   - Max files: 0 (commands only)
   - Estimated tokens: ~500

### **Context Optimization Strategies**
- **Use terminal commands** for file operations (ls, mv, rm)
- **Use grep** to search for references before reading files
- **Read only if needed** to verify content
- **Focus on specific files** rather than entire directories

### **Validation Commands (No Runtime Needed)**
```bash
# List directories
ls -la <directory>

# Find files
find <directory> -name "<pattern>"

# Search for references
grep -rn "<pattern>" <directory>

# Check git status
git status

# Check git history
git log --all --full-history -- <file>

# Verify file removal
ls -la <file>  # Should fail if removed
```

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

