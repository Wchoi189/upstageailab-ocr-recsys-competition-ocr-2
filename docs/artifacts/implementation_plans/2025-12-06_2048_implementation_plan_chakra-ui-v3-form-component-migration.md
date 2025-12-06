---
title: "Chakra Ui V3 Form Component Migration"
date: "2025-12-06 20:48 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Chakra UI v3 Form Component Migration**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Migration completed successfully").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Chakra UI v3 Form Component Migration

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Phase 2 Complete - Form Migration Successful
- **CURRENT STEP:** Phase 2 Complete – SchemaForm.tsx successfully migrated to Field API
- **LAST COMPLETED TASK:** Phase 2 (Task 2.1, 2.2, 2.3) – Migrated all form components (Wrapper, TextInput, Number, Checkbox, Select, Slider) to Chakra UI v3 Field API
- **NEXT TASK:** Phase 3 - Testing & Validation (pending separate session for remaining Tab component errors)
- **BLOCKER:** Chakra UI v3 compatibility - Phase 3 components use deprecated FormControl/FormLabel/FormHelperText that need migration to Field component

### Implementation Outline (Checklist)

#### **Phase 1: Research & Preparation (Day 1)**
1. [ ] **Task 1.1: Research Chakra UI v3 Field Component API**
   - [ ] Review Chakra UI v3 documentation for Field component
   - [ ] Identify migration path from FormControl/FormLabel/FormHelperText to Field API
   - [ ] Document breaking changes and new patterns
   - [ ] Create migration reference guide

2. [ ] **Task 1.2: Audit Current Form Component Usage**
   - [ ] Search codebase for all FormControl/FormLabel/FormHelperText usage
   - [ ] Identify all affected files (primary: SchemaForm.tsx)
   - [ ] Document form component patterns used
   - [ ] Create inventory of migration targets

3. [ ] **Task 1.3: Set Up Testing Strategy**
   - [ ] Identify test cases for form functionality
   - [ ] Set up test environment for visual regression testing
   - [ ] Create checklist for validation criteria

#### **Phase 2: Core Migration (Day 2)**
4. [x] **Task 2.1: Migrate Wrapper Component**
   - [x] Update `Wrapper` function in SchemaForm.tsx to use Field.Root
   - [x] Replace FormLabel with Field.Label
   - [x] Replace FormHelperText with Field.HelperText
   - [x] Replace error display with Field.ErrorText
   - [x] Test component rendering

5. [x] **Task 2.2: Update Field Components**
   - [x] Migrate TextInputField to use Field API
   - [x] Migrate NumberField to use Field API (NumberInput.Root + NumberInput.Input)
   - [x] Migrate CheckboxField to use Field API
   - [x] Migrate SelectField to use Field API
   - [x] Migrate SliderField to use Field API (Slider.Root + Slider.Track + Slider.Range + Slider.Thumb)
   - [x] Verify InfoField doesn't need changes (no changes needed)

6. [x] **Task 2.3: Fix TypeScript Types**
   - [x] Update imports to use Field component from @chakra-ui/react
   - [x] Remove deprecated FormControl, FormLabel, FormHelperText imports
   - [x] Fix TypeScript errors (all resolved)
   - [x] Verify type safety (no linter errors)

#### **Phase 3: Testing & Validation (Day 3)**
7. [ ] **Task 3.1: Visual Testing**
   - [ ] Test form rendering in development mode
   - [ ] Verify all form fields render correctly
   - [ ] Check spacing and layout consistency
   - [ ] Verify error states display correctly

8. [ ] **Task 3.2: Functional Testing**
   - [ ] Test form value updates
   - [ ] Test form validation
   - [ ] Test conditional visibility
   - [ ] Test help text display
   - [ ] Test error message display

9. [ ] **Task 3.3: Build Verification**
   - [ ] Run `npm run build` in playground-console
   - [ ] Verify production build completes successfully
   - [ ] Check for any build warnings
   - [ ] Verify bundle size is acceptable

#### **Phase 4: Cleanup & Documentation (Day 4)**
10. [ ] **Task 4.1: Code Cleanup**
    - [ ] Remove unused imports
    - [ ] Ensure consistent code style
    - [ ] Update comments if needed
    - [ ] Run linter and fix issues

11. [ ] **Task 4.2: Documentation Update**
    - [ ] Update Next.js migration plan to mark blocker as resolved
    - [ ] Document migration patterns for future reference
    - [ ] Update any relevant documentation

12. [ ] **Task 4.3: Final Verification**
    - [ ] Run full test suite
    - [ ] Verify production build one more time
    - [ ] Test in actual browser environment
    - [ ] Confirm all blockers are resolved

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Use Chakra UI v3 Field component API (Field.Root, Field.Label, Field.HelperText, Field.ErrorText)
- [ ] Maintain existing form functionality and behavior
- [ ] Preserve TypeScript type safety
- [ ] Follow Chakra UI v3 best practices

### **Integration Points**
- [ ] SchemaForm component works with existing CommandSchema types
- [ ] Form integration with CommandBuilderClient unchanged
- [ ] Error handling works correctly
- [ ] Visibility evaluation logic remains functional

### **Quality Assurance**
- [ ] All form fields render correctly visually
- [ ] Form functionality works as expected
- [ ] Production build completes without errors (0 build errors)
- [ ] No TypeScript errors
- [ ] No runtime errors in development or production

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] All form fields render correctly with proper labels and help text
- [ ] Form value updates work correctly
- [ ] Form validation displays errors correctly
- [ ] Conditional visibility works as expected
- [ ] Form submission and command building works end-to-end

### **Technical Requirements**
- [ ] `npm run build` completes successfully with 0 errors
- [ ] All TypeScript types are correct
- [ ] No deprecated Chakra UI APIs are used
- [ ] Code follows project conventions
- [ ] Bundle size is acceptable (no significant increase)

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM

### **Active Mitigation Strategies**:
1. **Incremental Migration**: Migrate one field type at a time, test after each
2. **Comprehensive Testing**: Test both visually and functionally after migration
3. **Reference Documentation**: Keep Chakra UI v3 docs open during migration
4. **Version Control**: Commit after each successful field type migration

### **Fallback Options**:
1. If Field component API doesn't work as expected, research alternative patterns
2. If migration breaks critical functionality, revert and document issues
3. If TypeScript types are problematic, use type assertions as temporary fix

### **Known Risks**:
- **API Changes**: Field component API might differ from FormControl in unexpected ways
- **Styling Differences**: Visual appearance might change slightly (should be minimal)
- **Type Compatibility**: TypeScript types might need adjustment

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

**TASK:** Research Chakra UI v3 Field component API

**OBJECTIVE:** Understand the Field component API and create a migration strategy from FormControl/FormLabel/FormHelperText to Field components.

**APPROACH:**
1. Review Chakra UI v3 documentation for Field component
   - Check official docs: https://chakra-ui.com/docs/components/field
   - Understand Field.Root, Field.Label, Field.HelperText, Field.ErrorText API
   - Review examples and migration guides
2. Identify all current FormControl usage patterns in SchemaForm.tsx
   - Document how FormControl, FormLabel, FormHelperText are currently used
   - Note any special props or patterns
3. Create migration mapping:
   - FormControl → Field.Root
   - FormLabel → Field.Label
   - FormHelperText → Field.HelperText
   - Error Text → Field.ErrorText
4. Verify Field component is available in @chakra-ui/react v3.29.0

**SUCCESS CRITERIA:**
- ✅ Migration strategy documented
- ✅ Field component API understood
- ✅ SchemaForm migration completed successfully

---

## ✅ **Phase 2 Completion Summary (2025-11-20)**

**Completed Tasks:**
- ✅ Task 2.1: Wrapper component migrated (FormControl → Field.Root)
- ✅ Task 2.2: All field components migrated (TextInput, Number, Checkbox, Select, Slider)
- ✅ Task 2.3: TypeScript types fixed, imports updated

**Key Changes:**
- `FormControl` → `Field.Root`
- `FormLabel` → `Field.Label`
- `FormHelperText` → `Field.HelperText`
- Error text → `Field.ErrorText`
- `NumberInputField` → `NumberInput.Input` (within `NumberInput.Root`)
- `SliderTrack/SliderFilledTrack/SliderThumb` → `Slider.Root/Slider.Track/Slider.Range/Slider.Thumb`

**Results:**
- Form-related build errors: **27 → 0** (all resolved for SchemaForm.tsx)
- Remaining build errors: **16** (Tab components and GTM provider - separate issue)
- File: `apps/playground-console/src/components/command-builder/SchemaForm.tsx` ✅ Complete

**Next Steps:**
- Remaining errors are in Tab components (CommandBuilderClient.tsx, PreviewPanel.tsx) and GTM provider
- See separate implementation plan: Chakra UI v3 Tab Component Migration & Build Fix

---

## 📚 **Reference Information**

### **Current Blocker Details:**
- **File:** `apps/playground-console/src/components/command-builder/SchemaForm.tsx`
- **Lines Affected:** 116-134 (Wrapper component), and all field renderer functions
- **Build Errors:** 27 errors related to FormControl/FormLabel/FormHelperText exports not existing
- **Chakra UI Version:** @chakra-ui/react v3.29.0

### **Migration Mapping (Initial Research):**
- `FormControl` → `Field.Root`
- `FormLabel` → `Field.Label`
- `FormHelperText` → `Field.HelperText`
- Error Text (custom) → `Field.ErrorText`

### **Related Documentation:**
- Next.js Migration Plan: `artifacts/implementation_plans/2025-11-19_1957_next.js-console-migration-and-chakra-adoption-plan.md`
- Chakra UI v3 Docs: https://chakra-ui.com/docs/components/field

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
