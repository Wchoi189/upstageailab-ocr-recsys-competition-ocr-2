## Audit-Driven Investigation Approach

### Strategy: Systematic Evidence Gathering

**Core Philosophy:** Build complete understanding before proposing solutions

### Investigation Phases

#### Phase 1: Configuration Archaeology (Evidence Collection)
**Goal:** Document what EXISTS in configs, not what should exist

**Method:**
1. Hydra composition analysis for both detection and recognition
2. Side-by-side structural comparison
3. Catalog ALL config keys at each tier (model, train, data, domain)
4. Trace interpolation chains end-to-end
5. Document defaults hierarchy

**Deliverable:** Config comparison matrix

---

#### Phase 2: Code Structure Analysis (Component Mapping)
**Goal:** Understand recognition domain implementation patterns

**Method:**
1. Dependency graph analysis (identify coupling)
2. AST dumps for architecture classes (PARSeq, components)
3. Constructor signature analysis (expected vs actual params)
4. Hydra instantiation pattern inventory
5. Module registration verification

**Deliverable:** Component dependency map + interface contracts

---

#### Phase 3: Reference Comparison (V5 Gold Standard)
**Goal:** Identify deviations from detection pipeline patterns

**Method:**
1. Detection config as V5 reference
2. Pattern matching: atomic vs legacy instantiation
3. Structural alignment check (same tier organization?)
4. Missing component identification
5. Design intent inference from detection

**Deliverable:** V5 compliance gap analysis

---

#### Phase 4: Error Trace Analysis (Symptom Investigation)
**Goal:** Understand actual vs reported failures

**Method:**
1. Reproduce optimizer error with instrumentation
2. Add debug logging to orchestrator, PARSeq.__init__
3. Dry-run component instantiation in isolation
4. Forward pass simulation with None components
5. Trace execution path to actual failure point

**Deliverable:** Root cause determination

---

#### Phase 5: Historical Context Review (Design Intent)
**Goal:** Understand WHY recognition was deferred

**Method:**
1. Review 2026-01-16 debug session (pre-refactor)
2. Check refactor tracking notes on recognition
3. Identify architectural concerns vs config issues
4. Separate "deferred" (intentional) from "broken" (bugs)

**Deliverable:** Context document with design decisions

---

### Audit Execution Plan

**Week 1: Data Gathering**
- Days 1-2: Phase 1 (Config archaeology)
- Days 3-4: Phase 2 (Code structure)
- Day 5: Phase 3 (Reference comparison)

**Week 2: Analysis & Planning**
- Days 1-2: Phase 4 (Error traces)
- Day 3: Phase 5 (Historical context)
- Days 4-5: Synthesis + implementation plan creation

### Success Criteria

**Audit Complete When:**
1. ✅ All config differences documented
2. ✅ All component interfaces mapped
3. ✅ All V5 deviations identified
4. ✅ Root cause conclusively determined
5. ✅ Design intent understood
6. ✅ Implementation plan boundaries clear

### Risk Mitigation

**Risk:** Scope creep into architectural redesign
**Mitigation:** Strict boundary enforcement - only V5 structural fixes

**Risk:** Missing context from absent documentation
**Mitigation:** Infer from code + detection reference, document assumptions

**Risk:** Overwhelming detail paralysis
**Mitigation:** Time-box each phase, focus on actionable findings

### Tooling Strategy

**Primary Tools:**
- ADT meta queries (automated analysis)
- Hydra composition (ground truth configs)
- Custom diagnostic scripts (targeted investigation)

**Secondary Tools:**
- Existing audit scripts (validation)
- Debug session artifacts (historical context)
- Manual code review (gap filling)

### Deliverables

1. **Audit Report** - Comprehensive findings document
2. **Component Inventory** - What exists, what's missing, what's broken
3. **V5 Compliance Matrix** - Detection vs Recognition comparison
4. **Implementation Plan** - Scoped fixes with test strategy
5. **Knowledge Base** - Artifacts for future troubleshooting
