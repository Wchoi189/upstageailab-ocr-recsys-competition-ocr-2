---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'architecture', 'development']
title: "Domain-Driven Separation for AgentQMS Dashboard Integration"
date: "2025-12-14 02:20 (KST)"
branch: "feature/agentqms-dashboard-integration"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Domain-Driven Separation for AgentQMS Dashboard Integration**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Domain-Driven Separation for AgentQMS Dashboard Integration

## Progress Tracker
- **STATUS:** Planning Phase
- **CURRENT STEP:** Phase 1, Task 1.1 - Architecture Analysis
- **LAST COMPLETED TASK:** None (Initial Plan)
- **NEXT TASK:** Analyze current architecture and identify domain boundaries
- **BRANCH STRATEGY:** Work on `feature/agentqms-dashboard-integration` branch for all implementation

### Implementation Outline (Checklist)

#### **Phase 1: Architecture Analysis & Documentation (Week 1)**
1. [ ] **Task 1.1: Domain Boundary Identification**
   - [ ] Document existing application domains (OCR Playground, AgentQMS Dashboard, Playground Console)
   - [ ] Identify shared dependencies and coupling points
   - [ ] Map current directory structure to domain responsibilities
   - [ ] Analyze backend service separation (`apps/backend` vs `apps/agentqms-dashboard/backend`)

2. [ ] **Task 1.2: Domain Model Definition**
   - [ ] Define bounded contexts for each domain (OCR, QMS, Common)
   - [ ] Specify domain interfaces and contracts
   - [ ] Document integration points between domains
   - [ ] Create domain architecture diagrams

3. [ ] **Task 1.3: Gap Analysis**
   - [ ] Identify architectural violations (cross-domain dependencies)
   - [ ] Document missing abstractions or interfaces
   - [ ] List technical debt related to domain coupling
   - [ ] Prioritize refactoring needs

#### **Phase 2: Backend Separation (Week 2-3)**
4. [ ] **Task 2.1: Backend Service Architecture**
   - [ ] Define clear separation between OCR services and QMS services
   - [ ] Establish shared utilities layer (path resolution, logging, config)
   - [ ] Create domain-specific router organization
   - [ ] Document API versioning strategy

5. [ ] **Task 2.2: AgentQMS Backend Bridge Enhancement**
   - [ ] Expand minimal `apps/agentqms-dashboard/backend/server.py` stub
   - [ ] Implement artifact management API endpoints (aligned with DATA_CONTRACTS.md)
   - [ ] Add compliance validation endpoints
   - [ ] Integrate with AgentQMS agent_tools (validation, tracking)

6. [ ] **Task 2.3: Shared Infrastructure**
   - [ ] Extract common middleware (CORS, logging, error handling)
   - [ ] Create shared data models (Pydantic schemas)
   - [ ] Implement shared utilities (path resolution, config loading)
   - [ ] Establish consistent error handling patterns

#### **Phase 3: Frontend Separation (Week 3-4)**
7. [ ] **Task 3.1: Frontend Architecture Review**
   - [ ] Document frontend domains (OCR Playground UI, AgentQMS Dashboard UI, Playground Console)
   - [ ] Review current component organization and patterns
   - [ ] Identify shared UI components and utilities
   - [ ] Assess state management approaches

8. [ ] **Task 3.2: Shared Component Library**
   - [ ] Extract common UI components (buttons, cards, forms, layouts)
   - [ ] Create shared TypeScript types and interfaces
   - [ ] Establish shared styling approach (Tailwind, Chakra UI)
   - [ ] Document component usage patterns

9. [ ] **Task 3.3: API Client Separation**
   - [ ] Create domain-specific API client modules
   - [ ] Implement OCR-specific API client (`apps/frontend/src/api`)
   - [ ] Implement QMS-specific API client (`apps/agentqms-dashboard/frontend/services`)
   - [ ] Document API integration patterns

#### **Phase 4: Integration & Testing (Week 4-5)**
10. [ ] **Task 4.1: Integration Point Validation**
    - [ ] Test OCR Playground backend with OCR frontend
    - [ ] Test AgentQMS backend with AgentQMS dashboard frontend
    - [ ] Verify CORS configuration for all domains
    - [ ] Test shared utilities across domains

11. [ ] **Task 4.2: Domain Contract Testing**
    - [ ] Create contract tests for OCR API endpoints
    - [ ] Create contract tests for QMS API endpoints
    - [ ] Test API versioning and backward compatibility
    - [ ] Document breaking changes and migration paths

12. [ ] **Task 4.3: End-to-End Workflow Testing**
    - [ ] Test complete OCR training workflow (UI → API → Backend)
    - [ ] Test complete QMS artifact workflow (Dashboard → API → Validation)
    - [ ] Test cross-domain scenarios (if any)
    - [ ] Performance test with realistic loads

#### **Phase 5: Documentation & Finalization (Week 5-6)**
13. [ ] **Task 5.1: Architecture Documentation**
    - [ ] Create comprehensive architecture documentation
    - [ ] Document domain boundaries and responsibilities
    - [ ] Create deployment diagrams for each domain
    - [ ] Document API contracts and versioning

14. [ ] **Task 5.2: Developer Guidelines**
    - [ ] Create domain-specific development guides
    - [ ] Document how to add new features to each domain
    - [ ] Create troubleshooting guides
    - [ ] Update README files for each domain

15. [ ] **Task 5.3: Migration Guide**
    - [ ] Document changes from previous architecture
    - [ ] Create migration guide for developers
    - [ ] Document deprecated patterns and alternatives
    - [ ] Create FAQ for common questions

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Clear bounded contexts for each domain (OCR, QMS, Common/Shared)
- [ ] Domain-driven directory structure following DDD principles
- [ ] Pydantic V2 models for all API contracts
- [ ] YAML-driven configuration for each service
- [ ] Consistent error handling across all domains
- [ ] API versioning strategy (v1, v2, etc.)
- [ ] Separation of concerns: routes → services → domain logic

### **Integration Points**
- [ ] Integration with AgentQMS agent_tools (validation, compliance, tracking)
- [ ] Integration with OCR training pipeline (playground API)
- [ ] FastAPI routers for each domain service
- [ ] CORS configuration for cross-origin requests (frontend ↔ backend)
- [ ] Shared utilities layer (path resolution, logging, config)
- [ ] Use existing `ocr.utils.path_utils.setup_project_paths` for OCR domain
- [ ] Use AgentQMS validation tools for QMS domain

### **Quality Assurance**
- [ ] Contract tests for all API endpoints (>80% coverage)
- [ ] Integration tests for domain interactions
- [ ] End-to-end tests for critical workflows
- [ ] Performance tests: API response time <500ms for 95th percentile
- [ ] Security: Input validation, path traversal prevention
- [ ] Documentation: API docs auto-generated (FastAPI /docs)

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] OCR Playground operates independently with clear API boundaries
- [ ] AgentQMS Dashboard operates independently with artifact management API
- [ ] Shared utilities can be used by both domains without coupling
- [ ] All existing workflows continue to work after refactoring
- [ ] New features can be added to each domain without affecting others
- [ ] API endpoints follow consistent naming and versioning conventions

### **Technical Requirements**
- [ ] Code is fully type-hinted (Python) and typed (TypeScript)
- [ ] All API endpoints documented in OpenAPI specification
- [ ] Integration tests pass with >80% coverage
- [ ] Performance metrics met (API <500ms, frontend load <2s)
- [ ] No circular dependencies between domains
- [ ] Clear separation: `apps/backend/services/playground_api` (OCR domain), `apps/agentqms-dashboard/backend` (QMS domain)
- [ ] Shared code in dedicated common library (if needed)
- [ ] Documentation covers all domain boundaries and integration points

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
**Rationale**: Significant refactoring of existing working systems (OCR playground, QMS dashboard). Risk of breaking existing workflows while separating domains.

### **Active Mitigation Strategies**:
1. **Incremental Refactoring**: Start with documentation and analysis before code changes
2. **Feature Branch Isolation**: All work on `feature/agentqms-dashboard-integration` branch
3. **Contract-First Design**: Define interfaces before implementation
4. **Comprehensive Testing**: Integration and contract tests for all API endpoints
5. **Backward Compatibility**: Maintain existing APIs during migration

### **Identified Risks & Fallbacks**:

#### **Risk 1: Breaking Existing OCR Playground Functionality**
- **Probability**: Medium (refactoring shared utilities and backend structure)
- **Impact**: High (breaks production OCR training workflows)
- **Mitigation**: 
  - Test existing OCR workflows before and after each change
  - Keep existing API endpoints unchanged during separation
  - Add integration tests for critical OCR workflows
- **Fallback**: Revert domain separation and use namespace-based organization instead

#### **Risk 2: AgentQMS Dashboard Integration Complexity**
- **Probability**: Medium (minimal backend exists, needs full implementation)
- **Impact**: Medium (delays dashboard but doesn't affect OCR)
- **Mitigation**:
  - Use existing AgentQMS agent_tools as foundation
  - Follow DATA_CONTRACTS.md specification
  - Start with read-only endpoints (artifact listing) before write operations
- **Fallback**: Phase dashboard integration, focus on OCR domain separation first

#### **Risk 3: Shared Code Duplication**
- **Probability**: Low-Medium (temptation to duplicate rather than share properly)
- **Impact**: Medium (technical debt, maintenance overhead)
- **Mitigation**:
  - Create explicit shared utilities package
  - Document what should be shared vs domain-specific
  - Code review focusing on DRY principles
- **Fallback**: Accept some duplication if coupling is worse than duplication

#### **Risk 4: Performance Degradation**
- **Probability**: Low (adding layers of abstraction)
- **Impact**: Medium (slower API responses, worse UX)
- **Mitigation**:
  - Profile API performance before and after changes
  - Use async/await patterns throughout
  - Implement caching where appropriate
- **Fallback**: Optimize hot paths, remove unnecessary abstractions

#### **Risk 5: Documentation Drift**
- **Probability**: High (complex refactoring, multiple domains)
- **Impact**: Low-Medium (confusion, slower onboarding)
- **Mitigation**:
  - Update documentation concurrently with code changes
  - Use AgentQMS artifact workflow for tracking
  - Create architecture decision records (ADRs)
- **Fallback**: Conduct documentation sprint at end of project

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

**TASK:** Phase 1, Task 1.1 - Domain Boundary Identification

**OBJECTIVE:** Document and analyze the current application architecture to identify clear domain boundaries and coupling points between OCR Playground, AgentQMS Dashboard, and shared infrastructure.

**APPROACH:**
1. **Directory Structure Analysis**:
   ```bash
   # Map current application structure
   tree -L 3 apps/
   
   # Identify Python modules and their dependencies
   find apps/ -name "*.py" -type f | head -20
   
   # Check TypeScript/React structure
   find apps/ -name "*.tsx" -o -name "*.ts" | head -20
   ```

2. **Dependency Mapping**:
   ```bash
   # Check imports in OCR backend
   grep -r "^from\|^import" apps/backend/services/playground_api/ | head -30
   
   # Check imports in AgentQMS backend
   grep -r "^from\|^import" apps/agentqms-dashboard/backend/
   
   # Identify shared dependencies
   grep -r "ocr\." apps/agentqms-dashboard/ 2>/dev/null
   ```

3. **API Endpoint Inventory**:
   - Document all OCR Playground API endpoints (`apps/backend/services/playground_api/routers/`)
   - Document AgentQMS Dashboard API (minimal in `apps/agentqms-dashboard/backend/server.py`)
   - Identify port configurations and CORS settings

4. **Create Architecture Documentation**:
   - Create domain diagram showing current state
   - List all integration points and dependencies
   - Identify architectural violations (if any)
   - Document in `docs/architecture/` or as assessment artifact

**SUCCESS CRITERIA:**
- ✅ Complete inventory of all applications and their purposes
- ✅ Dependency graph showing which modules import from which domains
- ✅ List of shared utilities and their current locations
- ✅ Documentation of current API endpoints and their domains
- ✅ Identification of at least 3 architectural issues or improvement opportunities

**EXPECTED FINDINGS:**
- **OCR Domain**: `apps/backend/services/playground_api/` + `apps/frontend/`
- **QMS Domain**: `apps/agentqms-dashboard/` (frontend + minimal backend)
- **Playground Console**: `apps/playground-console/` (purpose to be determined)
- **Shared Code**: Likely in `ocr.utils.*`, needs extraction to common layer

---

## 📚 **Reference Documentation**

### **Related Implementation Plans**
- `docs/artifacts/implementation_plans/2025-12-08_0231_implementation_plan_dashboard-integration-testing.md` - Dashboard integration context
- `docs/artifacts/implementation_plans/2025-11-11_0158_implementation_plan_breakdown-architecture-refactoring.md` - Architecture refactoring foundation

### **Architecture Documentation**
- `apps/agentqms-dashboard/frontend/docs/architecture/2025-12-08-1430_arch-frontend-patterns.md` - Frontend architecture guidelines
- `apps/agentqms-dashboard/frontend/docs/api/2025-12-08-1430_api-contracts-spec.md` - API contracts specification
- `.agentqms/state/architecture.yaml` - AgentQMS component architecture

### **AgentQMS Framework References**
- `AgentQMS/knowledge/agent/system.md` - Framework Single Source of Truth
- `AgentQMS/agent_tools/` - Tools for artifact validation, compliance, tracking
- `docs/artifacts/` - Artifact organization and conventions

### **Backend Code References**
- `apps/backend/services/playground_api/app.py` - OCR Playground API setup
- `apps/agentqms-dashboard/backend/server.py` - AgentQMS Dashboard backend stub
- `ocr/utils/path_utils.py` - Shared path resolution utilities

---

## 🔍 **Context & Background**

### **Project Overview**
This project contains multiple applications serving different purposes:
1. **OCR Competition Workspace** - Training and inference for OCR models
2. **OCR Playground** - Web UI for experimenting with OCR models and configurations
3. **AgentQMS Dashboard** - Quality management system for AI agents and artifacts

### **Current Architecture Issues**
- **Mixed Responsibilities**: Backend services not clearly separated by domain
- **Unclear Boundaries**: Shared utilities (`ocr.utils`) used across domains without clear interface
- **Incomplete Separation**: AgentQMS Dashboard has minimal backend, relies on direct filesystem access
- **Coupling Risk**: Frontend applications may be tightly coupled to specific backend implementations

### **Goals of Domain-Driven Separation**
1. **Clear Bounded Contexts**: Each domain (OCR, QMS) has clear boundaries and responsibilities
2. **Independent Evolution**: Domains can evolve independently without breaking each other
3. **Explicit Interfaces**: Integration points are well-defined and documented
4. **Reduced Coupling**: Minimize dependencies between domains
5. **Maintainability**: Easier to understand, test, and modify each domain

### **Alignment with AgentQMS Dashboard Integration**
This refactoring supports the broader dashboard integration effort (branch: `feature/agentqms-dashboard-integration`) by:
- Establishing clear API boundaries for AgentQMS backend
- Separating OCR-specific logic from QMS-specific logic
- Enabling independent deployment and scaling of each domain
- Providing foundation for comprehensive dashboard integration testing

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*