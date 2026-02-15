# Tasks: WandB Configuration Logging Constraints

**Input**: Design documents from `/workspaces/specs/001-wandb-config-logging/`
**Prerequisites**: spec.md ✅, plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: No new test infrastructure required - validation uses existing training runners and semantic search tools

**Organization**: Tasks grouped by user story priority (P1 → P2). User Story 3 (P3) is future enhancement, not included in MVP.

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2)
- Exact file paths included in all task descriptions

---

## Phase 1: Setup (Prerequisites & Environment Check)

**Purpose**: Verify environment and tooling before implementation

- [ ] T001 Verify Hydra configuration structure in /workspaces/configs/ is accessible
- [ ] T002 Confirm WandB logger config exists at /workspaces/configs/train/logger/wandb.yaml
- [ ] T003 Verify context bundling tools are operational (uv run python AgentQMS/tools/utilities/suggest_context.py --help)

**Checkpoint**: Environment ready for implementation

---

## Phase 2: Foundational (Specification Update)

**Purpose**: Establish constraint documentation in tier2 framework spec (BLOCKS all user stories - discovery depends on this)

**⚠️ CRITICAL**: This documentation must exist before code changes so inline comments can reference it

- [ ] T004 Add section "## 5. Serialization Constraints" after section 4 in /workspaces/AgentQMS/specs/tier2-framework/configuration.spec.md
- [ ] T005 Document CONFIG-WANDB-001 constraint (WandB log_config + _target_ incompatibility) in new section with rationale, default value, and override risks

**Checkpoint**: Constraint specification exists and is referenceable by code comments

---

## Phase 3: User Story 1 - Default Training Without Config Override (Priority: P1) 🎯 MVP

**Goal**: Enable training runs to launch successfully with default configuration settings, without manual overrides or serialization errors

**Independent Test**: Launch any training experiment with default config (no CLI overrides) → Verify trainer initializes successfully without serialization errors

### Implementation for User Story 1

- [ ] T006 [US1] Change log_config value from true to false in /workspaces/configs/train/logger/wandb.yaml (line 10)
- [ ] T007 [US1] Add inline comment documenting CONFIG-WANDB-001 constraint and spec reference above the log_config field in /workspaces/configs/train/logger/wandb.yaml
- [ ] T008 [US1] Add 7-line comment block before line 179 in /workspaces/ocr/pipelines/orchestrator.py explaining WandB config logging constraint, serialization risk, and spec reference

**Checkpoint**: Default training launches without serialization errors. Essential config values visible via run naming convention.

---

## Phase 4: User Story 2 - AI Agent Discovery of Constraints (Priority: P2)

**Goal**: Enable AI agents to discover configuration logging constraints through framework documentation and context bundles

**Independent Test**: Query context bundling system with "WandB configuration logging" → Verify HYDRA-CONFIGURATION bundle is recommended and references constraint documentation

### Implementation for User Story 2

- [ ] T009 [P] [US2] Add keyword "serialization" to triggers.keywords list in /workspaces/AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml (after line 30)
- [ ] T010 [P] [US2] Add keyword "wandb" to triggers.keywords list in /workspaces/AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml
- [ ] T011 [P] [US2] Add keyword "log_config" to triggers.keywords list in /workspaces/AgentQMS/.agentqms/plugins/context_bundles/hydra-configuration.yaml

**Checkpoint**: AI agents querying configuration or WandB constraints discover HYDRA-CONFIGURATION bundle and specification within 2 semantic searches

---

## Phase 5: Validation & Testing (Success Criteria)

**Purpose**: Verify all functional requirements and success criteria are met

### SC-001: Training Launches Without Manual Overrides

- [ ] T012 Run default parseq_flash_fast experiment without CLI overrides: `uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.max_epochs=1 trainer.limit_train_batches=10`
- [ ] T013 Verify trainer initializes successfully with exit code 0 and no serialization errors in output

### SC-002: AI Agent Discovery Within 2 Searches

- [ ] T014 Test semantic context suggestion: `uv run python AgentQMS/tools/utilities/suggest_context.py "WandB configuration logging constraints"`
- [ ] T015 Verify HYDRA-CONFIGURATION bundle appears in recommended bundles (should be first or second result)
- [ ] T016 Test grep search for constraint ID: `grep -r "CONFIG-WANDB-001" /workspaces/AgentQMS/specs/tier2-framework/`
- [ ] T017 Verify configuration.spec.md appears in search results with correct constraint documentation

### SC-003: Zero Serialization Failures Across Experiments

- [ ] T018 Run parseq_flash_fast experiment with default config: `uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.max_epochs=1 trainer.limit_train_batches=5`
- [ ] T019 Run rec_optimized_rtx3090 experiment with default config (if exists): `uv run python scripts/runners/train.py experiment=rec_optimized_rtx3090 trainer.max_epochs=1 trainer.limit_train_batches=5`
- [ ] T020 Run vlm_base experiment with default config (if exists): `uv run python scripts/runners/train.py experiment=vlm_base trainer.max_epochs=1 trainer.limit_train_batches=5`
- [ ] T021 Verify all 3 experiments initialize successfully without serialization errors (check exit codes and logs)

### SC-004: Essential Config Visible in WandB Dashboard

- [ ] T022 Launch one complete training run: `uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.max_epochs=1 trainer.limit_train_batches=20`
- [ ] T023 Verify WandB run name encodes: model architecture, batch size, learning rate, optimizer (check WandB dashboard or console output)
- [ ] T024 Verify metrics (loss, accuracy) are logged normally in WandB dashboard Metrics tab
- [ ] T025 Verify full config still accessible in checkpoint files or local outputs/ directory

### SC-005: No New Utility Modules or Abstractions

- [ ] T026 Verify no new Python modules created in /workspaces/ocr/ or /workspaces/AgentQMS/
- [ ] T027 Verify no new utility functions added to existing modules (only comments and config value changes)

**Checkpoint**: All success criteria validated. Feature ready for integration.

---

## Phase 6: Documentation Validation (Quickstart Guide)

**Purpose**: Verify quickstart guide accurately reflects implemented behavior

- [ ] T028 Follow quickstart.md "For Users: Default Behavior" section → Run default experiment and confirm behavior matches documentation
- [ ] T029 Follow quickstart.md "For AI Agents: Discovery Pattern" section → Test semantic search commands and verify context bundle discovery works as documented
- [ ] T030 Test quickstart.md troubleshooting scenarios → Verify config visibility workarounds (run name, files tab, local outputs)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001-T003) - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (T004-T005) - needs spec to exist for code comments to reference
- **User Story 2 (Phase 4)**: Depends on Foundational (T004-T005) - needs spec to exist for context bundle to reference
- **Validation (Phase 5)**: Depends on User Stories 1 & 2 completion (T006-T011)
- **Documentation Validation (Phase 6)**: Depends on Validation (Phase 5) - final verification

### User Story Dependencies

- **User Story 1 (P1)**: Independent after Foundational phase - can be tested standalone
- **User Story 2 (P2)**: Independent after Foundational phase - can be tested standalone via semantic search
- **User Story 3 (P3)**: Future enhancement - NOT included in MVP scope

### Within Each Phase

- **Foundational**: T004 must complete before T005 (section must exist before adding constraint details)
- **User Story 1**: T006 and T007 affect same file (sequential), T008 is different file (can be parallel with T006-T007)
- **User Story 2**: T009, T010, T011 modify same file but different lines (can be done together in one edit)
- **Validation**: Most tasks are independent test runs (can parallelize by running different experiments)

### Parallel Opportunities

- **Phase 1 (Setup)**: All 3 verification tasks (T001-T003) can run in parallel
- **Phase 3 (US1)**: T008 (orchestrator.py comments) can run in parallel with T006-T007 (wandb.yaml changes)
- **Phase 4 (US2)**: All 3 keyword additions (T009-T011) should be done in one file edit operation
- **Phase 5 (Validation)**: Experiment runs (T018-T020) can run in parallel if system resources allow

---

## Parallel Example: User Story 1

```bash
# Terminal 1: Modify WandB config file
# T006 + T007: Change log_config and add comment
vim /workspaces/configs/train/logger/wandb.yaml

# Terminal 2 (parallel): Add orchestrator documentation
# T008: Add comment block explaining constraint
vim /workspaces/ocr/pipelines/orchestrator.py
```

---

## Parallel Example: Validation Phase

```bash
# Run multiple experiments in parallel (if resources allow)
# Terminal 1:
uv run python scripts/runners/train.py experiment=parseq_flash_fast trainer.max_epochs=1 trainer.limit_train_batches=5

# Terminal 2:
uv run python scripts/runners/train.py experiment=rec_optimized_rtx3090 trainer.max_epochs=1 trainer.limit_train_batches=5

# Terminal 3:
uv run python scripts/runners/train.py experiment=vlm_base trainer.max_epochs=1 trainer.limit_train_batches=5
```

---

## Implementation Strategy

### MVP Scope (User Stories 1 & 2 Only)

1. **Complete Phase 1: Setup** → Verify environment (5 minutes)
2. **Complete Phase 2: Foundational** → Add specification documentation (5 minutes)
3. **Complete Phase 3: User Story 1** → Change config default + document constraint in code (10 minutes)
4. **Complete Phase 4: User Story 2** → Update context bundle keywords (5 minutes)
5. **Complete Phase 5: Validation** → Run success criteria tests (20 minutes)
6. **Complete Phase 6: Documentation** → Verify quickstart guide (5 minutes)

**Total Estimated Time**: 50 minutes (30 minutes implementation + 20 minutes validation)

### Delivery Checkpoints

- **After Phase 2**: Constraint is documented and discoverable by AI agents
- **After Phase 3**: Training runs work by default without serialization errors (MVP!)
- **After Phase 4**: AI agents can discover constraints through context bundling
- **After Phase 5**: All success criteria validated and tested
- **After Phase 6**: User and AI agent documentation verified

### User Story 3 (Future Enhancement)

**Not included in MVP** - Selective scalar whitelisting (P3 priority):
- Requires new configuration schema for whitelist paths
- Needs runtime validation of whitelisted fields
- Adds complexity (violates simplification principle)
- Deferred until user demand justifies the overhead

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tasks** | 30 |
| **Setup/Foundational** | 5 tasks (T001-T005) |
| **User Story 1 (P1)** | 3 implementation tasks (T006-T008) |
| **User Story 2 (P2)** | 3 implementation tasks (T009-T011) |
| **Validation** | 17 validation tasks (T012-T028) |
| **Documentation** | 3 validation tasks (T029-T031) |
| **Files Modified** | 4 files (0 new files) |
| **Parallel Opportunities** | 3 phases: Setup (3 tasks), US1 partial (1 task), Validation (3 tasks) |
| **Estimated Time** | 50 minutes total (30 min implementation + 20 min validation) |
| **MVP Scope** | User Stories 1 & 2 (P1 & P2) |

## Acceptance Criteria Summary

✅ **FR-001**: Default config disables log_config (T006)
✅ **FR-002**: Framework spec documents constraint (T004-T005)
✅ **FR-003**: Logger code includes explanatory comments (T008)
✅ **FR-004**: Context bundle references constraint (T009-T011)
✅ **FR-005**: WandB config sets log_config: false (T006-T007)
✅ **FR-006**: Run naming maintains config visibility (validated T023)
✅ **FR-007**: Documentation provides decision criteria (T004-T005)
⏭️ **FR-008**: Selective whitelisting deferred to P3 (future enhancement)

---

## Notes

- All tasks provide exact file paths for easy navigation
- [P] tasks touch different files and can run in parallel
- [Story] labels map tasks to user stories for traceability
- No new abstractions or utilities (meets SC-005 simplification requirement)
- Validation tasks use existing tooling (training runners, semantic search)
- User Story 3 (P3) intentionally excluded from MVP - documented as future enhancement
- Each checkpoint allows independent validation of delivered value
