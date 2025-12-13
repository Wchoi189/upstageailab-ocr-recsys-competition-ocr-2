---
title: "Data Contract Enforcement Implementation"
date: "2025-12-06 18:09 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---







# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Data Contract Enforcement Implementation**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Data Contract Enforcement Implementation

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Create ValidatedPolygonData Model
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Review existing PolygonData model and add bounds validation

## Executive Summary

This implementation plan addresses critical data contract gaps identified in the Data Contract Enforcement Assessment (2025-11-12). The plan implements validation for:

1. **Critical Data Quality Contracts** (Phase 1): Polygon bounds validation, tensor shape/device/range validation
2. **Configuration Contracts** (Phase 2): YAML/OmegaConf validation models
3. **Data Quality Contracts** (Phase 3): Synthetic data and submission format validation
4. **Runtime Pipeline Extensions** (Phase 4): Training loop and API validation

**Timing Recommendation**: Execute **BEFORE** refactoring (see Refactoring Assessment 2025-11-12). Data contracts are validation layers that will help ensure refactoring maintains data integrity. Contracts can be moved along with code during refactoring.

**Total Estimated Effort**: 10-14 days
**Risk Level**: Low-Medium (incremental implementation with existing patterns)
**Business Impact**: High (prevents 26.5% data corruption, CUDA crashes, 8000x performance degradation)

### Implementation Outline (Checklist)

#### **Phase 1: Critical Data Quality Contracts (1-2 days, Critical Priority)**

**Objective**: Implement polygon bounds validation and tensor validation to address immediate production issues (BUG-20251110-001, BUG-20251112-001, BUG-20251112-013).

1. [ ] **Task 1.1: Create ValidatedPolygonData Model**
   - [ ] Review existing `PolygonData` model in `ocr/datasets/schemas.py`
   - [ ] Create `ValidatedPolygonData` extending `PolygonData` with bounds checking
   - [ ] Add `image_height` and `image_width` fields to validation context
   - [ ] Implement `@field_validator` for polygon coordinates to ensure within bounds
   - [ ] Add validation error messages indicating which coordinates are out of bounds
   - [ ] Write unit tests for valid polygons (within bounds)
   - [ ] Write unit tests for invalid polygons (out of bounds)
   - [ ] Write unit tests for edge cases (coordinates at exact boundaries)

2. [ ] **Task 1.2: Integrate ValidatedPolygonData into Dataset Pipeline**
   - [ ] Review `ocr/datasets/base.py` `_load_image_data()` method
   - [ ] Update polygon loading to use `ValidatedPolygonData` with image dimensions
   - [ ] Add validation error handling with clear error messages
   - [ ] Add logging for validation failures (log polygon index, coordinates, image dimensions)
   - [ ] Test with corrupted dataset from BUG-20251110-001 (867 training images with out-of-bounds coordinates)
   - [ ] Verify validation catches all 867 corrupted images
   - [ ] Update dataset documentation to reflect new validation

3. [ ] **Task 1.3: Create ValidatedTensorData Model**
   - [ ] Create new file `ocr/validation/tensor_models.py`
   - [ ] Create `ValidatedTensorData` Pydantic model with fields:
     - `tensor`: torch.Tensor
     - `expected_shape`: tuple[int, ...] | None
     - `expected_device`: torch.device | str | None
     - `expected_dtype`: torch.dtype | None
     - `value_range`: tuple[float, float] | None (for range validation)
   - [ ] Implement `@field_validator` for shape validation
   - [ ] Implement `@field_validator` for device validation
   - [ ] Implement `@field_validator` for dtype validation
   - [ ] Implement `@field_validator` for value range validation (e.g., [0, 1] for predictions)
   - [ ] Add clear error messages for each validation failure
   - [ ] Write unit tests for all validation scenarios

4. [ ] **Task 1.4: Integrate Tensor Validation into Loss Functions**
   - [ ] Review `ocr/losses/` directory for loss function implementations
   - [ ] Identify loss functions affected by BUG-20251112-001 (Dice loss) and BUG-20251112-013 (BCE loss)
   - [ ] Add `ValidatedTensorData` validation at loss function entry points
   - [ ] Validate prediction tensors are in [0, 1] range before Dice loss
   - [ ] Validate tensor shapes match before loss computation
   - [ ] Validate tensor devices match before loss computation
   - [ ] Add validation error handling with clear error messages
   - [ ] Write unit tests for loss function validation
   - [ ] Test with corrupted predictions from bug reports

5. [ ] **Task 1.5: Add Tensor Validation to Lightning Module**
   - [ ] Review `ocr/lightning_modules/ocrpl_module.py` step methods
   - [ ] Add `ValidatedTensorData` validation in `training_step()` before loss computation
   - [ ] Add `ValidatedTensorData` validation in `validation_step()` before loss computation
   - [ ] Validate model outputs before passing to loss functions
   - [ ] Add validation error handling with clear error messages
   - [ ] Write unit tests for Lightning module validation
   - [ ] Test with corrupted model outputs

6. [ ] **Task 1.6: Phase 1 Testing and Validation**
   - [ ] Run full test suite: `pytest tests/unit/test_validation_models.py -v`
   - [ ] Run dataset tests: `pytest tests/unit/test_datasets.py -v`
   - [ ] Run loss function tests: `pytest tests/unit/test_losses.py -v`
   - [ ] Run Lightning module tests: `pytest tests/unit/test_lightning_modules.py -v`
   - [ ] Test with corrupted dataset from BUG-20251110-001
   - [ ] Verify validation catches all known issues from bug reports
   - [ ] Measure validation performance overhead (target: <5% overhead)
   - [ ] Update documentation with new validation models

#### **Phase 2: Configuration Contracts (2-3 days, High Priority)**

**Objective**: Implement YAML/OmegaConf validation models to catch configuration errors early (addresses BUG-20251112-001, BUG-20251112-013).

7. [ ] **Task 2.1: Create TrainerConfig Model**
   - [ ] Review existing trainer configuration in `configs/trainer/default.yaml`
   - [ ] Create `ocr/validation/config_models.py` file
   - [ ] Create `TrainerConfig` Pydantic model with fields:
     - `max_epochs`: int (positive)
     - `batch_size`: int (positive, reasonable limits)
     - `learning_rate`: float (positive, <1.0)
     - `weight_decay`: float (non-negative)
     - `gradient_clip_val`: float | None (positive if provided)
     - `accumulate_grad_batches`: int (positive)
     - `gpus`: int | list[int] | None
     - `precision`: int | str (16, 32, "bf16")
   - [ ] Add field validators for each field with appropriate constraints
   - [ ] Write unit tests for valid configurations
   - [ ] Write unit tests for invalid configurations (negative learning rates, etc.)

8. [ ] **Task 2.2: Create ModelConfig Model**
   - [ ] Review existing model configuration patterns
   - [ ] Create `ModelConfig` Pydantic model with fields:
     - `backbone`: str (valid backbone names)
     - `num_classes`: int (positive)
     - `input_size`: tuple[int, int] (positive dimensions)
     - `pretrained`: bool
     - `freeze_backbone`: bool
   - [ ] Add field validators for each field
   - [ ] Write unit tests for valid/invalid configurations

9. [ ] **Task 2.3: Create DataLoaderConfig Model**
   - [ ] Review existing data loader configuration
   - [ ] Create `DataLoaderConfig` Pydantic model with fields:
     - `batch_size`: int (positive, reasonable limits)
     - `num_workers`: int (non-negative)
     - `pin_memory`: bool
     - `shuffle`: bool
     - `drop_last`: bool
   - [ ] Add field validators for each field
   - [ ] Write unit tests for valid/invalid configurations

10. [ ] **Task 2.4: Integrate Configuration Validation with Hydra/OmegaConf**
    - [ ] Review Hydra configuration loading in training scripts
    - [ ] Create `ocr/utils/config_validator.py` utility module
    - [ ] Create `validate_config()` function that converts OmegaConf to Pydantic models
    - [ ] Integrate validation into training script entry points
    - [ ] Add validation error handling with clear error messages
    - [ ] Test with valid configuration files
    - [ ] Test with invalid configuration files (negative learning rates, etc.)
    - [ ] Verify validation catches configuration errors before training starts

11. [ ] **Task 2.5: Phase 2 Testing and Validation**
    - [ ] Run configuration validation tests: `pytest tests/unit/test_config_validation.py -v`
    - [ ] Test with all existing configuration files in `configs/`
    - [ ] Verify validation catches configuration errors from bug reports
    - [ ] Measure validation performance overhead (target: <1% overhead)
    - [ ] Update configuration documentation

#### **Phase 3: Data Quality Contracts (3-4 days, High Priority)**

**Objective**: Implement synthetic data validation and submission format validation to prevent data corruption issues.

12. [ ] **Task 3.1: Create ValidatedTextRegion Model**
    - [ ] Review synthetic data generation code (locate text region structures)
    - [ ] Create `ocr/validation/synthetic_models.py` file
    - [ ] Create `ValidatedTextRegion` Pydantic model with fields:
      - `text`: str (non-empty)
      - `bbox`: tuple[float, float, float, float] (x1, y1, x2, y2, all non-negative)
      - `polygon`: list[tuple[float, float]] | None (at least 3 points if provided)
      - `confidence`: float (0.0-1.0)
    - [ ] Add field validators for bbox bounds checking
    - [ ] Add field validators for polygon bounds checking
    - [ ] Write unit tests for valid/invalid text regions

13. [ ] **Task 3.2: Create ValidatedSyntheticImage Model**
    - [ ] Review synthetic image generation code
    - [ ] Create `ValidatedSyntheticImage` Pydantic model with fields:
      - `image`: np.ndarray (valid image array)
      - `text_regions`: list[ValidatedTextRegion]
      - `image_id`: str
      - `metadata`: dict[str, Any] | None
    - [ ] Add field validator to ensure all text regions are within image bounds
    - [ ] Write unit tests for valid/invalid synthetic images

14. [ ] **Task 3.3: Integrate Synthetic Data Validation**
    - [ ] Locate synthetic data generation functions
    - [ ] Add `ValidatedSyntheticImage` validation after image generation
    - [ ] Add validation error handling with clear error messages
    - [ ] Add logging for validation failures
    - [ ] Test with synthetic data generation pipeline
    - [ ] Verify validation catches corrupted synthetic data

15. [ ] **Task 3.4: Create SubmissionFormat Model**
    - [ ] Review submission format requirements (competition format)
    - [ ] Create `ocr/validation/submission_models.py` file
    - [ ] Create `SubmissionFormat` Pydantic model with fields:
      - `image_id`: str
      - `polygons`: list[list[tuple[float, float]]] (list of polygons)
      - `texts`: list[str] (matching polygon count)
      - `confidences`: list[float] (0.0-1.0, matching polygon count)
    - [ ] Add field validators for polygon bounds checking
    - [ ] Add field validators for array length matching
    - [ ] Add field validators for confidence score ranges
    - [ ] Write unit tests for valid/invalid submission formats

16. [ ] **Task 3.5: Integrate Submission Format Validation**
    - [ ] Locate submission generation code
    - [ ] Add `SubmissionFormat` validation before writing submission files
    - [ ] Add validation error handling with clear error messages
    - [ ] Test with submission generation pipeline
    - [ ] Verify validation catches invalid submission formats

17. [ ] **Task 3.6: Phase 3 Testing and Validation**
    - [ ] Run synthetic data validation tests: `pytest tests/unit/test_synthetic_validation.py -v`
    - [ ] Run submission format validation tests: `pytest tests/unit/test_submission_validation.py -v`
    - [ ] Test with synthetic data generation pipeline
    - [ ] Test with submission generation pipeline
    - [ ] Verify validation catches all known data corruption issues
    - [ ] Measure validation performance overhead
    - [ ] Update documentation

#### **Phase 4: Runtime Pipeline Extensions (4-5 days, Medium Priority)**

**Objective**: Implement training loop and API validation to improve training stability.

18. [ ] **Task 4.1: Create TrainingBatch Model**
    - [ ] Review training batch structure in Lightning module
    - [ ] Create `ocr/validation/training_models.py` file
    - [ ] Create `TrainingBatch` Pydantic model with fields:
      - `images`: torch.Tensor (batch of images)
      - `targets`: dict[str, torch.Tensor] (batch of targets)
      - `metadata`: list[dict[str, Any]] | None
    - [ ] Add field validators for tensor shapes and devices
    - [ ] Write unit tests for valid/invalid training batches

19. [ ] **Task 4.2: Create ValidationBatch Model**
    - [ ] Review validation batch structure
    - [ ] Create `ValidationBatch` Pydantic model (similar to TrainingBatch)
    - [ ] Add field validators for tensor shapes and devices
    - [ ] Write unit tests for valid/invalid validation batches

20. [ ] **Task 4.3: Integrate Batch Validation into Training Loop**
    - [ ] Review Lightning module `training_step()` and `validation_step()` methods
    - [ ] Add `TrainingBatch` validation at start of `training_step()`
    - [ ] Add `ValidationBatch` validation at start of `validation_step()`
    - [ ] Add validation error handling with clear error messages
    - [ ] Test with training pipeline
    - [ ] Verify validation catches batch-related issues

21. [ ] **Task 4.4: Create API Request/Response Validation Models**
    - [ ] Review API endpoints (inference services)
    - [ ] Create `ocr/validation/api_models.py` file
    - [ ] Create request validation models for each API endpoint
    - [ ] Create response validation models for each API endpoint
    - [ ] Add field validators for all models
    - [ ] Write unit tests for valid/invalid API requests/responses

22. [ ] **Task 4.5: Integrate API Validation**
    - [ ] Locate API endpoint handlers
    - [ ] Add request validation at API entry points
    - [ ] Add response validation at API exit points
    - [ ] Add validation error handling with clear error messages
    - [ ] Test with API endpoints
    - [ ] Verify validation catches API-related issues

23. [ ] **Task 4.6: Phase 4 Testing and Validation**
    - [ ] Run training batch validation tests: `pytest tests/unit/test_training_validation.py -v`
    - [ ] Run API validation tests: `pytest tests/unit/test_api_validation.py -v`
    - [ ] Test with full training pipeline
    - [ ] Test with API endpoints
    - [ ] Verify validation catches all known issues
    - [ ] Measure validation performance overhead
    - [ ] Update documentation

#### **Phase 5: Integration, Testing, and Documentation (2-3 days)**

24. [ ] **Task 5.1: End-to-End Integration Testing**
    - [ ] Run full test suite: `pytest tests/ -v`
    - [ ] Test with complete training pipeline from data loading to model output
    - [ ] Test with inference pipeline
    - [ ] Test with submission generation pipeline
    - [ ] Verify all validation models work together correctly
    - [ ] Measure overall validation performance overhead (target: <10% overhead)

25. [ ] **Task 5.2: Performance Optimization**
    - [ ] Profile validation overhead in production scenarios
    - [ ] Implement lazy validation for non-critical paths (debug mode only)
    - [ ] Implement validation caching where appropriate
    - [ ] Optimize validation error message generation
    - [ ] Verify performance targets are met

26. [ ] **Task 5.3: Documentation Updates**
    - [ ] Update `docs/pipeline/data_contracts.md` with new validation models
    - [ ] Create validation model reference documentation
    - [ ] Update developer guide with validation best practices
    - [ ] Add examples of validation error handling
    - [ ] Update API documentation with validation requirements

27. [ ] **Task 5.4: Success Metrics Validation**
    - [ ] Verify data corruption rate reduced from 26.5% to <1%
    - [ ] Verify training completion rate improved from 80% to 95%+
    - [ ] Verify model performance stability (no 8000x degradation)
    - [ ] Verify configuration errors caught at source (95%+)
    - [ ] Verify debug time reduced by 75%
    - [ ] Document success metrics in assessment report

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Use existing Pydantic v2 patterns (extend `ocr/validation/models.py`)
- [x] Follow existing validation model structure and naming conventions
- [x] Maintain backward compatibility with existing code
- [x] Use field validators for complex validation logic
- [x] Provide clear, actionable error messages

### **Integration Points**
- [x] Integrate with existing dataset pipeline (`ocr/datasets/base.py`)
- [x] Integrate with Lightning module (`ocr/lightning_modules/ocrpl_module.py`)
- [x] Integrate with loss functions (`ocr/losses/`)
- [x] Integrate with Hydra/OmegaConf configuration system
- [x] Integrate with synthetic data generation pipeline
- [x] Integrate with submission generation pipeline
- [x] Integrate with API endpoints

### **Quality Assurance**
- [x] Unit test coverage goal: >90% for all validation models
- [x] Integration tests for each phase
- [x] Performance tests to measure validation overhead
- [x] Test with corrupted data from bug reports
- [x] Test with edge cases and boundary conditions

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [x] Polygon bounds validation catches 100% of out-of-bounds coordinates (BUG-20251110-001)
- [x] Tensor validation prevents CUDA memory access errors (BUG-20251112-013)
- [x] Tensor range validation prevents Dice loss assertion errors (BUG-20251112-001)
- [x] Configuration validation catches invalid hyperparameters before training
- [x] Synthetic data validation prevents corrupted training data
- [x] Submission format validation prevents invalid competition submissions

### **Technical Requirements**
- [x] All validation models follow Pydantic v2 patterns
- [x] Validation overhead <10% in production scenarios
- [x] Clear, actionable error messages for all validation failures
- [x] Comprehensive test coverage (>90%)
- [x] Backward compatibility maintained
- [x] Documentation updated

### **Business Impact Metrics**
- [x] Data corruption rate: <1% (down from 26.5%)
- [x] Training completion rate: 95%+ (up from 80%)
- [x] Model performance stability: No 8000x degradation
- [x] Configuration error detection: 95%+ caught at source
- [x] Debug time reduction: 75% faster issue resolution

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW-MEDIUM

### **Active Mitigation Strategies**:
1. **Incremental Implementation**: Implement one phase at a time with testing after each phase
2. **Comprehensive Testing**: Test with corrupted data from bug reports to verify validation effectiveness
3. **Performance Monitoring**: Measure validation overhead and optimize as needed
4. **Backward Compatibility**: Maintain existing interfaces while adding validation
5. **Feature Flags**: Allow disabling validation in production if needed (debug mode only)

### **Fallback Options**:
1. **Lazy Validation**: If performance overhead is too high, implement lazy validation (debug mode only)
2. **Sampling**: If full validation is too expensive, validate only subset of data (1% sampling)
3. **Gradual Rollout**: Roll out validation incrementally, starting with most critical paths
4. **Validation Caching**: Cache validation results for repeated data structures

### **Known Risks**:
- **Performance Overhead**: Validation may slow down training/inference (mitigated by performance monitoring and optimization)
- **Breaking Changes**: Validation may reject previously accepted data (mitigated by comprehensive testing and backward compatibility)
- **Integration Complexity**: Integration with existing code may be complex (mitigated by incremental implementation)

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)
- Performance issue discovered (optimize or adjust approach)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed
5. Document any blockers or issues encountered

---

## 🚀 **Immediate Next Action**

**TASK:** Task 1.1 - Create ValidatedPolygonData Model

**OBJECTIVE:** Create a Pydantic model that extends PolygonData with bounds checking to validate polygon coordinates are within image dimensions.

**APPROACH:**
1. Review existing `PolygonData` model in `ocr/datasets/schemas.py`
2. Create `ValidatedPolygonData` class extending `PolygonData`
3. Add `image_height` and `image_width` fields to validation context
4. Implement `@field_validator` for polygon coordinates
5. Write unit tests for valid/invalid polygons

**SUCCESS CRITERIA:**
- `ValidatedPolygonData` model created with bounds checking
- Unit tests pass for valid polygons (within bounds)
- Unit tests pass for invalid polygons (out of bounds)
- Validation error messages are clear and actionable

---

## 📅 **Timing Recommendation: Execute BEFORE Refactoring**

**Rationale:**
1. **Validation Layer**: Data contracts are validation layers that don't change code structure, making them safe to add before refactoring
2. **Refactoring Safety**: Contracts will help ensure refactoring maintains data integrity by catching issues immediately
3. **Smaller Scope**: Data contract implementation (10-14 days) is much smaller than refactoring (3 months), so it can be completed first
4. **Code Movement**: Contracts can be moved along with code during refactoring without breaking functionality
5. **Critical Issues**: Contracts address immediate production issues (26.5% data corruption, CUDA crashes) that should be fixed before refactoring

**Recommended Timeline:**
- **Weeks 1-2**: Complete Phase 1 (Critical Data Quality Contracts) - addresses immediate production issues
- **Weeks 3-4**: Complete Phase 2 (Configuration Contracts) - prevents configuration errors
- **Weeks 5-7**: Complete Phase 3 (Data Quality Contracts) - prevents data corruption
- **Weeks 8-10**: Complete Phase 4 (Runtime Pipeline Extensions) - improves training stability
- **Weeks 11-12**: Complete Phase 5 (Integration, Testing, Documentation)
- **After Data Contracts**: Begin refactoring (3-month effort) with contracts in place to ensure data integrity

**Refactoring Assessment Reference:**
- See `docs/2025-11-12_refactoring_assessment.md` for refactoring plan
- Refactoring will change locations of scripts and functions
- Data contracts will be moved along with code during refactoring
- Contracts will help validate that refactoring maintains data integrity

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
