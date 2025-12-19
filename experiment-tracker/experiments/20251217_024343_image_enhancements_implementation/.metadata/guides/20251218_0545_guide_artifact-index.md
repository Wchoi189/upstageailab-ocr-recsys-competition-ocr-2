---
ads_version: "1.0"
type: "guide"
experiment_id: "20251217_024343_image_enhancements_implementation"
status: "complete"
created: "2025-12-18T05:45:00Z"
updated: "2025-12-18T05:45:00Z"
tags: ['index', 'navigation', 'artifact-catalog']
commands: []
prerequisites: []
---

# Artifact Index: Image Enhancement Experiment

**Purpose**: Navigation map for all experiment documentation
**Audience**: AI agents, implementation executors
**Status**: Phase 1 Week 1 Day 1 Ready

---

## Document Hierarchy

### Level 1: Master References (Start Here)
1. **[README.md](README.md)** - Experiment overview, quick start, status tracking
2. **[20251218_0530_implementation_plan_21step-image-enhancement-pipeline.md](20251218_0530_implementation_plan_21step-image-enhancement-pipeline.md)** - Complete implementation plan (PRIMARY REFERENCE)

### Level 2: Phase-Specific Details
3. **[20251217_0243_assessment_priority-plan-revised.md](20251217_0243_assessment_priority-plan-revised.md)** - Week 1-3 detailed plan with code examples
4. **[20251217_0243_assessment_master-roadmap.md](20251217_0243_assessment_master-roadmap.md)** - 21-step pipeline tracking
5. **[20251217_0243_assessment_executive-summary.md](20251217_0243_assessment_executive-summary.md)** - Strategic overview, success metrics

### Level 3: Technical Context
6. **[20251217_0243_assessment_current-state-summary.md](20251217_0243_assessment_current-state-summary.md)** - Capabilities matrix, codebase analysis
7. **[20251217_0243_guide_vlm-integration-guide.md](20251217_0243_guide_vlm-integration-guide.md)** - VLM validation workflows
8. **[20251217_0243_assessment_enhancement-quick-reference.md](20251217_0243_assessment_enhancement-quick-reference.md)** - Quick lookup tables

---

## Document Usage Matrix

| Task | Primary Document | Supporting Documents |
|------|------------------|---------------------|
| **Start experiment** | README.md | Implementation Plan |
| **Week 1-3 implementation** | Implementation Plan (Phase 1) | Priority Plan Revised, VLM Guide |
| **Week 4+ implementation** | Implementation Plan (Phases 2-4) | Master Roadmap |
| **VLM validation** | VLM Integration Guide | Implementation Plan (Validation sections) |
| **Track progress** | Master Roadmap | Executive Summary |
| **Understand codebase** | Current State Summary | Implementation Plan (Dependencies) |
| **Quick lookups** | Enhancement Quick Reference | Priority Plan Revised |
| **Strategic decisions** | Executive Summary | Master Roadmap |

---

## Information Flow

```
README.md (Entry point)
    ↓
Implementation Plan (Execution blueprint)
    ↓
    ├── Priority Plan Revised (Week 1-3 details)
    ├── Master Roadmap (21-step tracking)
    ├── VLM Integration Guide (Validation workflows)
    └── Current State Summary (Technical context)
```

---

## Document Summaries

### [README.md](README.md) (250 lines)
**Purpose**: Experiment overview, quick start guide
**Key Sections**:
- Current status (4/21 steps complete)
- Quick start (Week 1 Day 1 commands)
- Directory structure
- Phase breakdown
- Success criteria

**Use When**: First time accessing experiment, checking status, running initial commands

---

### [Implementation Plan](20251218_0530_implementation_plan_21step-image-enhancement-pipeline.md) (750 lines)
**Purpose**: Complete 10-12 week implementation blueprint
**Key Sections**:
- Progress tracker
- Phase 1-4 checklists (Week 1 Day 1 → Week 10 Day 5)
- Code examples and pseudocode
- VLM validation framework
- Success metrics per phase
- Risk mitigation

**Use When**: Executing implementation tasks, need code examples, checking validation workflows

**Critical Features**:
- Autonomous execution protocol
- Week-by-week task breakdown
- Detailed success criteria
- Helper script specifications

---

### [Priority Plan Revised](20251217_0243_assessment_priority-plan-revised.md) (693 lines)
**Purpose**: Detailed 3-week plan for Phase 1 based on production observations
**Key Sections**:
- Critical issues prioritized (tinted backgrounds #1, text slant #2)
- 3-week roadmap with daily tasks
- Code pseudocode for gray-world, edge-based, Hough transform
- Integration strategies

**Use When**: Implementing Week 1-3, need algorithm details, understanding priorities

**Critical Features**:
- Data-driven prioritization (vs. theoretical)
- Production observation evidence
- Incremental approach with go/no-go gates

---

### [Master Roadmap](20251217_0243_assessment_master-roadmap.md) (562 lines)
**Purpose**: 21-step pipeline tracking with phase breakdown
**Key Sections**:
- Full 21-step sequence ([Geometry] → [Tone/Color] → [Background/Text] → [Edge/Detail] → [Cleanup])
- Current status: 4/21 (19%) → Phase 1 target: 8/21 (38%)
- Phase dependencies
- Timeline estimation (Q1-Q2 2026 completion)

**Use When**: Tracking overall progress, planning future phases, understanding pipeline structure

**Critical Features**:
- Phase completion percentages
- Step dependencies
- Office Lens methodology mapping

---

### [Executive Summary](20251217_0243_assessment_executive-summary.md) (359 lines)
**Purpose**: Strategic overview with success metrics
**Key Sections**:
- Key findings from production
- Revised priority order (data-driven)
- 3-week implementation roadmap
- Go/No-Go decision criteria
- Risk assessment

**Use When**: Making strategic decisions, reporting to stakeholders, understanding rationale

**Critical Features**:
- Expected OCR gains per enhancement
- Complexity vs. impact matrix
- Phase gates with success thresholds

---

### [Current State Summary](20251217_0243_assessment_current-state-summary.md) (431 lines)
**Purpose**: Capabilities matrix and codebase analysis
**Key Sections**:
- What's already working (perspective correction: 100% success)
- Preprocessing pipeline architecture
- Extension points for integration
- Testing infrastructure
- Code references with line numbers

**Use When**: Understanding existing codebase, identifying integration points, code archaeology

**Critical Features**:
- Detailed code locations (files + line counts)
- Extension points identified
- Architectural analysis

---

### [VLM Integration Guide](20251217_0243_guide_vlm-integration-guide.md) (550 lines)
**Purpose**: VLM validation workflows and helper scripts
**Key Sections**:
- Setup instructions
- 3 workflow modes (baseline, validation, debugging)
- Helper scripts (bash + python)
- Integration into phases
- Best practices

**Use When**: Running VLM validation, creating before/after comparisons, debugging failures

**Critical Features**:
- Complete bash commands
- Python aggregation script
- Prompt mode descriptions
- Troubleshooting section

---

### [Enhancement Quick Reference](20251217_0243_assessment_enhancement-quick-reference.md) (200+ lines)
**Purpose**: Quick lookup tables for algorithms
**Key Sections**:
- Top 3 enhancements with implementation checklists
- Code snippets for each method
- Parameter recommendations
- Expected processing times

**Use When**: Need quick algorithm reference, checking parameters, implementation checklists

---

## Scripts Reference

### Helper Scripts (✅ Ready)
| Script | Purpose | Lines | Usage |
|--------|---------|-------|-------|
| **vlm_baseline_assessment.sh** | Batch VLM baseline on 10 images | 40 | `bash scripts/vlm_baseline_assessment.sh` |
| **vlm_validate_enhancement.sh** | VLM validation on comparisons | 45 | `bash scripts/vlm_validate_enhancement.sh phase1_bg_norm` |
| **aggregate_vlm_validations.py** | Aggregate VLM reports to summary | 120 | `python scripts/aggregate_vlm_validations.py --input ... --output ...` |

### Implementation Scripts (📋 TODO)
| Script | Purpose | Phase | Implementation Week |
|--------|---------|-------|---------------------|
| **background_normalization.py** | Gray-world, edge-based, illumination | Phase 1 | Week 1 Day 2-3 |
| **text_deskewing.py** | Projection profile, Hough transform | Phase 1 | Week 2 Day 1-4 |
| **create_before_after_comparison.py** | Side-by-side visualizations | Phase 1 | Week 1 Day 5 |
| **compute_accuracy.py** | OCR accuracy with delta tracking | Phase 1 | Week 1 Day 1 |
| **validate_coordinates.py** | Coordinate alignment verification | Phase 1 | Week 3 Day 4 |

### Parent Experiment Scripts (✅ Copied)
| Script | Purpose | Lines | Notes |
|--------|---------|-------|-------|
| **mask_only_edge_detector.py** | Robust rectangle fitting | 1339 | From perspective correction |
| **perspective_transformer.py** | Transform utilities | ~200 | Coordinate matrix ops |
| **run_perspective_correction.py** | Batch processing | ~300 | Metrics, configurability |
| **test_worst_performers.py** | Validation on 25 images | ~250 | Proven test set |

---

## VLM Prompts Reference

### Prompt Files (AgentQMS/vlm/prompts/markdown/)
| Prompt | Mode | Purpose | Output Format |
|--------|------|---------|---------------|
| **image_quality_analysis.md** | `image_quality` | Baseline quality assessment | 1-10 scores, RGB estimates, angle detection, priority ranking |
| **enhancement_validation.md** | `enhancement_validation` | Before/after comparison | Δ metrics, ✅/⚠️/❌ indicators, recommendations |
| **preprocessing_diagnosis.md** | `preprocessing_diagnosis` | Failure root cause analysis | Hypotheses, remediation strategies, validation plans |

### Prompt Usage
```bash
# Baseline quality
uv run python -m AgentQMS.vlm.cli.analyze_defects \
  --image data/image.jpg \
  --mode image_quality \
  --backend openrouter \
  --output vlm_reports/baseline/image_quality.md

# Enhancement validation
uv run python -m AgentQMS.vlm.cli.analyze_defects \
  --image outputs/comparisons/comparison.jpg \
  --mode enhancement_validation \
  --backend openrouter \
  --output vlm_reports/phase1_validation/validation.md

# Debugging diagnosis
uv run python -m AgentQMS.vlm.cli.analyze_defects \
  --image outputs/comparisons/failed_comparison.jpg \
  --mode preprocessing_diagnosis \
  --backend openrouter \
  --output vlm_reports/debugging/diagnosis.md \
  --context "Applied gray-world white-balance, result shows blue tint"
```

---

## Phase Execution Flow

### Phase 1 (Current): Background Normalization + Deskewing

```
Week 1 Day 1: Baseline Assessment
    ├─ README.md (Quick start commands)
    ├─ Implementation Plan (Day 1 checklist)
    └─ VLM Guide (Baseline workflow)
         ↓
    Execute: bash scripts/vlm_baseline_assessment.sh
         ↓
Week 1 Day 2-3: Implement Background Normalization
    ├─ Implementation Plan (Code examples)
    ├─ Priority Plan Revised (Algorithm details)
    └─ Enhancement Quick Reference (Parameters)
         ↓
    Create: scripts/background_normalization.py
         ↓
Week 1 Day 4-5: Validation
    ├─ VLM Guide (Validation workflow)
    ├─ Implementation Plan (Success criteria)
    └─ Execute: bash scripts/vlm_validate_enhancement.sh
         ↓
Week 2: Text Deskewing
    [Similar flow with deskewing scripts]
         ↓
Week 3: Integration & Ablation
    ├─ Implementation Plan (Integration section)
    ├─ Executive Summary (Go/No-Go criteria)
    └─ Master Roadmap (Update completion %)
```

---

## Document Maintenance

### Update Triggers
| Trigger | Document to Update | Action |
|---------|-------------------|--------|
| **Step completed** | Master Roadmap | Mark step complete, update % |
| **Week completed** | Implementation Plan | Update progress tracker |
| **Phase completed** | Executive Summary, Master Roadmap | Update timeline, success metrics |
| **Method selected** | Priority Plan Revised | Document chosen approach |
| **Issue discovered** | README.md, Implementation Plan | Add to known issues |

### Version Control
- All documents use ADS v1.0 frontmatter
- Track updates in `updated` field
- Link related artifacts in frontmatter

---

## Quick Navigation

### By Role
- **Implementer**: Implementation Plan → Priority Plan Revised → Enhancement Quick Reference
- **Validator**: VLM Integration Guide → Implementation Plan (Validation sections)
- **Tracker**: Master Roadmap → Executive Summary → README.md
- **Architect**: Current State Summary → Implementation Plan (Architecture sections)

### By Activity
- **Starting experiment**: README.md → Implementation Plan
- **Coding**: Implementation Plan → Priority Plan Revised
- **Testing**: VLM Integration Guide → Implementation Plan
- **Reporting**: Executive Summary → Master Roadmap
- **Debugging**: VLM Integration Guide (Diagnosis workflow) → Current State Summary

---

**Navigation Tip**: Start with README.md, drill down to Implementation Plan for execution, reference supporting docs as needed.

**Last Updated**: 2025-12-18 05:45 KST
