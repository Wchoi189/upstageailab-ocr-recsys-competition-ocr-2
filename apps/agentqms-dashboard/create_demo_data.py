#!/usr/bin/env python3
"""
Create demo data for AgentQMS Dashboard deployment.
Generates sample artifacts for isolated demo environment.
"""
import os
from pathlib import Path
from datetime import datetime

# Sample artifacts with frontmatter
ARTIFACTS = {
    "implementation_plans/2025-12-01_1000_plan-ocr-feature.md": """---
title: "Implementation Plan: OCR Feature Enhancement"
type: implementation_plan
status: active
created: 2025-12-01 10:00 (KST)
updated: 2025-12-01 10:00 (KST)
phase: 2
priority: high
tags: [ocr, feature, enhancement]
---

# Implementation Plan: OCR Feature Enhancement

## Objective
Enhance OCR accuracy for handwritten text detection using advanced deep learning models.

## Background
Current OCR system achieves 89% accuracy on printed text but struggles with handwritten content (65% accuracy). This plan outlines improvements to reach 95%+ accuracy.

## Tasks

### Phase 1: Research & Preparation
- [x] Literature review of SOTA handwriting recognition models
- [x] Evaluate TrOCR, PARSeq, and CRAFT models
- [ ] Collect benchmark datasets (IAM, RIMES, FUNSD)

### Phase 2: Implementation
- [ ] Implement TrOCR-based text detection pipeline
- [ ] Train on synthetic handwriting data (100K samples)
- [ ] Fine-tune on domain-specific documents

### Phase 3: Testing & Optimization
- [ ] Evaluate on validation set (5K images)
- [ ] Optimize inference time (target: <100ms)
- [ ] A/B test with existing system

## Timeline
- **Week 1**: Research and dataset preparation
- **Week 2-3**: Model implementation and training
- **Week 4**: Testing and optimization
- **Week 5**: Deployment and monitoring

## Success Metrics
- ✅ Accuracy > 95% on handwritten text
- ✅ Inference time < 100ms per image
- ✅ Model size < 500MB
- ✅ Pass all integration tests

## Resources Required
- GPU: 1x NVIDIA A100 (40GB VRAM)
- Storage: 200GB for datasets and checkpoints
- Compute time: ~80 GPU-hours estimated

## Risks & Mitigation
- **Risk**: Model too large for production deployment
  - **Mitigation**: Apply quantization and pruning techniques
- **Risk**: Training data quality issues
  - **Mitigation**: Implement data validation pipeline

## Dependencies
- PyTorch 2.0+
- Transformers library
- Custom data augmentation pipeline

## Next Steps
1. Set up training environment
2. Download and preprocess datasets
3. Begin baseline model training

---
*Implementation plan follows Blueprint Protocol Template (PROTO-GOV-003)*
""",

    "assessments/2025-12-01_1100_assessment-model-performance.md": """---
title: "Assessment: Model Performance Analysis Q4 2025"
type: assessment
status: complete
created: 2025-12-01 11:00 (KST)
updated: 2025-12-01 11:00 (KST)
category: evaluation
tags: [assessment, performance, analysis, ocr]
---

# Assessment: Model Performance Analysis Q4 2025

## Executive Summary
Comprehensive evaluation of OCR model performance across 10,000 test images spanning multiple document types and quality levels. Overall system demonstrates strong performance with room for targeted improvements.

## Methodology
- **Test Set**: 10,000 images (receipts: 4K, forms: 3K, handwritten: 2K, mixed: 1K)
- **Metrics**: F1 Score, Character Error Rate (CER), Word Error Rate (WER)
- **Hardware**: NVIDIA A100 GPU, 64GB RAM
- **Evaluation Period**: Nov 20 - Nov 30, 2025

## Performance Results

### Overall Metrics
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Detection F1 | 0.94 | 0.90 | ✅ PASS |
| Recognition Accuracy | 0.89 | 0.85 | ✅ PASS |
| Processing Speed | 85ms/image | <100ms | ✅ PASS |
| Memory Usage | 2.3GB | <3GB | ✅ PASS |

### By Document Type
- **Receipts**: F1 0.96, CER 0.03 (Excellent)
- **Forms**: F1 0.93, CER 0.05 (Good)
- **Handwritten**: F1 0.78, CER 0.15 (Needs improvement)
- **Mixed Content**: F1 0.91, CER 0.07 (Good)

## Key Findings

### Strengths
1. **Printed Text Performance**: Achieves near-perfect accuracy (98%) on high-quality scanned documents
2. **Speed Optimization**: 40% faster than previous version while maintaining accuracy
3. **Robustness**: Handles various image quality levels (50-300 DPI) effectively

### Weaknesses
1. **Handwriting Recognition**: Accuracy drops significantly for cursive writing (65%)
2. **Low-Light Images**: Performance degrades by 20% in poorly lit conditions
3. **Multi-Language Support**: Limited accuracy for non-Latin scripts (75% vs 94% for English)

### Edge Cases Discovered
- Tables with merged cells cause alignment issues
- Watermarks occasionally misidentified as text
- Vertical text orientation not consistently detected

## Recommendations

### Immediate Actions (Week 1-2)
1. **Fine-tune on domain-specific data**: Focus on handwritten receipts dataset (5K samples)
2. **Implement preprocessing pipeline**: Add image enhancement for low-light conditions
3. **Update post-processing rules**: Filter false positives from watermarks

### Short-Term Improvements (Month 1)
1. Integrate text-line detection model for better layout understanding
2. Add ensemble model for handwriting (combine TrOCR + PARSeq)
3. Implement confidence-based quality gating

### Long-Term Enhancements (Month 2+)
1. Multi-language model training (Korean, Japanese, Chinese priority)
2. Develop specialized models per document type
3. Active learning pipeline for continuous improvement

## Cost-Benefit Analysis
- **Training Cost**: $450 (GPU compute + data labeling)
- **Expected Improvement**: +8% handwriting accuracy
- **ROI Timeline**: 2 months (based on reduced manual review time)

## Success Metrics
- **Handwriting Accuracy**: Target 85% (from current 78%)
- **Processing Speed**: Maintain <100ms
- **Model Size**: Keep under 500MB for deployment

## Conclusion
Model demonstrates production-ready performance for printed text but requires targeted improvements for handwriting and low-quality images. Recommended enhancements are feasible within existing budget and timeline.

## Next Steps
- [x] Share findings with engineering team
- [ ] Implement preprocessing pipeline (Sprint 12)
- [ ] Begin handwriting model fine-tuning (Sprint 13)
- [ ] Retest on expanded dataset (Sprint 14)

---
*Assessment follows standardized evaluation format (v2.0)*
""",

    "bug_reports/2025-12-01_0900_BUG_001_unicode-error.md": """---
title: "BUG-001: Unicode Encoding Error in Korean Text Processing"
type: bug_report
status: resolved
created: 2025-12-01 09:00 (KST)
updated: 2025-12-01 14:00 (KST)
severity: medium
priority: high
tags: [bug, unicode, encoding, korean, i18n]
assignee: ai-agent
---

# BUG-001: Unicode Encoding Error in Korean Text Processing

## Description
Application crashes with `UnicodeDecodeError` when processing Korean documents containing special characters (e.g., ㈜, ㉮, ①). Error occurs during file reading and text extraction phases.

## Impact
- **Severity**: Medium (blocks Korean document processing)
- **Affected Users**: ~200 users (Korean language accounts)
- **Workaround**: Manual conversion to UTF-8 before upload (not user-friendly)

## Environment
- **OS**: Ubuntu 22.04
- **Python**: 3.11.14
- **Module**: `ocr/utils/file_reader.py`
- **First Reported**: 2025-11-28

## Steps to Reproduce
1. Upload Korean PDF with company name "㈜테스트컴퍼니"
2. Run OCR extraction pipeline: `python runners/extract.py --input korean_doc.pdf`
3. Observe crash at file reading stage
4. Error message:
   ```
   UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa4 in position 15: invalid start byte
   ```

## Expected Behavior
- Korean text with special characters should process without errors
- All Unicode code points should be handled correctly
- System should detect and convert non-UTF-8 encodings automatically

## Actual Behavior
- Application crashes with encoding error
- Processing halts, file not extracted
- No automatic encoding detection or conversion

## Root Cause Analysis

### Investigation Findings
1. **File encoding mismatch**: Input files often encoded in CP949/EUC-KR (legacy Korean encoding)
2. **Hard-coded UTF-8**: File reader assumes UTF-8 without detection
3. **No fallback mechanism**: Missing try/except for encoding errors

### Code Review
**File**: `ocr/utils/file_reader.py` (Line 42)
```python
# Problematic code
with open(file_path, 'r', encoding='utf-8') as f:  # ❌ Assumes UTF-8
    content = f.read()
```

## Solution Implemented

### Fix Applied
Added encoding detection with fallback logic:

```python
import chardet

def read_file_with_encoding(file_path):
    """Read file with automatic encoding detection."""
    # Try UTF-8 first (most common)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Detect encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            detected = chardet.detect(raw_data)
            encoding = detected['encoding']

        # Read with detected encoding
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
```

### Changes Made
1. Added `chardet` dependency to `requirements.txt`
2. Updated `file_reader.py` with detection logic
3. Added unit tests for CP949, EUC-KR, UTF-8 files
4. Updated documentation with encoding guidelines

## Testing

### Test Cases
- [x] UTF-8 Korean document (baseline)
- [x] CP949 encoded document with ㈜, ㉮ characters
- [x] EUC-KR encoded document
- [x] Mixed encoding (header UTF-8, body CP949)
- [x] Large file (50MB) encoding detection performance

### Regression Tests
- [x] English documents still process correctly
- [x] Performance impact < 5% (acceptable)
- [x] Memory usage unchanged

## Resolution

### Deployment
- **Commit**: `a1b2c3d4e5f6` on `fix/unicode-encoding`
- **PR**: #234 (reviewed and merged)
- **Deployed**: 2025-12-01 14:00 KST
- **Rollout**: Production (all regions)

### Verification
- ✅ 50 Korean documents processed successfully
- ✅ Zero encoding errors in last 24 hours
- ✅ User reports confirmed fix

## Lessons Learned
1. **Never assume encoding**: Always detect or allow configuration
2. **i18n testing**: Add non-English test cases to CI pipeline
3. **Error handling**: Implement graceful fallbacks for encoding issues

## Follow-Up Actions
- [ ] Add encoding detection to all file I/O operations
- [ ] Create comprehensive i18n testing suite
- [ ] Document encoding best practices in dev guide

## Related Issues
- Related to #198 (Japanese text processing)
- Blocks #245 (Multi-language OCR feature)

---
**Status**: ✅ RESOLVED
**Resolution Time**: 5 hours (reported 09:00, fixed 14:00)
**Fix Verified**: Production deployment successful
""",

    "audits/2025-12-01_1200_audit-code-quality.md": """---
title: "Audit: Code Quality Review Q4 2025"
type: audit
status: active
created: 2025-12-01 12:00 (KST)
updated: 2025-12-01 12:00 (KST)
category: quality
tags: [audit, quality, review, compliance]
auditor: ai-agent
scope: codebase
---

# Audit: Code Quality Review Q4 2025

## Executive Summary
Quarterly code quality audit covering 150K lines of Python code across OCR pipeline, AgentQMS framework, and dashboard application. Overall health: **GOOD** with 3 areas requiring attention.

## Audit Scope
- **Period**: Q4 2025 (Oct 1 - Dec 1)
- **Modules Reviewed**: 247 Python files, 15 frontend components
- **Tools Used**: pylint, mypy, ruff, pytest, coverage.py
- **Standards**: PEP 8, project style guide v2.1

## Metrics Summary

| Category | Current | Target | Status |
|----------|---------|--------|--------|
| Type Hints Coverage | 95% | 90% | ✅ PASS |
| Docstring Coverage | 78% | 90% | ⚠️ WARN |
| Unit Test Coverage | 87% | 85% | ✅ PASS |
| Linting Score | 9.2/10 | 8.0/10 | ✅ PASS |
| Cyclomatic Complexity | Avg 4.2 | <10 | ✅ PASS |
| Code Duplication | 3 instances | 0 | ⚠️ WARN |

## Detailed Findings

### ✅ Strengths

#### 1. Type Hints (95% Coverage)
Excellent adoption of type hints across codebase. Recent improvements:
- All public APIs have complete type signatures
- Generic types properly specified
- Return types documented

**Example** (`ocr/models/detector.py`):
```python
def predict(
    self,
    images: torch.Tensor,
    threshold: float = 0.5
) -> tuple[list[np.ndarray], list[float]]:
    """Type hints enable better IDE support and catch errors early."""
    ...
```

#### 2. Unit Testing (87% Line Coverage)
Robust test suite with good edge case coverage:
- Critical path: 95% coverage
- Utils/helpers: 82% coverage
- Integration tests: 45 scenarios

#### 3. Code Organization
- Clear module boundaries
- Dependency injection used appropriately
- Configuration management centralized

### ⚠️ Areas for Improvement

#### 1. Docstring Coverage (78%)
**Gap**: Utility modules and internal helpers lack documentation.

**Missing Docstrings**:
- `ocr/utils/polygon_utils.py`: 12 functions (0% documented)
- `AgentQMS/agent_tools/utilities/`: 8 modules (45% documented)
- Dashboard backend routes: 3 files (60% documented)

**Recommendation**:
```python
# Before (no docstring)
def normalize_polygon(points):
    return sorted(points, key=lambda p: (p[1], p[0]))

# After (with docstring)
def normalize_polygon(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Normalize polygon points to canonical order.

    Sorts points top-to-bottom, left-to-right to ensure consistent
    ordering regardless of input annotation order.

    Args:
        points: List of (x, y) coordinate tuples

    Returns:
        Sorted list of points in reading order

    Example:
        >>> normalize_polygon([(10, 20), (5, 15), (10, 15)])
        [(5, 15), (10, 15), (10, 20)]
    """
    return sorted(points, key=lambda p: (p[1], p[0]))
```

#### 2. Code Duplication (3 Instances)
**Duplicate Logic Found**:

1. **Validation Functions** (2 instances)
   - Location: `ocr/validation/schema.py` + `AgentQMS/agent_tools/compliance/validate_artifacts.py`
   - Issue: Frontmatter validation duplicated
   - Impact: Maintenance burden, inconsistent behavior
   - **Fix**: Extract to shared `validation_utils.py`

2. **Path Resolution** (1 instance)
   - Location: `backend/fs_utils.py` + `AgentQMS/agent_tools/utils/paths.py`
   - Issue: Similar path normalization logic
   - Impact: Duplicate security checks
   - **Fix**: Unify into single utility module

#### 3. Test Data Fixtures
**Gap**: Insufficient shared test fixtures lead to duplicated test setup code.

**Recommendation**: Expand `conftest.py` with:
```python
@pytest.fixture
def sample_korean_document():
    """Fixture for Korean text processing tests."""
    return {
        "text": "㈜테스트컴퍼니",
        "encoding": "cp949",
        "expected_utf8": "㈜테스트컴퍼니"
    }
```

## Compliance Checks

### Framework Conventions
- ✅ **Artifact Naming**: 100% compliant (YYYY-MM-DD_HHMM format)
- ✅ **Frontmatter Format**: 98% compliant (3 files missing version tag)
- ✅ **Directory Structure**: Fully compliant with AgentQMS v0.3.1

### Code Standards
- ✅ **Import Order**: isort verified (PEP 8 compliant)
- ✅ **Line Length**: 95% under 100 chars (5% justified exceptions)
- ⚠️ **Magic Numbers**: 12 instances (should use constants)

**Example Fix**:
```python
# Before
if confidence > 0.85:  # Magic number
    accept_prediction()

# After
CONFIDENCE_THRESHOLD = 0.85  # Configurable constant
if confidence > CONFIDENCE_THRESHOLD:
    accept_prediction()
```

## Security Review
- ✅ No SQL injection vectors (using parameterized queries)
- ✅ Path traversal protection in file utilities
- ✅ Input validation on all API endpoints
- ⚠️ Consider rate limiting for public API endpoints (Phase 5)

## Performance Review
- ✅ No N+1 query issues
- ✅ Proper database indexing
- ✅ Caching implemented for expensive operations
- Benchmarks stable within 5% variance

## Action Items

### High Priority (This Sprint)
- [ ] Add docstrings to `ocr/utils/polygon_utils.py`
- [ ] Refactor duplicate validation logic into shared module
- [ ] Create comprehensive test fixtures in `conftest.py`

### Medium Priority (Next Sprint)
- [ ] Update CI to enforce 90% docstring coverage
- [ ] Add pylint rule for magic numbers
- [ ] Document encoding best practices

### Low Priority (Backlog)
- [ ] Migrate remaining type: ignore comments to proper types
- [ ] Add performance regression tests
- [ ] Set up automated dependency vulnerability scanning

## Metrics Trend (vs Q3 2025)

| Metric | Q3 | Q4 | Change |
|--------|----|----|--------|
| Type Hints | 88% | 95% | +7% ⬆️ |
| Docstrings | 75% | 78% | +3% ➡️ |
| Test Coverage | 83% | 87% | +4% ⬆️ |
| Linting Score | 8.9 | 9.2 | +0.3 ⬆️ |

**Trend**: ⬆️ Overall improvement, docstring coverage needs focus.

## Recommendations for Next Quarter

1. **Documentation Sprint**: Allocate 1 week to docstring coverage
2. **Refactoring**: Extract common utilities to reduce duplication
3. **Automation**: Add pre-commit hooks for style enforcement
4. **Training**: Share code review best practices with team

## Conclusion
Codebase demonstrates strong engineering practices with consistent improvement trend. Primary focus areas: documentation coverage and eliminating code duplication. No critical issues identified.

**Overall Status**: ✅ COMPLIANT with 2 actionable recommendations

---
**Next Audit**: 2026-03-01 (Q1 2026 Review)
**Auditor**: AI Agent (AgentQMS Framework)
**Report Version**: 2.0
""",

    "design_documents/2025-12-01_1300_design-api-architecture.md": """---
title: "Design: AgentQMS Dashboard API Architecture"
type: design
status: active
created: 2025-12-01 13:00 (KST)
updated: 2025-12-01 13:00 (KST)
category: architecture
tags: [design, api, architecture, rest, fastapi]
version: "2.0"
---

# Design: AgentQMS Dashboard API Architecture

## Overview
RESTful API architecture for AgentQMS Manager Dashboard, providing programmatic access to artifact management, validation tools, and tracking database. Built with FastAPI for high performance and automatic OpenAPI documentation.

## Design Goals
1. **Developer Experience**: Intuitive endpoints with comprehensive documentation
2. **Performance**: <100ms response time for 95th percentile
3. **Reliability**: 99.9% uptime with graceful error handling
4. **Security**: Authentication, authorization, and input validation
5. **Extensibility**: Easy to add new endpoints and features

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│                  (Port 3000)                            │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP/REST
                 │ /api/v1/*
┌────────────────▼────────────────────────────────────────┐
│              FastAPI Backend                            │
│              (Port 8000)                                │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │          API Routes (v1)                         │  │
│  │  ┌────────────┬────────────┬─────────────────┐  │  │
│  │  │ Artifacts  │ Compliance │   Tracking      │  │  │
│  │  │  /api/v1/  │   /api/v1/ │   /api/v1/     │  │  │
│  │  │ artifacts/ │ compliance/│   tracking/    │  │  │
│  │  └─────┬──────┴─────┬──────┴────┬────────────┘  │  │
│  │        │            │           │                │  │
│  │  ┌─────▼────────────▼───────────▼────────────┐  │  │
│  │  │      Utilities & Services                  │  │  │
│  │  │  • fs_utils (file system ops)             │  │  │
│  │  │  • tool_runner (subprocess wrapper)       │  │  │
│  │  │  • validation (schema checks)             │  │  │
│  │  └───────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Middleware                               │  │
│  │  • CORS (Cross-Origin)                          │  │
│  │  • Error Handling                               │  │
│  │  • Logging                                      │  │
│  │  • Rate Limiting (TODO: Phase 5)                │  │
│  └──────────────────────────────────────────────────┘  │
└────────────┬────────────────────────────────────────────┘
             │
    ┌────────▼──────────┐  ┌───────────────────┐
    │  File System      │  │  AgentQMS Tools   │
    │  (artifacts/)     │  │  (make commands)  │
    └───────────────────┘  └───────────────────┘
```

## API Endpoints

### Base URL
- **Development**: `http://localhost:8000/api/v1`
- **Production**: `https://api.agentqms.com/api/v1`

### 1. Artifacts Module (`/api/v1/artifacts/`)

#### List Artifacts
```http
GET /api/v1/artifacts/list?type={type}&status={status}
```

**Parameters**:
- `type` (optional): Filter by artifact type (plan, assessment, audit, bug_report, design)
- `status` (optional): Filter by status (active, completed, archived)

**Response**:
```json
{
  "artifacts": [
    {
      "id": "2025-12-01_1000_plan-ocr-feature",
      "title": "Implementation Plan: OCR Feature",
      "type": "implementation_plan",
      "status": "active",
      "created": "2025-12-01T10:00:00+09:00",
      "updated": "2025-12-01T10:00:00+09:00",
      "path": "docs/artifacts/implementation_plans/2025-12-01_1000_plan-ocr-feature.md"
    }
  ],
  "total": 42,
  "filtered": 5
}
```

#### Get Artifact
```http
GET /api/v1/artifacts/{artifact_id}
```

**Response**:
```json
{
  "id": "2025-12-01_1000_plan-ocr-feature",
  "frontmatter": { ... },
  "content": "# Implementation Plan...",
  "metadata": {
    "size": 4523,
    "last_modified": "2025-12-01T10:00:00+09:00"
  }
}
```

#### Create Artifact
```http
POST /api/v1/artifacts/create
Content-Type: application/json

{
  "type": "implementation_plan",
  "title": "New Feature Plan",
  "content": "# Plan content...",
  "tags": ["feature", "urgent"]
}
```

**Response**: `201 Created`
```json
{
  "id": "2025-12-02_0930_plan-new-feature",
  "path": "docs/artifacts/implementation_plans/2025-12-02_0930_plan-new-feature.md",
  "created": true
}
```

#### Update Artifact
```http
PATCH /api/v1/artifacts/{artifact_id}
Content-Type: application/json

{
  "status": "completed",
  "content": "Updated content..."
}
```

#### Delete Artifact
```http
DELETE /api/v1/artifacts/{artifact_id}
```

### 2. Compliance Module (`/api/v1/compliance/`)

#### Run Validation
```http
POST /api/v1/compliance/validate
Content-Type: application/json

{
  "scope": "all",  // or "frontmatter", "naming", "structure"
  "fix": false     // true to auto-fix issues
}
```

**Response**:
```json
{
  "status": "completed",
  "violations": [
    {
      "file": "2025-11-01_assessment.md",
      "rule": "frontmatter_missing_field",
      "field": "version",
      "severity": "warning"
    }
  ],
  "summary": {
    "total_files": 42,
    "violations": 3,
    "fixed": 0
  }
}
```

#### Check Compliance
```http
GET /api/v1/compliance/status
```

**Response**:
```json
{
  "compliant": true,
  "score": 98.5,
  "checks": {
    "naming_convention": "pass",
    "frontmatter_format": "pass",
    "directory_structure": "pass",
    "boundary_enforcement": "warning"
  }
}
```

### 3. Tools Module (`/api/v1/tools/`)

#### Execute Tool
```http
POST /api/v1/tools/exec
Content-Type: application/json

{
  "tool_id": "validate",
  "args": {
    "scope": "all",
    "verbose": true
  }
}
```

**Response**:
```json
{
  "success": true,
  "output": "=== Validation Report ===\n✅ 42 artifacts validated\n⚠️ 3 warnings found",
  "error": "",
  "execution_time_ms": 245
}
```

**Available Tools**:
- `validate` - Run artifact validation
- `compliance` - Check compliance status
- `boundary` - Check boundary enforcement
- `discover` - Discover available tools
- `status` - Get system status

### 4. Tracking Module (`/api/v1/tracking/`)

#### Get Tracking Status
```http
GET /api/v1/tracking/status?kind={kind}
```

**Parameters**:
- `kind`: `plan`, `experiment`, `debug`, `refactor`, or `all`

**Response**:
```json
{
  "kind": "all",
  "status": "=== Tracking Database ===\nPlans: 15 (3 active)\nExperiments: 8 completed\n...",
  "structured": {
    "plans": { "active": 3, "completed": 12 },
    "experiments": { "running": 1, "completed": 8 }
  },
  "success": true
}
```

### 5. System Module (`/api/v1/system/`)

#### Health Check
```http
GET /api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-12-01T12:00:00+09:00",
  "services": {
    "filesystem": "ok",
    "tracking_db": "ok",
    "tools": "ok"
  }
}
```

## Data Models

### Artifact Schema
```python
class Artifact(BaseModel):
    id: str
    title: str
    type: Literal["implementation_plan", "assessment", "audit", "bug_report", "design", "research"]
    status: Literal["active", "draft", "completed", "archived"]
    created: datetime
    updated: datetime
    tags: list[str]
    content: str
    frontmatter: dict[str, Any]
```

### Tool Execution Request
```python
class ToolExecRequest(BaseModel):
    tool_id: str
    args: dict[str, Any] = {}
```

### Validation Result
```python
class ValidationViolation(BaseModel):
    file: str
    rule: str
    severity: Literal["error", "warning", "info"]
    message: str
    fix_available: bool
```

## Error Handling

### HTTP Status Codes
- `200 OK` - Successful request
- `201 Created` - Resource created
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid artifact type",
    "details": {
      "field": "type",
      "allowed_values": ["plan", "assessment", "audit"]
    }
  }
}
```

## Security (Phase 5)

### Authentication
```http
Authorization: Bearer <JWT_TOKEN>
```

### Rate Limiting
- 100 requests/minute per IP
- 1000 requests/hour per API key
- Burst allowance: 20 requests

### Input Validation
- Path traversal prevention
- SQL injection protection (if DB added)
- XSS prevention in responses
- File size limits (10MB per artifact)

## Performance Targets

| Endpoint | Target | P95 | P99 |
|----------|--------|-----|-----|
| GET /artifacts/list | <50ms | 45ms | 80ms |
| GET /artifacts/{id} | <30ms | 28ms | 50ms |
| POST /artifacts/create | <100ms | 95ms | 150ms |
| POST /tools/exec | <500ms | 450ms | 1000ms |

## Monitoring & Logging

### Metrics Collected
- Request count by endpoint
- Response time (mean, p95, p99)
- Error rate by type
- Active connections

### Logging Format
```json
{
  "timestamp": "2025-12-01T12:00:00Z",
  "level": "INFO",
  "endpoint": "/api/v1/artifacts/list",
  "method": "GET",
  "status": 200,
  "duration_ms": 42,
  "user_id": "demo_user"
}
```

## Versioning Strategy

### URL Versioning
- Current: `/api/v1/*`
- Future: `/api/v2/*` (backward compatible)

### Deprecation Policy
- 6-month notice for breaking changes
- Sunset header in responses
- Migration guide provided

## Testing Strategy

### Unit Tests
- Mock file system operations
- Test validation logic
- Schema validation tests

### Integration Tests
- End-to-end API workflows
- Database integration (when added)
- Tool execution tests

### Performance Tests
- Load testing (1000 req/s)
- Concurrency testing
- Memory leak detection

## Future Enhancements (Roadmap)

### Phase 4 (Q1 2026)
- [ ] WebSocket support for real-time updates
- [ ] Batch operations endpoint
- [ ] Export artifacts (PDF, ZIP)

### Phase 5 (Q2 2026)
- [ ] GraphQL alternative API
- [ ] OAuth2 authentication
- [ ] Webhook notifications
- [ ] API usage analytics dashboard

## References
- FastAPI Documentation: https://fastapi.tiangolo.com
- OpenAPI Specification: https://spec.openapis.org/oas/v3.1.0
- REST API Design Best Practices: https://restfulapi.net

---
**API Version**: 1.0.0
**Last Updated**: 2025-12-01
**Owner**: AgentQMS Team
"""
}

def create_demo_stubs():
    """Create demo stub scripts."""
    stubs_dir = Path("demo_scripts")
    stubs_dir.mkdir(exist_ok=True)

    stubs = {
        "validate_stub.py": '''#!/usr/bin/env python3
"""Demo stub: artifact validation."""
import sys

def main():
    print("=" * 60)
    print("ARTIFACT VALIDATION REPORT")
    print("=" * 60)
    print()
    print("✅ Frontmatter validation: PASS (5/5 artifacts)")
    print("✅ Naming convention: PASS (5/5 files)")
    print("⚠️  Boundary check: WARN (demo mode - no boundaries)")
    print()
    print("Total artifacts scanned: 5")
    print("Violations found: 0 errors, 1 warning")
    print()
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
''',
        "compliance_stub.py": '''#!/usr/bin/env python3
"""Demo stub: compliance check."""
import sys

def main():
    print("=" * 60)
    print("COMPLIANCE CHECK REPORT")
    print("=" * 60)
    print()
    print("📋 Framework Conventions:")
    print("  ✅ Artifact naming: COMPLIANT")
    print("  ✅ Frontmatter format: COMPLIANT")
    print("  ✅ Directory structure: COMPLIANT")
    print()
    print("📋 Demo Artifacts:")
    print("  ✅ 5 artifacts validated")
    print("  ✅ 100% compliance rate")
    print()
    print("Overall Status: ✅ COMPLIANT")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
''',
        "tracking_stub.py": '''#!/usr/bin/env python3
"""Demo stub: tracking database status."""
import sys
import json

def main():
    status = {
        "plans": {"active": 1, "completed": 4, "total": 5},
        "experiments": {"running": 0, "completed": 3, "failed": 0, "total": 3},
        "debug_sessions": {"active": 0, "completed": 2, "total": 2}
    }

    print("=" * 60)
    print("TRACKING DATABASE STATUS (DEMO)")
    print("=" * 60)
    print()
    print(f"📊 Implementation Plans: {status['plans']['active']} active, {status['plans']['completed']} complete")
    print(f"🧪 Experiments: {status['experiments']['running']} running, {status['experiments']['completed']} complete")
    print(f"🐛 Debug Sessions: {status['debug_sessions']['active']} active, {status['debug_sessions']['completed']} complete")
    print()
    print("Detailed Status:")
    print(json.dumps(status, indent=2))
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    }

    for filename, content in stubs.items():
        stub_path = stubs_dir / filename
        stub_path.write_text(content)
        stub_path.chmod(0o755)  # Make executable
        print(f"✅ Created: {stub_path}")

def main():
    """Create demo data and stubs."""
    print("=" * 60)
    print("CREATING DEMO DATA FOR AGENTQMS DASHBOARD")
    print("=" * 60)
    print()

    # Create demo_data directory structure
    base_dir = Path("demo_data/artifacts")

    for artifact_path, content in ARTIFACTS.items():
        full_path = base_dir / artifact_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        print(f"✅ Created: {full_path}")

    print()
    print(f"✅ Created {len(ARTIFACTS)} sample artifacts")
    print()

    # Create demo stubs
    create_demo_stubs()

    print()
    print("=" * 60)
    print("DEMO DATA CREATION COMPLETE")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("1. Set environment: export DEMO_MODE=true")
    print("2. Start backend: cd backend && python server.py")
    print("3. Start frontend: cd frontend && npm run dev")
    print("4. Access demo: http://localhost:3000")
    print()
    print("See DEMO_DEPLOYMENT_GUIDE.md for deployment instructions.")
    print()

if __name__ == "__main__":
    main()
