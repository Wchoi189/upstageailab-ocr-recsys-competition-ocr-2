#!/bin/bash
# Create demo data for AgentQMS Dashboard

mkdir -p demo_data/artifacts/{implementation_plans,assessments,audits,bug_reports,design_documents}
mkdir -p demo_scripts

# Sample artifacts created directly via heredoc
cat > demo_data/artifacts/implementation_plans/2025-12-01_1000_plan-ocr-feature.md << 'EOF'
---
title: "Implementation Plan: OCR Feature Enhancement"
type: implementation_plan
status: active
created: 2025-12-01 10:00 (KST)
phase: 2
priority: high
tags: [ocr, feature, enhancement]
---

# Implementation Plan: OCR Feature Enhancement

## Objective
Enhance OCR accuracy for handwritten text detection.

## Tasks
- [x] Research SOTA models
- [ ] Implement text detection pipeline
- [ ] Train on synthetic data  
- [ ] Evaluate on validation set

## Timeline
- Week 1: Research
- Week 2-3: Implementation
- Week 4: Testing

## Success Metrics
- Accuracy > 95%
- Inference time < 100ms
EOF

cat > demo_data/artifacts/assessments/2025-12-01_1100_assessment-model-performance.md << 'EOF'
---
title: "Assessment: Model Performance Analysis"
type: assessment
status: complete
created: 2025-12-01 11:00 (KST)
category: evaluation
tags: [assessment, performance]
---

# Assessment: Model Performance Analysis

## Summary
Evaluated OCR model performance on test dataset.

## Findings
- Detection F1: 0.94
- Recognition accuracy: 0.89
- Processing speed: 85ms/image

## Recommendations
1. Fine-tune on domain-specific data
2. Optimize inference pipeline
3. Add post-processing rules
EOF

cat > demo_data/artifacts/bug_reports/2025-12-01_0900_BUG_001_unicode-error.md << 'EOF'
---
title: "BUG-001: Unicode Encoding Error"
type: bug_report
status: resolved
created: 2025-12-01 09:00 (KST)
severity: medium
tags: [bug, unicode, encoding]
---

# BUG-001: Unicode Encoding Error

## Description
Application crashes when processing Korean text with special characters.

## Root Cause
File encoding mismatch (CP949 vs UTF-8).

## Fix
Added explicit UTF-8 encoding in file reader.

## Status
✅ Resolved in commit a1b2c3d
EOF

cat > demo_data/artifacts/audits/2025-12-01_1200_audit-code-quality.md << 'EOF'
---
title: "Audit: Code Quality Review"
type: audit
status: active
created: 2025-12-01 12:00 (KST)
tags: [audit, quality]
---

# Audit: Code Quality Review

## Scope
Review codebase for compliance with style guide.

## Findings
- ✅ Type hints: 95% coverage
- ⚠️ Docstrings: 78% coverage (target: 90%)
- ✅ Unit tests: 87% line coverage

## Action Items
- [ ] Add docstrings to utility modules
- [ ] Refactor duplicate validation logic
EOF

cat > demo_data/artifacts/design_documents/2025-12-01_1300_design-api-architecture.md << 'EOF'
---
title: "Design: API Architecture"
type: design
status: active
created: 2025-12-01 13:00 (KST)
category: architecture
tags: [design, api]
---

# Design: API Architecture

## Overview
RESTful API for AgentQMS artifact management.

## Endpoints

### Artifacts
- GET /api/v1/artifacts/list
- POST /api/v1/artifacts/create
- GET /api/v1/artifacts/{id}

### Tools
- POST /api/v1/tools/exec

### System
- GET /api/v1/health
EOF

# Create demo stubs
cat > demo_scripts/validate_stub.py << 'EOF'
#!/usr/bin/env python3
import sys

print("=" * 60)
print("ARTIFACT VALIDATION REPORT")
print("=" * 60)
print()
print("✅ Frontmatter validation: PASS (5/5 artifacts)")
print("✅ Naming convention: PASS (5/5 files)")
print()
print("Total artifacts scanned: 5")
print("Violations found: 0")
print("=" * 60)
sys.exit(0)
EOF

cat > demo_scripts/compliance_stub.py << 'EOF'
#!/usr/bin/env python3
import sys

print("=" * 60)
print("COMPLIANCE CHECK REPORT")
print("=" * 60)
print()
print("📋 Framework Conventions:")
print("  ✅ Artifact naming: COMPLIANT")
print("  ✅ Frontmatter format: COMPLIANT")
print()
print("Overall Status: ✅ COMPLIANT")
print("=" * 60)
sys.exit(0)
EOF

cat > demo_scripts/tracking_stub.py << 'EOF'
#!/usr/bin/env python3
import sys
import json

status = {
    "plans": {"active": 1, "completed": 4},
    "experiments": {"running": 0, "completed": 3}
}

print("=" * 60)
print("TRACKING DATABASE STATUS")
print("=" * 60)
print(f"📊 Plans: {status['plans']['active']} active, {status['plans']['completed']} complete")
print(f"🧪 Experiments: {status['experiments']['running']} running")
print(json.dumps(status, indent=2))
print("=" * 60)
sys.exit(0)
EOF

chmod +x demo_scripts/*.py

echo ""
echo "✅ Created 5 sample artifacts in demo_data/artifacts/"
echo "✅ Created 3 demo stubs in demo_scripts/"
echo ""
echo "Next: export DEMO_MODE=true && make dev"
