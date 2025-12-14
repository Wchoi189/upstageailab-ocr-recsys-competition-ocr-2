# AgentQMS Ultra-Concise Usage Guide

**Location**: Run all commands from `AgentQMS/interface/`.

## 🛠️ Core Artifact Workflows

### 1. Start a Task (Planning)
**ALWAYS** start here for multi-step work.
```bash
make create-plan NAME=my-feature TITLE="Implement Feature X"
```

### 2. Validate Work
**ALWAYS** run before finishing.
```bash
make validate          # Check validity of artifacts
make compliance        # Full compliance check
```

## 📋 Other Artifacts
- **Assessment** (Research/Analysis):
  `make create-assessment NAME=topic-analysis TITLE="Analysis of X"`
- **Audit** (Code Review/System Check):
  `make create-audit NAME=security-audit TITLE="Security Review"`
- **Bug Report** (Found an issue?):
  `make create-bug-report NAME=issue-desc TITLE="Fixing Issue Y"`

## 🔍 Discovery
- **List Tools**: `make discover`
- **Help**: `make help`

## 🧩 Context
- **Load Context**: `make context TASK="debugging ocr pipeline"`
