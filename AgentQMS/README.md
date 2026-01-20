# AgentQMS - Quality Management System for AI Agents

**Version:** 0.3.0
**Status:** Production Ready
**Last Updated:** 2026-01-20

---

## 🎯 Overview

AgentQMS is a unified quality management system for AI agents that provides:
- **Path-aware standard discovery** (~85% token reduction)
- **Unified CLI interface** (single `qms` command)
- **Artifact lifecycle management** (create, validate, monitor)
- **Compliance monitoring and reporting**

---

## 🚀 Quick Start

### Installation

The `qms` CLI is globally accessible:

```bash
# Verify installation
qms --version

# Get help
qms --help
```

### Basic Usage

```bash
# Create an artifact
qms artifact create --type implementation_plan --name my-feature --title "My Feature"

# Validate all artifacts
qms validate --all

# Monitor compliance
qms monitor --check

# Generate path-aware configuration
qms generate-config --path ocr/inference
```

---

## 📚 Commands Reference

### Artifact Management

```bash
# Create artifacts
qms artifact create --type <type> --name <name> --title "<title>"

# Validate artifacts
qms artifact validate --all
qms validate --file path/to/artifact.md
qms validate --directory docs/artifacts

# Update indexes
qms artifact update-indexes

# Check compliance
qms artifact check-compliance
```

**Supported artifact types:**
- `implementation_plan`
- `assessment`
- `design_document`
- `bug_report`
- `audit`
- `walkthrough`

### Validation & Compliance

```bash
# Run validation
qms validate --all              # Validate all artifacts
qms validate --file <path>      # Validate specific file
qms validate --check-naming     # Check naming only

# Monitor compliance
qms monitor --check             # Run compliance check
qms monitor --report            # Generate compliance report
qms monitor --alert             # Generate alerts
qms monitor --fix-suggestions   # Show fix suggestions
```

### Feedback & Quality

```bash
# Report issues
qms feedback report \
  --issue-type "bug" \
  --description "Issue description" \
  --file-path "path/to/file.md" \
  --severity "high"

# Suggest improvements
qms feedback suggest \
  --area "testing" \
  --current "Manual tests" \
  --suggested "Automated tests" \
  --rationale "Reduce manual effort"

# List feedback
qms feedback list --status open

# Quality checks
qms quality --check             # Run all quality checks
qms quality --consistency       # Check consistency only
qms quality --tool-paths        # Check for outdated paths
```

### Path-Aware Configuration

```bash
# Generate effective config for current context
qms generate-config --path ocr/inference

# Preview without writing
qms generate-config --path tests --dry-run

# Custom output location
qms generate-config --path configs --output custom-effective.yaml
```

---

## 🏗️ Architecture

### Core Components

```
AgentQMS/
├── bin/
│   ├── qms                          # Unified CLI (main entry point)
│   ├── generate-effective-config.py # Path-aware config generator
│   ├── validate-registry.py         # Registry validator
│   └── monitor-token-usage.py       # Token usage analysis
├── standards/
│   └── registry.yaml                # Single source of truth for standards
├── .agentqms/
│   ├── settings.yaml                # Configuration
│   ├── effective.yaml               # Generated effective configuration
│   ├── plugins/                     # Plugin definitions
│   └── state/                       # Runtime state
└── tools/
    ├── core/                        # Core utilities
    ├── compliance/                  # Compliance tools
    └── utils/                       # Utility modules
```

### Standards Registry

**File:** `AgentQMS/standards/registry.yaml`

The registry is the single source of truth for:
- Standard file mappings
- Path patterns for discovery
- Keywords for context matching
- Task-specific standard sets

**Path-Aware Discovery:**
- Automatically loads relevant standards based on working directory
- Reduces token usage from ~12,000 to ~1,500 tokens (87.5% reduction)
- Example: Working in `ocr/inference/` loads only 3 inference-related standards

---

## 📊 Performance

### Token Usage Improvements

| Metric | Before | After | Savings |
|--------|--------|-------|---------|
| **Tool Interface** | 5 separate tools (~500 tokens) | 1 unified CLI (~200 tokens) | 300 tokens |
| **Standards Loading** | All 24 standards (~12,000 tokens) | 3 relevant standards (~1,500 tokens) | 10,500 tokens |
| **Total per Session** | ~12,500 tokens | ~1,700 tokens | **~10,800 tokens (85.6%)** |

### Monitor Token Usage

```bash
# Analyze token savings
python AgentQMS/bin/monitor-token-usage.py

# Test path-aware discovery
python AgentQMS/bin/monitor-token-usage.py --path ocr/inference

# Detailed analysis
python AgentQMS/bin/monitor-token-usage.py --detailed
```

---

## 📖 Documentation

- **This README** - Primary reference (you are here)
- `AGENTS.yaml` - AI agent entrypoint
- `MIGRATION_GUIDE.md` - Historical migration reference
- `DEPRECATION_PLAN.md` - Legacy system deprecation (COMPLETED)
- `standards/registry.yaml` - Standards reference

---

## 🔧 Configuration

### Settings File

**Location:** `AgentQMS/.agentqms/settings.yaml`

Key settings:
```yaml
framework:
  validation:
    strict_mode: true

context_integration:
  enabled: true
  auto_load_threshold: 5

paths:
  artifacts: docs/artifacts
  docs: docs

tool_mappings:
  tool_mappings:
    qms:
      path: ../bin/qms
      description: Unified AgentQMS CLI tool
```

### Effective Configuration

Generated dynamically with path-aware discovery:

```bash
# Generate for current context
qms generate-config --path $(pwd)

# View result
cat AgentQMS/.agentqms/effective.yaml
```

---

## 🧪 Examples

### Create and Validate an Artifact

```bash
# Create implementation plan
qms artifact create \
  --type implementation_plan \
  --name user-authentication \
  --title "User Authentication System"

# Validate it
qms validate --file docs/artifacts/implementation_plans/user-authentication.md

# Run full compliance check
qms monitor --check
```

### Path-Aware Workflow

```bash
# Working on OCR inference
cd ocr/inference/

# Generate context-aware config
qms generate-config --path . --dry-run

# See only relevant standards loaded:
# - python-core.yaml
# - file-placement-rules.yaml
# - tool-catalog.yaml
# (87.5% reduction from loading all 24 standards)
```

### Feedback Loop

```bash
# Report a bug
qms feedback report \
  --issue-type "bug" \
  --description "Validation fails on valid frontmatter" \
  --file-path "AgentQMS/tools/compliance/validate.py" \
  --severity "high"

# Check feedback status
qms feedback list --status open
```

---

## 🚨 Troubleshooting

### qms command not found

If `qms` is not accessible:

```bash
# Option 1: Use full path
./AgentQMS/bin/qms --help

# Option 2: Add to PATH
export PATH="$PWD/AgentQMS/bin:$PATH"

# Option 3: Recreate symlink (requires sudo)
sudo ln -sf $(pwd)/AgentQMS/bin/qms /usr/local/bin/qms
```

### Validation errors

```bash
# Run with verbose output
qms validate --all

# Check specific file
qms validate --file path/to/artifact.md

# Get fix suggestions
qms monitor --fix-suggestions
```

### Token usage not reducing

```bash
# Verify path-aware discovery is working
qms generate-config --path your/working/directory --dry-run

# Check active standards count (should be 3-5, not 24)
python AgentQMS/bin/monitor-token-usage.py --path your/working/directory
```

---

## 🤝 Contributing

### Reporting Issues

```bash
qms feedback report \
  --issue-type "bug|feature|question" \
  --description "Description" \
  --severity "low|medium|high"
```

### Suggesting Improvements

```bash
qms feedback suggest \
  --area "Area for improvement" \
  --current "Current state" \
  --suggested "Suggested change" \
  --rationale "Why this is better"
```

---

## 📜 License

Part of the Upstage OCR Competition project.

---

## 🔗 Related

- **Project Compass:** `project_compass/` - Project navigation and management
- **MCP Server:** `AgentQMS/mcp_server.py` - Model Context Protocol integration
- **Standards Registry:** `AgentQMS/standards/registry.yaml` - Standards reference

---

## 📞 Support

For questions or issues:
1. Use `qms feedback report` for bugs
2. Use `qms feedback suggest` for improvements
3. Check `AgentQMS/MIGRATION_GUIDE.md` for historical context
4. Review `AgentQMS/standards/registry.yaml` for standards reference

---

**AgentQMS v0.3.0** - Unified, efficient, unambiguous.
