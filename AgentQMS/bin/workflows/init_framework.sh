#!/usr/bin/env bash
#
# AgentQMS Framework Initialization Script
#
# Scaffolds a minimal AgentQMS structure into a new project for rapid adoption.
# Copies essential framework files while remaining project-agnostic.
#
# Usage:
#   ./init_framework.sh [target_directory]
#   make agentqms-init TARGET=/path/to/project
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTQMS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Target directory (default: current directory)
TARGET_DIR="${1:-.}"

echo -e "${BLUE}🚀 AgentQMS Framework Initialization${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Validate target directory
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}⚠️  Target directory does not exist: $TARGET_DIR${NC}"
    echo "Creating directory..."
    mkdir -p "$TARGET_DIR"
fi

cd "$TARGET_DIR"
TARGET_DIR="$(pwd)"

echo -e "Target: ${GREEN}$TARGET_DIR${NC}"
echo ""

# Check if AgentQMS already exists
if [ -d "$TARGET_DIR/AgentQMS" ]; then
    echo -e "${YELLOW}⚠️  AgentQMS directory already exists in target${NC}"
    echo "Aborting to avoid overwriting existing framework."
    exit 1
fi

echo -e "${BLUE}📦 Creating AgentQMS structure...${NC}"

# Create directory structure
mkdir -p "$TARGET_DIR/AgentQMS/agent_tools"
mkdir -p "$TARGET_DIR/AgentQMS/knowledge/agent"
mkdir -p "$TARGET_DIR/AgentQMS/knowledge/references"
mkdir -p "$TARGET_DIR/AgentQMS/interface"
mkdir -p "$TARGET_DIR/.agentqms/plugins"
mkdir -p "$TARGET_DIR/.agentqms/state"
mkdir -p "$TARGET_DIR/docs/artifacts"

echo -e "${GREEN}✅ Directory structure created${NC}"

# Copy essential files
echo -e "${BLUE}📄 Copying essential framework files...${NC}"

# Core agent knowledge
if [ -f "$AGENTQMS_ROOT/knowledge/agent/system.md" ]; then
    cp "$AGENTQMS_ROOT/knowledge/agent/system.md" "$TARGET_DIR/AgentQMS/knowledge/agent/"
    echo "  ✓ system.md"
fi

if [ -f "$AGENTQMS_ROOT/knowledge/agent/README.md" ]; then
    cp "$AGENTQMS_ROOT/knowledge/agent/README.md" "$TARGET_DIR/AgentQMS/knowledge/agent/"
    echo "  ✓ README.md"
fi

# Interface Makefile
if [ -f "$AGENTQMS_ROOT/interface/Makefile" ]; then
    cp "$AGENTQMS_ROOT/interface/Makefile" "$TARGET_DIR/AgentQMS/interface/"
    echo "  ✓ Makefile"
fi

# Plugin registry
if [ -f "$AGENTQMS_ROOT/../.agentqms/plugins/registry.yaml" ]; then
    cp "$AGENTQMS_ROOT/../.agentqms/plugins/registry.yaml" "$TARGET_DIR/.agentqms/plugins/"
    echo "  ✓ plugins/registry.yaml"
fi

# Artifact rules
if [ -f "$AGENTQMS_ROOT/../.agentqms/plugins/artifact_rules.yaml" ]; then
    cp "$AGENTQMS_ROOT/../.agentqms/plugins/artifact_rules.yaml" "$TARGET_DIR/.agentqms/plugins/"
    echo "  ✓ plugins/artifact_rules.yaml"
fi

# Workflow triggers
if [ -f "$AGENTQMS_ROOT/../.copilot/context/workflow-triggers.yaml" ]; then
    mkdir -p "$TARGET_DIR/.copilot/context"
    cp "$AGENTQMS_ROOT/../.copilot/context/workflow-triggers.yaml" "$TARGET_DIR/.copilot/context/"
    echo "  ✓ workflow-triggers.yaml"
fi

echo -e "${GREEN}✅ Essential files copied${NC}"

# Copy agent_tools (selective - core utilities only)
echo -e "${BLUE}🔧 Copying core agent tools...${NC}"

# Create __init__.py files
touch "$TARGET_DIR/AgentQMS/__init__.py"
touch "$TARGET_DIR/AgentQMS/agent_tools/__init__.py"

# Core utilities (small, self-contained)
CORE_UTILS=(
    "smart_populate.py"
    "suggest_context.py"
    "plan_progress.py"
    "deprecated_registry.py"
    "tracking_integration.py"
)

mkdir -p "$TARGET_DIR/AgentQMS/agent_tools/utilities"
touch "$TARGET_DIR/AgentQMS/agent_tools/utilities/__init__.py"

for util in "${CORE_UTILS[@]}"; do
    if [ -f "$AGENTQMS_ROOT/agent_tools/utilities/$util" ]; then
        cp "$AGENTQMS_ROOT/agent_tools/utilities/$util" "$TARGET_DIR/AgentQMS/agent_tools/utilities/"
        echo "  ✓ $util"
    fi
done

echo -e "${GREEN}✅ Core tools copied${NC}"

# Create quickstart guide
echo -e "${BLUE}📝 Creating quickstart guide...${NC}"

cat > "$TARGET_DIR/AgentQMS/knowledge/agent/quickstart.md" << 'EOF'
# AgentQMS Quick Start Guide

## 5-Minute Setup

### 1. Verify Installation

```bash
cd AgentQMS/interface
make help
```

### 2. Create Your First Artifact

```bash
# Create an implementation plan
make create-plan NAME=my-feature TITLE="My Feature Implementation"

# Create an assessment
make create-assessment NAME=my-assessment TITLE="My Assessment"
```

### 3. Validate Artifacts

```bash
# Validate all artifacts
make validate

# Check compliance
make compliance
```

### 4. Use Smart Features

```bash
# Suggest context for a task
make context-suggest TASK="implement authentication"

# Track plan progress
make plan-progress-show FILE=docs/artifacts/implementation_plans/my-plan.md
```

## Common Workflows

### Planning Workflow
1. Create implementation plan
2. Load planning context: `make context-plan`
3. Track progress: `make plan-progress-show`
4. Mark tasks complete: `make plan-progress-complete`

### Quality Workflow
1. Create artifacts via Makefile
2. Auto-validate during creation
3. Run compliance check: `make compliance`
4. Fix any validation errors

### Migration Workflow
1. Find legacy artifacts: `make artifacts-find`
2. Migrate with preview: `make artifacts-migrate-dry`
3. Apply migration: `make artifacts-migrate`

## Next Steps

- Read `AgentQMS/knowledge/agent/system.md` for complete rules
- Explore `AgentQMS/interface/Makefile` for all available commands
- Check `.agentqms/plugins/` for extension points

## Support

For issues or questions, refer to the main project documentation.
EOF

echo -e "${GREEN}✅ Quickstart guide created${NC}"

# Create README for artifacts directory
cat > "$TARGET_DIR/docs/artifacts/README.md" << 'EOF'
# Artifacts Directory

This directory contains project artifacts managed by AgentQMS.

## Structure

- `implementation_plans/` - Feature implementation plans
- `assessments/` - Technical assessments and analyses
- `bug_reports/` - Bug reports and investigations
- `design_documents/` - Design specifications
- `research/` - Research documentation

## Creating Artifacts

Always use the AgentQMS Makefile to create artifacts:

```bash
cd AgentQMS/interface
make create-plan NAME=feature-name TITLE="Feature Title"
```

## Naming Convention

All artifacts must follow: `YYYY-MM-DD_HHMM_[type]_name.md`

Example: `2025-12-06_1430_implementation_plan_my-feature.md`

## Validation

Artifacts are automatically validated during creation.
Run `make validate` to check all artifacts.
EOF

echo ""
echo -e "${GREEN}✅ AgentQMS Framework Initialized Successfully!${NC}"
echo ""
echo -e "${BLUE}📚 Next Steps:${NC}"
echo -e "  1. cd $TARGET_DIR/AgentQMS/interface"
echo -e "  2. make help"
echo -e "  3. Read AgentQMS/knowledge/agent/quickstart.md"
echo ""
echo -e "${YELLOW}💡 Customize .agentqms/plugins/ for your project-specific rules${NC}"
echo ""
