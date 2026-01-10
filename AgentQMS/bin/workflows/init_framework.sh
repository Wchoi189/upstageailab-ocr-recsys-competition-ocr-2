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
PROJECT_ROOT="$(cd "$AGENTQMS_ROOT/.." && pwd)"

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
mkdir -p "$TARGET_DIR/AgentQMS/bin"
mkdir -p "$TARGET_DIR/AgentQMS/tools"
mkdir -p "$TARGET_DIR/AgentQMS/standards"
mkdir -p "$TARGET_DIR/AgentQMS/.agentqms/plugins/artifact_types"
mkdir -p "$TARGET_DIR/AgentQMS/.agentqms/plugins/context_bundles"
mkdir -p "$TARGET_DIR/.agentqms/schemas"
mkdir -p "$TARGET_DIR/docs/artifacts"

echo -e "${GREEN}✅ Directory structure created${NC}"

# Copy essential files
echo -e "${BLUE}📄 Copying essential framework files...${NC}"

# Copy bin/Makefile
if [ -f "$AGENTQMS_ROOT/bin/Makefile" ]; then
    cp "$AGENTQMS_ROOT/bin/Makefile" "$TARGET_DIR/AgentQMS/bin/"
    echo "  ✓ bin/Makefile"
fi

# Copy bin/README
if [ -f "$AGENTQMS_ROOT/bin/README.md" ]; then
    cp "$AGENTQMS_ROOT/bin/README.md" "$TARGET_DIR/AgentQMS/bin/"
    echo "  ✓ bin/README.md"
fi

# Copy AGENTS.yaml
if [ -f "$AGENTQMS_ROOT/AGENTS.yaml" ]; then
    cp "$AGENTQMS_ROOT/AGENTS.yaml" "$TARGET_DIR/AgentQMS/"
    echo "  ✓ AGENTS.yaml"
fi

# Copy standards
if [ -d "$AGENTQMS_ROOT/standards" ]; then
    cp -r "$AGENTQMS_ROOT/standards"/* "$TARGET_DIR/AgentQMS/standards/" 2>/dev/null || true
    echo "  ✓ standards/"
fi

# Copy plugin examples
if [ -d "$AGENTQMS_ROOT/.agentqms/plugins/artifact_types" ]; then
    cp "$AGENTQMS_ROOT/.agentqms/plugins/artifact_types"/*.yaml "$TARGET_DIR/AgentQMS/.agentqms/plugins/artifact_types/" 2>/dev/null || true
    echo "  ✓ artifact_types plugins"
fi

# Copy validation schema
if [ -f "$PROJECT_ROOT/.agentqms/schemas/artifact_type_validation.yaml" ]; then
    cp "$PROJECT_ROOT/.agentqms/schemas/artifact_type_validation.yaml" "$TARGET_DIR/.agentqms/schemas/"
    echo "  ✓ artifact_type_validation.yaml"
fi

# Copy MCP server
if [ -f "$AGENTQMS_ROOT/mcp_server.py" ]; then
    cp "$AGENTQMS_ROOT/mcp_server.py" "$TARGET_DIR/AgentQMS/"
    echo "  ✓ mcp_server.py"
fi

echo -e "${GREEN}✅ Essential files copied${NC}"

# Create artifacts directory README
cat > "$TARGET_DIR/docs/artifacts/README.md" << 'EOF'
# Artifacts Directory

This directory contains project artifacts managed by AgentQMS.

## Creating Artifacts

Always use the AgentQMS Makefile to create artifacts:

```bash
cd AgentQMS/bin
make create-plan NAME=feature-name TITLE="Feature Title"
make create-assessment NAME=my-assessment TITLE="Assessment Title"
```

## Naming Convention

All artifacts follow: `YYYY-MM-DD_HHMM_{type}_descriptive-name.md`

Example: `2026-01-10_1430_implementation_plan_my-feature.md`

## Validation

Run validation:
```bash
cd AgentQMS/bin
make validate
make compliance
```

See `AgentQMS/bin/README.md` for complete documentation.
EOF

echo ""
echo -e "${GREEN}✅ AgentQMS Framework Initialized Successfully!${NC}"
echo ""
echo -e "${BLUE}📚 Next Steps:${NC}"
echo -e "  1. cd $TARGET_DIR/AgentQMS/bin"
echo -e "  2. make help"
echo -e "  3. Read AgentQMS/AGENTS.yaml for quick reference"
echo -e "  4. Read AgentQMS/bin/README.md for full documentation"
echo ""
echo -e "${YELLOW}💡 Note: Install Python dependencies and tools separately${NC}"
echo ""
