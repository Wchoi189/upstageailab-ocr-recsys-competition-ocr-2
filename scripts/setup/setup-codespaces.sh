#!/bin/bash
# Codespaces Setup Validation Script
# Validates GitHub Codespaces configuration and provides setup status

set -e

echo "🔍 GitHub Codespaces Setup Validation"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running in Codespaces
if [ -n "${CODESPACES}" ]; then
    echo -e "${GREEN}✓ Running in GitHub Codespaces${NC}"
    echo "  Codespace: ${CODESPACE_NAME:-unknown}"
    echo "  Repository: ${GITHUB_REPOSITORY:-unknown}"
else
    echo -e "${YELLOW}⚠️  Not running in Codespaces${NC}"
    echo "  This script is designed for GitHub Codespaces"
    echo "  It will still validate configuration files"
fi

echo ""

# Check devcontainer.json
echo "📋 Checking devcontainer.json..."
if [ -f ".devcontainer/devcontainer.json" ]; then
    echo -e "${GREEN}✓ devcontainer.json exists${NC}"

    # Check if using pre-built image
    if grep -q '"image":' .devcontainer/devcontainer.json; then
        IMAGE=$(grep '"image":' .devcontainer/devcontainer.json | sed 's/.*"image": *"\([^"]*\)".*/\1/')
        echo -e "${GREEN}✓ Using pre-built image: ${IMAGE}${NC}"
    elif grep -q '"build":' .devcontainer/devcontainer.json; then
        echo -e "${YELLOW}⚠️  Using build configuration (slower startup)${NC}"
        echo "  Consider switching to pre-built image for faster Codespaces startup"
    fi

    # Check for hostRequirements
    if grep -q 'hostRequirements' .devcontainer/devcontainer.json; then
        echo -e "${GREEN}✓ hostRequirements configured${NC}"
    else
        echo -e "${YELLOW}⚠️  hostRequirements not configured${NC}"
        echo "  Add hostRequirements for optimal Codespaces performance"
    fi
else
    echo -e "${RED}✗ devcontainer.json not found${NC}"
    exit 1
fi

echo ""

# Check container build workflow
echo "🔨 Checking container build workflow..."
if [ -f ".github/workflows/build-container.yml" ]; then
    echo -e "${GREEN}✓ Container build workflow exists${NC}"

    # Check if workflow builds development target
    if grep -q 'target: development' .github/workflows/build-container.yml || grep -q 'target.*development' .github/workflows/build-container.yml; then
        echo -e "${GREEN}✓ Workflow builds development target${NC}"
    else
        echo -e "${YELLOW}⚠️  Workflow may not specify development target${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Container build workflow not found${NC}"
    echo "  Create .github/workflows/build-container.yml to automate image builds"
fi

echo ""

# Check Dockerfile
echo "🐳 Checking Dockerfile..."
if [ -f "docker/Dockerfile" ]; then
    echo -e "${GREEN}✓ Dockerfile exists${NC}"

    # Check for multi-stage build
    if grep -q 'FROM.*AS.*development' docker/Dockerfile; then
        echo -e "${GREEN}✓ Multi-stage build with development stage${NC}"
    else
        echo -e "${YELLOW}⚠️  Development stage not found${NC}"
    fi

    # Check if dependencies are installed
    if grep -q 'uv sync' docker/Dockerfile; then
        echo -e "${GREEN}✓ Dependencies installation found${NC}"
    else
        echo -e "${YELLOW}⚠️  Dependencies may not be pre-installed${NC}"
        echo "  Pre-installing dependencies improves startup time"
    fi
else
    echo -e "${RED}✗ Dockerfile not found${NC}"
    exit 1
fi

echo ""

# Check environment validation script
echo "✅ Checking environment validation..."
if [ -f "scripts/validate_environment.py" ]; then
    echo -e "${GREEN}✓ Environment validation script exists${NC}"

    # Try to run it if in Codespaces
    if [ -n "${CODESPACES}" ]; then
        echo "  Running validation..."
        if python scripts/validate_environment.py; then
            echo -e "${GREEN}✓ Environment validation passed${NC}"
        else
            echo -e "${YELLOW}⚠️  Environment validation had issues${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Environment validation script not found${NC}"
fi

echo ""

# Summary
echo "======================================"
echo "📊 Setup Summary"
echo "======================================"
echo ""

if [ -n "${CODESPACES}" ]; then
    echo -e "${GREEN}✅ Codespaces environment detected${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify container image is accessible"
    echo "  2. Check that dependencies are pre-installed"
    echo "  3. Run: python scripts/validate_environment.py"
    echo "  4. Start developing!"
else
    echo -e "${YELLOW}ℹ️  Local development environment${NC}"
    echo ""
    echo "To use GitHub Codespaces:"
    echo "  1. Push this repository to GitHub"
    echo "  2. Go to repository → Code → Codespaces"
    echo "  3. Click 'Create codespace on main'"
    echo "  4. Wait for environment to start (~30 seconds with pre-built image)"
fi

echo ""
echo "For more information, see: docs/guides/codespaces-setup.md"
echo ""
