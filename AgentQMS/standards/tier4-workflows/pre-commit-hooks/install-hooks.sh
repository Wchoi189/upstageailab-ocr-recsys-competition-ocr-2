#!/bin/bash
# Master pre-commit hook installer
# Installs all AgentQMS + AI Documentation validation hooks

set -e

HOOK_DIR=".git/hooks"
HOOKS_SOURCE="AgentQMS/standards/tier4-workflows/pre-commit-hooks"

echo "🔧 Installing pre-commit hooks..."

# Create pre-commit hook if it doesn't exist
if [ ! -f "$HOOK_DIR/pre-commit" ]; then
    cat > "$HOOK_DIR/pre-commit" << 'EOF'
#!/bin/bash
# AgentQMS + AI Documentation Pre-commit Hook Master

set -e

HOOKS_DIR="AgentQMS/standards/tier4-workflows/pre-commit-hooks"

# Run all validation hooks
if [ -f "$HOOKS_DIR/naming-validation.sh" ]; then
    bash "$HOOKS_DIR/naming-validation.sh" || exit 1
fi

if [ -f "$HOOKS_DIR/placement-validation.sh" ]; then
    bash "$HOOKS_DIR/placement-validation.sh" || exit 1
fi

if [ -f "$HOOKS_DIR/ads-compliance.sh" ]; then
    bash "$HOOKS_DIR/ads-compliance.sh" || exit 1
fi

echo "✅ All pre-commit validations passed"
exit 0
EOF

    chmod +x "$HOOK_DIR/pre-commit"
    echo "✅ Created master pre-commit hook"
else
    echo "ℹ️  Pre-commit hook already exists"
fi

# Make all hook scripts executable
chmod +x "$HOOKS_SOURCE"/*.sh
echo "✅ Made all hook scripts executable"

echo ""
echo "🎉 Pre-commit hooks installed successfully!"
echo ""
echo "Installed hooks:"
echo "  - naming-validation.sh (blocks ALL-CAPS filenames)"
echo "  - placement-validation.sh (blocks files at docs/ root)"
echo "  - ads-compliance.sh (validates ADS v1.0 compliance)"
echo ""
echo "Test hooks: git commit (hooks run automatically)"
