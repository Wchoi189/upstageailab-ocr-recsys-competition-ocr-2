#!/bin/bash
# Verification script for apps/ directory setup

echo "=== Apps Directory Status ==="
echo ""

echo "1. Check if apps is a symlink or directory:"
if [ -L "/workspaces/apps" ]; then
    echo "   ❌ apps/ is a SYMLINK (should be directory)"
    readlink -f /workspaces/apps
else
    echo "   ✅ apps/ is a real DIRECTORY (correct!)"
fi

echo ""
echo "2. Check git tracking:"
if git ls-files --error-unmatch apps/ > /dev/null 2>&1; then
    echo "   ✅ apps/ is tracked by git (correct!)"
else
    echo "   ⚠️  apps/ is NOT tracked by git (check .gitignore)"
fi

echo ""
echo "3. Check node_modules volumes (should NOT exist in host):"
for app in "ocr_inference_console" "playground-console" "agentqms-dashboard/frontend" "mcp-visibility-extension"; do
    if [ -d "/workspaces/apps/$app/node_modules" ]; then
        echo "   ⚠️  /workspaces/apps/$app/node_modules exists on host (might cause sync issues)"
    else
        echo "   ✅ /workspaces/apps/$app/node_modules NOT on host (correct - lives in volume)"
    fi
done

echo ""
echo "4. Check entrypoint.sh FOLDERS array:"
if grep -q '"apps"' /workspaces/docker/entrypoint.sh; then
    echo "   ❌ 'apps' found in FOLDERS array (should be removed)"
else
    echo "   ✅ 'apps' NOT in FOLDERS array (correct!)"
fi

echo ""
echo "5. Symlinks configured in entrypoint.sh:"
grep 'FOLDERS=(' /workspaces/docker/entrypoint.sh

echo ""
echo "=== Summary ==="
echo "Your setup should have:"
echo "  • apps/ as a real directory in git"
echo "  • node_modules in Docker volumes (not on host)"
echo "  • apps/ NOT in the FOLDERS symlink array"
