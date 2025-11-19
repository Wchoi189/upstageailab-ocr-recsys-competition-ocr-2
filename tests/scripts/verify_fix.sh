#!/bin/bash
# Quick verification that the pandas import deadlock fix is in place

echo "=========================================="
echo "Verifying Streamlit Pandas Import Fix"
echo "=========================================="
echo ""

FILE="ui/apps/inference/components/results.py"

# Check if pandas is imported at global scope
echo "1. Checking for pandas import at global scope..."
if grep -q "^import pandas as pd" "$FILE"; then
    echo "   ✅ PASS: Found 'import pandas as pd' at global scope"
else
    echo "   ❌ FAIL: pandas not imported at global scope"
    exit 1
fi

# Check that there's no lazy import inside functions
echo ""
echo "2. Checking for lazy pandas imports inside functions..."
if grep -q "^[[:space:]]\+import pandas as pd" "$FILE"; then
    echo "   ❌ FAIL: Found lazy import of pandas inside a function"
    echo "   This will cause the deadlock issue!"
    exit 1
else
    echo "   ✅ PASS: No lazy imports of pandas found"
fi

# Show the import location
echo ""
echo "3. Import location details:"
echo "   File: $FILE"
IMPORT_LINE=$(grep -n "^import pandas as pd" "$FILE" | cut -d: -f1)
echo "   Line: $IMPORT_LINE"
echo ""
echo "   Context:"
grep -n -A 2 -B 2 "^import pandas as pd" "$FILE"

echo ""
echo "=========================================="
echo "✅ Verification PASSED"
echo "=========================================="
echo ""
echo "The pandas import deadlock fix is correctly in place."
echo "The app should no longer freeze after inference."
echo ""
echo "To test the app:"
echo "  cd ui/apps/inference"
echo "  uv run streamlit run app.py --server.port=8504"
echo ""
