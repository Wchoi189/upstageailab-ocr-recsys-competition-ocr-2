#!/bin/bash
set -e

echo "🔍 [1/4] Checking environment status..."
PYTHON_ORIGIN=$(python -c "import ocr; print(ocr.__file__)")

if [[ "$PYTHON_ORIGIN" == *"site-packages"* ]]; then
    echo "❌ ERROR: Ghost Code Detected. Running from $PYTHON_ORIGIN"
    echo "Fixing: Re-installing in editable mode..."
    pip install -e .
else
    echo "✅ Environment: Editable install verified."
fi

echo "🔍 [2/4] Running Master Audit..."
python scripts/audit/master_audit.py > audit_results.txt

echo "🔍 [3/4] Validating Hydra Recursion Safety..."
# Check if factories use _recursive_=False
BAD_INSTANCES=$(grep -r "hydra.utils.instantiate" ocr/ | grep -v "_recursive_=False" || true)
if [ ! -z "$BAD_INSTANCES" ]; then
    echo "⚠️  WARNING: Potential Recursive Instantiation traps found:"
    echo "$BAD_INSTANCES"
else
    echo "✅ Hydra: Recursion safety looks good."
fi

echo "📊 [4/4] Summary:"
grep "🚨" audit_results.txt || echo "No critical anomalies found."
