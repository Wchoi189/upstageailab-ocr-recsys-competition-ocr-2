#!/bin/bash
# Script to check logs and find freeze point

echo "=================================="
echo "CHECKING STREAMLIT LOGS FOR FREEZE"
echo "=================================="

echo ""
echo "1. Checking output log (last 50 lines):"
echo "----------------------------------------"
tail -50 logs/ui/inference_8504.out 2>/dev/null || echo "No output log found"

echo ""
echo "2. Checking error log (last 50 lines):"
echo "----------------------------------------"
tail -50 logs/ui/inference_8504.err 2>/dev/null || echo "No error log found"

echo ""
echo "3. Checking streamlit process:"
echo "----------------------------------------"
ps aux | grep "[s]treamlit run" || echo "No streamlit process running"

echo ""
echo "=================================="
echo "INSTRUCTIONS:"
echo "=================================="
echo "1. Start the app: make ui-infer"
echo "2. Run ONE inference"
echo "3. Wait for freeze"
echo "4. Run this script: bash CHECK_LOGS.sh"
echo "5. Share the LAST log message before freeze"
echo ""
echo "The last message will tell us EXACTLY where it freezes."
echo "=================================="
