#!/bin/bash
# Quick Configuration Compliance Audit
# Usage: ./scripts/quick_config_audit.sh [module_path]

MODULE="${1:-ocr/}"
REPORT_DIR="docs/reports"

echo "🔍 Quick Config Compliance Audit"
echo "Module: $MODULE"
echo ""

# Create report directory
mkdir -p "$REPORT_DIR"

echo "1️⃣ Checking isinstance(dict) violations..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -rn "isinstance.*dict" "$MODULE" --include="*.py" | \
    grep -v "DictConfig" | \
    grep -v "# type:" | \
    head -10
echo ""

echo "2️⃣ Checking OmegaConf.to_container() usage..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -rn "OmegaConf\.to_container" "$MODULE" --include="*.py" | \
    grep -v "ensure_dict" | \
    head -10
echo ""

echo "3️⃣ Checking dict() conversions on config objects..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
grep -rn "dict(.*cfg" "$MODULE" --include="*.py" | \
    grep -v "ensure_dict" | \
    grep -v "def ensure_dict" | \
    head -10
echo ""

echo "4️⃣ Files using proper utilities..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
COUNT=$(grep -r "from ocr.core.utils.config_utils import" "$MODULE" --include="*.py" | wc -l)
echo "✅ $COUNT files import config_utils"
echo ""

echo "5️⃣ Top violators (by file)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
{
    grep -r "isinstance.*dict\|OmegaConf\.to_container\|dict(.*cfg" "$MODULE" --include="*.py" | \
        cut -d: -f1 | \
        sort | uniq -c | \
        sort -rn | \
        head -5
} || echo "No violations found!"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 For detailed report, run:"
echo "   python scripts/audit_config_compliance.py"
echo ""
echo "📚 For guide, see:"
echo "   docs/reports/config_compliance_audit_guide.md"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
