#!/bin/bash
# scripts/vlm_validate_enhancement.sh
# Run VLM validation on before/after comparisons

set -e

PHASE=$1
if [ -z "$PHASE" ]; then
  echo "Usage: $0 <phase_name>"
  echo "Example: $0 phase1_bg_norm"
  exit 1
fi

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPARISON_DIR="${COMPARISON_DIR:-outputs/comparisons/$PHASE}"

echo "=== VLM Enhancement Validation ==="
echo "Phase: $PHASE"
echo "Comparisons: $COMPARISON_DIR"
echo

# Create output directory
mkdir -p "$EXPERIMENT_DIR/vlm_reports/${PHASE}_validation"

# Process comparisons
count=0
for cmp in "$COMPARISON_DIR"/*.jpg; do
  [ -f "$cmp" ] || continue

  basename=$(basename "$cmp" .jpg | sed 's/comparison_//')
  output="$EXPERIMENT_DIR/vlm_reports/${PHASE}_validation/${basename}_validation.md"

  echo "[$((count+1))] Validating: $basename"

  uv run python -m AgentQMS.vlm.cli.analyze_defects \
    --image "$cmp" \
    --mode enhancement_validation \
    --backend openrouter \
    --output-format markdown \
    --output "$output" \
    2>&1 | grep -v "^DEBUG" || true

  count=$((count+1))
done

echo
echo "=== Validation complete ==="
echo "Reports: $EXPERIMENT_DIR/vlm_reports/${PHASE}_validation/"
echo "Total: $count reports generated"
