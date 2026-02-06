#!/bin/bash
# scripts/vlm_baseline_assessment.sh
# Run baseline VLM quality assessment on worst performers

set -e

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_IMAGES="${TEST_IMAGES:-data/zero_prediction_worst_performers}"
MAX_IMAGES="${MAX_IMAGES:-10}"

echo "=== VLM Baseline Assessment ==="
echo "Experiment: $(basename "$EXPERIMENT_DIR")"
echo "Test images: $TEST_IMAGES"
echo "Max images: $MAX_IMAGES"
echo

# Create output directory
mkdir -p "$EXPERIMENT_DIR/vlm_reports/baseline"

# Process images
count=0
for img in "$TEST_IMAGES"/*.jpg; do
  [ -f "$img" ] || continue
  [ $count -ge $MAX_IMAGES ] && break

  basename=$(basename "$img" .jpg)
  output="$EXPERIMENT_DIR/vlm_reports/baseline/${basename}_quality.md"

  echo "[$((count+1))/$MAX_IMAGES] Analyzing: $basename"

  uv run python -m AgentQMS.vlm.cli.analyze_defects \
    --image "$img" \
    --mode image_quality \
    --backend openrouter \
    --output-format markdown \
    --output "$output" \
    2>&1 | grep -v "^DEBUG" || true

  count=$((count+1))
done

echo
echo "=== Baseline assessment complete ==="
echo "Reports: $EXPERIMENT_DIR/vlm_reports/baseline/"
echo "Total: $count reports generated"
