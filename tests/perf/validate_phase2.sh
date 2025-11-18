#!/bin/bash
# Phase 2 Performance Validation Script
#
# This script validates the worker pipeline performance requirements.
# Run this after Phase 2 implementation is complete.

set -e

echo "=== Phase 2 Worker Pipeline Performance Validation ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if sample data exists
if [ ! -d "data/train/images" ] && [ ! -f "outputs/playground/sample_manifest.json" ]; then
    echo -e "${YELLOW}WARNING: No sample data found${NC}"
    echo "Please create sample data using:"
    echo "  python scripts/datasets/sample_images.py --output outputs/playground/sample_manifest.json"
    echo ""
    echo "Or specify an image directory with:"
    echo "  python tests/perf/pipeline_bench.py --image-dir data/train/images --limit 16"
    echo ""
    exit 1
fi

# Run backend benchmark
echo "Step 1: Running backend preprocessing benchmark..."
if [ -f "outputs/playground/sample_manifest.json" ]; then
    python tests/perf/pipeline_bench.py \
        --manifest outputs/playground/sample_manifest.json \
        --limit 16 \
        --output outputs/playground/pipeline_bench.json
else
    python tests/perf/pipeline_bench.py \
        --image-dir data/train/images \
        --limit 16 \
        --output outputs/playground/pipeline_bench.json
fi

# Validate benchmark results
echo ""
echo "Step 2: Validating benchmark results..."

# Extract P95 latencies and validate
AUTOCONTRAST_P95=$(jq -r '.autocontrast.p95_ms' outputs/playground/pipeline_bench.json)
BLUR_P95=$(jq -r '.gaussian_blur.p95_ms' outputs/playground/pipeline_bench.json)
REMBG_P95=$(jq -r '.rembg_client.p95_ms' outputs/playground/pipeline_bench.json)

echo "  Auto Contrast P95: ${AUTOCONTRAST_P95}ms (target: <100ms)"
echo "  Gaussian Blur P95: ${BLUR_P95}ms (target: <100ms)"
echo "  rembg Client P95: ${REMBG_P95}ms (target: <400ms)"

PASS_COUNT=0
FAIL_COUNT=0

# Validate auto contrast
if (( $(echo "$AUTOCONTRAST_P95 < 100" | bc -l) )); then
    echo -e "  ${GREEN}✓ Auto Contrast PASS${NC}"
    ((PASS_COUNT++))
else
    echo -e "  ${RED}✗ Auto Contrast FAIL${NC}"
    ((FAIL_COUNT++))
fi

# Validate gaussian blur
if (( $(echo "$BLUR_P95 < 100" | bc -l) )); then
    echo -e "  ${GREEN}✓ Gaussian Blur PASS${NC}"
    ((PASS_COUNT++))
else
    echo -e "  ${RED}✗ Gaussian Blur FAIL${NC}"
    ((FAIL_COUNT++))
fi

# Validate rembg
if (( $(echo "$REMBG_P95 < 400" | bc -l) )); then
    echo -e "  ${GREEN}✓ rembg Client PASS${NC}"
    ((PASS_COUNT++))
else
    echo -e "  ${RED}✗ rembg Client FAIL${NC}"
    ((FAIL_COUNT++))
fi

echo ""
echo "Step 3: Manual frontend testing instructions..."
echo ""
echo "Please complete the following manual tests:"
echo ""
echo "  1. Start the SPA: python run_spa.py"
echo "  2. Navigate to: http://localhost:5173/preprocessing"
echo "  3. Upload a test image (1024x1024 recommended)"
echo "  4. Test each preprocessing option:"
echo "     - Auto Contrast: verify <100ms processing time"
echo "     - Gaussian Blur: spam the slider, verify smooth UI"
echo "     - Background Removal: verify routing (client vs backend)"
echo "  5. Check browser console for errors"
echo "  6. Verify worker queue depth stays <5 during slider spam"
echo ""
echo "See tests/perf/worker_validation.md for detailed testing instructions."
echo ""

# Summary
echo "=== Validation Summary ==="
echo "Backend Tests: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ Phase 2 backend performance requirements met!${NC}"
    echo "Please complete manual frontend testing before marking Phase 2 as complete."
    exit 0
else
    echo -e "${RED}✗ Phase 2 performance requirements not met${NC}"
    echo "Please investigate and optimize before proceeding."
    exit 1
fi
