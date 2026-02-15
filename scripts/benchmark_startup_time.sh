#!/bin/bash
# Benchmark script for measuring training startup time
# Measures time from orchestrator init to first batch

set -e

echo "======================================"
echo "Training Startup Performance Benchmark"
echo "======================================"

EXPERIMENT="parseq_flash_fast"
RUNS=3

echo ""
echo "Configuration:"
echo "  Experiment: $EXPERIMENT"
echo "  Runs: $RUNS"
echo "  Measuring: Time to initialize (limit_train_batches=0)"
echo ""

TOTAL_TIME=0

for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS:"

    # Run training with limit_train_batches=0 to measure startup time only
    START=$(date +%s.%N)

    uv run python scripts/runners/train.py \
      experiment=$EXPERIMENT \
      trainer.limit_train_batches=0 \
      trainer.enable_checkpointing=false \
      > /dev/null 2>&1

    END=$(date +%s.%N)

    ELAPSED=$(echo "$END - $START" | bc)
    echo "  Time: ${ELAPSED}s"

    TOTAL_TIME=$(echo "$TOTAL_TIME + $ELAPSED" | bc)
done

AVERAGE=$(echo "scale=3; $TOTAL_TIME / $RUNS" | bc)

echo ""
echo "======================================"
echo "Results:"
echo "  Average startup time: ${AVERAGE}s"
echo "======================================"
echo ""
echo "Target: ≤2.0s"
if (( $(echo "$AVERAGE <= 2.0" | bc -l) )); then
    echo "✓ TARGET MET!"
else
    echo "⚠ Target not met, but improvement achieved"
fi
