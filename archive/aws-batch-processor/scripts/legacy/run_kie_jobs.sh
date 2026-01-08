#!/bin/bash
# Commands to run KIE processing (Prebuilt Extraction) with rate limits

# TIER 1 KEY (UPSTAGE_API_KEY) - Max 3 RPS
# Recommended for larger datasets (Training)
# Usage:
#   export UPSTAGE_API_KEY="your-key"
#   ./run_kie_jobs.sh train

# TIER 2 KEY (UPSTAGE_API_KEY2) - Max 1 RPS
# Recommended for smaller datasets (Validation, Test)
# Usage:
#   export UPSTAGE_API_KEY2="your-key-2"
#   ./run_kie_jobs.sh val
#   ./run_kie_jobs.sh test

set -e

COMMAND=$1

case $COMMAND in
  train)
    echo "Processing baseline_train with TIER 1 limits (3 RPS)..."
    cd aws-batch-processor
    python scripts/runners/reprocess_with_prebuilt_extraction.py \
      --dataset baseline_train \
      --concurrency 3 \
      --api-key-env UPSTAGE_API_KEY
    ;;

  val)
    echo "Processing baseline_val_canonical with TIER 2 limits (1 RPS)..."
    cd aws-batch-processor
    python scripts/runners/reprocess_with_prebuilt_extraction.py \
      --dataset baseline_val_canonical \
      --concurrency 1 \
      --api-key-env UPSTAGE_API_KEY2
    ;;

  test)
    echo "Processing baseline_test with TIER 2 limits (1 RPS)..."
    cd aws-batch-processor
    python scripts/runners/reprocess_with_prebuilt_extraction.py \
      --dataset baseline_test \
      --concurrency 1 \
      --api-key-env UPSTAGE_API_KEY2
    ;;

  *)
    echo "Usage: $0 {train|val|test}"
    echo "  train: Uses UPSTAGE_API_KEY (3 concurrent)"
    echo "  val:   Uses UPSTAGE_API_KEY2 (1 concurrent)"
    echo "  test:  Uses UPSTAGE_API_KEY2 (1 concurrent)"
    exit 1
    ;;
esac
