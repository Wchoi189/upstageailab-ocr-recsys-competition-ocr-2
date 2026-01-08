#!/bin/bash
# Script to run Test dataset splits with Document Parse API

# Usage:
#   ./scripts/runners/run_test_dp_splits.sh <START_SPLIT> <END_SPLIT> <TIER>

set -e

START=$1
END=$2
TIER=$3

if [ -z "$START" ] || [ -z "$END" ] || [ -z "$TIER" ]; then
    echo "Usage: $0 <START_SPLIT> <END_SPLIT> <TIER>"
    echo "  START: Split number to start (1-4)"
    echo "  END:   Split number to end (1-4)"
    echo "  TIER:  1 (UPSTAGE_API_KEY, 3 RPS) or 2 (UPSTAGE_API_KEY2, 1 RPS)"
    exit 1
fi

SPLIT_DIR="splits_test"
BATCH_SIZE=20

if [ "$TIER" == "1" ]; then
    API_KEY_ENV="UPSTAGE_API_KEY"
    CONCURRENCY=3
    echo "Using TIER 1 Configuration (Max 3 RPS)"
elif [ "$TIER" == "2" ]; then
    API_KEY_ENV="UPSTAGE_API_KEY2"
    CONCURRENCY=1
    echo "Using TIER 2 Configuration (Max 1 RPS)"
else
    echo "Error: Tier must be 1 or 2"
    exit 1
fi

echo "Processing (Document Parse) Test Splits $START to $END"
echo "Concurrency: $CONCURRENCY"
echo "Key Env: $API_KEY_ENV"

for ((i=START; i<=END; i++)); do
    DATASET="${SPLIT_DIR}/baseline_test_p${i}"
    echo "----------------------------------------------------------------"
    echo "Processing Split $i/4: $DATASET"
    echo "----------------------------------------------------------------"

    cd aws-batch-processor || exit 1
    python scripts/runners/reprocess_with_document_parse.py \
      --dataset "$DATASET" \
      --concurrency "$CONCURRENCY" \
      --batch-size "$BATCH_SIZE" \
      --resume \
      --api-key-env "$API_KEY_ENV"

    cd ..

    if [ "$i" -lt "$END" ]; then
        echo "Cooling down for 2 seconds..."
        sleep 2
    fi
done

echo "Test Splits (Document Parse) $START-$END processing completed."
