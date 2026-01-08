#!/bin/bash
# Script to run Training dataset splits with specific Tier/Key configuration

# Usage:
#   ./scripts/runners/run_train_splits.sh <START_SPLIT> <END_SPLIT> <TIER>
# Example:
#   ./scripts/runners/run_train_splits.sh 1 5 1  (Run splits 1-5 with Tier 1 Key)
#   ./scripts/runners/run_train_splits.sh 6 10 2 (Run splits 6-10 with Tier 2 Key)

set -e

START=$1
END=$2
TIER=$3

if [ -z "$START" ] || [ -z "$END" ] || [ -z "$TIER" ]; then
    echo "Usage: $0 <START_SPLIT> <END_SPLIT> <TIER>"
    echo "  START: Split number to start (1-10)"
    echo "  END:   Split number to end (1-10)"
    echo "  TIER:  1 (UPSTAGE_API_KEY, 3 RPS) or 2 (UPSTAGE_API_KEY2, 1 RPS)"
    exit 1
fi

SPLIT_DIR="splits_train"
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

echo "Processing Splits $START to $END"
echo "Concurrency: $CONCURRENCY"
echo "Key Env: $API_KEY_ENV"

for ((i=START; i<=END; i++)); do
    DATASET="${SPLIT_DIR}/baseline_train_p${i}"
    echo "----------------------------------------------------------------"
    echo "Processing Split $i/10: $DATASET"
    echo "----------------------------------------------------------------"

    cd aws-batch-processor || exit 1
    python scripts/runners/reprocess_with_prebuilt_extraction.py \
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

echo "Splits $START-$END processing completed."
