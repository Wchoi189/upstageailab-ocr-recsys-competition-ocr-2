#!/bin/bash
# Quick status check for Prebuilt Extraction batch jobs

JOB_IDS=(
    "b1dabc78-5b09-49bf-9978-cdf31b293ef4"  # baseline_val (retry with concurrency=1)
    "9ce7e6d4-891d-4d08-a623-340884dbef44"  # baseline_test (retry with concurrency=1)
    "feb64e9f-5a0d-41ff-a72b-111597848b43"  # baseline_train (retry with concurrency=1)
)

DATASETS=("baseline_val" "baseline_test" "baseline_train")

echo "=================================================================================="
echo "Batch Job Status Check"
echo "=================================================================================="
echo ""

for i in "${!JOB_IDS[@]}"; do
    JOB_ID="${JOB_IDS[$i]}"
    DATASET="${DATASETS[$i]}"
    
    STATUS=$(aws batch describe-jobs --jobs "$JOB_ID" --query 'jobs[0].status' --output text 2>/dev/null)
    
    if [ "$STATUS" = "None" ] || [ -z "$STATUS" ]; then
        STATUS="UNKNOWN"
    fi
    
    case "$STATUS" in
        "SUBMITTED"|"PENDING"|"RUNNABLE")
            ICON="⏳"
            ;;
        "STARTING"|"RUNNING")
            ICON="🔄"
            ;;
        "SUCCEEDED")
            ICON="✅"
            ;;
        "FAILED")
            ICON="❌"
            ;;
        *)
            ICON="❓"
            ;;
    esac
    
    echo "$ICON $DATASET: $STATUS (Job: ${JOB_ID:0:8}...)"
done

echo ""
echo "=================================================================================="
echo "Quick Commands:"
echo "=================================================================================="
echo ""
echo "# Check detailed status:"
echo "  aws batch describe-jobs --jobs ${JOB_IDS[0]} ${JOB_IDS[1]} ${JOB_IDS[2]}"
echo ""
echo "# View logs for baseline_val:"
echo "  aws logs tail /aws/batch/job --log-stream-name ${JOB_IDS[0]} --follow"
echo ""
echo "# List all running jobs:"
echo "  aws batch list-jobs --job-queue batch-processor-queue --job-status RUNNING"
echo ""
