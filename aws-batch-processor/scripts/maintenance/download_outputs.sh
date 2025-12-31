#!/bin/bash
# Download Prebuilt Extraction output files from S3 when jobs complete

set -e

source aws/config.env

echo "=================================================================================="
echo "Downloading Prebuilt Extraction Output Files"
echo "=================================================================================="
echo ""

OUTPUT_DIR="data/output"
mkdir -p "$OUTPUT_DIR"

DATASETS=("baseline_val" "baseline_test" "baseline_train")

for dataset in "${DATASETS[@]}"; do
    S3_PATH="s3://$S3_BUCKET/data/processed/${dataset}_pseudo_labels.parquet"
    LOCAL_PATH="$OUTPUT_DIR/${dataset}_pseudo_labels.parquet"
    
    echo "Checking $dataset..."
    
    # Check if file exists in S3
    if aws s3 ls "$S3_PATH" > /dev/null 2>&1; then
        echo "  ✓ Found in S3, downloading..."
        aws s3 cp "$S3_PATH" "$LOCAL_PATH"
        
        if [ -f "$LOCAL_PATH" ]; then
            SIZE=$(stat -f%z "$LOCAL_PATH" 2>/dev/null || stat -c%s "$LOCAL_PATH" 2>/dev/null || echo "unknown")
            echo "  ✓ Downloaded: $LOCAL_PATH ($SIZE bytes)"
        else
            echo "  ✗ Download failed"
        fi
    else
        echo "  ⚠️  Not found in S3 yet (job may still be running)"
    fi
    echo ""
done

echo "=================================================================================="
echo "Download complete!"
echo "=================================================================================="
