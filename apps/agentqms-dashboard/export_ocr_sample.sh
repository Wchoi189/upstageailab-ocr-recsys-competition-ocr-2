#!/bin/bash
# Export OCR sample dataset for sharing

SAMPLE_DIR="ocr_sample_data"
OUTPUT_ZIP="ocr_sample_data.zip"
OUTPUT_TAR="ocr_sample_data.tar.gz"

echo "📦 Exporting OCR inference sample..."
echo ""

# Create ZIP archive
echo "Creating ZIP archive..."
zip -r -q "$OUTPUT_ZIP" "$SAMPLE_DIR" -x "*.pyc" "__pycache__/*"
echo "✅ Created: $OUTPUT_ZIP ($(du -h $OUTPUT_ZIP | cut -f1))"

# Create TAR.GZ archive  
echo "Creating TAR.GZ archive..."
tar -czf "$OUTPUT_TAR" "$SAMPLE_DIR" --exclude="*.pyc" --exclude="__pycache__"
echo "✅ Created: $OUTPUT_TAR ($(du -h $OUTPUT_TAR | cut -f1))"

# Create manifest
echo "Creating manifest..."
cat > OCR_SAMPLE_MANIFEST.txt << 'EOFMANIFEST'
OCR Inference Sample Dataset - Manifest
Created: 2025-12-11
Size: 44 KB (uncompressed)

Contents:
- images/ (39 KB)
  - sample_001.jpg (15 KB) - Synthetic receipt
  - sample_002.jpg (12 KB) - Document 1
  - sample_003.jpg (12 KB) - Document 2
- annotations.json (2 KB) - COCO format
- config.yaml - Configuration
- requirements.txt - Dependencies

Total Annotations: 5 text regions
Languages: English
Formats: JPEG + COCO JSON

For details: OCR_INFERENCE_GUIDE.md
EOFMANIFEST

echo "✅ Created: OCR_SAMPLE_MANIFEST.txt"

echo ""
echo "========================================"
echo "✅ EXPORT COMPLETE"
echo "========================================"
echo ""
echo "📁 Files created:"
ls -lh ocr_sample_data.* OCR_SAMPLE_MANIFEST.txt | awk '{print "   " $9 " (" $5 ")"}'
echo ""
echo "Ready to upload to Google AI Studio!"
