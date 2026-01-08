# Session Handover

## Completed
*   **Parallel Processing**: Successfully processed `baseline_train` using two accounts and sharding.
*   **Splits**:
    *   `baseline_train_serial`: Initial ~1244 images (Account #1).
    *   `baseline_train_pseudo_labels_acc2_part1`: 1014 images (Account #2).
    *   `baseline_train_pseudo_labels_acc2_part2`: 1014 images (Account #2).
*   **Merging**: `merge_shards.py` created to combine these.
*   **Missing Data Identified**: 20 images (Indices 0-20) were skipped due to checkpoint collisions.
*   **Bug Fix**: `reprocess_serial.py` logic updated to parse `boundingBoxes` from nested `properties`, fixing the "empty polygons" issue for `receipt-extraction` API.

## Known Issues
*   **Polygons Missing**: Confirmed that the `receipt-extraction` API (model version 3.2.0) **does not return bounding boxes** for top-level fields like store name, total price, etc. It serves purely as an information extractor, not a layout analyzer.
*   **Missing 20 Images**: Processed successfully in the final "missing" shard, but they also lack polygons due to the API limitation.

## Next Steps
1.  **Merge**: Run `uv run python3 scripts/merge_shards.py` to get the final `baseline_train_pseudo_labels.parquet`.
2.  **Strategic Decision**:
    *   **Text-Only KIE**: Proceed with training a model that only predicts entities from text sequences (e.g., encoded HTML/Markdown or simple sequence labeling) without visual coordinate features.
    *   **Augment with PCR**: Use the existing "Baseline OCR" (CRAFT/Tesseract) output to "inject" polygons. You can match the extracted text values (e.g., "12,300") to the OCR words to find their approximate coordinates.
    *   **Document Parse**: For future work requiring precise layout/coordinates, switch to the `document-parse` API.

## Artifacts
*   `task.md`: Checklists.
*   `implementation_plan.md`: Parallel execution plan.
*   `scripts/reprocess_serial.py`: **Updated with Polygon Fix**.
*   `scripts/merge_shards.py`: Merging logic.
