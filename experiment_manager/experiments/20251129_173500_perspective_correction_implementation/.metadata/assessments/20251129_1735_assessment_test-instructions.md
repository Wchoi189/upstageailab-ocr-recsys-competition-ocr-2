---
ads_version: '1.0'
type: assessment
experiment_id: 20251129_173500_perspective_correction_implementation
status: complete
created: '2025-12-17T17:59:48Z'
updated: '2025-12-17T17:59:48Z'
tags:
- perspective-correction
- testing
phase: phase_0
priority: medium
evidence_count: 0
---
# Test Instructions for Worst Performers

## Overview
This document describes how to run the perspective correction test on the worst performers list.

## Test Script
The test script is located at:
```
scripts/test_worst_performers.py
```

## How to Run

### Option 1: Direct Python Execution
```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
python3 experiment-tracker/experiments/20251129_173500_perspective_correction_implementation/scripts/test_worst_performers.py
```

### Option 2: Using uv (if available)
```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
uv run python experiment-tracker/experiments/20251129_173500_perspective_correction_implementation/scripts/test_worst_performers.py
```

## What the Script Does

1. **Reads the worst performers list** from:
   - `experiment-tracker/experiments/20251128_220100_perspective_correction/worst_performers_top25.txt`

2. **Searches for mask files** in multiple possible locations:
   - `output/improved_edge_approach/worst_force_improved/`
   - `output/improved_edge_approach/`
   - Current directory `worst_force_improved/`

3. **Processes each image**:
   - Loads the mask file
   - Loads the original image from the dataset
   - Runs edge detection using `fit_mask_rectangle`
   - Applies perspective transformation
   - Saves the warped result

4. **Generates results**:
   - Creates a timestamped output directory in `artifacts/`
   - Saves warped images as `{image_id}_warped.jpg`
   - Creates `results.json` with detailed statistics

## Output Structure

```
artifacts/
└── {timestamp}_worst_performers_test/
    ├── {image_id}_warped.jpg  (for each successful image)
    └── results.json            (summary statistics)
```

## Results JSON Format

```json
{
  "total": 25,
  "success": 20,
  "failed": 3,
  "missing_mask": 1,
  "missing_image": 1,
  "timestamp": "20251129_120000",
  "details": [
    {
      "image_id": "drp.en_ko.in_house.selectstar_000006",
      "status": "success",
      "reason": "Edge detection successful",
      "output_file": "path/to/warped.jpg"
    },
    ...
  ]
}
```

## Status Values

- `success`: Perspective correction applied successfully
- `failed`: Edge detection or warping failed
- `missing_mask`: Mask file not found
- `missing_image`: Original image not found

## Next Steps After Running

1. Review the `results.json` file for statistics
2. Inspect the warped images in the output directory
3. Record results using experiment tracker tools:
   ```bash
   # Add a task for reviewing results
   ./experiment-tracker/scripts/add-task.py --description "Review worst performers test results" --status completed

   # Record the results artifact
   ./experiment-tracker/scripts/record-artifact.py --path artifacts/{timestamp}_worst_performers_test/results.json --type test_results
   ```

4. Generate an assessment if needed:
   ```bash
   ./experiment-tracker/scripts/generate-assessment.py --template run-log-negative-result
   ```
