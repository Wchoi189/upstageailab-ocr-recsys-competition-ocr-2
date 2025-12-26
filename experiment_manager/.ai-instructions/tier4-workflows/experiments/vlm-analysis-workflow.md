# Border Removal Experiment - VLM Analysis Workflow

## Experiment Philosophy

**This experiment uses VLM extensively** to build reproducible, high-quality technical analysis.

**VLM is NOT constrained by latency** - use it as much as needed for:
- Baseline image quality assessment
- Border detection validation
- Method comparison analysis
- Failure case diagnosis
- Success case verification

## VLM Analysis Phases

### Phase 1: Baseline Assessment

**Goal**: Document current state of border-affected images

```bash
# For each border case in manifest, run VLM baseline analysis
for image in border_cases_manifest.json:
    uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
        --image $image \
        --mode image_quality \
        --backend dashscope \
        --output outputs/vlm_baseline/${image_name}_baseline.md
```

**VLM will assess**:
- Border presence and severity (1-10 scale)
- Impact on OCR readability
- Geometric distortions
- Skew angle estimation (visual)
- Content area visibility

**Output**: Baseline report for each image (~30-45s per image, acceptable for quality)

### Phase 2: Border Detection Validation

**Goal**: Verify each method's border detection accuracy

```bash
# After running border removal methods, validate detection
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image outputs/border_detection_visualization/${image}_canny_detection.jpg \
    --mode preprocessing_diagnosis \
    --backend dashscope \
    --output outputs/vlm_validation/${image}_canny_validation.md
```

**VLM will assess**:
- Is detected border box accurate?
- Are there false positives (content incorrectly marked as border)?
- Are there false negatives (border not detected)?
- Confidence score reasonable?

**Output**: Detection validation report per method per image

### Phase 3: Crop Quality Assessment

**Goal**: Verify cropped images preserve content and improve quality

```bash
# Side-by-side comparison analysis
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image outputs/comparisons/${image}_before_after_canny.jpg \
    --mode enhancement_validation \
    --backend dashscope \
    --output outputs/vlm_quality/${image}_canny_quality.md
```

**VLM will assess**:
- Is content preserved (no text cut off)?
- Is border completely removed?
- Is image quality improved?
- Any artifacts introduced?
- OCR readiness improvement

**Output**: Quality assessment report per method per image

### Phase 4: Method Comparison

**Goal**: Compare all 3 methods on same image

```bash
# Create comparison grid: original + 3 methods
# Then analyze with VLM
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image outputs/method_comparison/${image}_all_methods.jpg \
    --mode enhancement_validation \
    --backend dashscope \
    --output outputs/vlm_comparison/${image}_methods_comparison.md
```

**VLM will compare**:
- Which method has cleanest result?
- Which method best preserves content?
- Which method best removes border?
- Trade-offs between methods?

**Output**: Comparative analysis report per image

### Phase 5: Failure Analysis

**Goal**: Understand why methods fail on certain images

```bash
# For images where all methods failed
uv run python -m AgentQMS.vlm.cli.analyze_image_defects \
    --image outputs/failures/${image}_all_methods_failed.jpg \
    --mode preprocessing_diagnosis \
    --backend dashscope \
    --output outputs/vlm_failures/${image}_failure_analysis.md
```

**VLM will diagnose**:
- Why did detection fail?
- What makes this image challenging?
- Are borders ambiguous/complex?
- Is content too close to edges?
- Recommendations for improvement

**Output**: Failure analysis report per failed case

## VLM Workflow Script

Create helper script to automate VLM analysis:

```python
#!/usr/bin/env python3
"""
VLM analysis workflow for border removal experiment.

Runs comprehensive VLM analysis on all test images across all phases.
"""

import json
import subprocess
from pathlib import Path


def run_vlm_analysis(
    image_path: Path,
    mode: str,
    output_path: Path,
    backend: str = "dashscope"
) -> dict:
    """Run VLM analysis and return metrics."""

    cmd = [
        "uv", "run", "python", "-m", "AgentQMS.vlm.cli.analyze_image_defects",
        "--image", str(image_path),
        "--mode", mode,
        "--backend", backend,
        "--output", str(output_path),
    ]

    print(f"Running VLM analysis: {image_path.name} (mode: {mode})...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Saved to: {output_path}")
        return {"success": True, "output": str(output_path)}
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return {"success": False, "error": result.stderr}


def run_baseline_analysis(manifest_path: Path, output_dir: Path):
    """Phase 1: Baseline assessment."""

    print("\n" + "="*60)
    print("PHASE 1: VLM Baseline Assessment")
    print("="*60)

    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case in manifest["cases"][:10]:  # First 10 for baseline
        image_path = Path(case["image_path"])
        output_path = output_dir / f"{image_path.stem}_baseline.md"

        result = run_vlm_analysis(
            image_path=image_path,
            mode="image_quality",
            output_path=output_path,
        )
        results.append(result)

    print(f"\nBaseline assessment complete: {sum(1 for r in results if r['success'])}/{len(results)} succeeded")
    return results


def run_validation_analysis(
    processed_dir: Path,
    output_dir: Path,
    method: str
):
    """Phase 2: Border detection validation."""

    print("\n" + "="*60)
    print(f"PHASE 2: VLM Validation - {method.upper()} method")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all processed images for this method
    image_files = list(processed_dir.glob(f"*_{method}_*.jpg"))

    results = []
    for image_path in image_files:
        output_path = output_dir / f"{image_path.stem}_validation.md"

        result = run_vlm_analysis(
            image_path=image_path,
            mode="preprocessing_diagnosis",
            output_path=output_path,
        )
        results.append(result)

    print(f"\nValidation complete: {sum(1 for r in results if r['success'])}/{len(results)} succeeded")
    return results


def run_quality_analysis(
    comparison_dir: Path,
    output_dir: Path
):
    """Phase 3: Crop quality assessment."""

    print("\n" + "="*60)
    print("PHASE 3: VLM Quality Assessment")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all comparison images
    image_files = list(comparison_dir.glob("*_comparison_*.jpg"))

    results = []
    for image_path in image_files:
        output_path = output_dir / f"{image_path.stem}_quality.md"

        result = run_vlm_analysis(
            image_path=image_path,
            mode="enhancement_validation",
            output_path=output_path,
        )
        results.append(result)

    print(f"\nQuality assessment complete: {sum(1 for r in results if r['success'])}/{len(results)} succeeded")
    return results


def main():
    """Run full VLM analysis workflow."""

    base_dir = Path("experiment-tracker/experiments/20251218_1900_border_removal_preprocessing")

    # Phase 1: Baseline
    baseline_results = run_baseline_analysis(
        manifest_path=base_dir / "outputs/border_cases_manifest.json",
        output_dir=base_dir / "outputs/vlm_analysis/baseline",
    )

    # Phase 2: Validation (per method)
    for method in ["canny", "morph", "hough"]:
        validation_results = run_validation_analysis(
            processed_dir=base_dir / f"outputs/border_removed_{method}",
            output_dir=base_dir / f"outputs/vlm_analysis/validation_{method}",
            method=method,
        )

    # Phase 3: Quality
    quality_results = run_quality_analysis(
        comparison_dir=base_dir / "outputs/comparisons",
        output_dir=base_dir / "outputs/vlm_analysis/quality",
    )

    print("\n" + "="*60)
    print("VLM ANALYSIS COMPLETE")
    print("="*60)
    print(f"Baseline: {len(baseline_results)} reports")
    print(f"Validation: 3 methods analyzed")
    print(f"Quality: {len(quality_results)} reports")
    print()
    print("All reports saved to: outputs/vlm_analysis/")


if __name__ == "__main__":
    main()
```

## Expected VLM Report Structure

Each VLM report will contain:

### Baseline Report
```markdown
# Border Detection Baseline - {image_name}

## Image Quality Assessment
- Border presence: YES/NO
- Border severity: 1-10 scale
- Border type: black/white/colored/partial
- OCR readability impact: HIGH/MEDIUM/LOW

## Geometric Analysis
- Estimated skew angle: X.X°
- Content area visibility: X%
- Distortions observed: list

## Recommendations
- Should border be removed? YES/NO
- Expected difficulty: EASY/MEDIUM/HARD
- Special handling needed: notes
```

### Validation Report
```markdown
# Border Detection Validation - {method} - {image_name}

## Detection Accuracy
- Border box accurate: YES/NO
- Content preserved: YES/NO
- False positives: count
- False negatives: areas missed

## Confidence Assessment
- Detection confidence: X.XX (algorithm)
- Visual confidence: HIGH/MEDIUM/LOW (VLM)
- Discrepancies: notes

## Recommendations
- Proceed with crop: YES/NO
- Adjust parameters: suggestions
```

### Quality Report
```markdown
# Crop Quality Assessment - {method} - {image_name}

## Content Preservation
- Text fully visible: YES/NO
- No content cut off: YES/NO
- Border completely removed: YES/NO

## Quality Improvement
- OCR readability: IMPROVED/SAME/WORSE
- Geometric correction: IMPROVED/SAME/WORSE
- Overall quality: IMPROVED/SAME/WORSE

## Artifacts
- Edge artifacts: YES/NO
- Content distortion: YES/NO
- Quality degradation: YES/NO

## Verdict
- Use this result: YES/NO
- Score: X/10
- Notes: detailed observations
```

## Integration with Experiment Workflow

```bash
# Complete experiment workflow with VLM

# Step 1: Collect border cases
uv run python scripts/collect_border_cases.py

# Step 2: VLM baseline analysis
uv run python scripts/vlm_analysis_workflow.py --phase baseline

# Step 3: Run border removal methods
uv run python scripts/border_removal.py --method canny --all-cases
uv run python scripts/border_removal.py --method morph --all-cases
uv run python scripts/border_removal.py --method hough --all-cases

# Step 4: VLM validation analysis
uv run python scripts/vlm_analysis_workflow.py --phase validation

# Step 5: Create comparisons
uv run python scripts/create_comparisons.py

# Step 6: VLM quality analysis
uv run python scripts/vlm_analysis_workflow.py --phase quality

# Step 7: Generate final assessment report
# Step 7: Generate final assessment report
etk create assessment "Border Removal Validation" --type assessment \
    --phase execution
```

## Timeline with VLM Analysis

| Phase | Task | VLM Time | Total Time |
|-------|------|----------|------------|
| 1 | Collect cases | 0 | 5 min |
| 2 | VLM baseline (10 images) | 5-8 min | 10 min |
| 3 | Implement methods | 0 | 3 hours |
| 4 | Run methods (30 images) | 0 | 15 min |
| 5 | VLM validation (30×3 images) | 45-75 min | 2 hours |
| 6 | Create comparisons | 0 | 15 min |
| 7 | VLM quality (30 images) | 15-25 min | 45 min |
| 8 | Generate assessment | 0 | 30 min |

**Total VLM time**: ~1.5-2 hours (acceptable for quality analysis)
**Total experiment time**: ~8 hours (comprehensive validation)

## Benefits of VLM-Heavy Approach

1. **Reproducible**: All analysis documented in markdown reports
2. **High-quality**: Human-level visual assessment at scale
3. **Comprehensive**: Covers baseline, validation, quality, and comparison
4. **Traceable**: Every decision backed by VLM evidence
5. **Shareable**: Reports can be reviewed by humans for verification
