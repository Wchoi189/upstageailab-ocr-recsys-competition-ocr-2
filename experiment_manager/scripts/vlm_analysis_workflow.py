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

    # Check if the command would actually work (mock environment might not have uv or the module)
    # in a real environment we would run it. For now we will mock valid execution if the module is missing
    # to allow the script to be valid python
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        # Fallback for environments without uv
        print("  ! 'uv' not found, skipping actual execution")
        return {"success": False, "error": "uv not found"}

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

    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}")
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    # Handle both list and dict formats for manifest
    cases = manifest.get("cases", []) if isinstance(manifest, dict) else manifest

    for case in cases[:10]:  # First 10 for baseline
        # Handle simple string paths or dict objects
        path_str = case["image_path"] if isinstance(case, dict) else case
        image_path = Path(path_str)
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

    # Default to a generic experiment path if specific one not found
    base_dir = Path("experiment-tracker/experiments/20251218_1900_border_removal_preprocessing")
    if not base_dir.exists():
         # Fallback for current workspace structure
         base_dir = Path("experiment_manager/experiments/20251218_1900_border_removal_preprocessing")

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
    print(f"Quality: {len(quality_results) if 'quality_results' in locals() else 0} reports")
    print()
    print("All reports saved to: outputs/vlm_analysis/")


if __name__ == "__main__":
    main()
