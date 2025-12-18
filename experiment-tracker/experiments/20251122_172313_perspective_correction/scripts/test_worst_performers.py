#!/usr/bin/env python3
"""
Extract worst performing images and test rembg mask-based approach on them.
"""

import json
import logging
from pathlib import Path
from typing import Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def extract_worst_performers(results_path: Path, max_count: int = 50) -> list[dict[str, Any]]:
    """
    Extract worst performing images from results.

    Prioritizes:
    1. Post-validation failures with highest area loss
    2. Pre-validation failures with smallest area ratio
    3. Pre-validation failures with aspect ratio mismatch
    """
    with open(results_path) as f:
        results = json.load(f)

    # Filter failures
    failures = []
    for result in results:
        regular_valid = result.get("regular_method", {}).get("validation", {}).get("valid", False)
        doctr_valid = result.get("doctr_method", {}).get("validation", {}).get("valid", False)

        if not regular_valid and not doctr_valid:
            # Calculate worst metric
            worst_metric = None
            metric_value = 0.0

            # Check post-validation area loss
            regular_validation = result.get("regular_method", {}).get("validation", {})
            doctr_validation = result.get("doctr_method", {}).get("validation", {})

            if "area_ratio" in regular_validation:
                area_ratio = regular_validation["area_ratio"]
                if isinstance(area_ratio, (int, float)):
                    worst_metric = "area_loss"
                    metric_value = 1.0 - area_ratio  # Higher is worse
            elif "area_ratio" in doctr_validation:
                area_ratio = doctr_validation["area_ratio"]
                if isinstance(area_ratio, (int, float)):
                    worst_metric = "area_loss"
                    metric_value = 1.0 - area_ratio

            # Check pre-validation area ratio (smaller is worse)
            regular_pre = result.get("regular_method", {}).get("pre_validation", {})
            doctr_pre = result.get("doctr_method", {}).get("pre_validation", {})

            if "area_ratio" in regular_pre:
                area_ratio = regular_pre["area_ratio"]
                if isinstance(area_ratio, (int, float)) and (worst_metric is None or metric_value < (1.0 - area_ratio)):
                    worst_metric = "pre_area_ratio"
                    metric_value = 1.0 - area_ratio

            if "area_ratio" in doctr_pre:
                area_ratio = doctr_pre["area_ratio"]
                if isinstance(area_ratio, (int, float)) and (worst_metric is None or metric_value < (1.0 - area_ratio)):
                    worst_metric = "pre_area_ratio"
                    metric_value = 1.0 - area_ratio

            # Add failure reason
            failure_reason = "unknown"
            if regular_validation.get("failure_reason"):
                failure_reason = regular_validation["failure_reason"]
            elif doctr_validation.get("failure_reason"):
                failure_reason = doctr_validation["failure_reason"]
            elif regular_pre.get("reason"):
                failure_reason = regular_pre["reason"]
            elif doctr_pre.get("reason"):
                failure_reason = doctr_pre["reason"]

            failures.append(
                {
                    "input_path": result["input_path"],
                    "worst_metric": worst_metric or "unknown",
                    "metric_value": metric_value,
                    "failure_reason": failure_reason,
                    "result": result,
                }
            )

    # Sort by worst metric value (descending - highest area loss first)
    failures.sort(key=lambda x: x["metric_value"], reverse=True)

    # Return top N
    return failures[:max_count]


def main():
    import argparse
    import sys

    # Import the test function
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from analyze_failures_rembg_approach import GPU_AVAILABLE, OptimizedBackgroundRemover, test_rembg_based_correction

    # Setup path utils for proper path resolution
    script_path = Path(__file__).resolve()
    try:
        # Add tracker src to path
        tracker_root = script_path.parent.parent.parent.parent
        sys.path.insert(0, str(tracker_root / "src"))
        from experiment_tracker.utils.path_utils import setup_script_paths

        TRACKER_ROOT, EXPERIMENT_ID, EXPERIMENT_PATHS = setup_script_paths(script_path)
    except ImportError:
        # Fallback if path_utils not available
        TRACKER_ROOT = script_path.parent.parent.parent.parent
        EXPERIMENT_PATHS = None

    # Setup OCR project paths
    workspace_root = tracker_root.parent
    sys.path.insert(0, str(workspace_root))
    try:
        from ocr.utils.path_utils import get_path_resolver

        OCR_RESOLVER = get_path_resolver()
        workspace_root = OCR_RESOLVER.config.project_root
    except ImportError:
        OCR_RESOLVER = None
        workspace_root = TRACKER_ROOT.parent if EXPERIMENT_PATHS else Path.cwd()

    # Get default paths using OCR resolver if available
    if OCR_RESOLVER:
        default_results_json = OCR_RESOLVER.config.output_dir / "perspective_comprehensive_retest" / "results.json"
        default_input_dir = OCR_RESOLVER.config.images_dir / "train"
        default_output_dir = OCR_RESOLVER.config.output_dir / "worst_performers_test"
    else:
        default_results_json = workspace_root / "outputs" / "perspective_comprehensive_retest" / "results.json"
        default_input_dir = workspace_root / "data" / "datasets" / "images" / "train"
        default_output_dir = workspace_root / "outputs" / "worst_performers_test"

    parser = argparse.ArgumentParser(description="Extract worst performers and test rembg mask-based approach")
    parser.add_argument(
        "--results-json",
        type=Path,
        default=default_results_json,
        help="Path to results JSON file",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Input directory with original images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Output directory",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=50,
        help="Maximum number of worst performers to test",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for rembg",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list worst performers, don't test",
    )

    args = parser.parse_args()

    # Extract worst performers
    logger.info(f"Extracting worst performers from: {args.results_json}")
    worst_performers = extract_worst_performers(args.results_json, args.max_count)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"WORST {len(worst_performers)} PERFORMING IMAGES")
    logger.info(f"{'=' * 80}\n")

    # Display list
    for i, failure in enumerate(worst_performers, 1):
        image_name = Path(failure["input_path"]).name
        metric = failure["worst_metric"]
        value = failure["metric_value"]
        reason = failure["failure_reason"][:60]  # Truncate long reasons
        logger.info(f"{i:3d}. {image_name}")
        logger.info(f"     Metric: {metric}, Value: {value:.2%}, Reason: {reason}")

    # Save list
    args.output_dir.mkdir(parents=True, exist_ok=True)
    list_output = args.output_dir / "worst_performers_list.json"
    with open(list_output, "w") as f:
        json.dump(
            [
                {
                    "rank": i + 1,
                    "input_path": w["input_path"],
                    "image_name": Path(w["input_path"]).name,
                    "worst_metric": w["worst_metric"],
                    "metric_value": w["metric_value"],
                    "failure_reason": w["failure_reason"],
                }
                for i, w in enumerate(worst_performers)
            ],
            f,
            indent=2,
        )
    logger.info(f"\nList saved to: {list_output}")

    if args.list_only:
        return 0

    # Run tests
    logger.info(f"\n{'=' * 80}")
    logger.info("TESTING REMBG MASK-BASED APPROACH")
    logger.info(f"{'=' * 80}\n")

    # Initialize remover
    remover = OptimizedBackgroundRemover(
        model_name="silueta",
        max_size=2048,
        alpha_matting=False,
        use_gpu=args.use_gpu and GPU_AVAILABLE,
        use_tensorrt=False,
        use_int8=False,
    )

    # Test each image
    test_results = []
    successful = 0
    failed = 0

    for i, failure in enumerate(worst_performers, 1):
        image_path = Path(failure["input_path"])

        # Try to find image
        if not image_path.exists() and args.input_dir.exists():
            candidate = args.input_dir / image_path.name
            if candidate.exists():
                image_path = candidate
            else:
                logger.warning(f"  ⚠ Image not found: {image_path.name}")
                continue

        logger.info(f"[{i}/{len(worst_performers)}] Processing: {image_path.name}")

        try:
            result = test_rembg_based_correction(
                image_path,
                args.output_dir,
                remover,
            )
            test_results.append(result)

            if result.get("success"):
                successful += 1
                logger.info(f"  ✓ Success (area ratio: {result.get('area_ratio', 0):.2%})")
            else:
                failed += 1
                reason = result.get("failure_reason") or result.get("error", "Unknown")
                logger.warning(f"  ✗ Failed: {reason}")

        except Exception as e:
            failed += 1
            logger.error(f"  ✗ Error: {e}", exc_info=True)
            test_results.append(
                {
                    "input_path": str(image_path),
                    "success": False,
                    "error": str(e),
                }
            )

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total tested: {len(test_results)}")
    logger.info(f"Successful: {successful} ({100 * successful / len(test_results):.1f}%)")
    logger.info(f"Failed: {failed} ({100 * failed / len(test_results):.1f}%)")

    # Save results
    results_output = args.output_dir / "test_results.json"
    with open(results_output, "w") as f:
        json.dump(test_results, f, indent=2, default=str)
    logger.info(f"\nTest results saved to: {results_output}")

    return 0


if __name__ == "__main__":
    exit(main())
