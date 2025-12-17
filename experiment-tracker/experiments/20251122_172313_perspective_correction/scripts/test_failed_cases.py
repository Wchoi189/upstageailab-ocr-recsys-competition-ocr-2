#!/usr/bin/env python3
"""
Test improved perspective correction on previously failed cases.

This script:
1. Loads failed cases from previous test
2. Tests with improved parameters (min_area_ratio=0.3, pre-validation)
3. Compares results to see improvement
"""

import json
import sys
from pathlib import Path

# Import the comprehensive test script
# Note: Both scripts are in the same directory in experiment-tracker
sys.path.insert(0, str(Path(__file__).parent))
from test_perspective_comprehensive import process_image_comprehensive

if __name__ == "__main__":
    # Load failed cases
    failed_cases_path = Path("outputs/perspective_comprehensive/failed_cases.json")
    if not failed_cases_path.exists():
        print(f"Error: {failed_cases_path} not found")
        print(
            "Run: python3 -c \"import json; data = json.load(open('outputs/perspective_comprehensive/results.json')); failures = [r['input_path'] for r in data if not r.get('regular_method', {}).get('validation', {}).get('valid', False)]; json.dump(failures, open('outputs/perspective_comprehensive/failed_cases.json', 'w'), indent=2)\""
        )
        sys.exit(1)

    with open(failed_cases_path) as f:
        failed_cases = json.load(f)

    print(f"Found {len(failed_cases)} failed cases to retest")

    # Output directory for retest
    output_dir = Path("outputs/perspective_comprehensive_retest")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each failed case
    all_results = []
    for i, image_path_str in enumerate(failed_cases, 1):
        image_path = Path(image_path_str)
        if not image_path.exists():
            print(f"  [{i}/{len(failed_cases)}] Skipping (not found): {image_path}")
            continue

        print(f"  [{i}/{len(failed_cases)}] Processing: {image_path.name}")
        result = process_image_comprehensive(
            image_path=image_path,
            output_dir=output_dir,
            use_gpu=True,
        )
        all_results.append(result)

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Analysis
    print("\n" + "=" * 80)
    print("RETEST RESULTS")
    print("=" * 80)

    regular_valid = [r for r in all_results if r.get("regular_method", {}).get("validation", {}).get("valid")]
    doctr_valid = [r for r in all_results if r.get("doctr_method", {}).get("validation", {}).get("valid")]
    fallback_used = [r for r in all_results if r.get("fallback_used")]
    both_failed = [
        r
        for r in all_results
        if not r.get("regular_method", {}).get("validation", {}).get("valid")
        and not r.get("doctr_method", {}).get("validation", {}).get("valid")
    ]

    print(f"\nTotal retested: {len(all_results)}")
    print(f"Regular method now valid: {len(regular_valid)} ({100 * len(regular_valid) / len(all_results):.1f}%)")
    print(f"DocTR method now valid: {len(doctr_valid)} ({100 * len(doctr_valid) / len(all_results):.1f}%)")
    print(f"Fallback used: {len(fallback_used)} ({100 * len(fallback_used) / len(all_results):.1f}%)")
    print(f"Both methods still failed: {len(both_failed)} ({100 * len(both_failed) / len(all_results):.1f}%)")

    # Improvement
    improvement = len(regular_valid) / len(all_results) * 100
    print(f"\nImprovement: {improvement:.1f}% of previously failed cases now succeed")

    print(f"\nResults saved to: {results_path}")
    print(f"Output images in: {output_dir}")
