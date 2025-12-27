#!/usr/bin/env python3
"""Compare pseudo-label results across enhancement strategies."""

import pandas as pd
from pathlib import Path


def analyze_strategy(name: str, parquet_file: Path):
    """Analyze a single strategy's results."""
    df = pd.read_parquet(parquet_file)

    # Calculate statistics
    polygon_counts = df['polygons'].apply(len)
    text_counts = df['texts'].apply(len)
    empty_count = (polygon_counts == 0).sum()

    # Get image dimensions for context
    avg_width = df['width'].mean()
    avg_height = df['height'].mean()

    return {
        'name': name,
        'total_images': len(df),
        'avg_polygons': polygon_counts.mean(),
        'std_polygons': polygon_counts.std(),
        'min_polygons': polygon_counts.min(),
        'max_polygons': polygon_counts.max(),
        'avg_texts': text_counts.mean(),
        'empty_results': empty_count,
        'empty_rate': (empty_count / len(df)) * 100,
        'avg_width': avg_width,
        'avg_height': avg_height,
    }


def main():
    # Define strategies to compare
    strategies = [
        ("Baseline (No Enhancement)", "data/samples/pseudo_labels_baseline.parquet"),
        ("Sepia 0.85 (Moderate)", "data/samples/pseudo_labels_sepia_085.parquet"),
    ]

    results = []
    for name, path in strategies:
        path_obj = Path(path)
        if path_obj.exists():
            results.append(analyze_strategy(name, path_obj))
        else:
            print(f"⚠️  Not found: {path}")

    if not results:
        print("No results to compare!")
        return

    # Print comparison table
    print("\n" + "="*80)
    print("PSEUDO-LABEL COMPARISON")
    print("="*80 + "\n")

    for r in results:
        print(f"Strategy: {r['name']}")
        print(f"  Total Images: {r['total_images']}")
        print(f"  Avg Polygons: {r['avg_polygons']:.1f} ± {r['std_polygons']:.1f}")
        print(f"  Range: {r['min_polygons']} - {r['max_polygons']}")
        print(f"  Avg Texts: {r['avg_texts']:.1f}")
        print(f"  Empty Results: {r['empty_results']}/{r['total_images']} ({r['empty_rate']:.1f}%)")
        print(f"  Avg Dimensions: {r['avg_width']:.0f}×{r['avg_height']:.0f}px")
        print()

    # Calculate improvement
    if len(results) == 2:
        baseline, enhanced = results[0], results[1]
        polygon_improvement = ((enhanced['avg_polygons'] - baseline['avg_polygons']) / baseline['avg_polygons']) * 100

        print("-" * 80)
        print(f"Enhancement Impact:")
        print(f"  Polygon Count Change: {polygon_improvement:+.1f}%")
        print(f"  Empty Result Change: {baseline['empty_results']} → {enhanced['empty_results']}")

        # Validation checks
        print("\n" + "="*80)
        print("VALIDATION CHECKLIST")
        print("="*80 + "\n")

        checks = [
            ("Empty Results ≤ 2%", enhanced['empty_rate'] <= 2.0, f"{enhanced['empty_rate']:.1f}%"),
            ("Enhancement Benefit ≥ 5%", polygon_improvement >= 5.0, f"{polygon_improvement:+.1f}%"),
            ("No Empty Baseline Images", baseline['empty_results'] == 0, f"{baseline['empty_results']} empty"),
        ]

        for check_name, passed, value in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {check_name} ({value})")

        print()


if __name__ == "__main__":
    main()
