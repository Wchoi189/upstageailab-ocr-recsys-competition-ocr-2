#!/usr/bin/env python3
"""View parquet file safely, handling numpy array polygons correctly."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def format_polygon(poly):
    """Safely format polygon for display."""
    if poly is None:
        return "None"
    if isinstance(poly, (list, np.ndarray)):
        try:
            # Convert to list if numpy array
            if isinstance(poly, np.ndarray):
                poly = poly.tolist()
            # Format first few points
            if len(poly) > 4:
                return f"[{len(poly)} points]"
            return str(poly)
        except:
            return str(type(poly))
    return str(poly)


def view_parquet(filepath, n_rows=5):
    """View parquet file safely."""
    df = pd.read_parquet(filepath)

    print(f"\n{'='*80}")
    print(f"Parquet File: {filepath}")
    print(f"{'='*80}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst {n_rows} rows:\n")

    for idx, row in df.head(n_rows).iterrows():
        print(f"Row {idx}:")
        for col in df.columns:
            value = row[col]

            # Special handling for array/list columns
            if col == 'polygons' and isinstance(value, (list, np.ndarray)):
                print(f"  {col}: {len(value)} polygons")
                if len(value) > 0:
                    print(f"    First: {format_polygon(value[0])}")
            elif col == 'texts' and isinstance(value, (list, np.ndarray)):
                print(f"  {col}: {len(value)} texts")
                if len(value) > 0:
                    print(f"    First: {value[0][:50] if len(value[0]) > 50 else value[0]}")
            elif isinstance(value, (list, np.ndarray)):
                print(f"  {col}: [{len(value)} items]")
            else:
                # Truncate long strings
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"  {col}: {value_str}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_parquet.py <parquet_file> [n_rows]")
        print("\nExample:")
        print("  python view_parquet.py data/processed/test_50.parquet 3")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    view_parquet(filepath, n_rows)
