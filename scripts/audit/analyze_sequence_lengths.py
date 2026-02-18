#!/usr/bin/env python3
"""Non-mutating sequence-length analyzer.

Iterates the training LMDB directly and computes label length distribution,
truncation rate, and percentile breakdown.

Output: data/audit/truncation_analysis.json

Usage:
  uv run python scripts/audit/analyze_sequence_lengths.py \
    --lmdb_path data/processed/recognition/aihub_lmdb_validation \
    --tokenizer_max_len 25 \
    --output data/audit/truncation_analysis.json

Gate dependency: Gate 0 — truncation_analysis.json required for exit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

try:
    from ocr.core.utils.path_utils import PROJECT_ROOT
    _project_root = PROJECT_ROOT
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_project_root))

from scripts.data.quality.manifest_io import iter_lmdb_samples, lmdb_num_samples


def _percentile(sorted_vals: list[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def analyze(lmdb_path: Path, tokenizer_max_len: int = 25) -> dict:
    lengths: list[int] = []
    for sample in iter_lmdb_samples(lmdb_path, skip_images=True):
        lengths.append(len(sample.label))

    if not lengths:
        return {"error": "no samples found"}

    lengths.sort()
    n = len(lengths)
    at_max = sum(1 for l in lengths if l >= tokenizer_max_len)
    near_max = sum(1 for l in lengths if tokenizer_max_len - 3 <= l < tokenizer_max_len)

    freq = Counter(lengths)
    length_histogram = {str(k): v for k, v in sorted(freq.items())}

    return {
        "lmdb_path": str(lmdb_path),
        "total": n,
        "tokenizer_max_len": tokenizer_max_len,
        "truncation_rate": round(at_max / n, 6),
        "near_max_rate": round(near_max / n, 6),
        "at_max_count": at_max,
        "near_max_count": near_max,
        "length_stats": {
            "min": lengths[0],
            "max": lengths[-1],
            "mean": round(sum(lengths) / n, 4),
            "p50": _percentile(lengths, 50),
            "p75": _percentile(lengths, 75),
            "p90": _percentile(lengths, 90),
            "p95": _percentile(lengths, 95),
            "p99": _percentile(lengths, 99),
        },
        "length_histogram": length_histogram,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lmdb_path",
        default="data/processed/recognition/aihub_lmdb_validation",
        help="Path to LMDB directory",
    )
    ap.add_argument("--tokenizer_max_len", type=int, default=25)
    ap.add_argument(
        "--output",
        default="data/audit/truncation_analysis.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        print(f"ERROR: LMDB path not found: {lmdb_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing sequence lengths: {lmdb_path}")
    result = analyze(lmdb_path, tokenizer_max_len=args.tokenizer_max_len)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Written: {out_path}")
    stats = result.get("length_stats", {})
    print(f"  total={result['total']} truncation_rate={result['truncation_rate']:.4f} p95={stats.get('p95')}")


if __name__ == "__main__":
    main()
