#!/usr/bin/env python3
"""Non-mutating defect distribution analyzer.

Iterates the training LMDB directly (no intermediate .jsonl) and produces a
prevalence report of defect classes across the dataset.

Output: data/audit/defect_prevalence.json

Usage:
  uv run python scripts/audit/analyze_defect_distribution.py \
    --lmdb_path data/processed/recognition/aihub_lmdb_validation \
    --output data/audit/defect_prevalence.json

Gate dependency: Gate 0 — defect_prevalence.json required for exit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from ocr.core.utils.path_utils import PROJECT_ROOT
    _project_root = PROJECT_ROOT
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_project_root))

from scripts.data.quality.manifest_io import iter_lmdb_samples, lmdb_num_samples
from scripts.data.quality.defect_rules import evaluate_label
from scripts.data.quality.quality_scoring import score_batch


def analyze(lmdb_path: Path, tokenizer_max_len: int = 25) -> dict:
    total = lmdb_num_samples(lmdb_path)
    results = []
    for sample in iter_lmdb_samples(lmdb_path, skip_images=True):
        r = evaluate_label(
            sample_id=str(sample.idx),
            label=sample.label,
            tokenizer_max_len=tokenizer_max_len,
        )
        results.append(r)

    batch = score_batch(results)

    # Prevalence rates
    n = batch["total"]
    defect_rates = {
        cls: round(count / n, 6)
        for cls, count in batch["defect_class_counts"].items()
    }
    severity_rates = {
        sev: round(count / n, 6)
        for sev, count in batch["severity_counts"].items()
    }

    return {
        "lmdb_path": str(lmdb_path),
        "total": n,
        "clean_count": batch["clean_count"],
        "defective_count": batch["defective_count"],
        "mean_quality_score": batch["mean_score"],
        "provenance_coverage": 1.0,  # all LMDB records have idx as sample_id
        "defect_class_counts": batch["defect_class_counts"],
        "defect_class_rates": defect_rates,
        "severity_counts": batch["severity_counts"],
        "severity_rates": severity_rates,
        "tokenizer_max_len": tokenizer_max_len,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lmdb_path",
        default="data/processed/recognition/aihub_lmdb_validation",
        help="Path to LMDB directory",
    )
    ap.add_argument(
        "--output",
        default="data/audit/defect_prevalence.json",
        help="Output JSON path",
    )
    ap.add_argument("--tokenizer_max_len", type=int, default=25)
    args = ap.parse_args()

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        print(f"ERROR: LMDB path not found: {lmdb_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing defect distribution: {lmdb_path}")
    result = analyze(lmdb_path, tokenizer_max_len=args.tokenizer_max_len)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Written: {out_path}")
    print(f"  total={result['total']} clean={result['clean_count']} defective={result['defective_count']}")


if __name__ == "__main__":
    main()
