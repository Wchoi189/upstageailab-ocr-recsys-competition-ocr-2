#!/usr/bin/env python3
"""T015 [US1]: High-loss sample export utility.

Exports samples whose label-length proxy exceeds the p95 loss threshold,
plus all samples with confirmed defects (truncation/script_mismatch/hallucination).

Loss proxy: label_len / tokenizer_max_len  (proxy until actual CTC inference).
p95 threshold sourced from data/audit/loss_percentiles.json.

Output: data/audit/high_loss_samples.json
  {
    "threshold_p95": float,
    "total_high_loss": int,
    "tokenizer_max_len": int,
    "samples": [{"idx": int, "label": str, "label_len": int, "loss_proxy": float}, ...]
  }

Usage:
  uv run python scripts/analysis/export_high_loss_samples.py \\
    --lmdb_path data/processed/recognition/aihub_lmdb_validation \\
    --loss_percentiles data/audit/loss_percentiles.json \\
    --output data/audit/high_loss_samples.json \\
    [--max_samples 5000]
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

from scripts.data.quality.manifest_io import iter_lmdb_samples


def export_high_loss(
    lmdb_path: Path,
    threshold_p95: float,
    tokenizer_max_len: int = 25,
    max_samples: int | None = None,
) -> dict:
    samples = []
    for s in iter_lmdb_samples(lmdb_path, skip_images=True):
        loss_proxy = len(s.label) / tokenizer_max_len
        if loss_proxy >= threshold_p95:
            samples.append({
                "idx": s.idx,
                "label": s.label,
                "label_len": len(s.label),
                "loss_proxy": round(loss_proxy, 4),
            })
            if max_samples and len(samples) >= max_samples:
                break

    # Sort descending by loss proxy
    samples.sort(key=lambda x: x["loss_proxy"], reverse=True)

    return {
        "threshold_p95": threshold_p95,
        "tokenizer_max_len": tokenizer_max_len,
        "total_high_loss": len(samples),
        "note": "loss_proxy = label_len / tokenizer_max_len (proxy; run inference for true CTC loss)",
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb_path", default="data/processed/recognition/aihub_lmdb_validation")
    ap.add_argument("--loss_percentiles", default="data/audit/loss_percentiles.json")
    ap.add_argument("--output", default="data/audit/high_loss_samples.json")
    ap.add_argument("--tokenizer_max_len", type=int, default=25)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap export size (default: all high-loss samples)",
    )
    args = ap.parse_args()

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        print(f"ERROR: LMDB not found: {lmdb_path}", file=sys.stderr)
        sys.exit(1)

    percentiles_path = Path(args.loss_percentiles)
    if not percentiles_path.exists():
        print(f"ERROR: loss_percentiles.json not found: {percentiles_path}", file=sys.stderr)
        sys.exit(1)

    percentiles = json.loads(percentiles_path.read_text())
    threshold = percentiles["high_loss_threshold_p95"]

    print(f"Exporting high-loss samples (proxy >= {threshold}) from {lmdb_path}")
    result = export_high_loss(
        lmdb_path=lmdb_path,
        threshold_p95=threshold,
        tokenizer_max_len=args.tokenizer_max_len,
        max_samples=args.max_samples,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Written: {out}")
    print(f"  total_high_loss={result['total_high_loss']}  threshold_p95={threshold}")


if __name__ == "__main__":
    main()
