#!/usr/bin/env python3
"""Non-mutating loss distribution analyzer.

PRIMARY MODE: Label-length distribution as CTC-loss proxy.
  - No inference run required; no GPU needed.
  - Label length is strongly correlated with CTC loss for Korean OCR.
  - Output is explicitly labeled as a proxy until an inference pass is run.

INFERENCE MODE (--loss_log): Accepts a pre-computed per-sample loss JSON/CSV
  from a training/eval run if available.

Output: data/audit/loss_percentiles.json

Usage (proxy mode):
  uv run python scripts/audit/compute_loss_distribution.py \
    --lmdb_path data/processed/recognition/aihub_lmdb_validation \
    --output data/audit/loss_percentiles.json

Usage (inference mode — requires pre-computed loss log):
  uv run python scripts/audit/compute_loss_distribution.py \
    --loss_log path/to/per_sample_loss.json \
    --output data/audit/loss_percentiles.json

Gate dependency: Gate 0 — loss_percentiles.json required for exit.
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


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def analyze_proxy(lmdb_path: Path, tokenizer_max_len: int = 25) -> dict:
    """Use label length as CTC-loss proxy — no inference required."""
    values: list[float] = []
    for sample in iter_lmdb_samples(lmdb_path, skip_images=True):
        # Normalised label length as proxy: longer labels → higher CTC loss
        values.append(len(sample.label) / tokenizer_max_len)

    if not values:
        return {"error": "no samples found"}

    values.sort()
    n = len(values)
    p95 = _percentile(values, 95)
    high_loss = [v for v in values if v >= p95]

    return {
        "mode": "label_length_proxy",
        "note": "Per-sample CTC loss not available. Label-length/max_len ratio used as proxy. Run inference for true loss distribution.",
        "lmdb_path": str(lmdb_path),
        "tokenizer_max_len": tokenizer_max_len,
        "total": n,
        "loss_percentiles": {
            "p50": round(_percentile(values, 50), 6),
            "p75": round(_percentile(values, 75), 6),
            "p90": round(_percentile(values, 90), 6),
            "p95": round(p95, 6),
            "p99": round(_percentile(values, 99), 6),
        },
        "high_loss_threshold_p95": round(p95, 6),
        "high_loss_count": len(high_loss),
        "high_loss_rate": round(len(high_loss) / n, 6),
        "stats": {
            "min": round(values[0], 6),
            "max": round(values[-1], 6),
            "mean": round(sum(values) / n, 6),
        },
    }


def analyze_from_log(loss_log_path: Path) -> dict:
    """Load per-sample loss from a pre-computed JSON log.

    Expected format: list of {"sample_id": str, "loss": float}
    """
    records = json.loads(loss_log_path.read_text())
    if isinstance(records, dict):
        # Support {sample_id: loss} dict format
        records = [{"sample_id": k, "loss": v} for k, v in records.items()]

    values = sorted(r["loss"] for r in records)
    n = len(values)
    p95 = _percentile(values, 95)
    high_loss = [v for v in values if v >= p95]

    return {
        "mode": "inference_log",
        "loss_log": str(loss_log_path),
        "total": n,
        "loss_percentiles": {
            "p50": round(_percentile(values, 50), 6),
            "p75": round(_percentile(values, 75), 6),
            "p90": round(_percentile(values, 90), 6),
            "p95": round(p95, 6),
            "p99": round(_percentile(values, 99), 6),
        },
        "high_loss_threshold_p95": round(p95, 6),
        "high_loss_count": len(high_loss),
        "high_loss_rate": round(len(high_loss) / n, 6),
        "stats": {
            "min": round(values[0], 6),
            "max": round(values[-1], 6),
            "mean": round(sum(values) / n, 6),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--lmdb_path",
        default="data/processed/recognition/aihub_lmdb_validation",
        help="Path to LMDB directory (proxy mode)",
    )
    ap.add_argument(
        "--loss_log",
        default=None,
        help="Pre-computed per-sample loss JSON (inference mode); overrides --lmdb_path",
    )
    ap.add_argument("--tokenizer_max_len", type=int, default=25)
    ap.add_argument(
        "--output",
        default="data/audit/loss_percentiles.json",
        help="Output JSON path",
    )
    args = ap.parse_args()

    if args.loss_log:
        loss_log = Path(args.loss_log)
        if not loss_log.exists():
            print(f"ERROR: loss_log not found: {loss_log}", file=sys.stderr)
            sys.exit(1)
        print(f"Analyzing loss distribution from log: {loss_log}")
        result = analyze_from_log(loss_log)
    else:
        lmdb_path = Path(args.lmdb_path)
        if not lmdb_path.exists():
            print(f"ERROR: LMDB path not found: {lmdb_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Analyzing loss distribution (proxy): {lmdb_path}")
        result = analyze_proxy(lmdb_path, tokenizer_max_len=args.tokenizer_max_len)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Written: {out_path}")
    p = result.get("loss_percentiles", {})
    print(f"  p95={p.get('p95')} high_loss_rate={result.get('high_loss_rate')}")


if __name__ == "__main__":
    main()
