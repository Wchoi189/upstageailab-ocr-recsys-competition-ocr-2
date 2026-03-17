#!/usr/bin/env python3
"""RISK-01: Per-sample CTC loss surrogate inference using PaddleOCR.

Uses PaddleOCR recognition model (trained) to compute per-sample
CTC loss surrogate: loss = -log(confidence) for each holdout sample.

This resolves RISK-01 by providing actual model-inference-derived loss
distribution (vs. label-length proxy used in Gate 0).

Output:
  data/audit/risk01_ctc_loss_surrogate.json  — per-sample loss log
  data/audit/risk01_loss_percentiles.json    — distribution analysis

Usage:
  cd /workspaces
  python dev_tools/experiment_manager/experiments/\
20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/\
risk01_ctc_loss_inference.py

Flags:
  --holdout   path to holdout_clean_v2.jsonl (default: canonical)
  --img_dir   local image directory (default: candidates_images/)
  --output    output JSON path (default: data/audit/risk01_ctc_loss_surrogate.json)
  --lang      PaddleOCR language (default: korean)
  --batch     progress report interval (default: 50)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_HOLDOUT = PROJECT_ROOT / "data/processed/recognition/holdout_clean_v2.jsonl"
DEFAULT_IMG_DIR = PROJECT_ROOT / "data/processed/recognition/candidates_images"
DEFAULT_OUTPUT  = PROJECT_ROOT / "data/audit/risk01_ctc_loss_surrogate.json"
DEFAULT_DIST_OUTPUT = PROJECT_ROOT / "data/audit/risk01_loss_percentiles.json"

MIN_CONFIDENCE = 1e-10  # floor to avoid log(0)


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def compute_distribution(records: list[dict]) -> dict:
    values = sorted(r["loss"] for r in records)
    n = len(values)
    p95 = _percentile(values, 95)
    high_loss = [v for v in values if v >= p95]
    return {
        "mode": "paddle_ocr_ctc_surrogate",
        "note": "CTC loss surrogate: loss = -log(confidence). PaddleOCR recognition model inference.",
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
    ap.add_argument("--holdout", default=str(DEFAULT_HOLDOUT))
    ap.add_argument("--img_dir", default=str(DEFAULT_IMG_DIR))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--dist_output", default=str(DEFAULT_DIST_OUTPUT))
    ap.add_argument("--lang", default="korean")
    ap.add_argument("--batch", type=int, default=50)
    args = ap.parse_args()

    holdout_path = Path(args.holdout)
    img_dir = Path(args.img_dir)
    output_path = Path(args.output)
    dist_output_path = Path(args.dist_output)

    if not holdout_path.exists():
        print(f"ERROR: holdout not found: {holdout_path}", file=sys.stderr)
        sys.exit(1)
    if not img_dir.exists():
        print(f"ERROR: img_dir not found: {img_dir}", file=sys.stderr)
        sys.exit(1)

    records = [json.loads(l) for l in holdout_path.open()]
    print(f"Holdout: {len(records)} samples")
    print(f"Image dir: {img_dir}")
    print(f"Lang: {args.lang}")

    # ── Init PaddleOCR (recognition only) ──────────────────────────────────
    print("Loading PaddleOCR recognition model...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang=args.lang,
        det=False,   # recognition only — images are already cropped
        rec=True,
        cls=False,
        show_log=False,
    )
    print("Model loaded.")

    # ── Inference loop ──────────────────────────────────────────────────────
    results: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()

    for i, record in enumerate(records):
        sample_id = record["sample_id"]
        img_name = Path(record["image_path"]).name
        img_path = img_dir / img_name

        if not img_path.exists():
            errors.append({"sample_id": sample_id, "error": "image_not_found"})
            continue

        try:
            res = ocr.ocr(str(img_path), det=False, rec=True, cls=False)
            # res: [[text, confidence]] or [[[text, confidence]]]
            # Flatten PaddleOCR output structure
            if res and res[0]:
                item = res[0][0]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    text_pred, conf = item[0], float(item[1])
                elif isinstance(item, str):
                    text_pred, conf = item, 0.5  # fallback
                else:
                    text_pred, conf = str(item), 0.5
            else:
                text_pred, conf = "", 0.0

            conf = max(float(conf), MIN_CONFIDENCE)
            loss = -math.log(conf)

            results.append({
                "sample_id": sample_id,
                "loss": round(loss, 6),
                "confidence": round(conf, 6),
                "paddle_text": text_pred,
                "gt_text": record["gt_text"],
                "validation_cer": record["cer"],
                "validation_source": record.get("source", ""),
            })

        except Exception as e:
            errors.append({"sample_id": sample_id, "error": str(e)})

        if (i + 1) % args.batch == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(records) - i - 1) / rate
            print(f"  [{i+1}/{len(records)}] elapsed={elapsed:.1f}s rate={rate:.1f}/s eta={remaining:.0f}s")

    elapsed_total = time.time() - t0
    print(f"\nDone: {len(results)} success, {len(errors)} errors in {elapsed_total:.1f}s")

    # ── Save per-sample loss log ───────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Per-sample loss → {output_path}")

    if errors:
        err_path = output_path.parent / "risk01_errors.json"
        err_path.write_text(json.dumps(errors, indent=2))
        print(f"Errors ({len(errors)}) → {err_path}")

    # ── Distribution analysis ──────────────────────────────────────────────
    if results:
        dist = compute_distribution(results)
        dist_output_path.write_text(json.dumps(dist, indent=2, ensure_ascii=False))
        print(f"Distribution → {dist_output_path}")
        p = dist["loss_percentiles"]
        print(f"  p50={p['p50']} p90={p['p90']} p95={p['p95']}")
        print(f"  high_loss_rate={dist['high_loss_rate']} ({dist['high_loss_count']}/{dist['total']})")


if __name__ == "__main__":
    main()
