#!/usr/bin/env python3
"""DISC-01: Gate 4 synthetic-vs-real discriminator accuracy check.

Method: Confidence-distribution proxy (Option B from 20260221_1930_SESSION_HANDOVER.md).

Runs PaddleOCR rec-only on synthetic_v6 (1000 samples) and the first 1000
real candidates. Uses confidence score as the single discriminator feature.
Reports:
  - Per-group distribution stats
  - KS-test statistic + p-value
  - Optimal single-threshold classifier accuracy (best-case discriminator)
  - PASS / FAIL relative to <=70% accuracy criterion

Gate criterion: discriminator accuracy <= 70%
  PASS  → synthetic images indistinguishable from real by confidence proxy
  FAIL  → regenerate synthetic set

Output:
  data/audit/gate4_disc01_discriminator_results.json

Usage:
  cd /workspaces
  python dev_tools/experiment_manager/experiments/\\
20260217_154031_ocr_dq_exec_phase1_filtered/scripts/analysis/\\
disc01_discriminator_accuracy.py

Flags:
  --synthetic   path to synthetic JSONL (default: synthetic_v6.jsonl)
  --real        path to real candidates JSONL (default: candidates.jsonl)
  --n_real      number of real samples to use (default: 1000)
  --output      output JSON path (default: data/audit/gate4_disc01_discriminator_results.json)
  --lang        PaddleOCR language (default: korean)
  --batch       progress report interval (default: 50)
  --seed        random seed for real sample selection (default: 42)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT))

# ── Constants ───────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SYNTHETIC = PROJECT_ROOT / "data/generated/recognition/synthetic_v6.jsonl"
DEFAULT_REAL      = PROJECT_ROOT / "data/processed/recognition/candidates.jsonl"
DEFAULT_OUTPUT    = PROJECT_ROOT / "data/audit/gate4_disc01_discriminator_results.json"
DISC_THRESHOLD    = 0.70   # Gate criterion: accuracy must be <= this
MIN_CONFIDENCE    = 1e-10  # floor to avoid log(0)


# ── Utilities ───────────────────────────────────────────────────────────────

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def _dist_stats(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    var  = sum((x - mean) ** 2 for x in s) / n
    return {
        "n":    n,
        "mean": round(mean, 6),
        "std":  round(var ** 0.5, 6),
        "min":  round(s[0], 6),
        "p25":  round(_percentile(s, 25), 6),
        "p50":  round(_percentile(s, 50), 6),
        "p75":  round(_percentile(s, 75), 6),
        "p95":  round(_percentile(s, 95), 6),
        "max":  round(s[-1], 6),
    }


def _ks_test(a: list[float], b: list[float]) -> tuple[float, str]:
    """Two-sample KS statistic (simplified; no p-value computation)."""
    sa, sb = sorted(a), sorted(b)
    na, nb = len(sa), len(sb)
    all_vals = sorted(sa + sb)
    max_diff = 0.0
    ia = ib = 0
    for x in all_vals:
        while ia < na and sa[ia] <= x:
            ia += 1
        while ib < nb and sb[ib] <= x:
            ib += 1
        diff = abs(ia / na - ib / nb)
        if diff > max_diff:
            max_diff = diff
    # Approximate critical value for α=0.05: 1.36 * sqrt((na+nb)/(na*nb))
    critical = 1.36 * math.sqrt((na + nb) / (na * nb))
    significant = max_diff > critical
    return round(max_diff, 6), "significant" if significant else "not_significant"


def _optimal_threshold_accuracy(
    synth_confs: list[float],
    real_confs:  list[float],
) -> tuple[float, float, str]:
    """Binary classifier using confidence as single feature.

    For each candidate threshold, classify:
      conf >= threshold → real (label=1)
      conf <  threshold → synthetic (label=0)

    Returns (best_accuracy, best_threshold, direction).
    """
    labels_synth = [(c, 0) for c in synth_confs]
    labels_real  = [(c, 1) for c in real_confs]
    all_data = sorted(labels_synth + labels_real, key=lambda x: x[0])
    n_total  = len(all_data)

    best_acc  = 0.0
    best_thr  = 0.0
    best_dir  = ">=thr→real"

    # Sweep thresholds at each unique confidence value
    unique_thrs = sorted(set(c for c, _ in all_data))
    for thr in unique_thrs:
        # Direction A: conf >= thr → predict real (1)
        correct_a = sum(1 for c, lbl in all_data if (c >= thr) == (lbl == 1))
        # Direction B: conf < thr → predict real (1)
        correct_b = sum(1 for c, lbl in all_data if (c < thr)  == (lbl == 1))
        acc_a = correct_a / n_total
        acc_b = correct_b / n_total
        if acc_a > best_acc:
            best_acc, best_thr, best_dir = acc_a, thr, ">=thr→real"
        if acc_b > best_acc:
            best_acc, best_thr, best_dir = acc_b, thr, "<thr→real"

    return round(best_acc, 6), round(best_thr, 6), best_dir


# ── PaddleOCR inference ─────────────────────────────────────────────────────

def _run_ocr(
    records: list[dict],
    ocr,
    label: str,
    batch_size: int,
) -> tuple[list[float], list[dict]]:
    """Run PaddleOCR rec-only on records. Returns (confidences, errors)."""
    confs:  list[float] = []
    errors: list[dict]  = []
    t0 = time.time()

    for i, rec in enumerate(records):
        img_path = Path(rec["image_path"])
        # Remap /mnt/external_artifacts to project root if needed
        if not img_path.exists():
            img_name  = img_path.name
            img_parent = img_path.parent.name
            # Try common remaps
            remaps = [
                PROJECT_ROOT / "data/generated/recognition" / img_parent / img_name,
                PROJECT_ROOT / "data/processed/recognition" / img_parent / img_name,
            ]
            found = next((p for p in remaps if p.exists()), None)
            if found:
                img_path = found
            else:
                errors.append({"sample_id": rec.get("sample_id"), "error": "image_not_found", "path": str(img_path)})
                continue

        try:
            res = ocr.ocr(str(img_path), det=False, rec=True, cls=False)
            if res and res[0]:
                item = res[0][0]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    conf = float(item[1])
                elif isinstance(item, str):
                    conf = 0.5  # fallback — text but no confidence
                else:
                    conf = 0.5
            else:
                conf = 0.0
            confs.append(max(conf, MIN_CONFIDENCE))
        except Exception as exc:  # noqa: BLE001
            errors.append({"sample_id": rec.get("sample_id"), "error": str(exc)})
            continue

        if (i + 1) % batch_size == 0:
            elapsed = time.time() - t0
            print(f"  [{label}] {i+1}/{len(records)} done  ({elapsed:.1f}s)")

    print(f"  [{label}] COMPLETE — {len(confs)} ok, {len(errors)} errors  ({time.time()-t0:.1f}s)")
    return confs, errors


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", default=str(DEFAULT_SYNTHETIC))
    ap.add_argument("--real",      default=str(DEFAULT_REAL))
    ap.add_argument("--n_real",    type=int, default=1000)
    ap.add_argument("--output",    default=str(DEFAULT_OUTPUT))
    ap.add_argument("--lang",      default="korean")
    ap.add_argument("--batch",     type=int, default=50)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    synth_path  = Path(args.synthetic)
    real_path   = Path(args.real)
    output_path = Path(args.output)

    # ── Load datasets ──────────────────────────────────────────────────────
    if not synth_path.exists():
        print(f"ERROR: synthetic not found: {synth_path}", file=sys.stderr)
        sys.exit(1)
    if not real_path.exists():
        print(f"ERROR: real not found: {real_path}", file=sys.stderr)
        sys.exit(1)

    synth_records = [json.loads(l) for l in synth_path.open()]
    real_all      = [json.loads(l) for l in real_path.open()]

    rng = random.Random(args.seed)
    real_records = rng.sample(real_all, min(args.n_real, len(real_all)))

    print(f"Synthetic samples : {len(synth_records)}")
    print(f"Real samples      : {len(real_records)} (seed={args.seed})")
    print(f"Total to process  : {len(synth_records) + len(real_records)}")

    # ── Init PaddleOCR ─────────────────────────────────────────────────────
    print("Loading PaddleOCR recognition model...")
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang=args.lang,
        det=False,
        rec=True,
        cls=False,
        show_log=False,
    )
    print("Model loaded.")

    # ── Inference ──────────────────────────────────────────────────────────
    print("\n[Synthetic inference]")
    synth_confs, synth_errors = _run_ocr(synth_records, ocr, "synthetic", args.batch)

    print("\n[Real inference]")
    real_confs, real_errors = _run_ocr(real_records, ocr, "real", args.batch)

    # ── Analysis ───────────────────────────────────────────────────────────
    synth_stats = _dist_stats(synth_confs)
    real_stats  = _dist_stats(real_confs)
    mean_diff   = abs(synth_stats["mean"] - real_stats["mean"])

    ks_stat, ks_significance = _ks_test(synth_confs, real_confs)

    opt_acc, opt_thr, opt_dir = _optimal_threshold_accuracy(synth_confs, real_confs)

    gate_pass = opt_acc <= DISC_THRESHOLD

    print(f"\n{'='*60}")
    print(f"DISC-01 RESULTS")
    print(f"{'='*60}")
    print(f"Synthetic conf mean : {synth_stats['mean']:.4f}")
    print(f"Real conf mean      : {real_stats['mean']:.4f}")
    print(f"Mean gap            : {mean_diff:.4f}")
    print(f"KS statistic        : {ks_stat:.4f}  ({ks_significance})")
    print(f"Optimal threshold   : {opt_thr:.4f}  (rule: {opt_dir})")
    print(f"Best-case accuracy  : {opt_acc:.4f}  (threshold <= {DISC_THRESHOLD})")
    print(f"Gate PASS           : {gate_pass}")
    print(f"{'='*60}\n")

    # ── Audit record ───────────────────────────────────────────────────────
    audit = {
        "gate":              "gate4",
        "check":             "disc01_discriminator_accuracy",
        "method":            "confidence_distribution_proxy",
        "note":              (
            "Single-feature binary classifier using PaddleOCR confidence score. "
            "Optimal threshold = best-case discriminator. "
            "If accuracy <= 0.70, synthetic images are indistinguishable by this proxy."
        ),
        "synthetic_corpus":  str(synth_path),
        "real_corpus":       str(real_path),
        "n_synthetic":       len(synth_confs),
        "n_real":            len(real_confs),
        "n_synthetic_errors": len(synth_errors),
        "n_real_errors":      len(real_errors),
        "synthetic_conf_dist": synth_stats,
        "real_conf_dist":      real_stats,
        "confidence_mean_gap": round(mean_diff, 6),
        "ks_statistic":        ks_stat,
        "ks_significance":     ks_significance,
        "optimal_threshold_accuracy": opt_acc,
        "optimal_threshold":   opt_thr,
        "optimal_direction":   opt_dir,
        "disc_threshold":      DISC_THRESHOLD,
        "gate_pass":           gate_pass,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Audit written → {output_path}")

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
