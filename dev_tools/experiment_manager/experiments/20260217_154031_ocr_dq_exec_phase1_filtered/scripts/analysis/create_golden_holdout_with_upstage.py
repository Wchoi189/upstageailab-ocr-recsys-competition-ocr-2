#!/usr/bin/env python3
"""T036 [US4]: Golden holdout builder with tiered Upstage validation.

Constructs a versioned clean holdout set from a candidate pool manifest (JSONL)
using the tiered validation pipeline (golden_set_validator.py).

Pipeline:
  1. Load candidate pool from JSONL manifest
  2. Stratified sampling across text length buckets and script classes
  3. Route samples through TieredGoldenValidator (Tier 1–4 + manual fallback):
       Tier 1:   model confidence triage
       Tier 2:   PaddleOCR local inference
       Tier 2.5: Ollama VLM (olmocr2:7b-q8) — local, zero API cost
       Tier 3:   Upstage OCR API (dual-key pool: UPSTAGE_API_KEY 3rps + UPSTAGE_API_KEY2 1rps = 4rps)
  4. Emit clean holdout (auto_accept) and review queue (manual_review)
  5. Validate Gate 4.5: Upstage call ratio must be within 20%–40% of pool

Output artifacts:
  data/processed/recognition/holdout_clean_v{N}.jsonl
  data/processed/recognition/holdout_review_queue_v{N}.jsonl
  data/audit/holdout_construction_summary_v{N}.json

Usage:
  uv run python scripts/analysis/create_golden_holdout_with_upstage.py \\
    --candidates data/processed/recognition/candidates.jsonl \\
    --version 1 \\
    --sample_size 500 \\
    [--upstage_budget_ratio 0.40] \\
    [--tier1_threshold 0.95] \\
    [--seed 42] \\
    [--no_paddle] \\
    [--no_ollama] \\
    [--dry_run]

Environment variables:
  UPSTAGE_API_KEY  — Tier-3 primary key (3 rps)
  UPSTAGE_API_KEY2 — Tier-3 secondary key (1 rps); optional, extends pool to 4 rps
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path

try:
    from ocr.core.utils.path_utils import PROJECT_ROOT
    _project_root = PROJECT_ROOT
except ImportError:
    _project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(_project_root))

from scripts.data.quality.golden_set_validator import (
    CandidateSample,
    TieredGoldenValidator,
)
from scripts.data.quality.ollama_validator import OllamaOCRClient
from scripts.data.quality.paddle_validator import PaddleOCRValidator
from scripts.data.quality.upstage_validator import UpstageKeyPool

_GATE_45_MIN_RATIO = 0.20
_GATE_45_MAX_RATIO = 0.40


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

_LEN_BUCKETS = [(0, 5), (6, 10), (11, 20), (21, 999)]


def _len_bucket(text: str) -> str:
    n = len(text)
    for lo, hi in _LEN_BUCKETS:
        if lo <= n <= hi:
            return f"len_{lo}_{hi}"
    return "len_other"


def _script_class(text: str) -> str:
    """Classify text as 'korean', 'mixed', or 'ascii'."""
    has_korean = any("\uAC00" <= c <= "\uD7A3" for c in text)
    has_ascii = any(c.isascii() and c.isalnum() for c in text)
    if has_korean and has_ascii:
        return "mixed"
    if has_korean:
        return "korean"
    return "ascii"


def _stratified_sample(
    pool: list[dict],
    target_size: int,
    seed: int,
) -> list[dict]:
    """Stratified sample from pool by (len_bucket, script_class).

    Falls back to random sample if pool < target_size.
    """
    rng = random.Random(seed)

    # Group by stratum
    strata: dict[str, list[dict]] = {}
    for item in pool:
        gt = item.get("gt_text", "")
        key = f"{_len_bucket(gt)}|{_script_class(gt)}"
        strata.setdefault(key, []).append(item)

    n_strata = len(strata)
    per_stratum = max(1, target_size // n_strata) if n_strata else target_size

    selected: list[dict] = []
    for items in strata.values():
        rng.shuffle(items)
        selected.extend(items[:per_stratum])

    # Top-up or trim to target_size
    if len(selected) < target_size:
        remainder = [x for x in pool if x not in selected]
        rng.shuffle(remainder)
        selected.extend(remainder[: target_size - len(selected)])
    elif len(selected) > target_size:
        rng.shuffle(selected)
        selected = selected[:target_size]

    return selected


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _versioned_path(template: str, version: int, root: Path) -> Path:
    return root / template.format(N=version)


# ---------------------------------------------------------------------------
# Gate 4.5 check
# ---------------------------------------------------------------------------


def _check_gate_45(upstage_call_ratio: float) -> dict:
    """Return Gate 4.5 decision dict."""
    within_range = _GATE_45_MIN_RATIO <= upstage_call_ratio <= _GATE_45_MAX_RATIO
    return {
        "gate_id": "Gate 4.5",
        "upstage_call_ratio": round(upstage_call_ratio, 4),
        "threshold_min": _GATE_45_MIN_RATIO,
        "threshold_max": _GATE_45_MAX_RATIO,
        "decision": "pass" if within_range else "hold",
        "violations": (
            []
            if within_range
            else [
                f"upstage_call_ratio={upstage_call_ratio:.4f} outside [{_GATE_45_MIN_RATIO}, {_GATE_45_MAX_RATIO}]"
            ]
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build golden holdout via tiered Upstage validation")
    p.add_argument("--candidates", required=True, help="JSONL candidate pool manifest")
    p.add_argument("--version", type=int, default=1, help="Holdout version number (vN)")
    p.add_argument("--sample_size", type=int, default=500, help="Target stratified sample size")
    p.add_argument(
        "--upstage_budget_ratio",
        type=float,
        default=_GATE_45_MAX_RATIO,
        help=f"Max Upstage call ratio (Gate 4.5 max: {_GATE_45_MAX_RATIO})",
    )
    p.add_argument("--tier1_threshold", type=float, default=0.95, help="Tier-1 auto-accept threshold")
    p.add_argument("--seed", type=int, default=42, help="Random seed for stratified sampling")
    p.add_argument(
        "--no_paddle",
        action="store_true",
        help="Skip Tier-2 PaddleOCR (use when not installed)",
    )
    p.add_argument(
        "--no_ollama",
        action="store_true",
        help="Skip Tier-2.5 Ollama VLM (use when Ollama is unavailable or VRAM is constrained)",
    )
    p.add_argument("--dry_run", action="store_true", help="Skip validation API calls; emit empty outputs")
    p.add_argument("--output_root", default=".", help="Repository root for output paths")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"[ERROR] Candidate pool not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    # Output paths
    clean_out = _versioned_path(
        "data/processed/recognition/holdout_clean_v{N}.jsonl", args.version, root
    )
    review_out = _versioned_path(
        "data/processed/recognition/holdout_review_queue_v{N}.jsonl", args.version, root
    )
    summary_out = _versioned_path(
        "data/audit/holdout_construction_summary_v{N}.json", args.version, root
    )

    print(f"[T036] Loading candidates from {candidates_path}")
    pool = _load_jsonl(candidates_path)
    print(f"[T036] Pool size: {len(pool)}")

    # Stratified sample
    sampled = _stratified_sample(pool, args.sample_size, args.seed)
    print(f"[T036] Sampled {len(sampled)} candidates (stratified, seed={args.seed})")

    if args.dry_run:
        print("[T036] DRY_RUN: skipping validation calls")
        _write_jsonl([], clean_out)
        _write_jsonl([], review_out)
        summary = {
            "version": args.version,
            "dry_run": True,
            "pool_size": len(pool),
            "sampled": len(sampled),
            "stats": {},
            "gate_45": {},
        }
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"[T036] DRY_RUN artifacts written to {clean_out}, {review_out}, {summary_out}")
        return

    # --- Tier-2: PaddleOCR ---
    paddle = None if args.no_paddle else PaddleOCRValidator()

    # --- Tier-2.5: Ollama VLM ---
    ollama: OllamaOCRClient | None = None
    if not args.no_ollama:
        client = OllamaOCRClient()
        if client.is_available():
            ollama = client
            print("[T036] Tier-2.5: Ollama olmocr2:7b-q8 available — enabled")
        else:
            print("[T036] WARN: Ollama unavailable — Tier-2.5 disabled")

    # --- Tier-3: Upstage dual-key pool ---
    upstage_key = os.environ.get("UPSTAGE_API_KEY")
    upstage: UpstageKeyPool | None = None
    if upstage_key:
        upstage = UpstageKeyPool()  # reads UPSTAGE_API_KEY + UPSTAGE_API_KEY2 from env
        key_count = upstage.key_count
        print(f"[T036] Tier-3: UpstageKeyPool — {key_count} key(s) active")
    else:
        print("[T036] WARN: UPSTAGE_API_KEY not set — Tier-3 disabled; all ambiguous → manual_review")

    validator = TieredGoldenValidator(
        paddle_validator=paddle,
        ollama_client=ollama,
        upstage_client=upstage,
        tier1_accept_threshold=args.tier1_threshold,
        upstage_budget_ratio=args.upstage_budget_ratio,
    )

    # Convert sampled records to CandidateSample
    samples = [
        CandidateSample(
            sample_id=str(r.get("sample_id", r.get("idx", i))),
            image_path=r.get("image_path", ""),
            gt_text=r.get("gt_text", r.get("label", "")),
            model_confidence=r.get("model_confidence"),
        )
        for i, r in enumerate(sampled)
    ]

    print("[T036] Running tiered validation...")
    results, stats = validator.validate_batch(samples)
    print(f"[T036] Validation complete: {stats.as_dict()}")

    # Split outputs
    clean_records = [r for r in results if r.disposition in ("auto_accept", "auto_correct")]
    review_records = [r for r in results if r.disposition == "manual_review"]

    def _result_to_dict(r) -> dict:
        return {
            "sample_id": r.sample_id,
            "image_path": r.image_path,
            "gt_text": r.gt_text,
            "tier_used": r.tier_used,
            "ocr_text": r.ocr_text,
            "confidence": r.confidence,
            "cer": r.cer,
            "gt_match": r.gt_match,
            "disposition": r.disposition,
            "source": r.source,
            "review_status": "pending",
        }

    _write_jsonl([_result_to_dict(r) for r in clean_records], clean_out)
    _write_jsonl([_result_to_dict(r) for r in review_records], review_out)

    gate_45 = _check_gate_45(stats.upstage_call_ratio)
    summary = {
        "version": args.version,
        "seed": args.seed,
        "pool_size": len(pool),
        "sampled": len(sampled),
        "clean_count": len(clean_records),
        "review_queue_count": len(review_records),
        "stats": stats.as_dict(),
        "gate_45": gate_45,
        "outputs": {
            "clean": str(clean_out),
            "review_queue": str(review_out),
            "summary": str(summary_out),
        },
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"[T036] Clean holdout ({len(clean_records)}): {clean_out}")
    print(f"[T036] Review queue ({len(review_records)}): {review_out}")
    print(f"[T036] Summary: {summary_out}")
    print(f"[T036] Gate 4.5: {gate_45['decision'].upper()} (upstage_call_ratio={gate_45['upstage_call_ratio']})")

    if gate_45["decision"] != "pass":
        print(f"[T036] Gate 4.5 HOLD: {gate_45['violations']}", file=sys.stderr)


if __name__ == "__main__":
    main()
