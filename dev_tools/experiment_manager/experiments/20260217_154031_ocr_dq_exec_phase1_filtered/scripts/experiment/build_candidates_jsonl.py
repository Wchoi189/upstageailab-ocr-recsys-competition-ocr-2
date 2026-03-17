#!/usr/bin/env python3
"""Gate 2 prep: Build candidates JSONL from high_loss_samples.json + LMDB images.

Reads high-loss sample indices from the T015 export, extracts corresponding
JPEG images from the LMDB, writes them to a staging directory, and emits
a candidates JSONL manifest for use with create_golden_holdout_with_upstage.py.

Candidates JSONL schema per record:
  {
    "sample_id": str,     # LMDB 1-based idx as string
    "gt_text": str,       # ground-truth label
    "image_path": str,    # absolute path to extracted JPEG
    "loss_proxy": float   # from T015 export (informational)
  }

Usage:
  uv run python <exp>/scripts/experiment/build_candidates_jsonl.py \\
    [--high_loss_json data/audit/high_loss_samples.json] \\
    [--lmdb_path data/processed/recognition/aihub_lmdb_validation] \\
    [--image_dir data/processed/recognition/candidates_images] \\
    [--output data/processed/recognition/candidates.jsonl] \\
    [--pool_size 2000] \\
    [--sample_mode top]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from ocr.core.utils.path_utils import PROJECT_ROOT
    _root = PROJECT_ROOT
except ImportError:
    _root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(_root))

import lmdb


def _extract_images(
    lmdb_path: Path,
    indices: list[int],
    image_dir: Path,
    verbose: bool = True,
) -> dict[int, Path]:
    """Extract JPEG images for given indices from LMDB.

    Returns:
        mapping of idx → absolute image path (only for successfully extracted)
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)

    extracted: dict[int, Path] = {}
    missing = 0

    with env.begin(write=False) as txn:
        for i, idx in enumerate(indices):
            key = f"image-{idx:09d}".encode()
            img_bytes = txn.get(key)
            if img_bytes is None:
                missing += 1
                continue
            out_path = image_dir / f"{idx:09d}.jpg"
            out_path.write_bytes(img_bytes)
            extracted[idx] = out_path
            if verbose and (i + 1) % 200 == 0:
                print(f"  extracted {i + 1}/{len(indices)} images...", flush=True)

    if verbose:
        print(f"  done: {len(extracted)} extracted, {missing} missing in LMDB")
    return extracted


def _select_pool(samples: list[dict], pool_size: int, mode: str) -> list[dict]:
    """Select pool_size samples from the high-loss list.

    mode='top': take first pool_size (highest loss_proxy, already sorted desc)
    mode='uniform': sample uniformly across the sorted list for diversity
    """
    if len(samples) <= pool_size:
        return samples

    if mode == "top":
        return samples[:pool_size]

    if mode == "uniform":
        step = len(samples) / pool_size
        return [samples[int(i * step)] for i in range(pool_size)]

    raise ValueError(f"Unknown sample_mode: {mode!r}. Use 'top' or 'uniform'.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--high_loss_json",
        default="data/audit/high_loss_samples.json",
        help="T015 export (high_loss_samples.json)",
    )
    ap.add_argument(
        "--lmdb_path",
        default="data/processed/recognition/aihub_lmdb_validation",
        help="LMDB directory",
    )
    ap.add_argument(
        "--image_dir",
        default="data/processed/recognition/candidates_images",
        help="Output directory for extracted JPEGs",
    )
    ap.add_argument(
        "--output",
        default="data/processed/recognition/candidates.jsonl",
        help="Output candidates JSONL path",
    )
    ap.add_argument(
        "--pool_size",
        type=int,
        default=2000,
        help="Max candidates in JSONL (default: 2000)",
    )
    ap.add_argument(
        "--sample_mode",
        choices=["top", "uniform"],
        default="uniform",
        help="'top' = highest loss_proxy first; 'uniform' = spread across list (default)",
    )
    args = ap.parse_args()

    high_loss_path = Path(args.high_loss_json)
    lmdb_path = Path(args.lmdb_path)
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    if not high_loss_path.exists():
        print(f"ERROR: {high_loss_path} not found", file=sys.stderr)
        sys.exit(1)
    if not lmdb_path.exists():
        print(f"ERROR: LMDB not found: {lmdb_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[build_candidates] Loading {high_loss_path}")
    data = json.loads(high_loss_path.read_text())
    all_samples = data["samples"]
    print(f"[build_candidates] Total high-loss samples: {len(all_samples)}")
    print(f"[build_candidates] threshold_p95={data['threshold_p95']}  tokenizer_max_len={data['tokenizer_max_len']}")

    selected = _select_pool(all_samples, args.pool_size, args.sample_mode)
    print(f"[build_candidates] Pool selected: {len(selected)} (mode={args.sample_mode})")

    indices = [s["idx"] for s in selected]
    label_map = {s["idx"]: s for s in selected}

    print(f"[build_candidates] Extracting images from LMDB → {image_dir}")
    extracted = _extract_images(lmdb_path, indices, image_dir, verbose=True)

    # Build JSONL — only include samples with successfully extracted images
    records = []
    skipped = 0
    for idx in indices:
        if idx not in extracted:
            skipped += 1
            continue
        s = label_map[idx]
        records.append({
            "sample_id": str(idx),
            "gt_text": s["label"],
            "image_path": str(extracted[idx].resolve()),
            "loss_proxy": s["loss_proxy"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[build_candidates] Written {len(records)} candidates → {output_path}")
    if skipped:
        print(f"[build_candidates] WARN: {skipped} skipped (no image in LMDB)")

    # Summary
    label_lens = [len(r["gt_text"]) for r in records]
    print(f"[build_candidates] label_len stats: min={min(label_lens)} max={max(label_lens)} "
          f"mean={sum(label_lens)/len(label_lens):.1f}")


if __name__ == "__main__":
    main()
