#!/usr/bin/env python3
"""T016 [US1]: Defect taxonomy labeling utility.

Streams the LMDB and assigns canonical defect taxonomy labels to every sample.
Produces a structured summary keyed by defect class with sample indices and
representative examples.

Output: data/audit/defect_taxonomy.json
  {
    "total": int,
    "classes": {
      "<defect_class>": {
        "count": int,
        "rate": float,
        "severity": str,
        "examples": [{"idx": int, "label": str, "flags": dict}, ...]
      }
    }
  }

Usage:
  uv run python scripts/analysis/label_defect_classes.py \\
    --lmdb_path data/processed/recognition/aihub_lmdb_validation \\
    --output data/audit/defect_taxonomy.json \\
    [--max_examples 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    from ocr.core.utils.path_utils import PROJECT_ROOT
    _project_root = PROJECT_ROOT
except ImportError:
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_project_root))

from scripts.data.quality.manifest_io import iter_lmdb_samples
from scripts.data.quality.defect_rules import evaluate_label

# Canonical severity per defect class (from defect_rules._aggregate_severity)
_CLASS_SEVERITY = {
    "unreadable_sample": "critical",
    "truncation_misalignment": "high",
    "script_mismatch": "high",
    "hallucinated_gt_chars": "medium",
    "missing_char_due_to_clipping": "low",
}


def label_defects(
    lmdb_path: Path,
    tokenizer_max_len: int = 25,
    max_examples: int = 10,
) -> dict:
    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list[dict]] = defaultdict(list)
    total = 0

    for s in iter_lmdb_samples(lmdb_path, skip_images=True):
        total += 1
        result = evaluate_label(
            sample_id=str(s.idx),
            label=s.label,
            tokenizer_max_len=tokenizer_max_len,
        )
        for cls in result.defect_classes:
            counts[cls] += 1
            if len(examples[cls]) < max_examples:
                examples[cls].append({
                    "idx": s.idx,
                    "label": s.label,
                    "flags": result.flags,
                })

    classes = {}
    for cls in _CLASS_SEVERITY:
        count = counts.get(cls, 0)
        classes[cls] = {
            "count": count,
            "rate": round(count / total, 6) if total else 0.0,
            "severity": _CLASS_SEVERITY[cls],
            "examples": examples.get(cls, []),
        }

    return {
        "lmdb_path": str(lmdb_path),
        "total": total,
        "tokenizer_max_len": tokenizer_max_len,
        "classes": classes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lmdb_path", default="data/processed/recognition/aihub_lmdb_validation")
    ap.add_argument("--output", default="data/audit/defect_taxonomy.json")
    ap.add_argument("--tokenizer_max_len", type=int, default=25)
    ap.add_argument("--max_examples", type=int, default=10)
    args = ap.parse_args()

    lmdb_path = Path(args.lmdb_path)
    if not lmdb_path.exists():
        print(f"ERROR: LMDB not found: {lmdb_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Labeling defect taxonomy: {lmdb_path}")
    result = label_defects(
        lmdb_path=lmdb_path,
        tokenizer_max_len=args.tokenizer_max_len,
        max_examples=args.max_examples,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Written: {out}")
    for cls, data in result["classes"].items():
        if data["count"] > 0:
            print(f"  {cls}: {data['count']} ({data['rate']*100:.4f}%) [{data['severity']}]")


if __name__ == "__main__":
    main()
