from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="One or more metrics json files")
    p.add_argument("--output", default="artifacts/method_comparison.json")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for in_path in [Path(x) for x in args.inputs]:
        if not in_path.exists():
            continue
        runs.append(json.loads(in_path.read_text()))

    # Placeholder comparator.
    # Expected behavior: compute detection accuracy, false crops, latency, skew deltas.
    out = {
        "inputs": [str(Path(x)) for x in args.inputs],
        "runs": runs,
        "notes": "comparison logic not implemented in this scaffold",
    }

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
