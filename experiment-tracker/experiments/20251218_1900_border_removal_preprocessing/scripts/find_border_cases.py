from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output", default="artifacts/border_cases_manifest.json")
    p.add_argument("--skew_gate_deg", type=float, default=20.0)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # This is a placeholder collector.
    # Expected behavior: compute skew estimate per image; select abs(skew) > skew_gate_deg.
    images = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    payload = {
        "input_dir": str(input_dir),
        "skew_gate_deg": float(args.skew_gate_deg),
        "candidates": [{"path": str(p)} for p in images],
        "notes": "skew estimation not implemented in this scaffold",
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
