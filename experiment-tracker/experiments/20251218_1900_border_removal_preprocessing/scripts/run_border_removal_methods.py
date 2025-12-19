from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .border_remover import BorderRemover


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", default="outputs/border_removed")
    p.add_argument("--method", choices=["canny", "morph", "hough"], default="canny")
    p.add_argument("--metrics_out", default="artifacts/baseline_metrics.json")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    metrics_out = Path(args.metrics_out)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    remover = BorderRemover(method=args.method)

    images = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    results = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = remover.remove_border(img)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), res.image)
        results.append(
            {
                "path": str(img_path),
                "out": str(out_path),
                "metrics": {
                    "method": res.metrics.method,
                    "processing_time_ms": res.metrics.processing_time_ms,
                    "confidence": res.metrics.confidence,
                    "cropped_area_ratio": res.metrics.cropped_area_ratio,
                    "notes": res.metrics.notes,
                },
            }
        )

    payload = {
        "method": args.method,
        "count": len(results),
        "results": results,
        "notes": "border removal methods are scaffold placeholders",
    }
    metrics_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
