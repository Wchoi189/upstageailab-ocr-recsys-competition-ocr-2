from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", default="artifacts/synthetic")
    p.add_argument("--manifest", default="artifacts/synthetic_manifest.json")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder generator.
    # Expected behavior: take clean images and render borders (colors, widths) into output_dir.
    images = sorted([p for p in input_dir.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

    manifest = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "generated": [],
        "notes": "synthetic border rendering not implemented in this scaffold",
    }
    for pth in images:
        manifest["generated"].append({"source": str(pth), "outputs": []})

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
