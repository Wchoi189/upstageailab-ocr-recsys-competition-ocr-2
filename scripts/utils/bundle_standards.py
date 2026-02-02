import json
import os
from pathlib import Path

def bundle_standards():
    standards_dir = Path("archive/legacy_standards_dump")
    output_file = Path("AgentQMS/standards_bundle.json")

    bundle = {}

    for root, _, files in os.walk(standards_dir):
        for file in files:
            file_path = Path(root) / file
            if file.endswith((".yaml", ".yml", ".md")):
                key = str(file_path.relative_to(standards_dir))
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        bundle[key] = f.read()
                except Exception as e:
                    print(f"Skipping {key}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    print(f"Bundled {len(bundle)} files to {output_file}")

if __name__ == "__main__":
    bundle_standards()
