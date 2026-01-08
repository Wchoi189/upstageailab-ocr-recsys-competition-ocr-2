import sys
from pathlib import Path

sys.path.append(str(Path.cwd() / "ocr-etl-pipeline/src"))

from etl.core import process_single_json, infer_image_path
import logging

logging.basicConfig(level=logging.DEBUG)

json_path = (
    "data/raw/external/aihub_public_admin_doc/validation/labels/01.라벨링데이터(Json)/인.허가/5350093/2001/5350093-2001-0001-0009.json"
)
root = "data/raw/external/aihub_public_admin_doc/validation"

print(f"Testing JSON: {json_path}")
print(f"Root: {root}")

p_json = Path(json_path)
p_root = Path(root)

inferred = infer_image_path(p_json, p_root)
print(f"Inferred Image Path: {inferred}")

if inferred:
    print(f"Exists? {inferred.exists()}")

results = process_single_json(str(p_json), str(p_root))
print(f"Results count: {len(results)}")
if results:
    print(f"Sample 0 label: {results[0][1]}")
