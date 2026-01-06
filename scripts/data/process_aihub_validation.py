import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_aihub_validation(
    root_dir: str = "data/raw/external/aihub_public_admin_doc/validation", output_path: str = "data/processed/aihub_validation.parquet"
):
    """
    Processes AI Hub Public Administration Documents (Validation) into a KIE-compatible Parquet file.
    """
    root_path = Path(root_dir)
    labels_dir = root_path / "labels"
    images_dir = root_path / "images"  # Assumes images are unpacked here

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    # Images directory might not exist yet if check is done before unpacking, but script should run after.

    # Recursively find all JSON files
    json_files = list(labels_dir.rglob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files.")

    data_records = []

    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, encoding="utf-8") as f:
                content = json.load(f)

            # AI Hub format usually has 'images' and 'annotations'
            # But the sample we saw:
            # {
            #   "images": [ { "id": 1, "width": 2480, "height": 3508, "file_name": "..." } ],
            #   "annotations": [ ... ]
            # }
            # Another variant found:
            # {
            #   "images": [ { "image.file.name": "...", "image.width": ... } ] (NO ID)
            #   "annotations": [ { "annotation.text": "...", ... } ] (NO image_id)
            # }

            # Map image_id to valid image info
            image_map = {}
            images_list = content.get("images", [])

            # If no ID in images, assign synthetic IDs
            # If plain list of images, usually 1 file per JSON in this dataset
            for idx, img in enumerate(images_list):
                # Try to get ID or use idx+1
                img_id = img.get("id", idx + 1)

                # Try to find file_name
                file_name = img.get("file_name") or img.get("image.file.name")
                if not file_name:
                    # Fallback or skip?
                    continue

                width = img.get("width") or img.get("image.width")
                height = img.get("height") or img.get("image.height")

                image_map[img_id] = {"file_name": file_name, "width": width, "height": height}

            # Group annotations by image_id
            anns_by_image = {}
            annotations_list = content.get("annotations", [])

            for ann in annotations_list:
                # Try to get image_id
                img_id = ann.get("image_id")

                # If missing, and we have only 1 image, assign to that image
                if img_id is None:
                    if len(image_map) == 1:
                        img_id = list(image_map.keys())[0]
                    else:
                        # Ambiguous if multiple images and no linkage
                        continue

                if img_id not in anns_by_image:
                    anns_by_image[img_id] = []
                anns_by_image[img_id].append(ann)

            # Create records
            for img_id, img_info in image_map.items():
                file_name = img_info["file_name"]
                # Search for the image file in the images directory (recursive or direct?)
                # Assuming flattening or preserving structure.
                # Let's assume relative path from images_dir matches structure or just filename.
                # Usually AI Hub separates Label and Source, but filenames are unique.
                # For now, store relative path as just the filename or try to resolve.

                # Check if we can find the file
                # If images are scattered in subfolders matching labels:
                # labels/Category/ID/Year/xxx.json -> images/Category/ID/Year/xxx.jpg
                # We can try to deduce the relative path from the json_file path

                # Calculate relative path from the labels root
                rel_path_from_labels = json_file.relative_to(labels_dir).parent

                # Adjust for directory structure difference
                # Labels: 01.라벨링데이터(Json)/Category/...
                # Images: 02.원천데이터(Jpg)/Category/...

                parts = list(rel_path_from_labels.parts)
                if parts and parts[0] == "01.라벨링데이터(Json)":
                    parts[0] = "02.원천데이터(Jpg)"

                rel_image_dir = Path(*parts)
                expected_image_path = images_dir / rel_image_dir / file_name

                # record relative path from the validation root or just keep absolute?
                # The script usually saves 'image_path' relative to some root or absolute.
                # KIEDataset checks if absolute, else joins with image_dir.
                # Let's verify if we want absolute or relative to `images_dir`.

                # If we save relative to `images_dir`, then KIEDataset(image_dir=...) works.
                record_image_path = str(rel_image_dir / file_name)

                if not expected_image_path.exists():
                    # Log warning but maybe it's just not unzipped yet?
                    pass

                # Collect texts and boxes
                texts = []
                polygons = []
                labels = []

                current_anns = anns_by_image.get(img_id, [])

                for ann in current_anns:
                    text = ann.get("annotation.text", "")
                    bbox = ann.get("annotation.bbox", [])  # [x, y, w, h]

                    if not bbox or len(bbox) != 4:
                        continue

                    x, y, w, h = bbox
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h

                    texts.append(text)
                    polygons.append([x1, y1, x2, y2])
                    labels.append("text")  # Default label

                if not texts:
                    continue

                record = {
                    "image_path": record_image_path,
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "texts": texts,
                    "polygons": polygons,
                    "labels": labels,
                    "kie_labels": labels,
                }
                data_records.append(record)

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")

    logger.info(f"Processed {len(data_records)} images.")

    # Create DataFrame
    df = pd.DataFrame(data_records)

    # Save to Parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    process_aihub_validation()
