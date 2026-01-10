import json
import lmdb
import cv2
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CropData:
    """container for single crop data"""

    def __init__(self, image_bytes: bytes, label: str, metadata: dict):
        self.image_bytes = image_bytes
        self.label = label
        self.metadata = metadata


def infer_image_path(json_path: Path, input_root: Path) -> Path | None:
    """
    Infers the image path from the JSON path assuming AI Hub structure.
    JSON: .../labels/01.라벨링데이터(Json)/...
    Image: .../images/02.원천데이터(Jpg)/...
    """
    try:
        parts = json_path.parts
        # Naive string replacement based on observed structure
        # We look for the "labels" part and try to switch to "images" and valid subdirs

        # Construct relative path from the input root to use for reconstruction
        # but the subdirectories change names.
        # Strategy:
        # 1. Get the relative path of the json file from the 'labels' directory or '01.라벨링데이터(Json)'
        # 2. Try to find the corresponding 'images' directory equivalent.

        # Based on user's listing:
        # labels/01.라벨링데이터(Json)/...
        # images/02.원천데이터(Jpg)/...

        # So we replace:
        # "labels" -> "images"
        # "01.라벨링데이터(Json)" -> "02.원천데이터(Jpg)"
        # Suffix ".json" -> ".jpg" or ".JPG"

        new_parts = list(parts)

        # Find index of key directories
        try:
            lbl_idx = -1
            for i, p in enumerate(new_parts):
                if "01.라벨링데이터(Json)" in p:
                    lbl_idx = i
                    break

            if lbl_idx != -1:
                new_parts[lbl_idx] = "02.원천데이터(Jpg)"
                # Also check parent if 'labels' exists
                if lbl_idx > 0 and new_parts[lbl_idx - 1] == "labels":
                    new_parts[lbl_idx - 1] = "images"
            else:
                # Fallback: strictly replace 'labels' with 'images' if 01... not found
                # This might happen if user simplified path
                new_parts = [p.replace("labels", "images") for p in new_parts]

        except Exception:
            pass  # fallback to simple replacement

        # Reconstruct
        potential_path = Path(*new_parts).with_suffix(".jpg")
        if potential_path.exists():
            return potential_path

        potential_path_upper = Path(*new_parts).with_suffix(".JPG")
        if potential_path_upper.exists():
            return potential_path_upper

        return None

    except Exception as e:
        logger.debug(f"Path inference failed for {json_path}: {e}")
        return None


def process_single_json(json_path: str, input_root_str: str) -> list[tuple[bytes, str, dict]]:
    """
    Worker function to process a single JSON file.
    Returns a list of (jpeg_bytes, label, metadata).
    """
    results = []
    json_path_obj = Path(json_path)
    input_root_obj = Path(input_root_str)

    img_path = infer_image_path(json_path_obj, input_root_obj)

    if not img_path or not img_path.exists():
        return []

    try:
        # Load JSON (using standard json as orjson might not be strictly necessary here and strict dependency management in workers is easier)
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # Load Image
        # cv2.imread doesn't handle non-ascii paths well on some systems, use numpy
        img_array = np.fromfile(str(img_path), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return []

        h, w, _ = img.shape

        # Iterate annotations
        # Adjust based on the schema seen: data['annotations'] is a list
        annotations = data.get("annotations", [])

        for ann in annotations:
            # bbox: [x, y, w, h] as seen in user's file
            bbox = ann.get("annotation.bbox")
            text = ann.get("annotation.text")  # Key from user file: "annotation.text"

            if not bbox or not text:
                continue

            x, y, bw, bh = map(int, bbox)

            # Sanity check
            if bw <= 0 or bh <= 0 or x < 0 or y < 0:
                continue

            # Clamp to image
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            bw = min(bw, w - x)
            bh = min(bh, h - y)

            if bw <= 4 or bh <= 4:  # Too small
                continue

            crop = img[y : y + bh, x : x + bw]

            # Encode
            success, encoded_img = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                continue

            results.append((encoded_img.tobytes(), text, {"src": str(json_path_obj.name), "id": ann.get("id")}))

    except Exception as e:
        # Fail silently for individual file errors to keep pipeline moving
        logger.debug(f"Error processing {json_path}: {e}")
        return []

    return results


class LMDBConverter:
    def __init__(self, input_dir: str, output_dir: str, num_workers: int = 4, batch_size: int = 1000, limit: int = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.limit = limit
        self.state_file = self.output_dir / "state.json"

        self.processed_files = set()
        self.current_index = 1  # 1-based index for LMDB image-%09d

        # Initialize output
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                    self.processed_files = set(state.get("processed_files", []))
                    self.current_index = state.get("current_index", 1)
                logger.info(f"Resumed state: {len(self.processed_files)} files processed, Next Index: {self.current_index}")
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")

    def save_state(self):
        temp_file = self.state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump({"processed_files": list(self.processed_files), "current_index": self.current_index}, f)
        temp_file.replace(self.state_file)  # Atomic move

    def run(self):
        self.load_state()

        # Discovery
        logger.info("Scanning for JSON files...")
        all_files = sorted([str(p) for p in self.input_dir.rglob("*.json")])

        # Filter
        to_process = [f for f in all_files if f not in self.processed_files]
        if self.limit:
            to_process = to_process[: self.limit]

        logger.info(f"Found {len(all_files)} total files. {len(to_process)} to process.")

        if not to_process:
            logger.info("Nothing to process.")
            return

        # LMDB Env
        # Map size 1TB - LMDB doesn't allocate this, it's just virtual address space limit
        env = lmdb.open(str(self.output_dir), map_size=1099511627776)

        buffer_ops = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Chunking submissions to avoid memory overflow for huge file lists
            chunk_size = self.batch_size * 2

            for i in range(0, len(to_process), chunk_size):
                chunk_files = to_process[i : i + chunk_size]
                futures = {executor.submit(process_single_json, f, str(self.input_dir)): f for f in chunk_files}

                for future in as_completed(futures):
                    fname = futures[future]
                    try:
                        crops = future.result()
                        for img_bytes, label, meta in crops:
                            buffer_ops.append((img_bytes, label))

                        # Mark as processed regardless of whether it yielded crops (maybe it was empty)
                        self.processed_files.add(fname)

                        # Write batch if ready
                        if len(buffer_ops) >= self.batch_size:
                            self._write_batch(env, buffer_ops)
                            buffer_ops = []
                            self.save_state()
                            logger.info(
                                f"Processed {len(self.processed_files)}/{len(to_process)} files. Current samples: {self.current_index-1}"
                            )

                    except Exception as e:
                        logger.error(f"Worker failed for {fname}: {e}")

            # Flush remaining
            if buffer_ops:
                self._write_batch(env, buffer_ops)
                self.save_state()

        # Finalize
        with env.begin(write=True) as txn:
            txn.put(b"num-samples", str(self.current_index - 1).encode())

        env.close()
        logger.info("Conversion Complete.")

    def _write_batch(self, env, ops):
        with env.begin(write=True) as txn:
            for img_bytes, label in ops:
                image_key = f"image-{self.current_index:09d}".encode()
                label_key = f"label-{self.current_index:09d}".encode()

                txn.put(image_key, img_bytes)
                txn.put(label_key, label.encode("utf-8"))

                self.current_index += 1
