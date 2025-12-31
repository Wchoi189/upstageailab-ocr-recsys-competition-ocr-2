import os
import tarfile
import subprocess
import glob
import shutil
from pathlib import Path

# Configuration
RAW_DIR = Path("data/raw/external")
TARGET_DIR = RAW_DIR / "aihub_public_admin_doc/validation"
TEMP_DIR = RAW_DIR / "temp_unpack"

LABEL_TAR = RAW_DIR / "[라벨]validation_32mb.zip.tar"
SOURCE_TAR = RAW_DIR / "[원천]validation_7.47gb.zip.tar"

def run_command(cmd, shell=False):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=shell)

def main():
    if TARGET_DIR.exists():
        print(f"Target directory {TARGET_DIR} already exists. Cleaning up...")
        shutil.rmtree(TARGET_DIR)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    # Don't recreate TEMP_DIR if it has content, assuming we reuse it
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Extract Tars (Skip if already populated - check for known dir)
        base_path = TEMP_DIR / "032.공공행정문서_OCR/01.데이터/02.Validation"
        if base_path.exists():
            print("Temp directory appears populated. Skipping Tar extraction...")
        else:
            print("Extracting Tar archives...")
            with tarfile.open(LABEL_TAR) as tar:
                tar.extractall(path=TEMP_DIR)

            with tarfile.open(SOURCE_TAR) as tar:
                tar.extractall(path=TEMP_DIR)

        # 2. Find and Merge Split Zips
        # Use simple glob and filter
        print(f"Searching in {base_path}")
        all_parts = sorted(list(base_path.glob("*validation.zip.part*")))

        # Filter for labels
        label_parts = [p for p in all_parts if "라벨" in p.name]

        # Filter for source (images)
        source_parts = [p for p in all_parts if "원천" in p.name]

        # Labels
        print(f"Found {len(label_parts)} label parts")
        if label_parts:
            label_zip = TARGET_DIR / "labels.zip"
            with open(label_zip, "wb") as outfile:
                for part in label_parts:
                    with open(part, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)

            (TARGET_DIR / "labels").mkdir(exist_ok=True)
            run_command(["unzip", "-q", "-o", str(label_zip), "-d", str(TARGET_DIR / "labels")])
            label_zip.unlink()

        # Images
        print(f"Found {len(source_parts)} image parts")
        if source_parts:
            # Sort by byte offset (suffix number)
            # Filenames: ...part0, ...part1073741824
            source_parts.sort(key=lambda p: int(p.name.split("part")[-1]))

            source_zip = TARGET_DIR / "images.zip"
            with open(source_zip, "wb") as outfile:
                for part in source_parts:
                    print(f"Merging {part.name}...")
                    with open(part, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)

            (TARGET_DIR / "images").mkdir(exist_ok=True)
            run_command(["unzip", "-q", "-o", str(source_zip), "-d", str(TARGET_DIR / "images")])
            source_zip.unlink()

    finally:
         # Cleanup Temp
         # print("Cleaning up temp files...")
         # shutil.rmtree(TEMP_DIR)
         pass

if __name__ == "__main__":
    main()
