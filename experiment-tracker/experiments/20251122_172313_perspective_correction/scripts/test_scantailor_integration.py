#!/usr/bin/env python3
"""
Test ScanTailor integration for perspective correction.

ScanTailor is a C++ application for post-processing scanned pages.
This script provides a Python wrapper to test it.

Requirements:
    - ScanTailor must be installed on the system
    - Install: sudo apt-get install scantailor (Ubuntu/Debian)
    - Or build from source: https://github.com/scantailor/scantailor

Usage:
    python scripts/test_scantailor_integration.py --input-dir data/samples --output-dir outputs/scantailor_test
"""

import argparse
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_scantailor_available() -> bool:
    """Check if ScanTailor is available on the system."""
    return shutil.which("scantailor-cli") is not None or shutil.which("scantailor") is not None


def process_with_scantailor(
    image_path: Path,
    output_dir: Path,
    use_cli: bool = True,
) -> dict[str, Any]:
    """
    Process image with ScanTailor.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        use_cli: Use CLI version (scantailor-cli) if available

    Returns:
        Dictionary with results
    """
    logger.info(f"Processing: {image_path.name}")

    results = {
        "input_path": str(image_path),
        "success": False,
        "processing_time": 0.0,
        "error": None,
    }

    try:
        start_time = time.perf_counter()

        # Create temporary directory for ScanTailor project
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_dir = temp_path / "project"
            project_dir.mkdir()

            # Copy image to project directory
            project_image = project_dir / image_path.name
            shutil.copy(image_path, project_image)

            # Determine ScanTailor command
            if use_cli and shutil.which("scantailor-cli"):
                # Use CLI version (headless)
                cmd = [
                    "scantailor-cli",
                    "--margins=0.1",  # 10% margins
                    "--alignment=auto",
                    "--output-project",
                    str(project_dir / "project.st"),
                    str(project_image),
                ]
            elif shutil.which("scantailor"):
                # Use GUI version (may require X11)
                logger.warning("Using GUI version - may require X11 display")
                cmd = [
                    "scantailor",
                    "--margins=0.1",
                    "--alignment=auto",
                    str(project_image),
                    str(project_dir),
                ]
            else:
                raise RuntimeError("ScanTailor not found. Install with: sudo apt-get install scantailor")

            # Run ScanTailor
            logger.info(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"ScanTailor failed: {result.stderr}")

            # Find output image
            # ScanTailor outputs to project directory with modified name
            output_files = list(project_dir.glob("*.tif")) + list(project_dir.glob("*.png")) + list(project_dir.glob("*.jpg"))

            if not output_files:
                # Try looking in subdirectories
                for subdir in project_dir.iterdir():
                    if subdir.is_dir():
                        output_files.extend(list(subdir.glob("*.tif")))
                        output_files.extend(list(subdir.glob("*.png")))
                        output_files.extend(list(subdir.glob("*.jpg")))

            if not output_files:
                raise RuntimeError("No output file found from ScanTailor")

            # Use first output file
            output_file = output_files[0]

            # Copy to final output directory
            final_output = output_dir / f"{image_path.stem}_scantailor.jpg"

            # Convert if needed (ScanTailor may output TIFF)
            output_image = cv2.imread(str(output_file))
            if output_image is None:
                raise RuntimeError(f"Could not read ScanTailor output: {output_file}")

            cv2.imwrite(str(final_output), output_image)

            processing_time = time.perf_counter() - start_time

            results["success"] = True
            results["processing_time"] = processing_time
            results["output_path"] = str(final_output)
            results["output_shape"] = output_image.shape

            logger.info(f"  ✓ Success: {processing_time:.3f}s")
            logger.info(f"    Output: {final_output.name} ({output_image.shape[1]}x{output_image.shape[0]})")

    except subprocess.TimeoutExpired:
        results["error"] = "ScanTailor timeout (60s)"
        logger.error("  ✗ Timeout")
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"  ✗ Failed: {e}", exc_info=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Test ScanTailor integration for perspective correction")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory with images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/scantailor_test"),
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of images to process",
    )

    args = parser.parse_args()

    # Check if ScanTailor is available
    if not check_scantailor_available():
        logger.error("ScanTailor not found!")
        logger.error("\nInstallation options:")
        logger.error("  1. Build from source (recommended):")
        logger.error("     git clone https://github.com/scantailor/scantailor.git")
        logger.error("     cd scantailor")
        logger.error("     cmake . && make && sudo make install")
        logger.error("\n  2. Use AppImage (if available):")
        logger.error("     Download from: https://github.com/scantailor/scantailor/releases")
        logger.error("\n  3. Use Docker (alternative):")
        logger.error("     docker run -v $(pwd):/data scantailor/scantailor")
        logger.error("\nNote: This is an experimental integration.")
        logger.error("ScanTailor may not be in default repositories.")
        return 1

    logger.info("✓ ScanTailor found")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        image_files.extend(args.input_dir.glob(f"*{ext}"))
        image_files.extend(args.input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.error("No image files found")
        return 1

    image_files = sorted(image_files)[: args.num_samples]
    logger.info(f"Processing {len(image_files)} images")

    # Process
    logger.info("\n" + "=" * 80)
    logger.info("SCANTAILOR INTEGRATION TEST (EXPERIMENTAL)")
    logger.info("=" * 80)

    all_results = []
    for image_path in image_files:
        result = process_with_scantailor(image_path, args.output_dir)
        all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    successful = [r for r in all_results if r.get("success")]
    failed = [r for r in all_results if not r.get("success")]

    logger.info(f"\nTotal: {len(all_results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")

    if successful:
        avg_time = np.mean([r["processing_time"] for r in successful])
        logger.info(f"Average processing time: {avg_time:.3f}s")

    if failed:
        logger.info("\nFailures:")
        for r in failed:
            logger.info(f"  - {Path(r['input_path']).name}: {r.get('error', 'Unknown')}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
