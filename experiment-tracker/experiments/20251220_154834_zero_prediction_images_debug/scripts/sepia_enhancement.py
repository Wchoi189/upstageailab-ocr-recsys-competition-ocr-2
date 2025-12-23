#!/usr/bin/env python3
"""
Sepia tone enhancement for document images.

Implements multiple sepia methods as alternatives to gray-scale and normalization:
1. Classic Sepia - Traditional sepia transformation matrix
2. Adaptive Sepia - Intensity-based sepia with contrast preservation
3. Warm Sepia - Enhanced warm tones for document clarity
4. Sepia + Contrast - Sepia with automatic contrast adjustment

EDS v1.0 Experiment: 20251217_024343_image_enhancements_implementation
Phase: Week 2 Day 4-5 (Alternative to gray-scale normalization)
Target: Superior OCR results compared to gray-scale and normalization

Reference sample: drp.en_ko.in_house.selectstar_000712_sepia.jpg

Usage:
    python sepia_enhancement.py --input <dir> --method <method> --output <dir>
    python sepia_enhancement.py --input <image> --method all --output <dir>
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class SepiaEnhancer:
    """Document image sepia enhancement methods."""

    def __init__(self, method: str = "classic"):
        """
        Initialize sepia enhancer.

        Args:
            method: Sepia method [classic|adaptive|warm|contrast|all]
        """
        self.method = method
        self.valid_methods = ["classic", "adaptive", "warm", "clahe", "linear_contrast", "all"]

        if method not in self.valid_methods:
            raise ValueError(f"Invalid method: {method}. Choose from {self.valid_methods}")

    def enhance(self, img: np.ndarray) -> dict[str, tuple[np.ndarray, dict]]:
        """
        Apply sepia enhancement to document image.

        Args:
            img: Input image (BGR format)

        Returns:
            Dictionary of {method_name: (enhanced_image, metrics_dict)}
        """
        results = {}

        if self.method == "all":
            methods = ["classic", "adaptive", "warm", "clahe", "linear_contrast"]
        else:
            methods = [self.method]

        for method in methods:
            start_time = time.time()

            if method == "classic":
                result = self.sepia_classic(img)
            elif method == "adaptive":
                result = self.sepia_adaptive(img)
            elif method == "warm":
                result = self.sepia_warm(img)
            elif method == "clahe":
                result = self.sepia_clahe(img)
            elif method == "linear_contrast":
                result = self.sepia_linear_contrast(img)
            else:
                raise ValueError(f"Unknown method: {method}")

            processing_time = (time.time() - start_time) * 1000  # ms

            # Calculate metrics
            metrics = self._calculate_metrics(img, result, processing_time)
            results[method] = (result, metrics)

        return results

    def sepia_classic(self, img: np.ndarray) -> np.ndarray:
        """
        Classic sepia transformation using standard matrix.

        Formula:
            R' = 0.393*R + 0.769*G + 0.189*B
            G' = 0.349*R + 0.686*G + 0.168*B
            B' = 0.272*R + 0.534*G + 0.131*B

        Args:
            img: Input image (BGR)

        Returns:
            Sepia-toned image (BGR)
        """
        # Convert BGR to RGB for processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Classic sepia transformation matrix (transposed for OpenCV)
        sepia_matrix = np.array(
            [
                [0.393, 0.769, 0.189],  # Red channel
                [0.349, 0.686, 0.168],  # Green channel
                [0.272, 0.534, 0.131],  # Blue channel
            ]
        )

        # Apply transformation
        sepia = cv2.transform(img_rgb, sepia_matrix)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        # Convert back to BGR
        result = cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)

        return result

    def sepia_adaptive(self, img: np.ndarray) -> np.ndarray:
        """
        Adaptive sepia based on image intensity with contrast preservation.

        Adjusts sepia strength based on pixel intensity to preserve details.

        Args:
            img: Input image (BGR)

        Returns:
            Adaptive sepia-toned image (BGR)
        """
        # Apply classic sepia first
        sepia = self.sepia_classic(img)

        # Calculate intensity weights (preserve bright/dark contrasts)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        intensity = gray.astype(np.float32) / 255.0

        # Create adaptive blend weight (stronger sepia in mid-tones)
        # Preserve extreme darks and brights
        blend_weight = 1.0 - np.abs(intensity - 0.5) * 0.4  # Range: 0.8 to 1.0
        blend_weight = np.stack([blend_weight] * 3, axis=-1)

        # Blend original and sepia based on intensity
        result = sepia.astype(np.float32) * blend_weight + img.astype(np.float32) * (1 - blend_weight)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def sepia_warm(self, img: np.ndarray) -> np.ndarray:
        """
        Enhanced warm sepia tone optimized for document OCR.

        Stronger warm tones than classic sepia, reducing blue/cold tints
        that can interfere with OCR.

        Args:
            img: Input image (BGR)

        Returns:
            Warm sepia-toned image (BGR)
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Enhanced warm sepia matrix (stronger red/yellow channels)
        warm_matrix = np.array(
            [
                [0.450, 0.850, 0.200],  # Red channel (strong boost)
                [0.350, 0.750, 0.150],  # Green channel (boosted)
                [0.200, 0.450, 0.100],  # Blue channel (reduced)
            ]
        )

        # Apply transformation
        sepia = cv2.transform(img_rgb, warm_matrix)
        sepia = np.clip(sepia, 0, 255).astype(np.uint8)

        # Convert back to BGR
        result = cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)

        return result

    def sepia_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Sepia with CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Adaptive contrast adjustment optimized for document clarity.

        Args:
            img: Input image (BGR)

        Returns:
            Sepia-toned image with enhanced contrast (BGR)
        """
        # Apply warm sepia first
        sepia = self.sepia_warm(img)

        # Convert to LAB color space for contrast enhancement
        lab = cv2.cvtColor(sepia, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)

        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result

    def sepia_linear_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Sepia with global linear contrast adjustment (alpha/beta).

        Simple global contrast boost without local adaptation.

        Args:
            img: Input image (BGR)

        Returns:
            Sepia-toned image with global contrast adjustment (BGR)
        """
        # Apply warm sepia first
        sepia = self.sepia_warm(img)

        # Apply global contrast (gain=1.2, bias=10)
        # Formula: new_img = alpha*img + beta
        alpha = 1.2
        beta = 10
        result = cv2.convertScaleAbs(sepia, alpha=alpha, beta=beta)

        return result

    def _calculate_metrics(self, original: np.ndarray, enhanced: np.ndarray, processing_time: float) -> dict:
        """
        Calculate enhancement metrics.

        Args:
            original: Original image
            enhanced: Enhanced image
            processing_time: Processing time in ms

        Returns:
            Dictionary of metrics
        """
        # Color statistics
        orig_mean = original.mean(axis=(0, 1))
        enh_mean = enhanced.mean(axis=(0, 1))

        # Color tint score (deviation from neutral gray)
        orig_tint = np.std(orig_mean)
        enh_tint = np.std(enh_mean)

        # Background variance (spatial variation)
        orig_var = original.std()
        enh_var = enhanced.std()

        # Contrast (standard deviation of intensity)
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        orig_contrast = orig_gray.std()
        enh_contrast = enh_gray.std()

        # Brightness
        orig_brightness = orig_gray.mean()
        enh_brightness = enh_gray.mean()

        return {
            "processing_time_ms": round(processing_time, 2),
            "color_tint_before": round(orig_tint, 2),
            "color_tint_after": round(enh_tint, 2),
            "tint_change": round(enh_tint - orig_tint, 2),
            "variance_before": round(orig_var, 2),
            "variance_after": round(enh_var, 2),
            "variance_change": round(enh_var - orig_var, 2),
            "contrast_before": round(orig_contrast, 2),
            "contrast_after": round(enh_contrast, 2),
            "contrast_change": round(enh_contrast - orig_contrast, 2),
            "brightness_before": round(orig_brightness, 2),
            "brightness_after": round(enh_brightness, 2),
            "brightness_change": round(enh_brightness - orig_brightness, 2),
        }


def process_single_image(input_path: Path, method: str, output_dir: Path, verbose: bool = True) -> dict[str, dict]:
    """
    Process a single image with sepia enhancement.

    Args:
        input_path: Path to input image
        method: Sepia method to apply
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Dictionary of {method_name: metrics_dict}
    """
    # Load image
    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"Failed to load image: {input_path}")

    if verbose:
        print(f"Processing: {input_path.name}")
        print(f"  Input size: {img.shape[1]}x{img.shape[0]}")

    # Apply enhancement
    enhancer = SepiaEnhancer(method=method)
    results = enhancer.enhance(img)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    all_metrics = {}

    for method_name, (enhanced, metrics) in results.items():
        # Save enhanced image
        output_path = output_dir / f"{stem}_sepia_{method_name}.jpg"
        cv2.imwrite(str(output_path), enhanced)

        if verbose:
            print(f"\n  Method: {method_name}")
            print(f"    Output: {output_path.name}")
            print(f"    Processing time: {metrics['processing_time_ms']:.2f}ms")
            print(f"    Tint: {metrics['color_tint_before']:.1f} → {metrics['color_tint_after']:.1f} ({metrics['tint_change']:+.1f})")
            print(f"    Contrast: {metrics['contrast_before']:.1f} → {metrics['contrast_after']:.1f} ({metrics['contrast_change']:+.1f})")
            print(
                f"    Brightness: {metrics['brightness_before']:.1f} → {metrics['brightness_after']:.1f} ({metrics['brightness_change']:+.1f})"
            )

        all_metrics[method_name] = metrics

    return all_metrics


def process_directory(input_dir: Path, method: str, output_dir: Path, verbose: bool = True) -> dict[str, dict[str, dict]]:
    """
    Process all images in directory.

    Args:
        input_dir: Input directory
        method: Sepia method to apply
        output_dir: Output directory
        verbose: Print progress

    Returns:
        Dictionary of {image_name: {method_name: metrics_dict}}
    """
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        raise ValueError(f"No images found in {input_dir}")

    if verbose:
        print(f"\nFound {len(image_files)} images in {input_dir}")
        print(f"Method: {method}")
        print(f"Output: {output_dir}\n")
        print("=" * 80)

    all_results = {}

    for img_file in sorted(image_files):
        metrics = process_single_image(img_file, method, output_dir, verbose)
        all_results[img_file.name] = metrics

        if verbose:
            print("=" * 80)

    return all_results


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Apply sepia enhancement to document images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input image file or directory",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="classic",
        choices=["classic", "adaptive", "warm", "clahe", "linear_contrast", "all"],
        help="Sepia method to apply (default: classic)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for enhanced images",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Process input
    if args.input.is_file():
        process_single_image(args.input, args.method, args.output, verbose=not args.quiet)
    elif args.input.is_dir():
        process_directory(args.input, args.method, args.output, verbose=not args.quiet)
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    if not args.quiet:
        print("\n✓ Sepia enhancement complete!")


if __name__ == "__main__":
    main()
