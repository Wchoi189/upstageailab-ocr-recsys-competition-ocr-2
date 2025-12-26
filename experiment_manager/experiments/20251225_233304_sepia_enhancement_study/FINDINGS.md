# Findings: Sepia vs Grayscale Enhancement for OCR

## Overview
This experiment investigated preprocessing techniques to improve text detection on illumination-challenged images (e.g., severe lighting variation, shadows).

## Setup
- **Subject**: Image `1454` (known worst-case failure: 0 detections).
- **Tool**: Upstage Document AI OCR (API).
- **Conditions**:
    1.  **Baseline**: Raw image.
    2.  **Sepia**: Rembg + Perspective Warp + Sepia Tone.
    3.  **Grayscale**: Rembg + Perspective Warp + Grayscale.

## Key Findings
| Strategy | Detected Words | Observation |
| :--- | :---: | :--- |
| **Baseline** | 0 | Complete failure. |
| **Sepia** | **126** | **Best Performance**. The built-in brightness boost (coefficients > 1.0) lifted dark text out of the noise floor. |
| **Grayscale** | 123 | Very strong performance. A neutral alternative if brightness artifacts are undesirable. |

## Conclusion
**Sepia Enhancement** is the recommended default for maximum recall on this dataset.
- It solved the "0 detections" problem effectively.
- It outperformed Grayscale (textbook standard) by a small margin (+3 words) due to its favourable side-effects (brightness/contrast boost).

## Artifacts
- **Scripts**: Located in `src/`.
- **Visualizations**: Located in `artifacts/visualizations/`.
