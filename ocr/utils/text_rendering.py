"""UTF-8 text rendering utilities for Korean text visualization.

OpenCV's cv2.putText() doesn't support Korean characters. This module provides
PIL-based text rendering that supports UTF-8/Korean text while maintaining
compatibility with OpenCV image arrays.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Default font paths for Korean text support
KOREAN_FONT_PATHS = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
    "C:\\Windows\\Fonts\\malgun.ttf",  # Windows
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux alternative
]


def get_korean_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a TrueType font that supports Korean characters.

    Args:
        size: Font size in pixels

    Returns:
        PIL Font object (FreeTypeFont if Korean font found, else default ImageFont)
    """
    for font_path in KOREAN_FONT_PATHS:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue

    # Fallback to default font (limited character support)
    return ImageFont.load_default()


def put_text_utf8(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    color: tuple[int, int, int] = (255, 255, 255),
    bgcolor: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Draw UTF-8 text (including Korean) on an OpenCV image using PIL.

    Args:
        image: OpenCV/NumPy image array (BGR format)
        text: UTF-8 text to draw (supports Korean)
        position: (x, y) position for text placement
        font_size: Font size in pixels
        color: Text color in BGR format (default white)
        bgcolor: Optional background color in BGRA format

    Returns:
        Modified image with text drawn
    """
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Get Korean-compatible font
    font = get_korean_font(font_size)

    # Convert BGR color to RGB for PIL
    color_rgb = (color[2], color[1], color[0])

    # Draw background rectangle if specified
    if bgcolor is not None:
        # Get text bounding box
        bbox = draw.textbbox(position, text, font=font)
        # Draw background with alpha
        bg_rgb = (bgcolor[2], bgcolor[1], bgcolor[0])
        draw.rectangle(bbox, fill=bg_rgb)

    # Draw text
    draw.text(position, text, font=font, fill=color_rgb)

    # Convert back to BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


def put_text_with_outline(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_size: int = 20,
    text_color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
    outline_width: int = 2,
) -> np.ndarray:
    """Draw UTF-8 text with outline for better visibility.

    Args:
        image: OpenCV/NumPy image array (BGR format)
        text: UTF-8 text to draw
        position: (x, y) position
        font_size: Font size in pixels
        text_color: Text color in BGR
        outline_color: Outline color in BGR
        outline_width: Outline thickness in pixels

    Returns:
        Modified image with outlined text
    """
    # Convert BGR to RGB for PIL
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Get Korean-compatible font
    font = get_korean_font(font_size)

    # Convert BGR colors to RGB for PIL
    text_rgb = (text_color[2], text_color[1], text_color[0])
    outline_rgb = (outline_color[2], outline_color[1], outline_color[0])

    # Draw outline by drawing text multiple times with offset
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_rgb)

    # Draw main text
    draw.text(position, text, font=font, fill=text_rgb)

    # Convert back to BGR for OpenCV
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_bgr


__all__ = [
    "get_korean_font",
    "put_text_utf8",
    "put_text_with_outline",
]
