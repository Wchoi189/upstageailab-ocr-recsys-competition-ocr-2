import cv2
import numpy as np

from .text_rendering import put_text_with_outline


# Draw GT and Detection box
def draw_boxes(
    image_path,
    det_polys,
    gt_polys=None,
    det_color=(0, 255, 0),
    gt_color=(0, 0, 255),
    thickness=2,
):
    image = cv2.imread(image_path)

    # Draw GT Polygons
    if gt_polys is not None:
        image = put_text_with_outline(
            image,
            f"{len(gt_polys)} GTs",
            (15, 35),
            font_size=16,
            text_color=gt_color,
            outline_color=(0, 0, 0),
            outline_width=2,
        )
        for box in gt_polys:
            box = np.array(box).reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [box], True, gt_color, thickness=thickness + 1)

    # Draw Detected Polygons
    image = put_text_with_outline(
        image,
        f"{len(det_polys)} DETs",
        (15, 70),
        font_size=16,
        text_color=det_color,
        outline_color=(0, 0, 0),
        outline_width=2,
    )
    for box in det_polys:
        box = np.array(box).reshape(-1, 2).astype(np.int32)
        cv2.polylines(image, [box], True, det_color, thickness=thickness)

    return image
