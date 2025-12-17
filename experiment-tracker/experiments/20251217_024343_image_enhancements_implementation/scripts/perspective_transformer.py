import numpy as np
import cv2

def calculate_target_dimensions(pts: np.ndarray) -> tuple[int, int]:
    """
    Calculate the target width and height using the 'Max-Edge' aspect ratio rule.

    Args:
        pts: The 4 detected corners in order [TL, TR, BR, BL]

    Returns:
        (maxWidth, maxHeight) for the destination image.
    """
    (tl, tr, br, bl) = pts

    # 1. Calculate the width of the top and bottom edges
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # The final width is the maximum of the two
    maxWidth = max(int(widthA), int(widthB))

    # 2. Calculate the height of the left and right edges
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # The final height is the maximum of the two
    maxHeight = max(int(heightA), int(heightB))

    return maxWidth, maxHeight

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to the image using the provided corners.

    Args:
        image: Source image
        pts: The 4 detected corners in order [TL, TR, BR, BL]

    Returns:
        Warped image
    """
    # Obtain a consistent order of the points and unpack them separately
    # pts is assumed to be ordered [TL, TR, BR, BL]
    # Ensure pts is float32
    if pts.dtype != np.float32:
        pts = pts.astype(np.float32)

    maxWidth, maxHeight = calculate_target_dimensions(pts)

    # Construct the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)

    # Use INTER_LANCZOS4 for better quality on text documents
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LANCZOS4)

    return warped

