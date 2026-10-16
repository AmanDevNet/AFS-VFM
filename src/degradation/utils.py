"""
Utility functions for the AFS-VFM degradation engine.

Provides image I/O and dimension-preserving helpers used throughout
the degradation pipeline.
"""

import os
from pathlib import Path
from typing import Union
import cv2
import numpy as np


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and return it in RGB format.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        NumPy array of shape (H, W, 3) with dtype uint8, in RGB order.

    Raises:
        FileNotFoundError: If the image cannot be read by OpenCV.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    # OpenCV loads as BGR — convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(path: Union[str, Path], image: np.ndarray) -> None:
    """Save an RGB image array to disk in standard image format.

    Args:
        path: Destination file path.
        image: NumPy array of shape (H, W, 3) with dtype uint8, in RGB order.
    """
    dest_dir = os.path.dirname(os.path.abspath(str(path)))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr_image)


def ensure_dimensions(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad or crop an image so it matches the target height and width.

    The image is centred within the target canvas.  Any padding is
    filled with zeros (black).

    Args:
        image: Input image array of shape (H, W, 3).
        target_h: Desired output height in pixels.
        target_w: Desired output width in pixels.

    Returns:
        NumPy array of shape (target_h, target_w, 3) with dtype uint8.
    """
    h, w = image.shape[:2]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Compute centring offsets
    y_offset = max(0, (target_h - h) // 2)
    x_offset = max(0, (target_w - w) // 2)

    # Region of the source that fits on the canvas
    src_y_start = max(0, (h - target_h) // 2)
    src_x_start = max(0, (w - target_w) // 2)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)

    canvas[y_offset:y_offset + copy_h, x_offset:x_offset + copy_w] = \
        image[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

    return canvas
