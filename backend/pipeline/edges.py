import cv2
import numpy as np
from PIL import Image

def detect_edges(mask: np.ndarray):
    """
    Detect class boundaries from a discrete segmentation mask.

    The previous implementation ran Canny on the raw class-index values
    (0..3), which often yields near-black edges. Instead, we detect
    where adjacent pixels belong to different classes.
    """
    if mask is None:
        return None

    m = mask.astype(np.int32)

    # Horizontal/vertical boundaries between different class ids
    diff_x = m[:, 1:] != m[:, :-1]
    diff_y = m[1:, :] != m[:-1, :]

    edges = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint8)
    edges[:, 1:][diff_x] = 255
    edges[1:, :][diff_y] = 255

    # Dilate slightly for visibility
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Create colored overlay (green edges)
    edge_overlay = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    edge_overlay[edges > 0] = [0, 255, 0]  # Green edges

    return edge_overlay
