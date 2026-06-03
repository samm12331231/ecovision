import cv2
import numpy as np
from PIL import Image

def detect_edges(mask: np.ndarray):
    """
    Run Canny edge detection on segmentation mask.
    Returns colored boundary lines.
    """
    # Ensure mask is uint8
    mask_uint8 = mask.astype(np.uint8)
    
    # Canny edge detection
    edges = cv2.Canny(mask_uint8, 50, 150)
    
    # Dilate for visibility
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create colored overlay (green edges)
    edge_overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    edge_overlay[edges > 0] = [0, 255, 0]  # Green edges
    
    return edge_overlay