import cv2
import numpy as np
from PIL import Image

def detect_change(current_image: Image.Image, baseline_image: Image.Image = None):
    """
    Compare current image to baseline using OpenCV.
    If no baseline provided, returns empty heatmap.
    """
    if baseline_image is None:
        return np.zeros((current_image.size[1], current_image.size[0], 3), dtype=np.uint8)
    
    # Convert to numpy arrays
    curr = np.array(current_image)
    base = np.array(baseline_image)
    
    # Resize baseline to match current if needed
    if curr.shape != base.shape:
        base = cv2.resize(base, (curr.shape[1], curr.shape[0]))
    
    # Absolute difference
    diff = cv2.absdiff(curr, base)
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    # Morphology to clean noise
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    
    # Create red heatmap overlay
    heatmap = np.zeros_like(curr)
    heatmap[morphed > 0] = [255, 0, 0]  # Red for changes
    
    return heatmap