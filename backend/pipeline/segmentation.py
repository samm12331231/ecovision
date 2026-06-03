import numpy as np
from PIL import Image

class SegmentationModel:
    def __init__(self, model_path=None):
        # Placeholder - will load real model later
        self.model_path = model_path
    
    def predict(self, image: Image.Image):
        # MOCK: Returns random segmentation mask for now
        # Real version will load SegFormer and return 0-3 classes
        width, height = image.size
        mask = np.random.randint(0, 4, size=(height, width), dtype=np.uint8)
        return mask