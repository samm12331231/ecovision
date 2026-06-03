from PIL import Image
import numpy as np

class DetectionModel:
    def __init__(self, model_path=None):
        self.model_path = model_path
    
    def predict(self, image: Image.Image):
        # MOCK: Returns fake bounding boxes for now
        # Real version will load YOLOv8 and return [x1, y1, x2, y2, class, conf]
        width, height = image.size
        boxes = [
            {"x1": 50, "y1": 50, "x2": 150, "y2": 150, "class": "vegetation", "confidence": 0.85}
        ]
        return boxes