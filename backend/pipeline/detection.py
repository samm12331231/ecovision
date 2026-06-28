from ultralytics import YOLO
import numpy as np
from PIL import Image

class DetectionModel:
    def __init__(self, model_path="backend/models/yolo-runs/ecovision-yolo/weights/best.pt"):
        self.model = YOLO(model_path)
    
    def predict(self, image: Image.Image):
        results = self.model(image, conf=0.2)
        boxes = results[0].boxes
        
        # Map YOLO class id -> class name (if available)
        names = getattr(self.model, "names", None) or {}
        objects = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cls_id = int(box.cls[0]) if hasattr(box, "cls") and box.cls is not None else None
            cls_name = names.get(cls_id, str(cls_id)) if cls_id is not None else "unknown"

            objects.append({
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
                "class": cls_name,
                "confidence": float(box.conf[0])
            })
        
        return {
            "num_objects": len(objects),
            "objects": objects
        }
