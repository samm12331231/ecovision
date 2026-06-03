import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# UAE class names
ID2LABEL = {0: "vegetation", 1: "sand", 2: "water", 3: "urban"}

class SegmentationModel:
    def __init__(self, model_path="backend/models/segformer-b0-final"):
        self.processor = SegformerImageProcessor.from_pretrained(model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        self.model.eval()  # inference mode
        self.id2label = ID2LABEL
    
    def predict(self, image: Image.Image):
        # Preprocess and run inference
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():  # no gradients needed for inference
            outputs = self.model(**inputs)
        
        # Upsample logits to original image size
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        
        # Get predicted class for each pixel
        predicted_mask = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        
        return predicted_mask