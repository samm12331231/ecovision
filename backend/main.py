from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import time
import numpy as np
import cv2
import base64

# Import pipeline modules
from backend.pipeline.segmentation import SegmentationModel
from backend.pipeline.detection import DetectionModel
from backend.pipeline.change import detect_change
from backend.pipeline.edges import detect_edges

app = FastAPI(title="UAE EcoVision", version="0.1.0")

# Initialize models (placeholders for now)
seg_model = SegmentationModel()
det_model = DetectionModel()

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "UAE EcoVision",
        "version": "0.1.0",
        "models_loaded": False
    }

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    baseline: UploadFile = File(None),  # Optional baseline for change detection
    mode: str = Form("full")  # "full", "segmentation", "detection", "change", "edges"
):
    start_time = time.time()
    
    # Load uploaded image
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Load baseline if provided
    baseline_img = None
    if baseline:
        baseline_contents = await baseline.read()
        baseline_img = Image.open(io.BytesIO(baseline_contents)).convert("RGB")
    
    # Initialize response
    response = {
        "filename": image.filename,
        "image_size": img.size,
        "mode": mode,
        "processing_time_ms": 0,
        "results": {}
    }
    
    # Run pipeline based on mode
    if mode in ["full", "segmentation"]:
        # Task 1: Segmentation
        seg_mask = seg_model.predict(img)
        response["results"]["segmentation"] = {
            "mask_shape": seg_mask.shape,
            "classes_found": [int(c) for c in np.unique(seg_mask) if c != 255],
            "health_score": float(np.mean(seg_mask == 0) * 100)  # % vegetation
        }
        
        # Task 4: Edge Detection (on segmentation mask)
        if mode == "full":
            edges = detect_edges(seg_mask)
            response["results"]["edges"] = {
                "edge_pixels": int(np.sum(edges > 0) / 3)
            }
    
    if mode in ["full", "detection"]:
        # Task 2: Object Detection
        boxes = det_model.predict(img)
        response["results"]["detection"] = {
            "num_objects": len(boxes),
            "objects": boxes
        }
    
    if mode in ["full", "change"]:
        # Task 3: Change Detection
        heatmap = detect_change(img, baseline_img)
        response["results"]["change"] = {
            "has_baseline": baseline_img is not None,
            "changed_pixels": int(np.sum(heatmap > 0) / 3)
        }
    
    # Calculate latency
    response["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
    
    return JSONResponse(response)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "UAE EcoVision"}