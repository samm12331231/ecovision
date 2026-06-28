from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
import time
import base64
import logging
import numpy as np

# Import pipeline modules
from backend.pipeline.segmentation import SegmentationModel
from backend.pipeline.detection import DetectionModel
from backend.pipeline.change import detect_change
from backend.pipeline.edges import detect_edges

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecovision")

# Color map for segmentation classes (RGB)
COLOR_MAP = {
    0: [46, 204, 113],   # Vegetation - green
    1: [244, 208, 63],   # Sand - yellow  
    2: [52, 152, 219],   # Water - blue
    3: [231, 76, 60],    # Urban - red
    255: [0, 0, 0]       # Ignore - black
}

def _colorize_mask(mask):
    """Convert class index mask to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        rgb[mask == cls_id] = color
    return rgb

def _blend_overlay(original_rgb, color_mask, alpha=0.5):
    """Alpha-blend the class-color mask over the original image.

    Produces a semi-transparent segmentation overlay (terrain visible
    underneath) instead of a flat color block, which is the standard way
    to visualize semantic segmentation.
    """
    orig = original_rgb.astype(np.float32)
    cmask = color_mask.astype(np.float32)
    blended = (alpha * cmask + (1.0 - alpha) * orig)
    return np.clip(blended, 0, 255).astype(np.uint8)

def _img_to_b64(img_array):
    """Convert numpy array to base64 PNG string."""
    if img_array is None:
        return None
    
    arr = img_array
    
    # Normalize to 0-255 if needed
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1 else arr.astype(np.uint8)
    
    # Handle grayscale
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    
    # Ensure 3-channel
    if len(arr.shape) == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

app = FastAPI(title="UAE EcoVision", version="0.1.0")

# Initialize models. If a weights folder is missing or a model fails to load,
# log it and keep the server running — the affected stage reports an error per
# request instead of taking the whole API down at startup.
try:
    seg_model = SegmentationModel()
except Exception as e:
    logger.exception("Failed to load segmentation model")
    seg_model = None

try:
    det_model = DetectionModel()
except Exception as e:
    logger.exception("Failed to load detection model")
    det_model = None

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "UAE EcoVision",
        "version": "0.1.0",
        "models_loaded": {
            "segmentation": seg_model is not None,
            "detection": det_model is not None,
        },
    }

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    baseline: UploadFile = File(None),
    mode: str = Form("full")
):
    start_time = time.time()

    # ---- Load uploaded image (corrupt/invalid uploads -> 400, not 500) ----
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.exception("Could not read uploaded image")
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not read uploaded image: {e}"},
        )

    # ---- Load baseline if provided (optional; a bad baseline is non-fatal) ----
    baseline_img = None
    if baseline:
        try:
            baseline_contents = await baseline.read()
            baseline_img = Image.open(io.BytesIO(baseline_contents)).convert("RGB")
        except Exception as e:
            logger.warning("Could not read baseline image, ignoring it: %s", e)
            baseline_img = None

    response = {
        "filename": image.filename,
        "image_size": list(img.size),
        "mode": mode,
        "processing_time_ms": 0,
        "results": {},
        "errors": {},   # per-stage error messages, if any
    }

    # ---- Segmentation ----
    seg_mask = None
    if mode in ["full", "segmentation"]:
        try:
            if seg_model is None:
                raise RuntimeError("segmentation model not loaded")
            seg_mask = seg_model.predict(img)
            seg_color = _colorize_mask(seg_mask)
            # Blend the class colors over the original image so the overlay is
            # readable as a labeled map rather than a flat color block.
            seg_vis = _blend_overlay(np.array(img), seg_color, alpha=0.5)
            classes_present = [int(c) for c in np.unique(seg_mask) if c != 255]
            # Real per-class proportions over valid (non-ignore) pixels, so the
            # dashboard land-cover chart reflects actual model output instead of
            # an even split. Keys are strings to survive JSON round-tripping.
            valid = seg_mask != 255
            total_valid = int(np.sum(valid))
            class_distribution = {}
            if total_valid > 0:
                for c in classes_present:
                    pct = float(np.sum(seg_mask == c)) / total_valid * 100.0
                    class_distribution[str(c)] = round(pct, 2)
            response["results"]["segmentation"] = {
                "mask_shape": list(seg_mask.shape),
                "classes_found": classes_present,
                "class_distribution": class_distribution,
                "health_score": float(np.mean(seg_mask == 0) * 100),
                "overlay_b64": _img_to_b64(seg_vis),
            }
        except Exception as e:
            logger.exception("Segmentation stage failed")
            response["errors"]["segmentation"] = str(e)

    # ---- Edge detection (depends on the segmentation mask) ----
    if mode == "full":
        if seg_mask is not None:
            try:
                edges_overlay = detect_edges(seg_mask)
                response["results"]["edges"] = {
                    "edge_pixels": int(np.sum(edges_overlay[..., 1] > 0)),
                    "overlay_b64": _img_to_b64(edges_overlay),
                }
            except Exception as e:
                logger.exception("Edge detection stage failed")
                response["errors"]["edges"] = str(e)
        else:
            response["errors"]["edges"] = "skipped: segmentation unavailable"

    # ---- Object detection ----
    if mode in ["full", "detection"]:
        try:
            if det_model is None:
                raise RuntimeError("detection model not loaded")
            response["results"]["detection"] = det_model.predict(img)
        except Exception as e:
            logger.exception("Detection stage failed")
            response["errors"]["detection"] = str(e)

    # ---- Change detection ----
    if mode in ["full", "change"]:
        try:
            heatmap_overlay = detect_change(img, baseline_img)
            changed_pixels = int(np.sum(heatmap_overlay[..., 0] > 0))
            response["results"]["change"] = {
                "has_baseline": baseline_img is not None,
                "changed_pixels": changed_pixels,
                "overlay_b64": _img_to_b64(heatmap_overlay),
            }
        except Exception as e:
            logger.exception("Change detection stage failed")
            response["errors"]["change"] = str(e)

    response["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)

    return JSONResponse(response)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "UAE EcoVision"}