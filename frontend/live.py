"""
Live Video modality for UAE EcoVision — YOLO object detection on video frames.
Optimized for >10 FPS via frame skipping and resolution downsampling.
"""

import time
import tempfile
import inspect

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.pipeline.detection import DetectionModel


def _fill_kwargs(fn):
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    try:
        mj, mn = (int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        mj, mn = 0, 0
    if (mj, mn) >= (1, 49) and "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    if "use_column_width" in params:
        return {"use_column_width": True}
    return {}


_IMG_W = _fill_kwargs(st.image)

# Target resolution for faster inference (YOLO handles small inputs fine)
TARGET_W, TARGET_H = 416, 234  # ~480p downsampled


def process_frame_yolo(frame_bgr, detector):
    """Run YOLO detection on a single BGR frame and draw boxes."""
    # Resize for speed (YOLO is trained on 640x640, 416 is fast but accurate enough)
    small = cv2.resize(frame_bgr, (TARGET_W, TARGET_H))
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Run YOLO inference
    result = detector.predict(pil_img)
    
    # Scale boxes back to original resolution for display
    h_orig, w_orig = frame_bgr.shape[:2]
    scale_x = w_orig / TARGET_W
    scale_y = h_orig / TARGET_H
    
    for obj in result.get("objects", []):
        x1 = int(obj["x1"] * scale_x)
        y1 = int(obj["y1"] * scale_y)
        x2 = int(obj["x2"] * scale_x)
        y2 = int(obj["y2"] * scale_y)
        conf = obj.get("confidence", 0)
        label = obj.get("class", "vegetation").replace("uae-vegetation-satellite", "vegetation")
        
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame_bgr, text, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    return frame_bgr, result.get("num_objects", 0)


def _run_video(cap, detector, skip_frames=1):
    """
    Stream frames with YOLO detection.
    skip_frames=1 means process every frame, skip_frames=2 means every 2nd frame.
    """
    vid_col, stat_col = st.columns([3, 1])
    frame_slot = vid_col.empty()
    fps_slot = stat_col.empty()
    obj_slot = stat_col.empty()
    progress_slot = vid_col.empty()

    fps_display = 0.0
    prev = None
    frame_count = 0
    processed_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Skip frames to maintain target FPS
        if frame_count % skip_frames != 0:
            continue
        
        annotated, count = process_frame_yolo(frame, detector)
        processed_count += 1
        
        # FPS calculation (based on processed frames)
        now = time.time()
        if prev is not None:
            dt = now - prev
            if dt > 0:
                inst = 1.0 / dt
                fps_display = inst if fps_display == 0.0 else (0.9 * fps_display + 0.1 * inst)
        prev = now

        frame_slot.image(annotated, channels="BGR", **_IMG_W)
        fps_slot.metric("FPS", f"{fps_display:.1f}")
        obj_slot.metric("Objects detected", count)
        
        progress_slot.progress(min(frame_count / total_frames, 1.0), 
                               text=f"Frame {frame_count}/{total_frames}")

    cap.release()
    return processed_count


def live_section():
    st.header("🎥 Live Video — Vegetation Detection")
    st.caption(
        "Real-time vegetation detection on video using YOLOv8. "
        "Green boxes mark detected vegetation patches with confidence scores. "
        "Resolution downsampled to 416px width for >10 FPS performance."
    )

    video_file = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi", "mkv"]
    )
    
    # Frame skip control to tune performance
    skip = st.slider("Frame skip (higher = faster FPS)", 1, 5, 1,
                     help="Process every Nth frame. Set to 2-3 if FPS is below 10.")
    
    if video_file is not None and st.button("▶ Run Detection", type="primary"):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()
        
        with st.spinner("Loading YOLO model..."):
            detector = DetectionModel()
        
        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Could not open video file.")
            return
        
        with st.spinner("Processing video..."):
            total = _run_video(cap, detector, skip_frames=skip)
        
        st.success(f"✅ Processed {total} frames")