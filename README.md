# 🌿 UAE EcoVision

An environmental monitoring system for UAE satellite/aerial imagery. It runs a
four-stage computer-vision pipeline over an uploaded image and presents the
results in an interactive dashboard.

The system has two parts:

- **Backend** — a FastAPI service that loads the models once and exposes an
  `/analyze` endpoint.
- **Frontend** — a Streamlit app with an upload page and a results dashboard.

---

## Pipeline

| Stage | Model / Method | Output |
|-------|----------------|--------|
| **Segmentation** | SegFormer-B0 (fine-tuned) | Per-pixel land-cover mask (4 classes) + colored overlay |
| **Detection** | YOLOv8n (fine-tuned) | Object bounding boxes with class + confidence |
| **Change Detection** | OpenCV (absolute difference + morphology) | Red heatmap of changed regions vs. a baseline image |
| **Edge Detection** | OpenCV (class-boundary differencing) | Green outlines of land-cover boundaries |
| **Live Video** | OpenCV background subtraction (MOG2) | Moving-object detection with bounding boxes + live FPS |

The app has two input modes, selectable at the top of the page:

- **🖼️ Image Analysis** — runs the four-stage pipeline on an uploaded image.
- **🎥 Live Video** — processes a live webcam stream or an uploaded video file,
  detecting moving objects in real time.

### Land-cover classes

| ID | Class | Overlay color |
|----|-------|---------------|
| 0 | Vegetation | 🟢 green |
| 1 | Sand | 🟡 yellow |
| 2 | Water | 🔵 blue |
| 3 | Urban | 🔴 red |
| 255 | Ignore | ⬛ black |

---

## Project structure

```
ecovision/
├── backend/
│   ├── main.py                 # FastAPI app + /analyze endpoint
│   ├── models/                 # Trained weights (not in git)
│   │   ├── segformer-b0-final/
│   │   └── yolo-runs/ecovision-yolo/weights/best.pt
│   └── pipeline/
│       ├── segmentation.py     # SegFormer inference
│       ├── detection.py        # YOLOv8 inference
│       ├── change.py           # OpenCV change detection
│       └── edges.py            # OpenCV class-boundary edges
├── frontend/
│   ├── app.py                  # Streamlit entry point (2-column layout)
│   └── pages/
│       ├── upload.py           # Upload / camera input + Analyze button
│       └── dashboard.py        # Metrics, overlays, charts, JSON export
├── training/
│   ├── train_segformer_cpu.py  # SegFormer fine-tuning (LoveDA → UAE classes)
│   └── yolo/train_yolo.py      # YOLOv8 fine-tuning
└── requirements.txt
```

---

## Setup

Requires **Python 3.10+**.

```bash
# 1. (optional) create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt
```

The trained model weights are expected at:

- `backend/models/segformer-b0-final/`
- `backend/models/yolo-runs/ecovision-yolo/weights/best.pt`

These are large and excluded from git (see `.gitignore`). Retrain them with the
scripts in `training/` if they are missing.

---

## Running

The backend and frontend run as **two separate processes**. Start the backend
first.

```bash
# Terminal 1 — backend (http://localhost:8000)
uvicorn backend.main:app --port 8000 --reload

# Terminal 2 — frontend (http://localhost:8501)
streamlit run frontend/app.py
```

Then open the Streamlit URL, upload an image (optionally a baseline image for
change detection), and click **Analyze**.

> The frontend calls the backend at `http://localhost:8000/analyze`. If you
> change the backend port, update the URL in `frontend/pages/upload.py`.

---

## API

### `POST /analyze`

**Form fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | yes | Image to analyze |
| `baseline` | file | no | Reference image for change detection |
| `mode` | string | no | `full` (default), `segmentation`, `detection`, or `change` |

**Response (abridged)**

```json
{
  "filename": "scene.jpg",
  "image_size": [1024, 768],
  "mode": "full",
  "processing_time_ms": 842.5,
  "results": {
    "segmentation": { "classes_found": [0, 2, 3], "health_score": 41.2, "overlay_b64": "data:image/png;base64,..." },
    "edges":        { "edge_pixels": 18342, "overlay_b64": "..." },
    "detection":    { "num_objects": 3, "objects": [ { "x1": 50, "y1": 50, "x2": 150, "y2": 150, "class": "vegetation", "confidence": 0.85 } ] },
    "change":       { "has_baseline": false, "changed_pixels": 0, "overlay_b64": "..." }
  }
}
```

Other endpoints: `GET /` and `GET /health` for status checks.

---

## Training

**Segmentation** — `training/train_segformer_cpu.py` fine-tunes `nvidia/mit-b0`
on the LoveDA dataset, remapping its 7 classes onto the 4 UAE land-cover
classes, and saves to `backend/models/segformer-b0-final/`.

**Detection** — `training/yolo/train_yolo.py` fine-tunes `yolov8n.pt` from a
`data.yaml` and saves runs under `backend/models/yolo-runs/`.

---

## Troubleshooting

**Frontend shows "Cannot connect to backend"** — the FastAPI server isn't
running. Start it with the `uvicorn` command above.

**Segmentation overlay looks mostly one color** — the colors are rendered
correctly; an overlay dominated by one class means the SegFormer model is
predicting that class. This is a model-accuracy issue (limited training), not a
rendering bug. Retrain for longer / on more representative imagery to improve it.

**Change Detection section is hidden** — expected when no baseline image is
uploaded. Provide a baseline to enable it.

**`use_container_width` deprecation warnings** — the app uses the current
`width="stretch"` Streamlit API; ensure `streamlit>=1.49`.
