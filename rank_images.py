"""
Quick helper: run the real segmentation + detection models on every image in
the project and report the true land-cover breakdown for each.

Use it to pick the best (and honest) image for report/presentation screenshots.

Run from the project root:
    python rank_images.py
"""

import glob
import numpy as np
from PIL import Image

from backend.pipeline.segmentation import SegmentationModel
from backend.pipeline.detection import DetectionModel

CLASS_NAMES = {0: "Vegetation", 1: "Sand", 2: "Water", 3: "Urban"}


def main():
    images = sorted(
        glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
    )
    if not images:
        print("No images found in the project root.")
        return

    print("Loading models...")
    seg = SegmentationModel()
    det = DetectionModel()

    rows = []
    for path in images:
        img = Image.open(path).convert("RGB")
        mask = seg.predict(img)

        valid = mask != 255
        total = max(int(np.sum(valid)), 1)
        dist = {
            CLASS_NAMES.get(int(c), str(c)): round(float(np.sum(mask == c)) / total * 100, 1)
            for c in np.unique(mask) if c != 255
        }
        veg = dist.get("Vegetation", 0.0)

        try:
            n_obj = det.predict(img).get("num_objects", 0)
        except Exception:
            n_obj = "?"

        rows.append((path, veg, n_obj, dist))

    # Rank by vegetation %, highest first
    rows.sort(key=lambda r: r[1], reverse=True)

    print("\n" + "=" * 60)
    print("RANKED BY VEGETATION %  (best screenshot candidate first)")
    print("=" * 60)
    for path, veg, n_obj, dist in rows:
        print(f"\n{path}")
        print(f"  Vegetation: {veg}%   |   objects detected: {n_obj}")
        print(f"  Full land-cover: {dist}")

    best = rows[0]
    print("\n" + "-" * 60)
    print(f"Suggested hero image: {best[0]}  ({best[1]}% vegetation)")
    print("-" * 60)


if __name__ == "__main__":
    main()
