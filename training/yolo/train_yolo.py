from ultralytics import YOLO

print("Loading YOLOv8n pretrained model...")
model = YOLO("yolov8n.pt")

print("Starting training on UAE vegetation dataset...")
model.train(
    data="/Users/sampk/ecovision/data/yolo/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="ecovision-yolo",
    project="/Users/sampk/ecovision/backend/models/yolo-runs",
    device="cpu",
    patience=10,
    save=True,
    plots=True,
)

print("Exporting to ONNX...")
model.export(format="onnx")
print("Training complete. Model saved.")
