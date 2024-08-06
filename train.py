import torch
from ultralytics import YOLO

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("\n\nGPU is available. Using CUDA for training.\n\n")
else:
    device = torch.device("cpu")
    print("\n\nGPU is not available. Using CPU for training.\n\n")

# Creacion del modelo
model = YOLO("yolov8n.yaml")

# Train the model
results = model.train(data='Planos-craneales-prenatales-2/data.yaml', epochs=1, imgsz=640, device=device)
