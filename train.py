import os
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

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
data_file = f'{os.getenv('DATASET')}/data.yaml'
results = model.train(data=data_file, epochs=100, imgsz=640, device=device)
