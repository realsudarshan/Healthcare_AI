import io
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from torchvision import models, transforms

router = APIRouter(tags=["Legacy X-Ray Prediction"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "checkpoints", "checkpoint_epoch_10.pt")
CLASSES = ["Normal", "Pneumonia", "Tuberculosis"]

print("Initializing PyTorch ResNet101...")
model = models.resnet101()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device)["model_state_dict"]
    )
    print(f"Model loaded successfully on {device}")
else:
    print(
        f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}! Predictions will be random."
    )

model = model.to(device)
model.eval()

val_test_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@router.post("/predict")
async def predict_xray(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img_clahe = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        image_clahe = Image.fromarray(img_clahe)

        input_tensor = val_test_transforms(image_clahe).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)

        prediction = CLASSES[predicted.item()]
        confidence = float(probs[predicted.item()].cpu().numpy())

        return {
            "prediction": prediction,
            "confidence": confidence,
            "all_probs": {
                cls: float(prob) for cls, prob in zip(CLASSES, probs.cpu().numpy())
            },
        }
    except Exception as error:
        return {"error": str(error)}
