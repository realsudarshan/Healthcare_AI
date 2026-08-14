import os
import json
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path

# Setup router
router = APIRouter(prefix="/api/cxr", tags=["CXR Classification"])

# Load model and metadata globally at startup
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "datasets" # Using the datasets folder where best_model.pth was saved
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CXRPredictionResponse(BaseModel):
    predicted_class: str
    probabilities: dict[str, float]
    gradcam_base64: str
    disclaimer: str = "Research prototype, not a diagnostic device."

# Globals
model = None
metadata = None
target_layer = None

def load_resources():
    global model, metadata, target_layer
    meta_path = MODEL_DIR / "model_metadata.json"
    weights_path = MODEL_DIR / "best_model.pth"
    
    if not meta_path.exists() or not weights_path.exists():
        print("Warning: Model or metadata not found. CXR classifier endpoint will return 503 until trained.")
        return
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    target_layer = model.features.norm5

# Run loading
load_resources()

def preprocess_image(file_bytes, target_size=(224, 224)):
    """Preprocess identical to training data engineering script."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Invalid image file.")
        
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # Min-max stretch to 8-bit
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_8bit = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        img_8bit = img.astype(np.uint8)

    # Aspect ratio preserving resize
    h, w = img_8bit.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img_8bit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to target_size
    canvas = np.zeros(target_size, dtype=np.uint8)
    y_offs = (target_size[0] - new_h) // 2
    x_offs = (target_size[1] - new_w) // 2
    canvas[y_offs:y_offs+new_h, x_offs:x_offs+new_w] = resized
    
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    
    # ImageNet normalization
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (img_float - mean) / std
    
    tensor_img = torch.from_numpy(normalized).permute(2, 0, 1)
    return tensor_img

def get_gradcam(input_tensor, pred_label):
    activations = []
    def hook_fn(module, input, output):
        activations.append(output)
    hook = target_layer.register_forward_hook(hook_fn)
    
    gradients = []
    def backward_hook_fn(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    bw_hook = target_layer.register_full_backward_hook(backward_hook_fn)
    
    out = model(input_tensor)
    target_score = out[0, pred_label]
    model.zero_grad()
    target_score.backward()
    
    hook.remove()
    bw_hook.remove()
    
    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    for j in range(activations[0].shape[1]):
        activations[0][:, j, :, :] *= pooled_gradients[j]
        
    heatmap = torch.mean(activations[0], dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
    heatmap = heatmap.cpu().numpy()
    
    return heatmap

@router.post("/classify", response_model=CXRPredictionResponse)
async def classify_cxr(file: UploadFile = File(...)):
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
        
    if not file.content_type.startswith("image/") and file.content_type != "application/dicom":
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    contents = await file.read()
    try:
        tensor_img = preprocess_image(contents, target_size=tuple(metadata['preprocessing']['target_size']))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
    input_tensor = tensor_img.unsqueeze(0).to(device)
    
    with torch.enable_grad():
        out = model(input_tensor)
        probs = torch.softmax(out, dim=1).detach()[0].cpu().numpy()
        pred_label = int(np.argmax(probs))
    
        heatmap = get_gradcam(input_tensor, pred_label)
    
    # Format probabilities
    prob_dict = {metadata['class_mapping'][str(i)]: float(probs[i]) for i in range(3)}
    predicted_class = metadata['class_mapping'][str(pred_label)]
    
    # Overlay heatmap on original image for base64
    orig_img = tensor_img.permute(1, 2, 0).cpu().numpy()
    mean = np.array(metadata['preprocessing']['mean'])
    std = np.array(metadata['preprocessing']['std'])
    orig_img = std * orig_img + mean
    orig_img = np.clip(orig_img, 0, 1)
    orig_img = (orig_img * 255).astype(np.uint8)
    
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
    
    _, buffer = cv2.imencode('.jpg', superimposed)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    
    return CXRPredictionResponse(
        predicted_class=predicted_class,
        probabilities=prob_dict,
        gradcam_base64=b64_str
    )

if __name__ == "__main__":
    # 5. Preprocessing parity test
    print("Running preprocessing parity test...")
    from ml_pipeline.dataset import CXRDataset
    from ml_pipeline.data_engineering import harmonize_image
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
    _, buffer = cv2.imencode('.png', dummy_img)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(buffer.tobytes())
        dummy_path = f.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
        f2.write(f"image_path,label\n{dummy_path},0")
        dummy_manifest = f2.name
        
    try:
        # Inference pipeline transform
        inf_tensor = preprocess_image(buffer.tobytes(), target_size=(224, 224))
        
        # Training pipeline transform
        ds = CXRDataset(dummy_manifest, is_train=False)
        train_tensor, _ = ds[0]
        
        diff = torch.abs(inf_tensor - train_tensor).max().item()
        assert diff < 1e-4, f"Parity mismatch: max difference {diff}"
        print("Preprocessing parity test passed!")
    finally:
        os.remove(dummy_path)
        os.remove(dummy_manifest)
