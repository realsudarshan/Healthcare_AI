import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings('ignore')

TEST_PATH = "final_dataset/test"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_10.pt"

val_test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(TEST_PATH)

print("Loading model for Grad-CAM...")
model = models.resnet101()
model.fc = nn.Linear(model.fc.in_features, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['model_state_dict'])
model = model.to(device)
model.eval()

# We attach Grad-CAM to the final convolutional block of ResNet101
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
found_classes = []
idx = 0

print("Generating Heatmaps...")
while len(found_classes) < 3 and idx < len(dataset):
    img_pil, label = dataset[idx]
    cls_name = dataset.classes[label]
    
    if cls_name not in found_classes:
        found_classes.append(cls_name)
        
        input_tensor = val_test_transforms(img_pil).unsqueeze(0).to(device)
        
        # Prepare original image for overlay
        img_np = np.array(img_pil.resize((512, 512))).astype(np.float32) / 255.0
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,)*3, axis=-1)
            
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
        ax_idx = len(found_classes) - 1
        axes[ax_idx].imshow(visualization)
        axes[ax_idx].set_title(f"AI Focus Area: {cls_name}", fontsize=16, fontweight='bold')
        axes[ax_idx].axis('off')
        
    idx += 1

out_dir = r"C:\Users\sudar\.gemini\antigravity\brain\47a75e81-8372-4486-b739-fd982c3f56de\scratch"
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "gradcam_heatmaps.png"), dpi=300, bbox_inches='tight')
print("Grad-CAM heatmaps saved successfully!")
