import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

TEST_PATH = "final_dataset/test"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_10.pt"
BATCH_SIZE = 32
IMG_SIZE = (512, 512)

# Transforms (must match testing)
val_test_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading test dataset...")
test_dataset = datasets.ImageFolder(TEST_PATH, transform=val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load Model
print("Building model architecture...")
model = models.resnet101()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading weights to {device}...")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['model_state_dict'])
model = model.to(device)
model.eval()

y_true = []
y_pred = []

print("Running inference on test set (this may take a minute)...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n=== Classification Report ===")
report = classification_report(y_true, y_pred, target_names=test_dataset.classes)
print(report)

print("Generating confusion matrix plot...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=test_dataset.classes, 
            yticklabels=test_dataset.classes)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

out_dir = r"C:\Users\sudar\.gemini\antigravity\brain\47a75e81-8372-4486-b739-fd982c3f56de\scratch"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
print(f"Saved confusion matrix to {os.path.join(out_dir, 'confusion_matrix.png')}")
