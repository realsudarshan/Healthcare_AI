import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import os
import numpy as np

TEST_PATH = "final_dataset/test"
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch_10.pt"
BATCH_SIZE = 32

val_test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(TEST_PATH, transform=val_test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Loading model for ROC curves...")
model = models.resnet101()
model.fc = nn.Linear(model.fc.in_features, 3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)['model_state_dict'])
model = model.to(device)
model.eval()

y_true = []
y_score = []

print("Extracting probabilities on test set...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_score.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_score = np.array(y_score)
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
n_classes = 3

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, color, cls in zip(range(n_classes), colors, test_dataset.classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:0.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)

out_dir = r"C:\Users\sudar\.gemini\antigravity\brain\47a75e81-8372-4486-b739-fd982c3f56de\scratch"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
print("ROC Curve saved successfully!")
