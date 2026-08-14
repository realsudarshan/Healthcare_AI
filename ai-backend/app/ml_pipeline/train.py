import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from ml_pipeline.dataset import CXRDataset

def train_model(train_manifest, val_manifest, output_dir, epochs=20, batch_size=32, lr=1e-4, patience=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Datasets and Loaders
    train_ds = CXRDataset(train_manifest, is_train=True)
    val_ds = CXRDataset(val_manifest, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Compute class weights
    y_train = train_ds.df['label'].values
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class weights: {weights}")

    # Model definition: DenseNet121
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    # Replace classifier
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)

    # Early stopping trackers
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = -1
    best_val_acc = 0.0
    
    history = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_acc': epoch_train_acc,
            'val_loss': epoch_val_loss,
            'val_acc': epoch_val_acc,
            'lr': current_lr
        })
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {current_lr}")
              
        scheduler.step(epoch_val_loss)

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Save best model
    model_save_path = output_dir / "best_model.pth"
    torch.save(best_model_state, model_save_path)
    
    # Save training history
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    
    # Save metadata JSON
    metadata = {
        "best_epoch": best_epoch,
        "val_loss": best_val_loss,
        "val_acc": best_val_acc,
        "class_mapping": {0: "Normal", 1: "Pneumonia", 2: "Tuberculosis"},
        "preprocessing": {
            "target_size": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": "cv2.INTER_AREA",
            "bit_depth_scaling": "min-max 8-bit stretch"
        }
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Training complete. Best model saved to {model_save_path}")

if __name__ == "__main__":
    dataset_base = Path(__file__).parent.parent.parent / "datasets"
    train_man = dataset_base / "train_manifest.csv"
    val_man = dataset_base / "val_manifest.csv"
    if train_man.exists() and val_man.exists():
        train_model(str(train_man), str(val_man), str(dataset_base))
    else:
        print("Manifests not found. Please run data engineering first.")
