import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from ml_pipeline.dataset import CXRDataset

def evaluate_model(test_manifest, model_dir):
    model_dir = Path(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata
    with open(model_dir / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load model
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 3)
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    test_ds = CXRDataset(test_manifest, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    print("Running evaluation on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 4.1 Standard Metrics
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Pneumonia", "Tuberculosis"], yticklabels=["Normal", "Pneumonia", "Tuberculosis"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(model_dir / 'confusion_matrix.png')
    plt.close()
    
    def calculate_metrics(y_true, y_pred, y_prob):
        metrics = {}
        for i, class_name in metadata['class_mapping'].items():
            class_idx = int(i)
            y_true_binary = (y_true == class_idx).astype(int)
            y_pred_binary = (y_pred == class_idx).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # ROC AUC
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, class_idx])
            roc_auc = auc(fpr, tpr)
            
            metrics[class_name] = {
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "f1": float(f1),
                "auc": float(roc_auc)
            }
        return metrics

    pool_metrics = calculate_metrics(all_labels, all_preds, all_probs)
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(pool_metrics, f, indent=4)
        
    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    for i, class_name in metadata['class_mapping'].items():
        class_idx = int(i)
        y_true_binary = (all_labels == class_idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, all_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(model_dir / 'roc_curves.png')
    plt.close()

    # 4.2 Domain-shift stratified evaluation
    stratified_metrics = {}
    sources = test_ds.df['source_dataset'].unique()
    for src in sources:
        src_mask = (test_ds.df['source_dataset'] == src).values
        if np.sum(src_mask) == 0:
            continue
        src_labels = all_labels[src_mask]
        src_preds = all_preds[src_mask]
        src_probs = all_probs[src_mask]
        stratified_metrics[src] = calculate_metrics(src_labels, src_preds, src_probs)
        
    with open(model_dir / "stratified_metrics.json", "w") as f:
        json.dump(stratified_metrics, f, indent=4)

    # 4.3 Confidence intervals on minority-class (TB) metrics
    n_iterations = 1000
    n_size = len(all_labels)
    tb_idx = 2
    tb_stats = {'sensitivity': [], 'specificity': [], 'f1': []}
    
    print("Computing Bootstrap CIs...")
    for _ in range(n_iterations):
        indices = np.random.choice(n_size, size=n_size, replace=True)
        resampled_true = all_labels[indices]
        resampled_pred = all_preds[indices]
        
        y_true_binary = (resampled_true == tb_idx).astype(int)
        y_pred_binary = (resampled_pred == tb_idx).astype(int)
        if len(np.unique(y_true_binary)) < 2:
            continue
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        tb_stats['sensitivity'].append(sens)
        tb_stats['specificity'].append(spec)
        tb_stats['f1'].append(f1)
        
    bootstrap_ci = {}
    for metric_name, values in tb_stats.items():
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        bootstrap_ci[metric_name] = {"2.5%": float(lower), "97.5%": float(upper)}
        
    with open(model_dir / "bootstrap_ci.json", "w") as f:
        json.dump(bootstrap_ci, f, indent=4)

    # 4.4 Grad-CAM Generation
    generate_gradcam(model, test_ds, all_labels, all_preds, device, model_dir)
    print("Evaluation complete.")

def generate_gradcam(model, test_ds, all_labels, all_preds, device, output_dir):
    model.eval()
    
    # Select 5 correct and 5 incorrect
    correct_indices = np.where(all_labels == all_preds)[0]
    incorrect_indices = np.where(all_labels != all_preds)[0]
    
    np.random.shuffle(correct_indices)
    np.random.shuffle(incorrect_indices)
    
    selected_indices = list(correct_indices[:5]) + list(incorrect_indices[:5])
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Target layer for DenseNet121: model.features.norm5 (last batch-norm before global pooling)
    target_layer = model.features.norm5
    
    for i, idx in enumerate(selected_indices):
        tensor_img, true_label = test_ds[idx]
        input_tensor = tensor_img.unsqueeze(0).to(device)
        pred_label = all_preds[idx]
        
        # Forward hook
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)
        hook = target_layer.register_forward_hook(hook_fn)
        
        # Backward hook for gradients
        gradients = []
        def backward_hook_fn(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        bw_hook = target_layer.register_full_backward_hook(backward_hook_fn)
        
        out = model(input_tensor)
        
        # Target specific class for backprop
        target_score = out[0, pred_label]
        model.zero_grad()
        target_score.backward()
        
        hook.remove()
        bw_hook.remove()
        
        # Calculate Grad-CAM
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        for j in range(activations[0].shape[1]):
            activations[0][:, j, :, :] *= pooled_gradients[j]
            
        heatmap = torch.mean(activations[0], dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else 1
        heatmap = heatmap.cpu().numpy()
        
        # Unnormalize original image for visualization
        orig_img = tensor_img.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_img = std * orig_img + mean
        orig_img = np.clip(orig_img, 0, 1)
        orig_img = (orig_img * 255).astype(np.uint8)
        
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
        
        axes[i].imshow(superimposed)
        class_mapping = {0: "Normal", 1: "Pneumonia", 2: "TB"}
        axes[i].set_title(f"True: {class_mapping[true_label.item()]}\nPred: {class_mapping[pred_label]}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam_grid.png")
    plt.close()

if __name__ == "__main__":
    dataset_base = Path(__file__).parent.parent.parent / "datasets"
    test_man = dataset_base / "test_manifest.csv"
    if test_man.exists():
        evaluate_model(str(test_man), str(dataset_base))
    else:
        print("Test manifest not found.")
