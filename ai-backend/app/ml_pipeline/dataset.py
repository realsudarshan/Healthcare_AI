import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from ml_pipeline.data_engineering import harmonize_image

class CXRDataset(Dataset):
    def __init__(self, manifest_path, is_train=False, target_size=(224, 224)):
        """
        Args:
            manifest_path: path to the CSV manifest (train/val/test).
            is_train: Boolean to toggle data augmentation.
            target_size: default resize size.
        """
        self.df = pd.read_csv(manifest_path)
        self.is_train = is_train
        self.target_size = target_size
        
        # ImageNet normalization stats
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        
        # Data augmentation for training
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                self.normalize
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        label = row['label']

        # Harmonize on the fly
        canvas = harmonize_image(img_path, target_size=self.target_size)
        
        if canvas is None:
            # Fallback for missing/corrupt image
            canvas = np.zeros(self.target_size, dtype=np.uint8)
        
        # Convert 1-channel grayscale to 3-channel RGB representation
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        
        # Apply PyTorch transforms (which include ToTensor that scales to [0, 1] if input is numpy uint8)
        tensor_img = self.transform(img_rgb)
        
        return tensor_img, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Simple test logic for the assertion required in Step 2
    import tempfile
    
    # Create dummy manifest
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("image_path,label\ndummy.png,0")
        dummy_path = f.name
        
    ds_train = CXRDataset(dummy_path, is_train=True)
    ds_test = CXRDataset(dummy_path, is_train=False)
    
    # Assert val/test pipeline contains no random transforms
    assert not any(isinstance(t, (transforms.RandomRotation, transforms.RandomHorizontalFlip)) 
                   for t in ds_test.transform.transforms), "Test transform pipeline contains random augmentations!"
    assert any(isinstance(t, (transforms.RandomRotation, transforms.RandomHorizontalFlip)) 
                   for t in ds_train.transform.transforms), "Train transform pipeline is missing random augmentations!"
                   
    print("Dataset logic and augmentation assertions passed.")
    os.remove(dummy_path)
