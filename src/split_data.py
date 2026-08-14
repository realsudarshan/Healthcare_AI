import os
import glob
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict
from tqdm import tqdm

FINAL_DIR = "final_dataset"
CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
SPLITS = ["train", "val", "test"]

# Create target directories
for split in SPLITS:
    for cls in CLASSES:
        os.makedirs(os.path.join(FINAL_DIR, split, cls), exist_ok=True)

for cls in CLASSES:
    print(f"Splitting class: {cls}")
    class_dir = os.path.join(FINAL_DIR, "combined", cls)
    
    # Group by source
    source_to_files = defaultdict(list)
    for img_path in glob.glob(os.path.join(class_dir, "*.png")):
        basename = os.path.basename(img_path)
        source = basename.split("_")[0]
        source_to_files[source].append(img_path)
        
    for source, files in source_to_files.items():
        print(f"  Source: {source} - {len(files)} images")
        
        # If too few images for split, just put in train
        if len(files) < 3:
            for f in files:
                shutil.copy(f, os.path.join(FINAL_DIR, "train", cls, os.path.basename(f)))
            continue
            
        # Split 70/15/15
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        
        if len(temp_files) < 2:
            val_files = temp_files
            test_files = []
        else:
            val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
            
        # Copy files
        for f in train_files:
            shutil.copy(f, os.path.join(FINAL_DIR, "train", cls, os.path.basename(f)))
        for f in val_files:
            shutil.copy(f, os.path.join(FINAL_DIR, "val", cls, os.path.basename(f)))
        for f in test_files:
            shutil.copy(f, os.path.join(FINAL_DIR, "test", cls, os.path.basename(f)))

print("Data splitting complete!")
