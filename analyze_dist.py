import os
import glob
from collections import defaultdict

FINAL_DIR = "final_dataset"
CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

print("=== Dataset Source Distribution ===")
for cls in CLASSES:
    class_dir = os.path.join(FINAL_DIR, "combined", cls)
    if not os.path.exists(class_dir):
        print(f"Directory missing: {class_dir}")
        continue
        
    source_to_files = defaultdict(list)
    for img_path in glob.glob(os.path.join(class_dir, "*.png")):
        basename = os.path.basename(img_path)
        source = basename.split("_")[0]
        source_to_files[source].append(img_path)
    
    print(f"\nClass: {cls}")
    for source, files in sorted(source_to_files.items()):
        print(f"  Source: {source} - {len(files)} images")
