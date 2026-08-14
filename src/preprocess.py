import os
import cv2
import glob
import imagehash
from PIL import Image
from tqdm import tqdm
import shutil

# Target directory for combined data
FINAL_DIR = "final_dataset"
CLASSES = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]

# Create combined dataset structure
for cls in CLASSES:
    os.makedirs(os.path.join(FINAL_DIR, "combined", cls), exist_ok=True)

# Helper function to process and save a single image
def process_and_save(src_path, dest_dir, filename_prefix, target_class):
    try:
        # Read with unchanged bit depth
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
        
        # Normalize to 8-bit if not
        if img.dtype != 'uint8':
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
        # Convert to grayscale if it's RGB/BGR
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This standardizes contrast across all different X-ray machines
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
            
        # Resize to 512x512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Determine output filename
        basename = os.path.basename(src_path)
        name, _ = os.path.splitext(basename)
        out_name = f"{filename_prefix}_{name}.png"
        out_path = os.path.join(FINAL_DIR, "combined", target_class, out_name)
        
        # Save as PNG
        cv2.imwrite(out_path, img)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

# 1. Process jtiptj (Kaggle) dataset
print("Processing jtiptj (Kaggle) dataset...")
for split in ["train", "test", "val"]:
    for cls in CLASSES:
        # Original kaggle dataset has a typo in one folder: TURBERCULOSIS
        orig_cls = "TURBERCULOSIS" if cls == "TUBERCULOSIS" else cls
        src_dir = os.path.join("raw_data", "jtiptj", split, orig_cls)
        
        if not os.path.exists(src_dir):
            continue
            
        for img_path in tqdm(glob.glob(os.path.join(src_dir, "*.*"))):
            process_and_save(img_path, FINAL_DIR, f"jtiptj_{split}", cls)

# 2. Process Rahman et al (dataset1)
print("Processing Rahman et al dataset...")
rahman_dir = os.path.join("raw_data", "dataset1", "TB_Chest_Radiography_Database")
# Normal
for img_path in tqdm(glob.glob(os.path.join(rahman_dir, "Normal", "*.*"))):
    process_and_save(img_path, FINAL_DIR, "rahman", "NORMAL")
# Tuberculosis
for img_path in tqdm(glob.glob(os.path.join(rahman_dir, "Tuberculosis", "*.*"))):
    process_and_save(img_path, FINAL_DIR, "rahman", "TUBERCULOSIS")

# 3. Process Shenzhen (dataset2)
print("Processing Shenzhen dataset...")
shenzhen_dir = os.path.join("raw_data", "dataset2", "images")
if os.path.exists(shenzhen_dir):
    for img_path in tqdm(glob.glob(os.path.join(shenzhen_dir, "*.*"))):
        name = os.path.basename(img_path)
        if name.endswith("_0.png"):
            process_and_save(img_path, FINAL_DIR, "shenzhen", "NORMAL")
        elif name.endswith("_1.png"):
            process_and_save(img_path, FINAL_DIR, "shenzhen", "TUBERCULOSIS")

# 4. Process Mendeley Pakistan (dataset3)
print("Processing Mendeley Pakistan dataset...")
mendeley_dir = os.path.join("raw_data", "dataset3", "Dataset of Tuberculosis Chest X-rays Images")
# Normal
for img_path in tqdm(glob.glob(os.path.join(mendeley_dir, "Normal Chest X-rays", "*.*"))):
    process_and_save(img_path, FINAL_DIR, "mendeley", "NORMAL")
# Tuberculosis
for img_path in tqdm(glob.glob(os.path.join(mendeley_dir, "TB Chest X-rays", "*.*"))):
    process_and_save(img_path, FINAL_DIR, "mendeley", "TUBERCULOSIS")

print("Deduplication phase...")
# Deduplication using perceptual hashing
for cls in CLASSES:
    print(f"Deduplicating class: {cls}")
    class_dir = os.path.join(FINAL_DIR, "combined", cls)
    hashes = {}
    duplicates_removed = 0
    
    files = glob.glob(os.path.join(class_dir, "*.png"))
    files.sort(key=lambda x: (1 if "jtiptj" in os.path.basename(x) else 0, x))
    
    for img_path in tqdm(files):
        try:
            img = Image.open(img_path)
            h = str(imagehash.phash(img))
            if h in hashes:
                os.remove(img_path)
                duplicates_removed += 1
            else:
                hashes[h] = img_path
        except Exception as e:
            print(f"Error hashing {img_path}: {e}")
            
    print(f"Removed {duplicates_removed} duplicates from {cls}")

print("Preprocessing complete. Data is in final_dataset/combined/")
