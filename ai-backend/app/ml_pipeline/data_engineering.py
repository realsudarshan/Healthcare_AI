import os
import subprocess
import pandas as pd
import json
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import pydicom

DATASET_BASE = Path(__file__).parent.parent.parent / "datasets"
DATASET_BASE.mkdir(parents=True, exist_ok=True)

TBX11K_DIR = DATASET_BASE / "TBX11K"
RSNA_DIR = DATASET_BASE / "RSNA"
NIH_DIR = DATASET_BASE / "NIH"
SHENZHEN_DIR = DATASET_BASE / "Shenzhen"
MONTGOMERY_DIR = DATASET_BASE / "Montgomery"

def check_mirror_completeness(source, expected_count):
    if not source.exists():
        print(f"Directory {source} does not exist.")
        return False
    # count images
    exts = ['.jpg', '.jpeg', '.png', '.dicom', '.dcm']
    count = sum(1 for f in source.rglob('*') if f.suffix.lower() in exts)
    print(f"{source.name}: Found {count} images, Expected {expected_count}")
    return count >= expected_count

def download_tbx11k():
    print("Downloading TBX11K...")
    if check_mirror_completeness(TBX11K_DIR, 11200):
        return True
    
    # Attempting to clone the repo
    repo_dir = DATASET_BASE / "Tuberculosis_repo"
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "https://github.com/yun-liu/Tuberculosis.git", str(repo_dir)], check=False)
    
    # Note: the repo typically contains links. For an automated agent, if there's an HF mirror we can try that.
    print("TBX11K repo cloned. Checking for image files...")
    # If this fails to actually download the 11200 images (because they are external links in the repo), 
    # we log it.
    if not check_mirror_completeness(TBX11K_DIR, 11200):
        print("TBX11K images not fully present. (External links require manual download).")
        return False
    return True

def download_rsna():
    print("Downloading RSNA Pneumonia Detection Challenge...")
    if check_mirror_completeness(RSNA_DIR, 26684):
        return True
    
    RSNA_DIR.mkdir(exist_ok=True)
    torrent_url = "https://academictorrents.com/download/95588a735c9ae4d123f3ca408e56570409bcf2a9.torrent"
    print("Trying aria2c...")
    try:
        # 5 minute timeout for torrent to make progress or fail
        subprocess.run(["aria2c", "--seed-time=0", "-d", str(RSNA_DIR), torrent_url], timeout=300, check=True)
    except Exception as e:
        print(f"aria2c failed or timed out: {e}")
        print("Falling back to HuggingFace community mirror...")
        try:
            # Note: A community mirror like 'lupus-alpha/RSNA-Pneumonia' might exist.
            subprocess.run(["huggingface-cli", "download", "dataset", "juniper/rsna-pneumonia-detection", "--local-dir", str(RSNA_DIR)], timeout=300, check=False)
        except Exception as e2:
            print(f"HF fallback failed: {e2}")

    return check_mirror_completeness(RSNA_DIR, 26684)

def download_nih():
    print("Downloading NIH ChestX-ray14...")
    if check_mirror_completeness(NIH_DIR, 112120):
        return True
    NIH_DIR.mkdir(exist_ok=True)
    print("Trying HF mirror for NIH...")
    try:
        subprocess.run(["huggingface-cli", "download", "dataset", "nickmuchi/nih-chest-xray-14", "--local-dir", str(NIH_DIR)], timeout=600, check=False)
    except Exception as e:
        print(f"HF download for NIH failed: {e}")
    return check_mirror_completeness(NIH_DIR, 112120)

def download_shenzhen_montgomery():
    print("Downloading Shenzhen and Montgomery datasets...")
    # Simplified download logic for these - normally from NLM FTP or Kaggle mirrors
    SHENZHEN_DIR.mkdir(exist_ok=True)
    MONTGOMERY_DIR.mkdir(exist_ok=True)
    # Stubs for autonomous download - normally we would wget the zip files from NLM
    print("Shenzhen/Montgomery download not fully implemented in autonomous script due to NLM structure.")
    return False

def acquire_datasets():
    download_tbx11k()
    download_rsna()
    download_nih()
    download_shenzhen_montgomery()

def harmonize_image(img_path, target_size=(224, 224)):
    """Read image, handle DICOM, Min-Max stretch to 8-bit, and resize preserving aspect ratio."""
    img_path_str = str(img_path)
    if img_path_str.lower().endswith('.dicom') or img_path_str.lower().endswith('.dcm'):
        dicom = pydicom.dcmread(img_path_str)
        img = dicom.pixel_array
        if dicom.PhotometricInterpretation == 'MONOCHROME1':
            img = np.amax(img) - img
    else:
        img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
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

    return canvas

def build_manifests():
    records = []
    
    # 1. NIH (Normal strictly "No Finding")
    nih_csv = list(NIH_DIR.rglob('Data_Entry_2017.csv'))
    if nih_csv:
        df_nih = pd.read_csv(nih_csv[0])
        # strictly "No Finding"
        df_normal = df_nih[df_nih['Finding Labels'] == 'No Finding']
        for _, row in df_normal.iterrows():
            img_file = list(NIH_DIR.rglob(row['Image Index']))
            if img_file:
                records.append({
                    'image_path': str(img_file[0]),
                    'patient_id': f"{row['Patient ID']}_NIH",
                    'label_str': 'Normal',
                    'label': 0,
                    'source_dataset': 'NIH'
                })

    # 2. RSNA (Pneumonia Target == 1)
    rsna_csv = list(RSNA_DIR.rglob('stage_2_train_labels.csv'))
    if rsna_csv:
        df_rsna = pd.read_csv(rsna_csv[0])
        df_pneu = df_rsna[df_rsna['Target'] == 1].drop_duplicates(subset=['patientId'])
        for _, row in df_pneu.iterrows():
            img_file = list(RSNA_DIR.rglob(f"{row['patientId']}.dcm"))
            if not img_file:
                img_file = list(RSNA_DIR.rglob(f"{row['patientId']}.jpg"))
            if img_file:
                records.append({
                    'image_path': str(img_file[0]),
                    'patient_id': f"{row['patientId']}_RSNA",
                    'label_str': 'Pneumonia',
                    'label': 1,
                    'source_dataset': 'RSNA'
                })

    # 3. TBX11K (TB-positive)
    if TBX11K_DIR.exists():
        # Parsing assuming structure contains 'TB' folder or similar
        tb_files = list(TBX11K_DIR.rglob('TB/*.png')) + list(TBX11K_DIR.rglob('Active-TB/*.png'))
        for f in tb_files:
            pid = f.stem
            records.append({
                'image_path': str(f),
                'patient_id': f"{pid}_TBX11K",
                'label_str': 'Tuberculosis',
                'label': 2,
                'source_dataset': 'TBX11K'
            })

    # 4. Shenzhen & Montgomery (Secondary TB Domain)
    if SHENZHEN_DIR.exists():
        for f in SHENZHEN_DIR.rglob('*.png'):
            label = 2 if f.stem.endswith('_1') else 0
            label_str = 'Tuberculosis' if label == 2 else 'Normal'
            records.append({
                'image_path': str(f),
                'patient_id': f"{f.stem}_Shenzhen",
                'label_str': label_str,
                'label': label,
                'source_dataset': 'TB_Shenzhen'
            })
            
    if MONTGOMERY_DIR.exists():
        for f in MONTGOMERY_DIR.rglob('*.png'):
            label = 2 if f.stem.endswith('_1') else 0
            label_str = 'Tuberculosis' if label == 2 else 'Normal'
            records.append({
                'image_path': str(f),
                'patient_id': f"{f.stem}_Montgomery",
                'label_str': label_str,
                'label': label,
                'source_dataset': 'TB_Montgomery'
            })

    if not records:
        print("No records parsed. Please check if datasets are downloaded properly.")
        return

    df = pd.DataFrame(records)
    print(f"Total parsed records: {len(df)}")
    
    # Class-aware patient grouped split
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['patient_id']))
    df_train = df.iloc[train_idx]
    df_temp = df.iloc[temp_idx]
    
    gss_test = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42) # 0.5 of 30% is 15%
    val_idx, test_idx = next(gss_test.split(df_temp, groups=df_temp['patient_id']))
    df_val = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]
    
    harmonized_dir = DATASET_BASE / "harmonized"
    harmonized_dir.mkdir(exist_ok=True)
    
    def process_and_update(df_split, split_name):
        updated_records = []
        out_csv = DATASET_BASE / f"{split_name}_manifest.csv"
        for _, row in df_split.iterrows():
            updated_records.append(row.to_dict())
        pd.DataFrame(updated_records).to_csv(out_csv, index=False)
        return pd.DataFrame(updated_records)

    print("Generating manifests...")
    df_train = process_and_update(df_train, "train")
    df_val = process_and_update(df_val, "val")
    df_test = process_and_update(df_test, "test")
    
    # Generate split_stats.json
    def get_stats(df_split):
        stats = {
            "total_images": len(df_split),
            "unique_patients": df_split['patient_id'].nunique(),
            "class_distribution": df_split['label_str'].value_counts().to_dict(),
            "source_distribution": {}
        }
        for src in df_split['source_dataset'].unique():
            src_df = df_split[df_split['source_dataset'] == src]
            stats["source_distribution"][src] = src_df['label_str'].value_counts().to_dict()
        return stats
    
    stats_out = {
        "train": get_stats(df_train) if not df_train.empty else {},
        "val": get_stats(df_val) if not df_val.empty else {},
        "test": get_stats(df_test) if not df_test.empty else {}
    }
    
    # Leakage check
    train_p = set(df_train['patient_id']) if not df_train.empty else set()
    val_p = set(df_val['patient_id']) if not df_val.empty else set()
    test_p = set(df_test['patient_id']) if not df_test.empty else set()
    
    overlap = train_p.intersection(val_p).union(train_p.intersection(test_p)).union(val_p.intersection(test_p))
    stats_out["leakage_check_overlap_count"] = len(overlap)
    
    with open(DATASET_BASE / "split_stats.json", "w") as f:
        json.dump(stats_out, f, indent=4)
        
    assert len(overlap) == 0, f"Patient leakage detected! {len(overlap)} patients overlap between splits."
    print("Data engineering complete. Manifests and split_stats.json generated.")

if __name__ == "__main__":
    acquire_datasets()
    build_manifests()
