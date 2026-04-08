import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_ROOT = "dataset/raw/ds004584-download"
OUTPUT_ROOT = "dataset/processed"

LABEL_MAP = {
    "PD": "parkinsons",
    "CTL": "control",
    "Parkinson": "parkinsons",
    "Control": "control"
}

def safe_link(src, dest):
    """Attempts symlink, falls back to hard link for Windows Privilege Error 1314."""
    if os.path.exists(dest):
        return
    try:
        os.symlink(src, dest)
    except OSError as e:
        if getattr(e, 'winerror', None) == 1314:
            os.link(src, dest) 
        else:
            raise

def prepare_dataset():
    participants_path = os.path.join(DATASET_ROOT, "participants.tsv")
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"Missing {participants_path}")
        
    participants = pd.read_csv(participants_path, sep="\t")
    
    # Dynamically find the clinical label column
    label_col = None
    for col in participants.columns:
        if col.lower() in ["group", "diagnosis", "condition", "class"]:
            label_col = col
            break
            
    if not label_col:
        raise KeyError(f"Could not find label column. Found: {list(participants.columns)}")
        
    print(f"-> Using column '{label_col}' for labels.")

    subjects = []
    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        raw_label = str(row[label_col]).strip()
        
        group_label = None
        for key, val in LABEL_MAP.items():
            if key.lower() == raw_label.lower():
                group_label = val
                break
                
        if not group_label:
            continue
            
        eeg_file_path = os.path.join(DATASET_ROOT, subject_id, "eeg", f"{subject_id}_task-Rest_eeg.set")
        if os.path.exists(eeg_file_path):
            subjects.append({"path": eeg_file_path, "label": group_label})

    train, val = train_test_split(
        subjects, test_size=0.2, stratify=[s["label"] for s in subjects], random_state=42
    )

    def create_structure(data_list, split_name):
        for item in tqdm(data_list, desc=f"Linking {split_name}"):
            label_dir = os.path.join(OUTPUT_ROOT, split_name, item["label"])
            os.makedirs(label_dir, exist_ok=True)
            
            base_src = os.path.abspath(item["path"])
            link_dest_set = os.path.join(label_dir, os.path.basename(base_src))
            safe_link(base_src, link_dest_set)
                
            fdt_src = base_src.replace(".set", ".fdt")
            if os.path.exists(fdt_src):
                link_dest_fdt = os.path.join(label_dir, os.path.basename(fdt_src))
                safe_link(fdt_src, link_dest_fdt)

    create_structure(train, "train")
    create_structure(val, "val")
    print("\n[SUCCESS] Dataset structure built successfully!")

if __name__ == "__main__":
    prepare_dataset()