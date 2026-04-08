import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATASET_ROOT = "dataset/raw/ds004584-download"
OUTPUT_ROOT = "dataset/processed"
LABEL_MAP = {"PD": "parkinsons", "CTL": "control"}

def prepare_dataset():
    participants_path = os.path.join(DATASET_ROOT, "participants.tsv")
    if not os.path.exists(participants_path):
        raise FileNotFoundError(f"Missing {participants_path}")
        
    participants = pd.read_csv(participants_path, sep="\t")
    subjects = []
    
    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        group_label = LABEL_MAP.get(row["Group"])
        
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
            files_to_link = [base_src]
            fdt_src = base_src.replace(".set", ".fdt")
            if os.path.exists(fdt_src):
                files_to_link.append(fdt_src)
            
            for src in files_to_link:
                link_dest = os.path.join(label_dir, os.path.basename(src))
                if not os.path.exists(link_dest):
                    os.symlink(src, link_dest)

    create_structure(train, "train")
    create_structure(val, "val")

if __name__ == "__main__":
    prepare_dataset()