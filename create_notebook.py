import json
import os

def create_jupyter_notebook(filename="EEG_Parkinsons_Pipeline.ipynb"):
    cells = []

    def add_markdown(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split('\n')]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split('\n')]
        })

    # --- CELL 1: Title ---
    add_markdown("""
    # 🧠 NeuroAI: Parkinson's Disease EEG Classifier
    **End-to-End Clinical Machine Learning Pipeline**
    This notebook consolidates data processing, ResNet18 training, Grad-CAM Explainable AI (XAI), and clinical PDF report generation.
    """)

    # --- CELL 2: Installations ---
    add_markdown("### 1. Install Dependencies")
    add_code("""
    !pip install mne torch torchvision torchaudio monai scikit-learn matplotlib seaborn scipy fpdf grad-cam opencv-python pandas
    """)

    # --- CELL 3: Imports & Global Configs ---
    add_markdown("### 2. Imports & Configuration")
    add_code("""
    import os
    import glob
    import json
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms.functional as F
    import numpy as np
    import pandas as pd
    import mne
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.signal import spectrogram
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import GradScaler, autocast
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
    from fpdf import FPDF
    from datetime import datetime
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from monai.transforms import Compose, RandGaussianNoise, RandCoarseDropout, RandScaleIntensity, EnsureType
    from tqdm.notebook import tqdm

    # Configurations
    DATASET_ROOT = "dataset/raw/ds004584-download"
    OUTPUT_ROOT = "dataset/processed"
    BATCH_SIZE = 4
    NUM_WORKERS = 0 # Prevent RAM spikes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")
    """)

    # --- CELL 4: Data Builder ---
    add_markdown("### 3. Data Builder (BIDS to Processed Split)")
    add_code('''
    LABEL_MAP = {"PD": "parkinsons", "CTL": "control", "Parkinson": "parkinsons", "Control": "control"}

    def safe_link(src, dest):
        if os.path.exists(dest): return
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
            print(f"Skipping dataset build: Missing {participants_path}")
            return
            
        participants = pd.read_csv(participants_path, sep="\\t")
        label_col = next((col for col in participants.columns if col.lower() in ["group", "diagnosis", "condition", "class"]), None)
        
        subjects = []
        for _, row in participants.iterrows():
            subject_id = row["participant_id"]
            raw_label = str(row[label_col]).strip()
            group_label = next((val for key, val in LABEL_MAP.items() if key.lower() == raw_label.lower()), None)
            
            if not group_label: continue
                
            eeg_path = os.path.join(DATASET_ROOT, subject_id, "eeg", f"{subject_id}_task-Rest_eeg.set")
            if os.path.exists(eeg_path):
                subjects.append({"path": eeg_path, "label": group_label})

        train, val = train_test_split(subjects, test_size=0.2, stratify=[s["label"] for s in subjects], random_state=42)

        def create_structure(data_list, split_name):
            for item in data_list:
                label_dir = os.path.join(OUTPUT_ROOT, split_name, item["label"])
                os.makedirs(label_dir, exist_ok=True)
                base_src = os.path.abspath(item["path"])
                safe_link(base_src, os.path.join(label_dir, os.path.basename(base_src)))
                fdt_src = base_src.replace(".set", ".fdt")
                if os.path.exists(fdt_src):
                    safe_link(fdt_src, os.path.join(label_dir, os.path.basename(fdt_src)))

        create_structure(train, "train")
        create_structure(val, "val")
        print("[SUCCESS] Dataset structure built.")
    ''')

    # --- CELL 5: Dataset Pipeline ---
    add_markdown("### 4. EEG-to-Spectrogram Pipeline")
    add_code('''
    class EEGSpectrogramDataset(Dataset):
        def __init__(self, root_dir, is_training=False):
            self.samples = []
            self.labels = {"control": 0, "parkinsons": 1}
            self.is_training = is_training
            mne.set_log_level("ERROR")

            # MONAI Data Augmentations
            self.train_transforms = Compose([
                RandGaussianNoise(prob=0.5, mean=0.0, std=0.05),
                RandCoarseDropout(holes=2, spatial_size=(20,20), dropout_holes=True, fill_value=0, prob=0.5),
                RandScaleIntensity(factors=0.1, prob=0.5),
                EnsureType(dtype=torch.float32)
            ])

            for label_name, label_idx in self.labels.items():
                folder = os.path.join(root_dir, label_name)
                if not os.path.exists(folder): continue
                for f in os.listdir(folder):
                    if f.endswith(".set"):
                        self.samples.append((os.path.join(folder, f), label_idx))

        def __len__(self):
            return len(self.samples)

        def _eeg_to_spectrogram(self, eeg_data):
            channels_to_use = min(3, eeg_data.shape[0])
            images = []
            for ch in range(channels_to_use):
                _, _, Sxx = spectrogram(eeg_data[ch], fs=256, nperseg=256, noverlap=128)
                Sxx_log = np.log1p(Sxx)
                Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)
                images.append(Sxx_norm)

            while len(images) < 3: images.append(np.zeros_like(images[0]))

            img_tensor = torch.tensor(np.stack(images)).float()
            img_tensor = F.resize(img_tensor, [224, 224], antialias=True)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (img_tensor - mean) / std

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            raw = mne.io.read_raw_eeglab(path, preload=False)
            data = raw.get_data().astype(np.float32)
            img = self._eeg_to_spectrogram(data)
            if self.is_training: img = self.train_transforms(img)
            return img, torch.tensor(label, dtype=torch.long)
            
    def build_model(num_classes=2):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    ''')

    # --- CELL 6: Training Loop ---
    add_markdown("### 5. Training Pipeline (Weighted Loss & AdamW)")
    add_code('''
    def train_pipeline(epochs=30):
        train_ds = EEGSpectrogramDataset("dataset/processed/train", is_training=True)
        val_ds = EEGSpectrogramDataset("dataset/processed/val", is_training=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = build_model().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        class_weights = torch.tensor([2.5, 1.0], dtype=torch.float32).to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        scaler = GradScaler('cuda')

        best_acc = 0.0
        os.makedirs("outputs/checkpoints", exist_ok=True)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda'):
                    preds = model(x)
                    loss = criterion(preds, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
                    with autocast('cuda'):
                        out = model(x_val)
                        loss = criterion(out, y_val)
                    val_loss += loss.item()
                    _, predicted = torch.max(out.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(y_val.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "outputs/checkpoints/best_model.pt")
                print(f" -> Best model saved! ({best_acc:.4f})")
    ''')

    # --- CELL 7: XAI & PDF GENERATION ---
    add_markdown("### 6. Explainable AI (Grad-CAM) & PDF Report Generation")
    add_code('''
    def save_spectrogram_image(tensor, save_path):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_viz = tensor * std + mean
        
        img_2d = img_viz.mean(dim=0).numpy()
        v_max = np.percentile(img_2d, 95) 
        v_min = np.min(img_2d)
        
        plt.figure(figsize=(6, 4))
        plt.imshow(img_2d, aspect='auto', cmap='jet', origin='lower', vmin=v_min, vmax=v_max)
        
        zoom_limit = img_2d.shape[0] // 2
        plt.ylim(0, zoom_limit)
        
        plt.title("STFT Spectrogram (Raw Input)")
        plt.ylabel("Frequency Bins (Zoomed)")
        plt.xlabel("Time Windows")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def save_gradcam_image(model, input_tensor, save_path):
        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(input_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(input_tensor.device)
        rgb_img = input_tensor[0] * std + mean
        rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy()
        rgb_img = np.clip(rgb_img, 0, 1)
        
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        plt.figure(figsize=(6, 4))
        plt.imshow(visualization, aspect='auto', origin='lower')
        
        zoom_limit = visualization.shape[0] // 2
        plt.ylim(0, zoom_limit)
        
        plt.title("Grad-CAM: AI Attention Map")
        plt.ylabel("Frequency Bins (Zoomed)")
        plt.xlabel("Time Windows")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    class PDFReport(FPDF):
        def header(self):
            self.set_fill_color(0, 51, 102) 
            self.rect(0, 0, 210, 25, 'F')
            self.set_y(8)
            self.set_font('Arial', 'B', 16)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, 'NEURO-AI CLINICAL EEG REPORT', 0, 1, 'C')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cell(0, 10, f'Report Generated: {timestamp}  |  Page {self.page_no()}', 0, 0, 'C')

    def create_pdf(patient_id, prediction, confidence, spec_img_path, gradcam_img_path, output_pdf):
        pdf = PDFReport()
        pdf.set_margins(left=15, top=25, right=15)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        pdf.set_y(30)
        pdf.set_fill_color(240, 248, 255) 
        pdf.set_draw_color(70, 130, 180)  
        pdf.set_line_width(0.5)
        pdf.rect(15, 30, 180, 22, 'DF')
        
        pdf.set_y(32)
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(90, 5, ' FINAL DIAGNOSIS:', 0, 0, 'L')
        pdf.cell(90, 5, ' AI CONFIDENCE SCORE:', 0, 1, 'L')
        
        pdf.set_font('Arial', 'B', 16)
        if "Parkinson" in prediction:
            pdf.set_text_color(178, 34, 34) 
        else:
            pdf.set_text_color(34, 139, 34) 
        pdf.cell(90, 10, f" {prediction.upper()}", 0, 0, 'L')
        
        pdf.set_text_color(0, 51, 102) 
        pdf.cell(90, 10, f" {confidence:.2f}%", 0, 1, 'L')
        pdf.ln(8)
        
        pdf.set_fill_color(70, 130, 180) 
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, '  PATIENT & SCAN DEMOGRAPHICS', 0, 1, 'L', fill=True)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.set_y(pdf.get_y() + 2)
        
        col_w = 90
        pdf.cell(col_w, 7, f" Patient ID: {patient_id}", border='B', ln=0)
        pdf.cell(col_w, 7, f" Scan Date: {datetime.now().strftime('%Y-%m-%d')}", border='B', ln=1)
        pdf.cell(col_w, 7, " Sex: Unknown (Blind Scan)", border='B', ln=0)
        pdf.cell(col_w, 7, " Modality: Resting-State EEG (10-20 System)", border='B', ln=1)
        pdf.cell(0, 7, " Pipeline: MNE Extraction -> STFT -> ResNet18 CNN", border='B', ln=1)
        pdf.ln(8)
        
        pdf.set_fill_color(70, 130, 180) 
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, '  AI VISUALIZATION & EXPLAINABILITY (XAI)', 0, 1, 'L', fill=True)
        pdf.ln(4)
        
        start_y = pdf.get_y()
        pdf.image(spec_img_path, x=15, y=start_y, w=88)
        pdf.image(gradcam_img_path, x=107, y=start_y, w=88)
        
        pdf.set_y(start_y + 55)
        cm_path = "outputs/logs/confusion_matrix.png"
        roc_path = "outputs/logs/roc_curve.png"
        
        img_y = pdf.get_y()
        if os.path.exists(cm_path): pdf.image(cm_path, x=15, y=img_y, w=88)  
        if os.path.exists(roc_path): pdf.image(roc_path, x=107, y=img_y, w=88) 
            
        pdf.ln(60) 
        
        if pdf.get_y() > 210: 
            pdf.add_page()
            pdf.set_y(30)
            
        pdf.set_fill_color(70, 130, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, '  CLINICAL NARRATIVE & IMPRESSION', 0, 1, 'L', fill=True)
        pdf.ln(3)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        narrative = (
            "TECHNIQUE:\\n"
            "This AI-assisted analysis utilizes a deep Convolutional Neural Network (ResNet18) fine-tuned for neurological "
            "biomarker detection. The raw multi-channel EEG signal was transformed into a 2D Short-Time Fourier Transform (STFT) "
            "spectrogram. This isolates specific frequency-domain anomalies (e.g., Alpha/Beta band slowing or Theta spikes) "
            "associated with Parkinsonian neurodegeneration.\\n\\n"
            "FINDINGS & IMPRESSION:\\n"
            f"The AI pipeline analyzed the electrical signature and classified the scan as {prediction.upper()} with "
            f"a computed confidence probability of {confidence:.2f}%. "
            "The high-contrast STFT feature extraction map is provided above, alongside a Grad-CAM (Gradient-weighted Class "
            "Activation Mapping) overlay. The Grad-CAM heatmap explicitly highlights the specific spectral regions evaluated "
            "by the AI's convolutional layers to arrive at this specific diagnosis."
        )
        pdf.multi_cell(0, 5, narrative)
        pdf.ln(6)
        
        pdf.set_text_color(105, 105, 105) 
        pdf.set_font('Arial', 'I', 8)
        disclaimer = (
            "DISCLAIMER: This report was generated autonomously by the NeuroAI classification system (v1.0). "
            "It is intended strictly for research, triage, and educational purposes. It has NOT been reviewed, validated, "
            "or approved by a licensed clinical neurologist. AI systems can produce false positives and false negatives. "
            "Do not use this output for definitive medical decision-making."
        )
        pdf.multi_cell(0, 4, disclaimer)
        pdf.output(output_pdf, 'F')

    def run_clinical_inference(eeg_path, model_path):
        if not os.path.exists(model_path):
            print(f"ERROR: Model weights not found at {model_path}")
            return
            
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        print(f"Processing EEG signal: {os.path.basename(eeg_path)}")
        mne.set_log_level("ERROR")
        try:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=False)
            data = raw.get_data().astype("float32")
        except Exception as e:
            print(f"ERROR loading EEG: {e}")
            return
            
        dummy_ds = EEGSpectrogramDataset(root_dir="")
        img_tensor = dummy_ds._eeg_to_spectrogram(data)
        
        os.makedirs("outputs/reports", exist_ok=True)
        temp_spec_path = "outputs/reports/temp_spectrogram.png"
        temp_gradcam_path = "outputs/reports/temp_gradcam.png"
        
        save_spectrogram_image(img_tensor, temp_spec_path)
        
        img_batch = img_tensor.unsqueeze(0).to(DEVICE)
        save_gradcam_image(model, img_batch, temp_gradcam_path)
        
        with torch.no_grad():
            out = model(img_batch)
            prob = torch.softmax(out, dim=1)
            
        parkinsons_prob = prob[0][1].item()
        CLINICAL_THRESHOLD = 0.70 
        
        if parkinsons_prob >= CLINICAL_THRESHOLD:
            prediction = "Parkinson's Disease"
            confidence = parkinsons_prob * 100
        else:
            prediction = "Control (Healthy)"
            confidence = (1.0 - parkinsons_prob) * 100
        
        print(f"\\n-> Diagnosis: {prediction} ({confidence:.2f}%)")
        
        patient_id = os.path.basename(eeg_path).replace("_task-Rest_eeg.set", "")
        output_pdf = f"outputs/reports/NeuroAI_Report_{patient_id}.pdf"
        
        create_pdf(patient_id, prediction, confidence, temp_spec_path, temp_gradcam_path, output_pdf)
        
        if os.path.exists(temp_spec_path): os.remove(temp_spec_path)
        if os.path.exists(temp_gradcam_path): os.remove(temp_gradcam_path)
        print(f"[SUCCESS] Report Saved: {output_pdf}")
    ''')

    # --- CELL 8: Execution Wrapper ---
    add_markdown("### 7. Execution Area")
    add_code('''
    # 1. Build the dataset splits (uncomment if running for the very first time)
    # prepare_dataset()

    # 2. Train the model (uncomment to train)
    # train_pipeline(epochs=30)
    
    # 3. Run Clinical Inference & PDF Generation
    MODEL_PATH = "outputs/checkpoints/best_model.pt" 
    
    # Point this to a test file (e.g., sub-054 for Parkinson's, sub-149 for Control)
    TARGET_EEG_FILE = "dataset/processed/val/parkinsons/sub-054_task-Rest_eeg.set"
    
    run_clinical_inference(TARGET_EEG_FILE, MODEL_PATH)
    ''')

    # Construct and save the JSON structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(filename, 'w') as f:
        json.dump(notebook, f, indent=2)
    print(f"[SUCCESS] Jupyter Notebook successfully generated at: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_jupyter_notebook()