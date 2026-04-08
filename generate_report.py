import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from fpdf import FPDF
from datetime import datetime
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==========================================
# 1. MODEL & PREPROCESSING DEFS
# ==========================================
def build_model(num_classes=2):
    model = models.resnet18(weights=None) 
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def eeg_to_spectrogram(eeg_data):
    channels_to_use = min(3, eeg_data.shape[0])
    images = []
    for ch in range(channels_to_use):
        _, _, Sxx = spectrogram(eeg_data[ch], fs=256, nperseg=256, noverlap=128)
        Sxx_log = np.log1p(Sxx)
        Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)
        images.append(Sxx_norm)
        
    while len(images) < 3:
        images.append(np.zeros_like(images[0]))
        
    img_tensor = torch.tensor(np.stack(images)).float()
    img_tensor = F.resize(img_tensor, [224, 224], antialias=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor

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
    """Generates and saves the Grad-CAM XAI overlay."""
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

# ==========================================
# 2. PDF GENERATION LOGIC
# ==========================================
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
    
    # --- DIAGNOSIS HIGHLIGHT BOX ---
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
    
    # --- PATIENT DEMOGRAPHICS ---
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
    
    # --- AI FEATURE EXTRACTION IMAGING ---
    pdf.set_fill_color(70, 130, 180) 
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, '  AI VISUALIZATION & EXPLAINABILITY (XAI)', 0, 1, 'L', fill=True)
    pdf.ln(4)
    
    start_y = pdf.get_y()
    
    # Place STFT and Grad-CAM Side-by-Side
    pdf.image(spec_img_path, x=15, y=start_y, w=88)
    pdf.image(gradcam_img_path, x=107, y=start_y, w=88)
    
    pdf.set_y(start_y + 55)
    
    cm_path = "outputs/logs/confusion_matrix.png"
    roc_path = "outputs/logs/roc_curve.png"
    
    img_y = pdf.get_y()
    if os.path.exists(cm_path):
        pdf.image(cm_path, x=15, y=img_y+5, w=88)  
    if os.path.exists(roc_path):
        pdf.image(roc_path, x=107, y=img_y+5, w=88) 
        
    pdf.ln(60) 
    
    # --- CLINICAL IMPRESSION & NARRATIVE ---
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
        "TECHNIQUE:\n"
        "This AI-assisted analysis utilizes a deep Convolutional Neural Network (ResNet18) fine-tuned for neurological "
        "biomarker detection. The raw multi-channel EEG signal was transformed into a 2D Short-Time Fourier Transform (STFT) "
        "spectrogram. This isolates specific frequency-domain anomalies (e.g., Alpha/Beta band slowing or Theta spikes) "
        "associated with Parkinsonian neurodegeneration.\n\n"
        "FINDINGS & IMPRESSION:\n"
        f"The AI pipeline analyzed the electrical signature and classified the scan as {prediction.upper()} with "
        f"a computed confidence probability of {confidence:.2f}%. "
        "The high-contrast STFT feature extraction map is provided above, alongside a Grad-CAM (Gradient-weighted Class "
        "Activation Mapping) overlay. The Grad-CAM heatmap explicitly highlights the specific spectral regions evaluated "
        "by the AI's convolutional layers to arrive at this specific diagnosis."
    )
    pdf.multi_cell(0, 5, narrative)
    pdf.ln(6)
    
    # --- DISCLAIMER ---
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

# ==========================================
# 3. MAIN INFERENCE PIPELINE
# ==========================================
def run_clinical_inference(eeg_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing AI Pipeline on {device}...")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return
        
    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Processing EEG signal: {os.path.basename(eeg_path)}")
    mne.set_log_level("ERROR")
    try:
        raw = mne.io.read_raw_eeglab(eeg_path, preload=False)
        data = raw.get_data().astype("float32")
    except Exception as e:
        print(f"ERROR loading EEG: {e}")
        return
        
    img_tensor = eeg_to_spectrogram(data)
    
    os.makedirs("outputs/reports", exist_ok=True)
    temp_spec_path = "outputs/reports/temp_spectrogram.png"
    temp_gradcam_path = "outputs/reports/temp_gradcam.png"
    
    # Save standard STFT
    save_spectrogram_image(img_tensor, temp_spec_path)
    
    # Generate and save Grad-CAM overlay
    img_batch = img_tensor.unsqueeze(0).to(device)
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
    
    print(f"\n-> Diagnosis: {prediction} ({confidence:.2f}%)")
    
    patient_id = os.path.basename(eeg_path).replace("_task-Rest_eeg.set", "")
    output_pdf = f"outputs/reports/NeuroAI_Report_{patient_id}.pdf"
    
    # Pass both image paths to the PDF generator
    create_pdf(patient_id, prediction, confidence, temp_spec_path, temp_gradcam_path, output_pdf)
    
    # Cleanup temp images
    if os.path.exists(temp_spec_path): os.remove(temp_spec_path)
    if os.path.exists(temp_gradcam_path): os.remove(temp_gradcam_path)

if __name__ == "__main__":
    MODEL_PATH = "outputs/checkpoints/best_model.pt" 
    TARGET_EEG_FILE = "dataset/processed/val/control/sub-129_task-Rest_eeg.set"
    
    run_clinical_inference(TARGET_EEG_FILE, MODEL_PATH)