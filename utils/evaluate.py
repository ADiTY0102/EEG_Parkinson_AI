import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from torch.utils.data import DataLoader

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import build_model
from data_pipeline.loader import EEGSpectrogramDataset
import mne

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting clinical evaluation on: {device}")

    # 1. Load Validation Data (num_workers=0 prevents the RAM crash we solved earlier)
    val_ds = EEGSpectrogramDataset("dataset/processed/val", is_training=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    # 2. Load Trained Model
    model = build_model().to(device)
    checkpoint_path = "outputs/checkpoints/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    print("Running predictions on the holdout validation set...")
    mne.set_log_level("ERROR") # Suppress MNE warnings
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            # Updated autocast syntax to remove your previous terminal warnings
            with torch.amp.autocast('cuda'): 
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)
                
            # Get probability of Parkinson's (Class 1) for ROC curve
            scores = probs[:, 1].cpu().numpy() 
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_scores.extend(scores)

    # 3. Terminal Output: Classification Report
    target_names = ['Control (0)', 'Parkinsons (1)']
    print("\n" + "="*40)
    print("      CLINICAL CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=target_names))

    # 4. Generate & Save Visualizations
    os.makedirs("outputs/logs", exist_ok=True)
    
    # --- Plot 1: Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 16})
    plt.title('Validation Confusion Matrix', fontsize=14)
    plt.ylabel('Actual Clinical Diagnosis', fontsize=12)
    plt.xlabel('AI Predicted Diagnosis', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/logs/confusion_matrix.png', dpi=300)
    plt.close()
    
    # --- Plot 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/logs/roc_curve.png', dpi=300)
    plt.close()

    print("\n[SUCCESS] Visualizations generated safely!")
    print("-> Check your 'outputs/logs/' folder for the images.")

if __name__ == "__main__":
    evaluate_model()