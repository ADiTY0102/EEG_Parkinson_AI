import os
import sys
import torch
import json
import mne

# Add root directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import build_model
from data_pipeline.loader import EEGSpectrogramDataset

def predict(eeg_path, checkpoint_path="outputs/checkpoints/best_model.pt"):
    if not os.path.exists(checkpoint_path):
        return json.dumps({"error": f"No trained model found at {checkpoint_path}"})
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and load your best weights
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Initialize the dataset class just to access its STFT transformation function
    transformer = EEGSpectrogramDataset(root_dir="", is_training=False) 
    
    # Suppress MNE warnings for clean output
    mne.set_log_level("ERROR")
    
    try:
        raw = mne.io.read_raw_eeglab(eeg_path, preload=False)
        data = raw.get_data().astype("float32")
    except Exception as e:
        return json.dumps({"error": f"Failed to load EEG file: {str(e)}"})
    
    # Convert to spectrogram and add batch dimension [1, C, H, W]
    img = transformer._eeg_to_spectrogram(data)
    img = img.unsqueeze(0).to(device)
    
    # Forward Pass
    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        conf, pred = prob.max(1)
        
    labels = ["Control", "Parkinsons"]
    
    result = {
        "patient_id": os.path.basename(eeg_path).replace("_task-Rest_eeg.set", ""),
        "scan_type": "EEG",
        "prediction": labels[pred.item()],
        "confidence": round(float(conf.item()), 4)
    }
    
    return json.dumps(result, indent=4)

if __name__ == "__main__":
    import glob
    
    # Dynamically grab the first available .set file in the validation folder
    search_path = "dataset/processed/val/parkinsons/*.set"
    available_files = glob.glob(search_path)
    
    if not available_files:
        print(f"Error: No files found matching {search_path}")
    else:
        test_file = available_files[11]
        #test_file = "sub-108_task-Rest_eeg.set"
        print(f"\n--- Clinical AI Prediction ---")
        print(f"Testing File: {os.path.basename(test_file)}")
        print(predict(test_file))
        print("------------------------------\n")