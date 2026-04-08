import os
import mne
import torch
import numpy as np
from scipy.signal import spectrogram
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from monai.transforms import (
    Compose, RandGaussianNoise, RandCoarseDropout, RandScaleIntensity, EnsureType
)

class EEGSpectrogramDataset(Dataset):
    def __init__(self, root_dir, is_training=False):
        self.samples = []
        self.labels = {"control": 0, "parkinsons": 1}
        self.is_training = is_training
        mne.set_log_level("ERROR")
        
        self.train_transforms = Compose([
            RandGaussianNoise(prob=0.5, mean=0.0, std=0.05),
            RandCoarseDropout(holes=2, spatial_size=(20, 20), dropout_holes=True, fill_value=0, prob=0.5),
            RandScaleIntensity(factors=0.1, prob=0.5),
            EnsureType(dtype=torch.float32)
        ])
        
        for label_name, label_idx in self.labels.items():
            folder = os.path.join(root_dir, label_name)
            if not os.path.exists(folder):
                continue
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
            # Log transform + Min-Max Scaling to 0-1 range
            Sxx_log = np.log1p(Sxx)
            Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)
            images.append(Sxx_norm)
            
        while len(images) < 3:
            images.append(np.zeros_like(images[0]))
            
        img_tensor = torch.tensor(np.stack(images)).float()
        img_tensor = F.resize(img_tensor, [224, 224], antialias=True)
        
        # Standardize for ResNet ImageNet weights
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor
    
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        # preload=False prevents loading the massive file into RAM all at once
        raw = mne.io.read_raw_eeglab(path, preload=False)
        
        # Load data explicitly and cast to float32 immediately to halve RAM usage
        data = raw.get_data().astype(np.float32)
        
        img = self._eeg_to_spectrogram(data)
        
        if self.is_training:
            img = self.train_transforms(img)
            
        return img, torch.tensor(label, dtype=torch.long)