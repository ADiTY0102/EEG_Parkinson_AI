# """
# Global configuration for EEG Parkinson detection project
# """

# import os
# import torch

# # -----------------------
# # Project Paths
# # -----------------------

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "raw")
# PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "processed")

# CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
# LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "logs")

# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)

# # -----------------------
# # Dataset Parameters
# # -----------------------

# TRAIN_SPLIT = 0.8
# RANDOM_SEED = 42

# CLASS_NAMES = [
#     "control",
#     "parkinsons"
# ]

# NUM_CLASSES = len(CLASS_NAMES)

# # -----------------------
# # EEG Parameters
# # -----------------------

# SAMPLING_RATE = 500

# WINDOW_SECONDS = 2
# WINDOW_SIZE = SAMPLING_RATE * WINDOW_SECONDS

# WINDOW_STRIDE = WINDOW_SIZE // 2

# # -----------------------
# # Spectrogram Parameters
# # -----------------------

# N_FFT = 256
# HOP_LENGTH = 128
# N_MELS = 128

# # -----------------------
# # Model Parameters
# # -----------------------

# IMAGE_SIZE = 224
# MODEL_NAME = "resnet18"

# # -----------------------
# # Training Parameters
# # -----------------------

# BATCH_SIZE = 8
# EPOCHS = 30
# LEARNING_RATE = 1e-4

# NUM_WORKERS = 4
# PIN_MEMORY = True

# # -----------------------
# # Device
# # -----------------------

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Mixed precision
# USE_AMP = True

# # -----------------------
# # Logging
# # -----------------------

# PRINT_EVERY = 20
# SAVE_BEST_ONLY = True

import torch
from pathlib import Path
from pathlib import Path
import torch

# dataset root
DATASET_ROOT = Path("dataset/dataset")

# dataloader settings
BATCH_SIZE = 4
NUM_WORKERS = 2

# spectrogram parameters
SAMPLING_RATE = 500  # from eeg.json

DEVICE ="cpu"
# "cuda" if torch.cuda.is_available() else 