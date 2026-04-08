import numpy as np
import torch
from scipy.signal import spectrogram
from config.config import SAMPLING_RATE

TARGET_WIDTH = 300   # fixed time dimension


def eeg_to_spectrogram(eeg):

    signal = eeg[0]  # first channel

    f, t, Sxx = spectrogram(
        signal,
        fs=SAMPLING_RATE
    )

    Sxx = np.log1p(Sxx)

    # fix spectrogram width
    current_width = Sxx.shape[1]

    if current_width > TARGET_WIDTH:
        Sxx = Sxx[:, :TARGET_WIDTH]

    elif current_width < TARGET_WIDTH:
        pad_width = TARGET_WIDTH - current_width
        Sxx = np.pad(
            Sxx,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    tensor = torch.tensor(
        Sxx,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0)

    return tensor