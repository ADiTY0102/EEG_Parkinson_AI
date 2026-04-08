# NeuroAI: EEG Parkinson's Detection Pipeline

NeuroAI is a deep learning clinical pipeline that automatically detects biomarkers for Parkinson's Disease directly from resting-state EEG signals. The architecture extracts frequency-domain features from raw electrical signals and passes them through a Convolutional Neural Network (CNN) to yield diagnostic predictions and visually explainable AI (XAI) overlays.

## Tech Stack & Architecture

- **Signal Processing**: MNE-Python (Raw EEG Parsing), SciPy (STFT Spectrogram Generation)
- **Deep Learning**: PyTorch, TorchVision (ResNet18 Classification), Grad-CAM (Explainable AI)
- **Report Generation**: FPDF (Automated Clinical PDFs), Matplotlib (Visualizations)
- **Data Handling**: NumPy

## Pipeline Breakdown

1. **Extraction**: Raw multi-channel EEG files are parsed using `mne`.
2. **Frequency Transform**: Signals undergo Short-Time Fourier Transform (STFT) via `scipy.signal.spectrogram`, mapping time-domain signals into 2D localized frequency heatmap representations.
3. **CNN Classification**: A customized `ResNet18` model classifies the 2D STFT spectral images as "Control" or "Parkinson's Disease".
4. **XAI Mapping**: Grad-CAM structural overlays highlight the exact frequency-time areas the neural network focused on to make its diagnosis.
5. **Autoreporting**: An automated PDF clinical narrative is bundled with metrics, exporting directly to your local `outputs/reports/` directory.

## Representative Metrics & Output Samples

Below are outputs derived from the testing suite outlining performance capabilities.

### Evaluation Metrics (Test Logs)
<div align="center">
  <img src="outputs/logs/confusion_matrix.png" width="45%" alt="Confusion Matrix"/>
  <img src="outputs/logs/roc_curve.png" width="45%" alt="ROC Curve"/>
</div>

### XAI Clinical Overlays (Grad-CAM)
By inspecting the gradient-weighted class activation map (Grad-CAM) overlays, clinicians can interpret which spectral slowing signatures lead to positive flags.
<br>
<img src="outputs/reports/GradCAM_sub-054.png" width="75%" alt="GradCAM Output"/>

### Automated Clinical Report Snippet
The automated pipeline generates a comprehensive PDF summarizing patient diagnosis and providing the interpretability narrative alongside extraction graphs.

[📥 View a Sample NeuroAI Clinical PDF Report](outputs/reports/NeuroAI_Report_sub-054.pdf)
