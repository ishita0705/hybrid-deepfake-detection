"""Hybrid deepfake video detection, Phase 2.

Spatial EfficientNet-B0, a frequency FFT branch, and a Bi-LSTM.
Written to run in Google Colab (T4, PyTorch 2.x).

data_prep.py builds the cached faces. train.py fits the four ablation
variants. evaluate.py scores each test set once. explain.py writes Grad-CAM.
"""

__version__ = "0.3.0"
