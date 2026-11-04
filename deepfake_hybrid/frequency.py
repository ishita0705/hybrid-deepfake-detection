"""Frequency forensic map for one 224x224 face.

2D FFT -> fftshift -> log(1 + |F|) -> per-image min-max normalization.
The transform runs on a grayscale copy of the face. The saved jpg stays RGB.
"""

import cv2
import numpy as np


def frequency_map(face_rgb: np.ndarray) -> np.ndarray:
    """Return a float32 map with the same height and width as face_rgb.

    face_rgb is uint8 RGB, H x W x 3, already resized to the model input.
    """
    if face_rgb.ndim != 3 or face_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB face, got shape {face_rgb.shape}")

    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)

    low = float(magnitude.min())
    high = float(magnitude.max())
    if high > low:
        magnitude = (magnitude - low) / (high - low)
    else:
        magnitude = np.zeros_like(magnitude)
    return magnitude
