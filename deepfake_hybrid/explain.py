"""Grad-CAM on the EfficientNet-B0 branch.

Saves heatmap overlays for 4 real and 4 fake frames. Up to 2 videos of each
class come from the FF++ c23 test split. The rest come from SDFVD, so a class
that has only one test video still contributes a frame and the folder still
holds eight overlays. Frames are chosen in manifest order, one per video, so
the choice does not depend on the score.

The full hybrid checkpoint is used. The other variants share this spatial branch,
and the overlay is an explanation of a frame, not a test metric.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import deepfake_hybrid.config as config
from deepfake_hybrid.dataset import load_face, load_freq, read_manifest
from deepfake_hybrid.metrics import log_line
from deepfake_hybrid.model import build_model, checkpoint_file


class _FrameWrapper(nn.Module):
    """Run the detector on one face, repeated across the 10-frame window.

    Grad-CAM calls this with a batch of images shaped (B, 3, H, W). Repeating
    the same frame keeps the batch aligned with the EfficientNet activations.
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb is (B, 3, H, W). EfficientNet must see that batch, not B*T,
        # or Grad-CAM cannot line the activations up with the input image.
        spatial = self.detector.spatial(rgb)
        if self.detector.use_freq:
            freq = self.freq
            if freq.shape[0] == 1 and rgb.shape[0] != 1:
                freq = freq.expand(rgb.shape[0], -1, -1, -1)
            frame = torch.cat([spatial, self.detector.freq_cnn(freq)], dim=-1)
        else:
            frame = spatial
        sequence = frame.unsqueeze(1).expand(-1, config.SEQ_LEN, -1).contiguous()
        return self.detector.head(self.detector.pool(sequence))


class _FakeScore:
    """Grad-CAM target: the fake-class probability."""

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 2:
            return model_output[:, 0]
        return model_output


def _disable_inplace(module: nn.Module) -> None:
    # Inplace activations block the Grad-CAM backward hook.
    for layer in module.modules():
        if isinstance(layer, (nn.ReLU, nn.SiLU, nn.ReLU6)) and hasattr(layer, "inplace"):
            layer.inplace = False


def _pick(rows: list[dict], label: int, n: int) -> list[dict]:
    chosen = []
    seen = set()
    for row in rows:
        if int(row["label"]) != label:
            continue
        if row["video"] in seen:
            continue
        seen.add(row["video"])
        chosen.append(row)
        if len(chosen) == n:
            break
    return chosen


def _middle_frame(row: dict) -> dict:
    frames = row["frames"]
    return frames[len(frames) // 2]


def _denormalize(rgb: torch.Tensor) -> np.ndarray:
    mean = np.asarray(config.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(config.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    image = rgb.detach().cpu().numpy() * std + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def _allow_lstm_backward(model: nn.Module) -> None:
    """cuDNN can backprop through an LSTM only while that module is training.

    This LSTM has one layer and no dropout, so training mode does not change
    its output. BatchNorm and the head dropout stay in eval mode.
    """
    if getattr(model, "lstm", None) is not None:
        model.lstm.train()


def _load_detector(device: torch.device) -> nn.Module:
    path = checkpoint_file("full")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Full hybrid checkpoint not found at {path}. "
            "Train it with python -m deepfake_hybrid.train --all"
        )
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    model = build_model("full", pretrained=False).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    _disable_inplace(model)
    return model


def explain() -> None:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_detector(device)
    wrapper = _FrameWrapper(model).to(device)
    wrapper.eval()

    try:
        cam = GradCAM(model=wrapper, target_layers=[model.cam_layer])
    except TypeError:
        cam = GradCAM(model=wrapper, target_layers=[model.cam_layer], use_cuda=device.type == "cuda")
    # Grad-CAM puts the wrapper in eval mode. Turn the LSTM back to train
    # after that, or the backward pass raises on CUDA.
    _allow_lstm_backward(model)

    ffpp = read_manifest("ffpp_sequences_test.json")
    sdfvd = read_manifest("sdfvd_sequences_test.json")
    jobs = []
    for label in (config.LABEL_REAL, config.LABEL_FAKE):
        from_ffpp = _pick(ffpp, label, 2)
        from_sdfvd = _pick(sdfvd, label, 4 - len(from_ffpp))
        jobs.extend(("ffpp_c23", row) for row in from_ffpp)
        jobs.extend(("sdfvd", row) for row in from_sdfvd)

    n_real = sum(1 for _name, row in jobs if int(row["label"]) == config.LABEL_REAL)
    n_fake = sum(1 for _name, row in jobs if int(row["label"]) == config.LABEL_FAKE)
    n_ffpp = sum(1 for name, _row in jobs if name == "ffpp_c23")
    n_sdfvd = sum(1 for name, _row in jobs if name == "sdfvd")
    log_line(
        f"Grad-CAM frames selected: real={n_real} fake={n_fake} "
        f"ffpp_c23={n_ffpp} sdfvd={n_sdfvd}"
    )

    out_dir = os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS, "gradcam")
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for dataset_name, row in jobs:
        frame = _middle_frame(row)
        face = load_face(frame["face_path"])
        freq = load_freq(frame["freq_path"])
        rgb = face.unsqueeze(0).to(device)
        wrapper.freq = freq.unsqueeze(0).to(device)
        grayscale = cam(input_tensor=rgb, targets=[_FakeScore()])[0]
        overlay = show_cam_on_image(_denormalize(face), grayscale, use_rgb=True)
        label_name = "real" if int(row["label"]) == config.LABEL_REAL else "fake"
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(row["video"]))
        filename = f"{dataset_name}_{label_name}_{safe}_f{int(frame['frame_idx']):06d}.png"
        path = os.path.join(out_dir, filename)
        Image.fromarray(overlay).save(path)
        with torch.no_grad():
            probability = float(wrapper(rgb).detach().cpu().reshape(-1)[0])
        index.append(
            {
                "file": filename,
                "dataset": dataset_name,
                "video": row["video"],
                "label": int(row["label"]),
                "frame_idx": int(frame["frame_idx"]),
                "fake_probability": probability,
            }
        )
        log_line(f"saved {path} P(fake)={probability:.4f}")

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    log_line(f"wrote {index_path}")
    # Some grad-cam releases raise inside hook cleanup. The PNGs are already saved.
    try:
        if getattr(cam, "activations_and_grads", None) is not None:
            cam.activations_and_grads.release()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> None:
    del argv
    explain()


if __name__ == "__main__":
    main(sys.argv[1:])
