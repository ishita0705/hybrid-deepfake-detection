"""Four ablation variants of the hybrid detector.

1. spatial              EfficientNet-B0, mean over frames. This is the baseline.
2. spatial_freq         Spatial and frequency features concatenated, then mean-pooled.
3. spatial_temporal     Spatial features through a Bi-LSTM.
4. full                 Spatial + frequency, then Bi-LSTM.

Variants without a Bi-LSTM mean-pool the per-frame vectors.
The head is Linear -> ReLU -> Dropout -> Linear -> sigmoid.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import timm

import deepfake_hybrid.config as config


VARIANTS = {
    "spatial": {
        "use_freq": False,
        "use_temporal": False,
        "display": "spatial-only (baseline)",
    },
    "spatial_freq": {
        "use_freq": True,
        "use_temporal": False,
        "display": "spatial+frequency",
    },
    "spatial_temporal": {
        "use_freq": False,
        "use_temporal": True,
        "display": "spatial+temporal",
    },
    "full": {
        "use_freq": True,
        "use_temporal": True,
        "display": "full hybrid",
    },
}


class FrequencyCNN(nn.Module):
    """Three-layer CNN. One FFT map in, one feature vector out."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        # freq: (N, 1, H, W) -> (N, out_dim)
        return self.net(freq).flatten(1)


class HybridDetector(nn.Module):
    def __init__(self, variant: str, pretrained: bool = True):
        super().__init__()
        if variant not in VARIANTS:
            known = ", ".join(VARIANTS)
            raise ValueError(f"Unknown variant '{variant}'. Choose one of: {known}")
        self.variant = variant
        spec = VARIANTS[variant]
        self.use_freq = spec["use_freq"]
        self.use_temporal = spec["use_temporal"]
        self.display_name = spec["display"]

        # num_classes=0 returns the pooled feature instead of ImageNet logits.
        self.spatial = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.spatial_dim = int(self.spatial.num_features)
        # Last spatial conv. Grad-CAM hooks this module.
        if hasattr(self.spatial, "conv_head"):
            self.cam_layer = self.spatial.conv_head
        else:
            self.cam_layer = self.spatial.blocks[-1]

        frame_dim = self.spatial_dim
        self.freq_cnn = None
        if self.use_freq:
            self.freq_cnn = FrequencyCNN(config.FREQ_OUT_DIM)
            frame_dim += config.FREQ_OUT_DIM

        self.lstm = None
        if self.use_temporal:
            self.lstm = nn.LSTM(
                input_size=frame_dim,
                hidden_size=config.LSTM_HIDDEN,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            head_in = config.LSTM_HIDDEN * 2
        else:
            head_in = frame_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, config.HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HEAD_HIDDEN, 1),
            nn.Sigmoid(),
        )

    def encode_frames(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """rgb (B, T, 3, H, W), freq (B, T, 1, H, W) -> (B, T, F)."""
        batch, time, channels, height, width = rgb.shape
        flat = rgb.reshape(batch * time, channels, height, width)
        # Recompute EfficientNet activations on the backward pass so a 4GB GPU
        # can train a 10-frame clip.
        if self.training and flat.requires_grad:
            spatial = torch.utils.checkpoint.checkpoint(
                self.spatial, flat, use_reentrant=False
            )
        else:
            spatial = self.spatial(flat)
        spatial = spatial.reshape(batch, time, self.spatial_dim)
        if not self.use_freq:
            return spatial
        freq_flat = freq.reshape(batch * time, freq.shape[2], freq.shape[3], freq.shape[4])
        freq_feat = self.freq_cnn(freq_flat).reshape(batch, time, config.FREQ_OUT_DIM)
        return torch.cat([spatial, freq_feat], dim=-1)

    def pool(self, frame_feats: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> one vector per sequence."""
        if self.lstm is None:
            return frame_feats.mean(dim=1)
        _outputs, (hidden, _cell) = self.lstm(frame_feats)
        # hidden[0] is the forward direction, hidden[1] the backward direction.
        return torch.cat([hidden[0], hidden[1]], dim=-1)

    def forward(self, rgb: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """Return P(fake) with shape (B,)."""
        features = self.pool(self.encode_frames(rgb, freq))
        return self.head(features).squeeze(-1)


def build_model(variant: str | None = None, pretrained: bool = True) -> HybridDetector:
    name = variant or config.MODEL_VARIANT
    return HybridDetector(name, pretrained=pretrained)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def checkpoint_file(variant: str) -> str:
    return os.path.join(config.OUTPUT_DIR, config.DIR_CHECKPOINTS, variant, "best.pt")
