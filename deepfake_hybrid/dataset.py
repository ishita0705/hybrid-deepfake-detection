"""Load cached face sequences. Training never runs MTCNN or the FFT again.

Each manifest row is one non-overlapping 10-frame sequence. Face jpgs are
ImageNet-normalized here. Frequency .npy files are already per-image normalized.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import deepfake_hybrid.config as config


def manifest_path(name: str) -> str:
    return os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS, name)


def read_manifest(name: str) -> list[dict]:
    path = manifest_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run data prep before training: "
            "python -m deepfake_hybrid.data_prep --step all"
        )
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not rows:
        raise RuntimeError(f"{path} has no sequences.")
    return rows


def _abs(rel_path: str) -> str:
    return os.path.join(config.OUTPUT_DIR, rel_path)


_MEAN = torch.tensor(config.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


def load_face(rel_path: str) -> torch.Tensor:
    """RGB face jpg -> float tensor (3, 224, 224), ImageNet normalized."""
    path = _abs(rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return (tensor - _MEAN) / _STD


def load_freq(rel_path: str) -> torch.Tensor:
    """Cached FFT map -> float tensor (1, 224, 224)."""
    path = _abs(rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    array = np.load(path).astype(np.float32)
    tensor = torch.from_numpy(array)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


class SequenceDataset(Dataset):
    """One item is a clip: rgb (T, 3, H, W), freq (T, 1, H, W), label, video id."""

    def __init__(self, manifest_name: str):
        self.manifest_name = manifest_name
        self.rows = read_manifest(manifest_name)
        for row in self.rows:
            n_frames = len(row["frames"])
            if n_frames != config.SEQ_LEN:
                raise RuntimeError(
                    f"{manifest_name} sequence {row.get('video')} has {n_frames} frames, "
                    f"expected {config.SEQ_LEN}."
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        faces = []
        freqs = []
        for frame in row["frames"]:
            faces.append(load_face(frame["face_path"]))
            freqs.append(load_freq(frame["freq_path"]))
        return {
            "rgb": torch.stack(faces, dim=0),
            "freq": torch.stack(freqs, dim=0),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "video": str(row["video"]),
        }


def make_loader(
    manifest_name: str,
    *,
    batch_size: int | None = None,
    sampler=None,
    shuffle: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = SequenceDataset(manifest_name)
    # The sampler draws the epoch order. shuffle must stay off in that case.
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE if batch_size is None else batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )


def class_counts(dataset: SequenceDataset) -> dict[int, int]:
    counts = {config.LABEL_REAL: 0, config.LABEL_FAKE: 0}
    for row in dataset.rows:
        counts[int(row["label"])] += 1
    return counts
