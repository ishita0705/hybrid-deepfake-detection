"""Face crop or resize to 224x224.

Full frames: MTCNN bounding box only. Landmarks are not used, so there is no
alignment warp. The box is expanded by a small margin, made square, and resized.

Already-cropped faces: bilinear resize to 224x224.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

import deepfake_hybrid.config as config


def infer_face_mode(width: int, height: int) -> str:
    """Guess whether a sample is a full video frame or a face crop.

    FaceForensics++ frames are wide (for example 1920x1080). A face crop from
    this mirror is nearly square and small. The inspect step prints the size
    so FACE_INPUT_MODE can be set by hand if this guess is wrong.
    """
    aspect = width / float(height)
    long_side = max(width, height)
    nearly_square = 0.8 <= aspect <= 1.25
    if nearly_square and long_side <= 512:
        return "face_crop"
    return "full_frame"


def _square_box(x1, y1, x2, y2, width, height):
    """Expand a box to a square and shift it so it stays inside the image."""
    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    x1 = cx - side / 2.0
    y1 = cy - side / 2.0
    x2 = x1 + side
    y2 = y1 + side

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        shift = x2 - width
        x1 -= shift
        x2 = width
    if y2 > height:
        shift = y2 - height
        y1 -= shift
        y2 = height

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(width, int(round(x2)))
    y2 = min(height, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


class FacePrep:
    """Turn a PIL image into a 224x224 RGB uint8 array, or None when no face exists."""

    def __init__(self, mode: str):
        if mode not in ("full_frame", "face_crop"):
            raise ValueError(f"unknown face mode: {mode}")
        self.mode = mode
        self.image_size = config.IMAGE_SIZE
        self.margin = config.MTCNN_MARGIN
        self.keep_prob = config.MTCNN_KEEP_PROB
        self._mtcnn = None

    def _load_mtcnn(self):
        if self._mtcnn is not None:
            return
        from facenet_pytorch import MTCNN

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # keep_all / select_largest apply to MTCNN.forward, which also aligns.
        # We call detect() and crop the box ourselves, so landmarks are unused.
        self._mtcnn = MTCNN(
            image_size=self.image_size,
            margin=0,
            min_face_size=config.MTCNN_MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            keep_all=True,
            device=device,
        )
        self.device = device

    def _resize(self, image: Image.Image) -> np.ndarray:
        face = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        return np.asarray(face, dtype=np.uint8)

    def _crop_box(self, image: Image.Image, boxes, probs) -> np.ndarray | None:
        if boxes is None:
            return None
        box_arr = np.asarray(boxes, dtype=np.float32)
        if box_arr.size == 0:
            return None
        # One face comes back as shape (4,), several faces as (n, 4).
        if box_arr.ndim == 1:
            box_arr = box_arr.reshape(1, 4)
        if probs is None:
            score_arr = np.ones(len(box_arr), dtype=np.float32)
        else:
            score_arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        best = None
        best_area = -1.0
        for i, box in enumerate(box_arr):
            score = float(score_arr[i]) if i < score_arr.size else 0.0
            if score < self.keep_prob:
                continue
            x1, y1, x2, y2 = [float(v) for v in np.ravel(box)[:4]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        if best is None:
            return None

        x1, y1, x2, y2 = best
        x1 = max(0.0, x1 - self.margin)
        y1 = max(0.0, y1 - self.margin)
        x2 = min(float(image.width), x2 + self.margin)
        y2 = min(float(image.height), y2 + self.margin)
        square = _square_box(x1, y1, x2, y2, image.width, image.height)
        if square is None:
            return None
        face = image.convert("RGB").crop(square)
        face = face.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.asarray(face, dtype=np.uint8)

    def prepare_many(self, images: list[Image.Image]) -> list[np.ndarray | None]:
        """Prepare a small batch. Passing a list keeps MTCNN.detect in batch mode."""
        if not images:
            return []
        if self.mode == "face_crop":
            return [self._resize(im) for im in images]

        self._load_mtcnn()
        # Frames in one batch are not all the same size, and MTCNN refuses
        # mixed sizes. Detect one image at a time.
        prepared = []
        for image in images:
            boxes, probs = self._mtcnn.detect(image)
            prepared.append(self._crop_box(image, boxes, probs))
        return prepared

    def prepare(self, image: Image.Image) -> np.ndarray | None:
        return self.prepare_many([image])[0]
