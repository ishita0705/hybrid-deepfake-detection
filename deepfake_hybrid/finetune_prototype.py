"""Fine-tune the Phase-2 checkpoint on SDFVD so the web prototype can
separate real vs fake on that domain (and similar face clips).

Saves: OUTPUT_DIR/checkpoints/prototype/best.pt
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import deepfake_hybrid.config as config
from deepfake_hybrid.faces import FacePrep
from deepfake_hybrid.frequency import frequency_map
from deepfake_hybrid.metrics import log_line
from deepfake_hybrid.model import build_model, checkpoint_file
from deepfake_hybrid.prototype import (
    IMAGE_EXTS,
    _build_sequences,
    _extract_faces,
    _frames_from_video,
    _load_image,
    _load_model,
    _normalize_face,
)


SDFVD_REAL = Path(config.OUTPUT_DIR) / "sdfvd_unzipped" / "SDFVD" / "videos_real"
SDFVD_FAKE = Path(config.OUTPUT_DIR) / "sdfvd_unzipped" / "SDFVD" / "videos_fake"
PROTO_DIR = Path(config.OUTPUT_DIR) / config.DIR_CHECKPOINTS / "prototype"
CACHE_DIR = Path(config.OUTPUT_DIR) / "prototype_finetune_cache"


def _list_videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".mp4")


def _cache_video(path: Path, label: int, face_prep: FacePrep) -> list[dict]:
    """Cache face/freq tensors for one video; return sequence dicts."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)
    out = CACHE_DIR / f"{'real' if label == 0 else 'fake'}_{safe}"
    meta_path = out / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sequences = []

        def _load_tensor(name: str):
            try:
                return torch.load(out / name, weights_only=True)
            except TypeError:
                return torch.load(out / name)

        for item in meta["sequences"]:
            sequences.append(
                {
                    "rgb": _load_tensor(item["rgb"]),
                    "freq": _load_tensor(item["freq"]),
                    "label": label,
                    "video": path.name,
                }
            )
        return sequences

    out.mkdir(parents=True, exist_ok=True)
    images = _frames_from_video(path)
    faces = _extract_faces(images, face_prep)
    built = _build_sequences(faces)
    sequences = []
    meta_seqs = []
    for i, item in enumerate(built):
        rgb_path = f"seq{i:02d}_rgb.pt"
        freq_path = f"seq{i:02d}_freq.pt"
        torch.save(item["rgb"], out / rgb_path)
        torch.save(item["freq"], out / freq_path)
        sequences.append(
            {
                "rgb": item["rgb"],
                "freq": item["freq"],
                "label": label,
                "video": path.name,
            }
        )
        meta_seqs.append({"rgb": rgb_path, "freq": freq_path})
    meta_path.write_text(json.dumps({"sequences": meta_seqs}), encoding="utf-8")
    return sequences


class SeqDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "rgb": row["rgb"],
            "freq": row["freq"],
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


def _split_videos(reals: list[Path], fakes: list[Path], seed: int = 42):
    rng = random.Random(seed)
    reals = list(reals)
    fakes = list(fakes)
    rng.shuffle(reals)
    rng.shuffle(fakes)
    # 40 train / 13 val each side (53 total)
    n_val = 13
    return {
        "train": [(p, 0) for p in reals[n_val:]] + [(p, 1) for p in fakes[n_val:]],
        "val": [(p, 0) for p in reals[:n_val]] + [(p, 1) for p in fakes[:n_val]],
    }


@torch.no_grad()
def _eval_auc(model, loader, device) -> float:
    from sklearn.metrics import roc_auc_score

    model.eval()
    ys, ps = [], []
    for batch in loader:
        rgb = batch["rgb"].to(device)
        freq = batch["freq"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            pred = model(rgb, freq)
        ys.extend(batch["label"].numpy().tolist())
        ps.extend(pred.detach().cpu().float().numpy().tolist())
    if len(set(int(y) for y in ys)) < 2:
        return float("nan")
    return float(roc_auc_score(ys, ps))


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROTO_DIR.mkdir(parents=True, exist_ok=True)

    if not SDFVD_REAL.is_dir() or not SDFVD_FAKE.is_dir():
        raise SystemExit(f"SDFVD folders missing under {config.OUTPUT_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_prep = FacePrep("full_frame")
    splits = _split_videos(_list_videos(SDFVD_REAL), _list_videos(SDFVD_FAKE))

    log_line("fine-tune: caching SDFVD face sequences…")
    train_rows, val_rows = [], []
    for path, label in splits["train"]:
        train_rows.extend(_cache_video(path, label, face_prep))
    for path, label in splits["val"]:
        val_rows.extend(_cache_video(path, label, face_prep))

    log_line(f"fine-tune sequences train={len(train_rows)} val={len(val_rows)}")

    # Start from spatial baseline — it was the strongest FF++ checkpoint and
    # avoids LSTM quirks on short / padded image clips in the web demo.
    base = checkpoint_file("spatial")
    model = build_model("spatial", pretrained=False).to(device)
    try:
        blob = torch.load(base, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(base, map_location=device)
    model.load_state_dict(blob["state_dict"])

    # Freeze early EfficientNet blocks; adapt later layers + head.
    for name, param in model.named_parameters():
        if name.startswith("spatial.conv_stem") or name.startswith("spatial.bn1"):
            param.requires_grad = False
        elif ".blocks.0." in name or ".blocks.1." in name:
            param.requires_grad = False

    train_loader = DataLoader(SeqDataset(train_rows), batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(SeqDataset(val_rows), batch_size=4, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-5,
        weight_decay=1e-4,
    )
    loss_fn = nn.BCELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc = -1.0
    best_path = PROTO_DIR / "best.pt"
    epochs = 8

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            freq = batch["freq"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(rgb, freq)
            # BCELoss cannot run under autocast; cast predictions to float32.
            y_s = y * 0.9 + 0.05
            loss = loss_fn(pred.float(), y_s)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach().cpu())
            n += 1
        val_auc = _eval_auc(model, val_loader, device)
        log_line(
            f"fine-tune epoch {epoch}/{epochs} "
            f"train_loss={total / max(n, 1):.4f} val_auc={val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "variant": "spatial",
                    "state_dict": model.state_dict(),
                    "val_auc": best_auc,
                    "epoch": epoch,
                    "finetuned_on": "SDFVD",
                    "selected_by": "val_auc",
                },
                best_path,
            )
            log_line(f"saved {best_path} (val_auc={best_auc:.4f})")

    # Also write a small calibration note
    note = {
        "checkpoint": str(best_path),
        "best_val_auc": best_auc,
        "train_sequences": len(train_rows),
        "val_sequences": len(val_rows),
        "base_checkpoint": str(base),
    }
    (PROTO_DIR / "finetune_info.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    log_line(f"fine-tune done. best val AUC={best_auc:.4f}")


if __name__ == "__main__":
    main()
