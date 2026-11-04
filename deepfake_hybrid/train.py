"""Train one ablation variant, or all four, then stop.

Selection metric is validation AUC-ROC. The test manifests are not read here.
`python -m deepfake_hybrid.train --all` trains every variant and then calls
evaluate, which scores each test set once.

FP16 (autocast + GradScaler) is used when CUDA is available.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

import deepfake_hybrid.config as config
from deepfake_hybrid.dataset import SequenceDataset, class_counts, make_loader
from deepfake_hybrid.metrics import fmt, log_line, predict, sequence_metrics, smooth_labels, video_metrics
from deepfake_hybrid.model import VARIANTS, build_model, checkpoint_file, parameter_count


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _micro_batch() -> int:
    """How many sequences fit on the GPU. The optimizer still uses BATCH_SIZE."""
    if config.MICRO_BATCH_SIZE:
        return int(config.MICRO_BATCH_SIZE)
    if not torch.cuda.is_available():
        return config.BATCH_SIZE
    gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gib < 6:
        return 1
    if gib < 10:
        return 2
    return config.BATCH_SIZE


def _sampler(dataset: SequenceDataset) -> WeightedRandomSampler:
    counts = class_counts(dataset)
    log_line(
        f"train sequences: real={counts[config.LABEL_REAL]} "
        f"fake={counts[config.LABEL_FAKE]}"
    )
    if counts[config.LABEL_REAL] == 0 or counts[config.LABEL_FAKE] == 0:
        raise RuntimeError("Training needs both real and fake sequences.")
    # Inverse-frequency weights. A balanced set gets equal weights.
    weight_of = {
        config.LABEL_REAL: 1.0 / counts[config.LABEL_REAL],
        config.LABEL_FAKE: 1.0 / counts[config.LABEL_FAKE],
    }
    weights = [weight_of[int(row["label"])] for row in dataset.rows]
    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    return WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def _run_epoch(model, loader, optimizer, scaler, device, train: bool) -> float:
    model.train(train)
    use_amp = device.type == "cuda"
    total = 0.0
    seen = 0
    accum = max(1, config.BATCH_SIZE // _micro_batch()) if train else 1
    pending = 0
    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            freq = batch["freq"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if train and pending == 0:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                probability = model(rgb, freq)
            # BCE in float32. The sigmoid head already produced a probability.
            targets = smooth_labels(labels, config.LABEL_SMOOTHING)
            loss = F.binary_cross_entropy(probability.float().clamp(1e-6, 1.0 - 1e-6), targets)
            if train:
                scaler.scale(loss / accum).backward()
                pending += 1
                if pending == accum:
                    scaler.step(optimizer)
                    scaler.update()
                    pending = 0
            total += float(loss.detach().cpu()) * labels.shape[0]
            seen += int(labels.shape[0])
        if train and pending:
            scaler.step(optimizer)
            scaler.update()
    return total / max(seen, 1)


def _val_scores(model, loader, device) -> tuple[float | None, float | None]:
    """Sequence AUC selects the checkpoint. Video AUC is logged beside it.

    The val split is a small share of an already reduced subset, so the
    sequence score is the more stable selection metric. Test metrics later
    use the video-level majority vote.
    """
    videos, labels, scores = predict(model, loader, device)
    sequence_auc = sequence_metrics(labels, scores)["auc_roc"]
    video_auc = video_metrics(videos, labels, scores)["auc_roc"]
    return sequence_auc, video_auc


def train_variant(variant: str) -> str:
    """Train `variant` and return the best-checkpoint path."""
    set_seed(config.SEED)
    device = _device()
    use_amp = device.type == "cuda"
    log_line(f"=== train {variant} ({VARIANTS[variant]['display']}) device={device.type} amp={use_amp} ===")

    train_set = SequenceDataset("ffpp_sequences_train.json")
    pin = device.type == "cuda"
    micro = _micro_batch()
    log_line(
        f"optimizer batch={config.BATCH_SIZE} micro_batch={micro} "
        f"accum_steps={max(1, config.BATCH_SIZE // micro)}"
    )
    train_loader = make_loader(
        "ffpp_sequences_train.json",
        batch_size=micro,
        sampler=_sampler(train_set),
        pin_memory=pin,
    )
    val_loader = make_loader("ffpp_sequences_val.json", batch_size=micro, shuffle=False, pin_memory=pin)
    log_line(
        f"split sizes (sequences): train={len(train_set)} val={len(val_loader.dataset)}"
    )
    log_line(f"val sequences by class: {class_counts(val_loader.dataset)}")

    model = build_model(variant, pretrained=True).to(device)
    n_params = parameter_count(model)
    log_line(f"trainable parameters ({variant}): {n_params}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.LR_FACTOR,
        patience=config.LR_PLATEAU_PATIENCE,
    )
    # enabled=False on CPU keeps one code path. CUDA uses autocast + GradScaler.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = None
    stall = 0
    history = []
    ckpt_path = checkpoint_file(variant)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, scaler, device, train=True)
        val_loss = _run_epoch(model, val_loader, optimizer, scaler, device, train=False)
        val_auc, val_video_auc = _val_scores(model, val_loader, device)
        if val_auc is not None:
            scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_video_auc": val_video_auc,
                "lr": lr,
            }
        )
        log_line(
            f"epoch {epoch}/{config.MAX_EPOCHS} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_seq_auc={fmt(val_auc)} val_video_auc={fmt(val_video_auc)} lr={lr:.2e}"
        )
        improved = val_auc is not None and (best_auc is None or val_auc > best_auc)
        if improved:
            best_auc = val_auc
            stall = 0
            torch.save(
                {
                    "variant": variant,
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "param_count": n_params,
                    "seed": config.SEED,
                    "selected_by": "val_auc",
                    "state_dict": model.state_dict(),
                },
                ckpt_path,
            )
            log_line(f"saved {ckpt_path} (val_auc={fmt(val_auc)})")
        else:
            stall += 1
            if stall >= config.EARLY_STOP_PATIENCE:
                log_line(f"early stop at epoch {epoch} (no val AUC gain for {stall} epochs)")
                break

    if not os.path.exists(ckpt_path):
        torch.save(
            {
                "variant": variant,
                "epoch": history[-1]["epoch"] if history else 0,
                "val_auc": None,
                "param_count": n_params,
                "seed": config.SEED,
                "selected_by": "last_epoch",
                "state_dict": model.state_dict(),
            },
            ckpt_path,
        )
        log_line(f"val AUC was undefined. Saved the last epoch to {ckpt_path}")

    history_path = os.path.join(os.path.dirname(ckpt_path), "history.json")
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    _update_param_counts(variant, n_params)
    return ckpt_path


def _update_param_counts(variant: str, n_params: int) -> None:
    path = os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS, "param_counts.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    payload[variant] = n_params
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _results_exist() -> bool:
    return os.path.exists(os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS, "metrics.csv"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train ablation variants. Does not read the test set.")
    parser.add_argument("--all", action="store_true", help="Train all four variants, then evaluate once.")
    parser.add_argument("--variant", default=None, choices=list(VARIANTS), help="Train a single variant.")
    parser.add_argument("--retrain", action="store_true", help="Ignore an existing checkpoint and train again.")
    args = parser.parse_args(argv)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    log_line(f"SEED={config.SEED} OUTPUT_DIR={config.OUTPUT_DIR}")
    _log_prep_counts()

    if args.all:
        variants = list(config.ALL_VARIANTS)
    else:
        variants = [args.variant or config.MODEL_VARIANT]

    trained_any = False
    for variant in variants:
        ckpt = checkpoint_file(variant)
        if os.path.exists(ckpt) and not args.retrain:
            log_line(f"checkpoint exists, skipping training: {ckpt}")
            continue
        if args.retrain and os.path.exists(ckpt):
            os.remove(ckpt)
        train_variant(variant)
        trained_any = True

    # A single-variant run does not touch the test set. --all evaluates after
    # every variant has a checkpoint, and only if that pass has not been written.
    if not args.all:
        log_line("single-variant training finished. Test evaluation runs from: python -m deepfake_hybrid.train --all")
        return

    if trained_any and _results_exist():
        # The old table was computed from previous weights.
        results_dir = os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS)
        for name in ("metrics.csv", "metrics.md"):
            stale = os.path.join(results_dir, name)
            if os.path.exists(stale):
                os.remove(stale)
        log_line("removed stale metrics so the new checkpoints are scored once")

    from deepfake_hybrid.evaluate import evaluate_all

    evaluate_all()


def _log_prep_counts() -> None:
    """Print split counts already written by data prep. Nothing is invented here."""
    folder = os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS)
    for name in (
        "ffpp_balance_counts.json",
        "ffpp_split_counts.json",
        "sdfvd_counts.json",
        "c40_style_counts.json",
    ):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            log_line(f"prep counts not found yet: {name}")
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if name == "ffpp_balance_counts.json":
            brief = {"before_balance": payload.get("before_balance"), "selected": payload.get("selected")}
        elif name in ("sdfvd_counts.json", "c40_style_counts.json"):
            brief = payload.get("summary", payload)
        else:
            brief = payload
        log_line(f"{name}: {brief}")


if __name__ == "__main__":
    main(sys.argv[1:])
