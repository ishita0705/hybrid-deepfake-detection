"""Metrics, ROC curves, and confusion matrices.

Every number written here is computed from predictions. Missing AUC (only one
class in the set) is stored as null and printed as "undefined".
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

import deepfake_hybrid.config as config


def log_line(message: str) -> None:
    print(message, flush=True)
    path = os.path.join(config.OUTPUT_DIR, config.DIR_LOGS, "train.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def smooth_labels(labels: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Binary label smoothing. 0 -> epsilon, 1 -> 1 - epsilon."""
    return labels * (1.0 - epsilon) + (1.0 - labels) * epsilon


@torch.no_grad()
def predict(model, loader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Sequence-level labels and P(fake). Test loaders are only passed here once."""
    model.eval()
    videos: list[str] = []
    labels: list[float] = []
    scores: list[float] = []
    use_amp = device.type == "cuda"
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        freq = batch["freq"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            probability = model(rgb, freq)
        scores.extend(probability.float().detach().cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
        videos.extend(batch["video"])
    return videos, np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    curve_fpr, curve_tpr, _thresholds = roc_curve(y_true, y_score)
    return float(auc(curve_fpr, curve_tpr))


def sequence_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    threshold = config.DECISION_THRESHOLD
    y_pred = (y_score >= threshold).astype(np.int64)
    return _pack(y_true, y_score, y_pred)


def video_metrics(videos: list[str], y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Majority vote for the hard label. Mean sequence score for AUC."""
    threshold = config.DECISION_THRESHOLD
    grouped: dict[str, dict] = {}
    for video, label, score in zip(videos, y_true, y_score):
        bucket = grouped.setdefault(video, {"label": int(label), "scores": []})
        if bucket["label"] != int(label):
            raise RuntimeError(f"Video {video} has two different labels.")
        bucket["scores"].append(float(score))

    y_video = []
    score_video = []
    pred_video = []
    for bucket in grouped.values():
        scores = bucket["scores"]
        fake_votes = sum(score >= threshold for score in scores)
        real_votes = len(scores) - fake_votes
        if fake_votes > real_votes:
            prediction = 1
        elif real_votes > fake_votes:
            prediction = 0
        else:
            # Tie: fall back to the mean sequence score.
            prediction = int(float(np.mean(scores)) >= threshold)
        y_video.append(bucket["label"])
        score_video.append(float(np.mean(scores)))
        pred_video.append(prediction)

    packed = _pack(
        np.asarray(y_video, dtype=np.int64),
        np.asarray(score_video, dtype=np.float64),
        np.asarray(pred_video, dtype=np.int64),
    )
    packed["y_true"] = np.asarray(y_video, dtype=np.int64)
    packed["y_score"] = np.asarray(score_video, dtype=np.float64)
    packed["y_pred"] = np.asarray(pred_video, dtype=np.int64)
    return packed


def _pack(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {
            "n": 0,
            "auc_roc": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
        }
    return {
        "n": int(len(y_true)),
        "auc_roc": _auc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
    }


def fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{float(value):.4f}"


def save_roc(path: str, y_true: np.ndarray, y_score: np.ndarray, title: str) -> bool:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        log_line(f"ROC skipped (need both classes): {title}")
        return False
    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    score = auc(fpr, tpr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label=f"AUC = {score:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def save_confusion(path: str, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], ["real", "fake"])
    axis.set_yticks([0, 1], ["real", "fake"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_table(path_stem: str, rows: list[dict]) -> None:
    """Write CSV and markdown. `rows` already hold computed metrics."""
    os.makedirs(os.path.dirname(path_stem), exist_ok=True)
    fieldnames = [
        "variant",
        "display",
        "test_set",
        "level",
        "n",
        "auc_roc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]
    csv_path = path_stem + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    video_rows = [row for row in rows if row["level"] == "video"]
    sequence_rows = [row for row in rows if row["level"] == "sequence"]
    lines = [
        "# Results",
        "",
        "Video-level accuracy, precision, recall, and F1 use a majority vote",
        "over sequence predictions. Video-level AUC uses the mean sequence score.",
        "A tie in the vote uses that mean score. Threshold = "
        f"{config.DECISION_THRESHOLD}.",
        "",
        "Positive class = fake.",
        "",
        "## Video level (4 variants x 3 test sets)",
        "",
        _markdown_table(video_rows),
        "",
        "## Sequence level",
        "",
        _markdown_table(sequence_rows),
        "",
    ]
    md_path = path_stem + ".md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    log_line(f"wrote {csv_path}")
    log_line(f"wrote {md_path}")


def _markdown_table(rows: list[dict]) -> str:
    header = "| variant | test set | n | AUC-ROC | accuracy | precision | recall | F1 |"
    rule = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            "| {display} | {test_set} | {n} | {auc} | {acc} | {prec} | {rec} | {f1} |".format(
                display=row["display"],
                test_set=row["test_set"],
                n=row["n"],
                auc=fmt(row["auc_roc"]),
                acc=fmt(row["accuracy"]),
                prec=fmt(row["precision"]),
                rec=fmt(row["recall"]),
                f1=fmt(row["f1"]),
            )
        )
    return "\n".join([header, rule, *body])
