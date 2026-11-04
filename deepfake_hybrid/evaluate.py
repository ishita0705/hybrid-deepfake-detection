"""Score every trained variant on the three test sets, once.

Reads checkpoints only. Does not train and does not update weights.
If metrics.csv is already present, this prints it and returns, so a second
Colab run does not score the test set again.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch

import deepfake_hybrid.config as config
from deepfake_hybrid.dataset import make_loader
from deepfake_hybrid.metrics import (
    log_line,
    predict,
    save_confusion,
    save_roc,
    sequence_metrics,
    video_metrics,
    write_table,
)
from deepfake_hybrid.model import VARIANTS, build_model, checkpoint_file

# Order is the rows of the 4 x 3 table.
TEST_SETS = (
    {"key": "ffpp_c23", "display": "FF++ c23", "manifest": "ffpp_sequences_test.json"},
    {
        "key": "c40_style",
        "display": config.C40_LABEL,
        "manifest": "c40_style_sequences_test.json",
    },
    {"key": "sdfvd", "display": "SDFVD", "manifest": "sdfvd_sequences_test.json"},
)


def _results_csv() -> str:
    return os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS, "metrics.csv")


def _load_weights(variant: str, device: torch.device):
    path = checkpoint_file(variant)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint for {variant} at {path}. Train it first.")
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    model = build_model(variant, pretrained=False).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    log_line(
        f"loaded {variant} from epoch {blob.get('epoch')} "
        f"val_auc={blob.get('val_auc')} selected_by={blob.get('selected_by')} "
        f"params={blob.get('param_count')}"
    )
    return model


def evaluate_all(force: bool = False) -> None:
    csv_path = _results_csv()
    if os.path.exists(csv_path) and not force:
        log_line(f"test evaluation already written: {csv_path}")
        log_line("Leaving it in place so each variant is scored once.")
        _print_csv(csv_path)
        return

    for variant in config.ALL_VARIANTS:
        path = checkpoint_file(variant)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing checkpoint for {variant}. Run python -m deepfake_hybrid.train --all first."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    from deepfake_hybrid.train import _micro_batch

    micro = _micro_batch()
    rows = []
    results_dir = os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS)
    os.makedirs(results_dir, exist_ok=True)

    for variant in config.ALL_VARIANTS:
        model = _load_weights(variant, device)
        display = VARIANTS[variant]["display"]
        for spec in TEST_SETS:
            loader = make_loader(spec["manifest"], batch_size=micro, shuffle=False, pin_memory=pin)
            videos, labels, scores = predict(model, loader, device)
            sequence = sequence_metrics(labels, scores)
            video = video_metrics(videos, labels, scores)
            log_line(
                f"{display} | {spec['display']} | sequences={sequence['n']} "
                f"seq_auc={sequence['auc_roc']} | videos={video['n']} video_auc={video['auc_roc']}"
            )
            rows.append(_row(variant, display, spec["display"], "sequence", sequence))
            rows.append(_row(variant, display, spec["display"], "video", video))
            stem = f"{variant}__{spec['key']}"
            save_roc(
                os.path.join(results_dir, "roc", stem + "_video.png"),
                video["y_true"],
                video["y_score"],
                f"{display} — {spec['display']} (video)",
            )
            save_confusion(
                os.path.join(results_dir, "confusion", stem + "_video.png"),
                video["y_true"],
                video["y_pred"],
                f"{display} — {spec['display']} (video)",
            )
            save_roc(
                os.path.join(results_dir, "roc", stem + "_sequence.png"),
                sequence["y_true"],
                sequence["y_score"],
                f"{display} — {spec['display']} (sequence)",
            )
            save_confusion(
                os.path.join(results_dir, "confusion", stem + "_sequence.png"),
                sequence["y_true"],
                sequence["y_pred"],
                f"{display} — {spec['display']} (sequence)",
            )
            _write_predictions(results_dir, stem, videos, labels, scores)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_table(os.path.join(results_dir, "metrics"), rows)
    _append_dataset_sizes(os.path.join(results_dir, "metrics.md"))
    _print_csv(csv_path)


def _row(variant: str, display: str, test_set: str, level: str, metrics: dict) -> dict:
    return {
        "variant": variant,
        "display": display,
        "test_set": test_set,
        "level": level,
        "n": metrics["n"],
        "auc_roc": metrics["auc_roc"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def _write_predictions(results_dir: str, stem: str, videos, labels, scores) -> None:
    folder = os.path.join(results_dir, "predictions")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, stem + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "label", "score"])
        for video, label, score in zip(videos, labels, scores):
            writer.writerow([video, int(label), float(score)])


def _append_dataset_sizes(markdown_path: str) -> None:
    """Copy prep-run counts onto the end of the results markdown."""
    folder = os.path.join(config.OUTPUT_DIR, config.DIR_MANIFESTS)
    sections = []
    balance_path = os.path.join(folder, "ffpp_balance_counts.json")
    if os.path.exists(balance_path):
        with open(balance_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        sections.append(
            "FF++ video counts before balance: "
            + json.dumps(payload.get("before_balance"))
            + "\n\nFF++ selected video counts: "
            + json.dumps(payload.get("selected"))
        )
    for name, title in (
        ("ffpp_split_counts.json", "FF++ sequence counts by split"),
        ("sdfvd_counts.json", "SDFVD"),
        ("c40_style_counts.json", "c40-style (simulated)"),
    ):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if "summary" in payload:
            payload = payload["summary"]
        sections.append(f"{title}: {json.dumps(payload)}")

    param_path = os.path.join(config.OUTPUT_DIR, config.DIR_RESULTS, "param_counts.json")
    if os.path.exists(param_path):
        with open(param_path, encoding="utf-8") as handle:
            sections.append("Trainable parameters: " + json.dumps(json.load(handle)))

    if not sections:
        return
    with open(markdown_path, "a", encoding="utf-8") as handle:
        handle.write("\n## Counts recorded by the prep and training runs\n\n")
        for section in sections:
            handle.write(section + "\n\n")


def _print_csv(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        log_line(handle.read().rstrip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate all variants once.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Score the test sets again even if metrics.csv exists.",
    )
    args = parser.parse_args(argv)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    evaluate_all(force=args.force)


if __name__ == "__main__":
    main(sys.argv[1:])
