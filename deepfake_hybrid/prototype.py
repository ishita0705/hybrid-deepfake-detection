"""Prototype: drop a video or image, get a REAL/FAKE report.

Media folders (separate from training data):
  user_media/real/
  user_media/fake/
  user_media/to_check/
  user_media/reports/   <- HTML reports land here

Usage:
  python -m deepfake_hybrid.prototype path\\to\\file.mp4
  python -m deepfake_hybrid.prototype --scan
  python -m deepfake_hybrid.prototype --ui
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import deepfake_hybrid.config as config
from deepfake_hybrid.faces import FacePrep
from deepfake_hybrid.frequency import frequency_map
from deepfake_hybrid.metrics import log_line
from deepfake_hybrid.model import build_model, checkpoint_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

_PROJECT = Path(config.DRIVE_ROOT)
USER_MEDIA = _PROJECT / "user_media"
DIR_REAL = USER_MEDIA / "real"
DIR_FAKE = USER_MEDIA / "fake"
DIR_CHECK = USER_MEDIA / "to_check"
DIR_REPORTS = USER_MEDIA / "reports"

_MEAN = torch.tensor(config.IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor(config.IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)


def ensure_media_folders() -> None:
    for path in (DIR_REAL, DIR_FAKE, DIR_CHECK, DIR_REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def _normalize_face(face_uint8: np.ndarray) -> torch.Tensor:
    array = face_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
    return (tensor - _MEAN) / _STD


def _load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _frames_from_video(path: Path) -> list[Image.Image]:
    """Sample every Nth frame, capped like SDFVD prep."""
    every = config.SDFVD_FRAME_EVERY
    max_frames = config.SDFVD_MAX_FRAMES
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[Image.Image] = []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % every == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            index += 1
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return frames


def _extract_faces(images: list[Image.Image], face_prep: FacePrep) -> list[np.ndarray]:
    faces = []
    for image in images:
        face = face_prep.prepare(image)
        if face is not None:
            faces.append(face)
    return faces


def _build_sequences(faces: list[np.ndarray]) -> list[dict]:
    """Pack faces into SEQ_LEN windows (pad short clips by repeating last face)."""
    if not faces:
        return []
    seq_len = config.SEQ_LEN
    stride = config.SEQ_STRIDE
    max_seqs = config.MAX_SEQS_PER_VIDEO

    # Single image / few faces: repeat to fill one window so the LSTM still runs.
    if len(faces) < seq_len:
        padded = list(faces) + [faces[-1]] * (seq_len - len(faces))
        windows = [padded]
    else:
        windows = []
        start = 0
        while start + seq_len <= len(faces) and len(windows) < max_seqs:
            windows.append(faces[start : start + seq_len])
            start += stride

    sequences = []
    for seq_index, window in enumerate(windows):
        rgb = torch.stack([_normalize_face(f) for f in window], dim=0)
        freq = torch.stack(
            [torch.from_numpy(frequency_map(f)).unsqueeze(0) for f in window],
            dim=0,
        )
        sequences.append(
            {
                "seq_index": seq_index,
                "rgb": rgb,
                "freq": freq,
                "faces_uint8": window,
            }
        )
    return sequences


def prototype_checkpoint() -> Path:
    """Prefer the SDFVD-finetuned demo weights when present."""
    tuned = Path(config.OUTPUT_DIR) / config.DIR_CHECKPOINTS / "prototype" / "best.pt"
    return tuned


def _load_model(variant: str, device: torch.device) -> torch.nn.Module:
    """Load weights for inference.

    `auto` / `prototype`: use SDFVD-finetuned `checkpoints/prototype/best.pt`
    when it exists, otherwise the original full hybrid checkpoint.
    Other names load that ablation from the Phase-2 train run.
    """
    tuned = prototype_checkpoint()
    if variant in ("auto", "prototype"):
        if tuned.is_file():
            path = tuned
            arch = "spatial"
        else:
            path = Path(checkpoint_file("full"))
            arch = "full"
    else:
        path = Path(checkpoint_file(variant))
        arch = variant

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. Train first: python -m deepfake_hybrid.train --all"
            + (
                "; for the web demo also run: python -m deepfake_hybrid.finetune_prototype"
                if not tuned.is_file()
                else ""
            )
        )
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    arch = blob.get("variant", arch)
    model = build_model(arch, pretrained=False).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def _score_sequences(model, sequences: list[dict], device: torch.device) -> list[float]:
    scores = []
    use_amp = device.type == "cuda"
    for item in sequences:
        rgb = item["rgb"].unsqueeze(0).to(device)
        freq = item["freq"].unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            score = float(model(rgb, freq).detach().cpu().reshape(-1)[0])
        scores.append(score)
    return scores


def _png_b64(array_uint8: np.ndarray) -> str:
    image = Image.fromarray(array_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _try_gradcam(model, face_uint8: np.ndarray, freq_map: np.ndarray, device: torch.device) -> np.ndarray | None:
    """Grad-CAM overlay on one face. Returns RGB uint8 or None if unavailable."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from deepfake_hybrid.explain import _FrameWrapper, _FakeScore, _disable_inplace, _allow_lstm_backward, _denormalize
    except Exception:
        return None

    _disable_inplace(model)
    wrapper = _FrameWrapper(model).to(device)
    wrapper.eval()
    try:
        cam = GradCAM(model=wrapper, target_layers=[model.cam_layer])
    except TypeError:
        cam = GradCAM(model=wrapper, target_layers=[model.cam_layer], use_cuda=device.type == "cuda")
    _allow_lstm_backward(model)

    rgb = _normalize_face(face_uint8).unsqueeze(0).to(device)
    wrapper.freq = torch.from_numpy(freq_map).unsqueeze(0).unsqueeze(0).to(device)
    try:
        grayscale = cam(input_tensor=rgb, targets=[_FakeScore()])[0]
        overlay = show_cam_on_image(face_uint8.astype(np.float32) / 255.0, grayscale, use_rgb=True)
        return overlay
    except Exception:
        return None
    finally:
        try:
            if getattr(cam, "activations_and_grads", None) is not None:
                cam.activations_and_grads.release()
        except Exception:
            pass


def _verdict(mean_score: float) -> tuple[str, str]:
    if mean_score >= config.DECISION_THRESHOLD:
        return "FAKE", "likely manipulated / deepfake"
    return "REAL", "likely authentic"


def _write_report(
    media_path: Path,
    folder_hint: str,
    sequences: list[dict],
    scores: list[float],
    variant: str,
    faces_kept: int,
    frames_seen: int,
    gradcam_b64: str | None,
    elapsed_s: float,
    gradcam_array: np.ndarray | None = None,
) -> tuple[Path, dict]:
    ensure_media_folders()
    mean_score = float(np.mean(scores)) if scores else 0.0
    video_pred = "fake" if sum(1 for s in scores if s >= config.DECISION_THRESHOLD) * 2 >= len(scores) else "real"
    # Tie-break with mean score (same idea as evaluate.py)
    if scores and sum(1 for s in scores if s >= config.DECISION_THRESHOLD) * 2 == len(scores):
        video_pred = "fake" if mean_score >= config.DECISION_THRESHOLD else "real"
    label, detail = _verdict(mean_score)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in media_path.stem)[:80]
    out_dir = DIR_REPORTS / f"{stamp}_{safe}"
    out_dir.mkdir(parents=True, exist_ok=True)

    thumbs = []
    for i, item in enumerate(sequences):
        mid = item["faces_uint8"][len(item["faces_uint8"]) // 2]
        thumb_path = out_dir / f"seq{i:02d}_face.png"
        Image.fromarray(mid).save(thumb_path)
        thumbs.append(
            {
                "file": thumb_path.name,
                "seq_index": i,
                "score": scores[i],
                "b64": _png_b64(mid),
            }
        )

    if gradcam_array is not None:
        Image.fromarray(gradcam_array).save(out_dir / "gradcam.png")

    report_id = out_dir.name
    face_files = [thumb["file"] for thumb in thumbs]
    payload = {
        "id": report_id,
        "media": str(media_path.resolve()),
        "media_name": media_path.name,
        "folder_hint": folder_hint,
        "variant": variant,
        "frames_seen": frames_seen,
        "faces_kept": faces_kept,
        "n_sequences": len(scores),
        "sequence_scores": scores,
        "mean_score": mean_score,
        "threshold": config.DECISION_THRESHOLD,
        "video_majority": video_pred,
        "verdict": label,
        "verdict_detail": detail,
        "elapsed_seconds": elapsed_s,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "face_files": face_files,
        "has_gradcam": gradcam_array is not None,
        "report_html": "report.html",
        "note": (
            "Prototype score from the Phase-2 hybrid checkpoint. "
            "The model was trained on a small FF++ subset; treat unknown "
            "internet videos as a demo, not a forensic guarantee."
        ),
    }
    (out_dir / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    bar_rows = ""
    for i, score in enumerate(scores):
        width = max(2, int(round(score * 100)))
        color = "#c0392b" if score >= config.DECISION_THRESHOLD else "#27ae60"
        bar_rows += (
            f"<tr><td>Sequence {i}</td><td class='num'>{score:.4f}</td>"
            f"<td><div class='bar' style='width:{width}%;background:{color}'></div></td></tr>\n"
        )

    thumb_html = ""
    for thumb in thumbs:
        pred = "FAKE" if thumb["score"] >= config.DECISION_THRESHOLD else "REAL"
        thumb_html += (
            f"<figure><img src='data:image/png;base64,{thumb['b64']}' alt='face'/>"
            f"<figcaption>seq {thumb['seq_index']} · {pred} · {thumb['score']:.3f}</figcaption></figure>\n"
        )

    cam_html = ""
    if gradcam_b64:
        cam_html = (
            "<h2>Grad-CAM (where the spatial branch looked)</h2>"
            f"<img class='cam' src='data:image/png;base64,{gradcam_b64}' alt='gradcam'/>"
        )

    verdict_class = "fake" if label == "FAKE" else "real"
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Deepfake report — {html.escape(media_path.name)}</title>
<style>
body {{ font-family: Segoe UI, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
h1 {{ font-size: 1.4rem; }}
.verdict {{ display: inline-block; padding: 0.55rem 1rem; font-weight: 700; letter-spacing: 0.04em; }}
.verdict.real {{ background: #e8f8ef; color: #1e8449; }}
.verdict.fake {{ background: #fdecea; color: #c0392b; }}
.meta {{ color: #444; max-width: 48rem; }}
table {{ border-collapse: collapse; margin-top: 1rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.6rem; }}
th {{ background: #f3f3f3; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.bar {{ height: 12px; border-radius: 2px; }}
.gallery {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; }}
figure {{ margin: 0; width: 140px; }}
figure img {{ width: 140px; height: 140px; object-fit: cover; display: block; border: 1px solid #ddd; }}
figcaption {{ font-size: 0.8rem; color: #444; margin-top: 0.25rem; }}
img.cam {{ max-width: 320px; border: 1px solid #ddd; }}
.note {{ background: #f7f7f7; padding: 0.75rem 1rem; max-width: 48rem; font-size: 0.9rem; }}
code {{ background: #f0f0f0; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>Deepfake analysis report</h1>
<p class="verdict {verdict_class}">{label}</p>
<p class="meta">{html.escape(detail)} · P(fake) mean = <strong>{mean_score:.4f}</strong>
(threshold {config.DECISION_THRESHOLD}) · majority vote = <strong>{video_pred.upper()}</strong></p>

<h2>Input</h2>
<ul class="meta">
<li>File: <code>{html.escape(str(media_path))}</code></li>
<li>Folder hint: {html.escape(folder_hint or "—")}</li>
<li>Model variant: {html.escape(variant)}</li>
<li>Frames read: {frames_seen} · faces kept: {faces_kept} · sequences: {len(scores)}</li>
<li>Runtime: {elapsed_s:.1f}s · {html.escape(payload["created_at"])}</li>
</ul>

<h2>Sequence scores</h2>
<table>
<tr><th>Window</th><th>P(fake)</th><th>Bar</th></tr>
{bar_rows if bar_rows else "<tr><td colspan='3'>No sequences (no face found)</td></tr>"}
</table>

<h2>Face crops used</h2>
<div class="gallery">
{thumb_html if thumb_html else "<p>No faces detected.</p>"}
</div>

{cam_html}

<p class="note">{html.escape(payload["note"])}</p>
<p class="meta">JSON: <code>{html.escape(str(out_dir / "analysis.json"))}</code></p>
</body>
</html>
"""
    report_path = out_dir / "report.html"
    report_path.write_text(page, encoding="utf-8")
    return report_path, payload


def analyze_media(
    media_path: str | Path,
    variant: str = "auto",
    folder_hint: str = "",
    open_browser: bool = True,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> Path:
    """Run the full prototype on one video or image. Returns report.html path."""
    report, _payload = analyze_media_detailed(
        media_path,
        variant=variant,
        folder_hint=folder_hint,
        open_browser=open_browser,
        model=model,
        device=device,
    )
    return report


def analyze_media_detailed(
    media_path: str | Path,
    variant: str = "auto",
    folder_hint: str = "",
    open_browser: bool = True,
    model: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> tuple[Path, dict]:
    """Same as analyze_media, also returns the analysis.json payload."""
    ensure_media_folders()
    path = Path(media_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTS | VIDEO_EXTS:
        raise ValueError(f"Unsupported type {suffix}. Use image or video.")

    started = time.time()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_prep = FacePrep("full_frame")

    if suffix in IMAGE_EXTS:
        images = [_load_image(path)]
    else:
        images = _frames_from_video(path)
    frames_seen = len(images)
    faces = _extract_faces(images, face_prep)
    sequences = _build_sequences(faces)
    if not sequences:
        raise RuntimeError(
            f"No face found in {path.name}. Try a clearer face photo or another clip."
        )

    if model is None:
        model = _load_model(variant, device)
    scores = _score_sequences(model, sequences, device)

    mid_seq = sequences[len(sequences) // 2]
    mid_face = mid_seq["faces_uint8"][len(mid_seq["faces_uint8"]) // 2]
    mid_freq = frequency_map(mid_face)
    cam = _try_gradcam(model, mid_face, mid_freq, device)
    cam_b64 = _png_b64(cam) if cam is not None else None

    report, payload = _write_report(
        media_path=path,
        folder_hint=folder_hint,
        sequences=sequences,
        scores=scores,
        variant=variant,
        faces_kept=len(faces),
        frames_seen=frames_seen,
        gradcam_b64=cam_b64,
        elapsed_s=time.time() - started,
        gradcam_array=cam,
    )
    mean_score = float(np.mean(scores))
    label, _ = _verdict(mean_score)
    log_line(f"prototype {path.name}: {label} P(fake)={mean_score:.4f} -> {report}")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(report.as_uri())
        except Exception:
            pass
    return report, payload


def _iter_scan_files() -> list[tuple[Path, str]]:
    ensure_media_folders()
    found: list[tuple[Path, str]] = []
    for folder, hint in ((DIR_CHECK, "to_check"), (DIR_REAL, "real"), (DIR_FAKE, "fake")):
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                found.append((path, hint))
    return found


def launch_ui(variant: str = "full") -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(
            "Gradio is missing. Install with: pip install gradio\n"
            "Or analyze a file without the UI:\n"
            '  python -m deepfake_hybrid.prototype "user_media\\to_check\\file.mp4"'
        ) from exc

    ensure_media_folders()

    def _run(file_obj):
        if file_obj is None:
            return "Drop a video or image first.", None
        path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
        report = analyze_media(path, variant=variant, folder_hint="upload", open_browser=False)
        data = json.loads((report.parent / "analysis.json").read_text(encoding="utf-8"))
        summary = (
            f"{data['verdict']} — P(fake)={data['mean_score']:.4f}\n"
            f"faces={data['faces_kept']} sequences={data['n_sequences']}\n"
            f"report: {report}"
        )
        return summary, str(report)

    demo = gr.Interface(
        fn=_run,
        inputs=gr.File(label="Video or image", file_types=["image", "video"]),
        outputs=[
            gr.Textbox(label="Verdict"),
            gr.File(label="Download HTML report"),
        ],
        title="Deepfake prototype",
        description=(
            f"Put demo files in `{USER_MEDIA}` "
            "(real / fake / to_check), or upload here. "
            "Uses the Phase-2 full hybrid checkpoint."
        ),
    )
    demo.launch()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prototype deepfake checker + HTML report")
    parser.add_argument("path", nargs="?", help="Video or image to analyze")
    parser.add_argument("--scan", action="store_true", help="Analyze user_media real/fake/to_check")
    parser.add_argument("--ui", action="store_true", help="Open Gradio upload UI")
    parser.add_argument(
        "--variant",
        default="auto",
        choices=["auto", "prototype", *config.ALL_VARIANTS],
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    ensure_media_folders()

    if args.ui:
        launch_ui(variant=args.variant)
        return

    if args.scan:
        files = _iter_scan_files()
        if not files:
            print(f"No media in {DIR_CHECK}, {DIR_REAL}, or {DIR_FAKE}")
            print("Copy videos/photos there, then run again.")
            return
        for path, hint in files:
            report = analyze_media(
                path,
                variant=args.variant,
                folder_hint=hint,
                open_browser=not args.no_browser,
            )
            print(report)
        return

    if not args.path:
        parser.print_help()
        print(f"\nMedia folders ready under:\n  {USER_MEDIA}")
        return

    report = analyze_media(
        args.path,
        variant=args.variant,
        open_browser=not args.no_browser,
    )
    print(report)


if __name__ == "__main__":
    main(sys.argv[1:])
