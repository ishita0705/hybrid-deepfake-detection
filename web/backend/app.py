"""FastAPI backend for the deepfake prototype.

Serves the frontend, accepts uploads, runs the hybrid model, returns analysis JSON
and report assets from user_media/reports/.
"""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from deepfake_hybrid import config
from deepfake_hybrid.prototype import (
    DIR_CHECK,
    DIR_FAKE,
    DIR_REAL,
    DIR_REPORTS,
    IMAGE_EXTS,
    USER_MEDIA,
    VIDEO_EXTS,
    analyze_media_detailed,
    ensure_media_folders,
    _load_model,
)

WEB_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = WEB_ROOT / "frontend"
UPLOAD_DIR = USER_MEDIA / "uploads"

_lock = threading.Lock()
_model = None
_device = None
# Prefer SDFVD-finetuned prototype weights when available (see finetune_prototype).
_variant = "auto"


def _get_model():
    global _model, _device
    if _model is None:
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = _load_model(_variant, _device)
    return _model, _device


app = FastAPI(title="Deepfake Prototype API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    ensure_media_folders()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Warm the model so the first browser upload is faster.
    try:
        _get_model()
    except FileNotFoundError:
        # Checkpoint missing: API still starts; /api/health reports it.
        pass


@app.get("/api/health")
def health():
    import torch

    from deepfake_hybrid.prototype import prototype_checkpoint

    tuned = prototype_checkpoint()
    ckpt = tuned if tuned.is_file() else (
        Path(config.OUTPUT_DIR) / config.DIR_CHECKPOINTS / "full" / "best.pt"
    )
    model_ready = _model is not None
    return {
        "ok": True,
        "cuda": torch.cuda.is_available(),
        "device": str(_device) if _device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        "variant": _variant,
        "checkpoint_exists": ckpt.is_file(),
        "checkpoint": str(ckpt),
        "finetuned_sdfvd": tuned.is_file(),
        "model_loaded": model_ready,
        "user_media": str(USER_MEDIA),
    }


def _safe_report_dir(report_id: str) -> Path:
    if not report_id or ".." in report_id or "/" in report_id or "\\" in report_id:
        raise HTTPException(400, "Invalid report id")
    path = (DIR_REPORTS / report_id).resolve()
    if not str(path).startswith(str(DIR_REPORTS.resolve())):
        raise HTTPException(400, "Invalid report path")
    if not path.is_dir():
        raise HTTPException(404, "Report not found")
    return path


def _enrich(payload: dict, report_id: str | None = None) -> dict:
    rid = report_id or payload.get("id")
    if not rid:
        raise KeyError("report id missing")
    base = f"/api/reports/{rid}"
    payload = dict(payload)
    payload["id"] = rid
    if "face_files" not in payload:
        # Older reports may only embed faces in HTML.
        folder = DIR_REPORTS / rid
        payload["face_files"] = sorted(
            p.name for p in folder.glob("seq*_face.png")
        ) if folder.is_dir() else []
    if "has_gradcam" not in payload:
        payload["has_gradcam"] = (DIR_REPORTS / rid / "gradcam.png").is_file()
    if "media_name" not in payload and payload.get("media"):
        payload["media_name"] = Path(payload["media"]).name
    payload["urls"] = {
        "report_html": f"{base}/report.html",
        "analysis_json": f"{base}/analysis.json",
        "gradcam": f"{base}/gradcam.png" if payload.get("has_gradcam") else None,
        "faces": [f"{base}/{name}" for name in payload.get("face_files", [])],
    }
    return payload


@app.post("/api/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    folder_hint: str = Form("upload"),
):
    ensure_media_folders()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(
            400,
            f"Unsupported file type {suffix}. Use image or video.",
        )

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        with dest.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    finally:
        await file.close()

    # Also keep a copy under to_check for the user's media library.
    library = DIR_CHECK / Path(file.filename or dest.name).name
    if not library.exists():
        shutil.copy2(dest, library)

    try:
        with _lock:
            model, device = _get_model()
            _report, payload = analyze_media_detailed(
                dest,
                variant=_variant,
                folder_hint=folder_hint or "upload",
                open_browser=False,
                model=model,
                device=device,
            )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return _enrich(payload)


@app.post("/api/analyze-path")
def analyze_existing(relative_path: str = Form(...)):
    """Analyze a file already under user_media (e.g. to_check/clip.mp4)."""
    ensure_media_folders()
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(400, "Path must be relative under user_media")
    path = (USER_MEDIA / rel).resolve()
    if not str(path).startswith(str(USER_MEDIA.resolve())):
        raise HTTPException(400, "Path escapes user_media")
    if not path.is_file():
        raise HTTPException(404, f"File not found: {relative_path}")

    folder_hint = rel.parts[0] if rel.parts else ""
    try:
        with _lock:
            model, device = _get_model()
            _report, payload = analyze_media_detailed(
                path,
                variant=_variant,
                folder_hint=folder_hint,
                open_browser=False,
                model=model,
                device=device,
            )
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return _enrich(payload)


@app.get("/api/media")
def list_media():
    ensure_media_folders()
    items = []
    for folder, hint in (
        (DIR_CHECK, "to_check"),
        (DIR_REAL, "real"),
        (DIR_FAKE, "fake"),
    ):
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                items.append(
                    {
                        "name": path.name,
                        "folder": hint,
                        "relative_path": f"{hint}/{path.name}",
                        "size_bytes": path.stat().st_size,
                        "kind": "image" if path.suffix.lower() in IMAGE_EXTS else "video",
                    }
                )
    return {"items": items}


@app.get("/api/reports")
def list_reports():
    ensure_media_folders()
    rows = []
    if not DIR_REPORTS.is_dir():
        return {"items": []}
    for path in sorted(DIR_REPORTS.iterdir(), reverse=True):
        analysis = path / "analysis.json"
        if path.is_dir() and analysis.is_file():
            import json

            payload = json.loads(analysis.read_text(encoding="utf-8"))
            rows.append(_enrich(payload, report_id=path.name))
    return {"items": rows[:40]}


@app.get("/api/reports/{report_id}/analysis.json")
def get_analysis(report_id: str):
    import json

    path = _safe_report_dir(report_id) / "analysis.json"
    if not path.is_file():
        raise HTTPException(404, "analysis.json missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _enrich(payload)


@app.get("/api/reports/{report_id}/{filename}")
def get_report_file(report_id: str, filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    path = _safe_report_dir(report_id) / filename
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
