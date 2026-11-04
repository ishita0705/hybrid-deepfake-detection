# FrameForge — frontend + backend

Browser UI that uploads a video/image, calls the FastAPI backend, runs the Phase‑2 hybrid model, and shows the analysis report.

## Folders

| Path | Role |
| --- | --- |
| `web/frontend/` | HTML / CSS / JS UI |
| `web/backend/app.py` | FastAPI API |
| `user_media/real` `fake` `to_check` | Your videos & photos |
| `user_media/reports` | Generated reports |

## Start

From the project root, with the Python that has torch + project deps:

```powershell
cd "C:\Users\itl\Desktop\DL PROJECT"
$env:KMP_DUPLICATE_LIB_OK="TRUE"
pip install fastapi uvicorn python-multipart
python web/run.py
```

Open **http://127.0.0.1:7860**

## API

- `GET /api/health`
- `POST /api/analyze` — multipart file upload
- `POST /api/analyze-path` — analyze `user_media/...` file
- `GET /api/media` — list drop-folder files
- `GET /api/reports` — recent analyses
- `GET /api/reports/{id}/...` — report assets
