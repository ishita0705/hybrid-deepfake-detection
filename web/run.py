"""Start the FrameForge frontend + FastAPI backend.

  python web/run.py
  # or
  python -m web.run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    print(f"FrameForge -> http://{host}:{port}")
    print(f"Media folder -> {ROOT / 'user_media'}")
    uvicorn.run(
        "web.backend.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
