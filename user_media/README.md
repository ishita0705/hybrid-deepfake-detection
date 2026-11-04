# User media (prototype)

Drop your own files here. This folder is separate from the training pipeline.

| Folder | Use |
| --- | --- |
| `real/` | Known-real videos or photos (optional, for your own demos) |
| `fake/` | Known-fake / deepfake videos or photos |
| `to_check/` | Anything you want the prototype to score |
| `reports/` | HTML analysis reports written by the prototype |

Supported: `.mp4` `.avi` `.mov` `.mkv` `.jpg` `.jpeg` `.png` `.webp`

## Run the prototype

From the project folder (`DL PROJECT`):

```powershell
# Analyze one file
python -m deepfake_hybrid.prototype "user_media\to_check\your_video.mp4"

# Analyze every file in to_check / real / fake
python -m deepfake_hybrid.prototype --scan

# Open a small web UI (upload a video or image)
python -m deepfake_hybrid.prototype --ui
```

Needs the trained checkpoint at `D:\deepfake_phase2\checkpoints\full\best.pt` (from `train --all`).
