# Hybrid deepfake video detection — Phase 2

Spatial EfficientNet-B0, a frequency FFT branch, Bi-LSTM, and Grad-CAM. Runs in Google Colab (T4, PyTorch 2.x). Every path is set in `deepfake_hybrid/config.py`.

**Docs:** [Project report](docs/PROJECT_REPORT.md) · [Viva / interview Q&A](docs/VIVA_QA.md) · [Scope & deviations](scope_implemented.md)

## What is on disk

```
deepfake_hybrid/
  config.py        paths, ablation flag, training settings
  data_prep.py     inspect, FF++ subset, SDFVD, c40-style (simulated)
  faces.py         MTCNN box crop, or resize when the frame is already a face
  frequency.py     FFT magnitude map
  dataset.py       loads cached sequences (ImageNet norm on RGB)
  model.py         four ablation variants
  train.py         train, select by validation AUC
  evaluate.py      one pass over the three test sets
  explain.py       Grad-CAM overlays
  metrics.py       AUC, accuracy, precision, recall, F1, plots
notebooks/phase2_colab.ipynb
requirements.txt
scope_implemented.md
```

## Colab run order

1. Upload this project folder to Google Drive (or open it from Drive).
2. Put `sdfvd.zip` in `DRIVE_ROOT` (default: `MyDrive/sdfvd.zip`).
3. Open `notebooks/phase2_colab.ipynb` in Colab.
4. Mount Drive. Edit `DRIVE_ROOT` and `OUTPUT_DIR` in `deepfake_hybrid/config.py` if your Drive layout differs.
5. Install dependencies without replacing Colab's PyTorch:

```bash
pip install datasets Pillow opencv-python numpy tqdm timm scikit-learn matplotlib grad-cam
pip install facenet-pytorch --no-deps
```

Colab's system image already includes `ffmpeg`. The c40-style step needs `ffmpeg` on `PATH`.

6. From the project folder (the one that contains `deepfake_hybrid/`):

```bash
python -m deepfake_hybrid.data_prep --step inspect
```

Open `OUTPUT_DIR/inspect/sample_frame.jpg`. The log prints the image size and whether prep will treat frames as full frames or face crops. If that guess is wrong, set `FACE_INPUT_MODE` in `config.py` to `full_frame` or `face_crop` and run inspect again.

7. Prepare the three test sources, in this order:

```bash
python -m deepfake_hybrid.data_prep --step ffpp
python -m deepfake_hybrid.data_prep --step sdfvd
python -m deepfake_hybrid.data_prep --step c40
```

`python -m deepfake_hybrid.data_prep --step all` runs those four steps in the same order.

8. Train all four variants. The run downloads ImageNet weights for EfficientNet-B0 on the first variant, then scores each test set once:

```bash
python -m deepfake_hybrid.train --all
```

9. Grad-CAM overlays (full hybrid, EfficientNet-B0 branch):

```bash
python -m deepfake_hybrid.explain
```

`MODEL_VARIANT` in `config.py` selects a single variant for `python -m deepfake_hybrid.train --variant spatial`. That command does not read the test set. The 4×3 table is produced by `--all` (or by `python -m deepfake_hybrid.evaluate` after all four checkpoints exist).

## What each step writes

| Step | Role |
| --- | --- |
| inspect | Features, split sizes, category counts, one sample image, face-mode guess |
| ffpp | Balanced video subset, identity-aware 70/15/15 split, cached faces and FFT maps, sequence manifests. A video id is `category/name` (for example `Deepfakes/000_003`) so the four manipulation methods stay separate |
| sdfvd | Unzip, print the folder tree, every 5th frame, MTCNN, test-only sequences |
| c40 | Re-encode FF++ **test** frames with libx264 CRF 40, read them back, same face step |
| train | One checkpoint per variant, chosen by validation sequence AUC. Logs the parameter count |
| evaluate | `results/metrics.csv` and `results/metrics.md` (4 variants × 3 test sets), ROC curves, confusion matrices |
| explain | `results/gradcam/*.png` for 4 real and 4 fake frames, including SDFVD |

Counts and metrics are whatever the run computes. They are saved under `OUTPUT_DIR/manifests/` and `OUTPUT_DIR/results/`. Re-running prep skips face files that already exist when the prep settings are unchanged. Re-running `train --all` skips a variant that already has a checkpoint, and it does not score the test set a second time if `results/metrics.csv` is already there.

## Prototype web app (frontend + backend)

Separate media folders under `user_media/`. Browser UI + FastAPI:

```powershell
pip install fastapi uvicorn python-multipart
python web/run.py
```

Open **http://127.0.0.1:7860** — upload a video/image, or analyze files already in `user_media/to_check`, `real`, or `fake`. Reports land in `user_media/reports/`.

CLI still works: `python -m deepfake_hybrid.prototype path\to\file.mp4`

Needs `OUTPUT_DIR/checkpoints/full/best.pt`.

## Variants

| Flag | Model |
| --- | --- |
| `spatial` | EfficientNet-B0, mean over the 10 frames. Baseline |
| `spatial_freq` | Spatial and frequency vectors concatenated, then mean-pooled |
| `spatial_temporal` | Spatial vectors through a Bi-LSTM |
| `full` | Spatial + frequency, then Bi-LSTM |

The head is Linear → ReLU → Dropout → Linear → sigmoid.

## Split rule

Videos are grouped from names such as `000_003`. With `IDENTITY_GROUPING = "all_persons"` (the default), both ids stay in the same split, and the script asserts that no video and no person id appears in two splits. If the subset collapses to one identity component, set `IDENTITY_GROUPING = "source"` and re-run. Source mode groups by the first id only, still asserts zero source-id overlap, and prints any target-id overlap.
