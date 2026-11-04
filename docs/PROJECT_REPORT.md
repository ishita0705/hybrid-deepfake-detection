# Hybrid Deepfake Video Detection — Project Report

**Phase 2 mini-project**  
**Stack:** PyTorch · EfficientNet-B0 · FFT frequency CNN · Bi-LSTM · FastAPI web demo  
**Hardware used for the reported run:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), CUDA 11.8 PyTorch build

---

## 1. Abstract

This project builds a **hybrid deepfake detector** that combines three cues:

1. **Spatial** appearance features from a pretrained EfficientNet-B0  
2. **Frequency** forensic maps from a 2D FFT magnitude transform  
3. **Temporal** context via a bidirectional LSTM over short face sequences  

Four ablation variants are trained on a reduced FaceForensics++ (c23) subset and evaluated on (a) FF++ c23 test, (b) a simulated high-compression “c40-style” set, and (c) the cross-dataset SDFVD collection **without retraining**. A web prototype (FrameForge) lets a user upload a video or photo and receive a REAL/FAKE report with Grad-CAM explanation. For the interactive demo, the spatial checkpoint is further **fine-tuned on SDFVD** so in-browser checks separate real vs face-swap clips more reliably.

---

## 2. Problem statement

Deepfake tools can swap faces in video with high visual quality. Manual inspection does not scale. The goal of this mini-project is to:

- Detect manipulated face videos using deep learning  
- Compare spatial-only, frequency-augmented, temporal, and full hybrid models  
- Test compression robustness and cross-dataset generalization  
- Ship a small end-to-end demo (data → train → evaluate → web UI)

---

## 3. Objectives

| # | Objective | Status |
| --- | --- | --- |
| 1 | Prepare FF++ c23 frames with identity-aware splits | Done |
| 2 | Train 4 ablation architectures under one codebase | Done |
| 3 | Evaluate once on FF++ c23, c40-style, SDFVD | Done |
| 4 | Produce metrics tables, ROC, confusion, Grad-CAM | Done |
| 5 | Provide a frontend + backend upload prototype | Done |

---

## 4. Related work (brief)

- **FaceForensics++** is a standard benchmark with multiple manipulation methods and compression levels (c23 / c40).  
- **Frequency artifacts** (FFT / spectrum anomalies) are widely used because generative pipelines leave periodic or band-energy clues.  
- **CNN + RNN/LSTM** hybrids model both frame quality and temporal inconsistency.  
- **Grad-CAM** visualizes which face regions drive a CNN decision.  
- Cross-dataset testing (e.g. Celeb-DF / SDFVD) measures whether a model overfits one corpus.

---

## 5. Datasets

### 5.1 FaceForensics++ c23 (training + in-domain test)

- Source: Hugging Face mirror `TsienDragon/ffplusplus_c23_frames` (pre-extracted frames).  
- Kept categories: `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`.  
- Dropped: `FaceShifter`, `DeepFakeDetection`.  
- After “≥10 frames” filter and balancing: **20 real + 20 fake** videos (5 fakes per manipulation type).  
- Identity-aware split **70 / 15 / 15** by video (same person IDs never cross splits).  
- Sequences in the reported run: train **148**, val **30**, test **30**.

### 5.2 c40-style (simulated) — compression test

- FF++ **test** frames re-encoded with `libx264 -crf 40`, decoded, then face-cropped again.  
- Labelled everywhere as **c40-style (simulated)** — not the official FF++ c40 release.  
- Same 6 test videos → **30** sequences.

### 5.3 SDFVD — cross-dataset test / demo fine-tune

- **53 real** (Pexels) + **53 fake** (Remaker AI face-swap), CC BY 4.0.  
- Used as **test-only** in Phase-2 tables (no training on SDFVD for those metrics).  
- Later used to **fine-tune** the web prototype checkpoint (separate from the Phase-2 ablation table).

---

## 6. Methodology

### 6.1 Preprocessing pipeline

```
Video / frame set
  → sample frames (FF++: keep provided frame_idx; SDFVD: every 5th, max 60)
  → MTCNN box crop (no landmark alignment), margin, square, resize 224×224
  → cache RGB face JPG + FFT frequency .npy
  → build sequences: length 10, stride 10, max 6 per video
  → ImageNet normalize RGB at load time
```

**Frequency map:** grayscale → 2D FFT → `fftshift` → `log(1+|F|)` → per-image min–max norm.

### 6.2 Model architecture

| Variant | Spatial EfficientNet-B0 | Frequency CNN | Bi-LSTM | Pooling |
| --- | --- | --- | --- | --- |
| `spatial` | ✓ | | | mean over frames |
| `spatial_freq` | ✓ | ✓ | | mean |
| `spatial_temporal` | ✓ | | ✓ | LSTM hidden |
| `full` | ✓ | ✓ | ✓ | LSTM hidden |

**Head:** Linear → ReLU → Dropout → Linear → Sigmoid → **P(fake)**.

**Trainable parameters (architecture build):**  
spatial 4,335,741 · spatial_freq 4,461,629 · spatial_temporal 7,288,957 · full 7,644,221.

### 6.3 Training setup

- Loss: BCE with label smoothing (ε = 0.1)  
- Optimizer: Adam, lr `1e-4`, weight decay `1e-4`  
- Schedule: ReduceLROnPlateau on val sequence AUC  
- Early stopping patience 3, max 10 epochs  
- Effective batch 8 via micro-batch 4 + gradient accumulation (4 GB GPU)  
- FP16 autocast + GradScaler  
- Gradient checkpointing on EfficientNet during training  
- Weighted sampler when real/fake sequence counts differ  
- Checkpoint chosen by **best validation sequence AUC**

### 6.4 Evaluation protocol

- Each variant scored **once** on each of the three test sets.  
- **Sequence metrics:** AUC, accuracy, precision, recall, F1 (threshold 0.5).  
- **Video metrics:** mean sequence score for AUC; majority vote for accuracy / precision / recall / F1.  
- Positive class = **fake**.

### 6.5 Web prototype (FrameForge)

- **Frontend:** static HTML/CSS/JS upload UI  
- **Backend:** FastAPI (`web/backend/app.py`)  
- Upload → face prep → sequences → model → HTML/JSON report + Grad-CAM  
- Demo weights: `checkpoints/prototype/best.pt` (spatial model fine-tuned on SDFVD; val AUC ≈ **0.98** on the held-out SDFVD split used for that fine-tune)

---

## 7. Experimental results (Phase-2 ablation run)

Numbers below are from the completed local run written to `results/metrics.md` / `metrics.csv`. They are **not** hardcoded targets.

### 7.1 Video-level AUC-ROC (main table)

| Variant | FF++ c23 | c40-style | SDFVD |
| --- | ---: | ---: | ---: |
| spatial-only | **0.8000** | 0.4000 | 0.5041 |
| spatial+frequency | 0.6000 | 0.2000 | 0.5039 |
| spatial+temporal | 0.4000 | 0.0000 | 0.4745 |
| full hybrid | 0.4000 | 0.2000 | 0.4950 |

### 7.2 Observations

1. **In-domain (FF++ c23):** the spatial baseline achieved the highest video AUC (0.80) on a tiny 6-video test set (5 real / 1 fake). Precision/recall at 0.5 are unstable because of that imbalance.  
2. **Compression (c40-style):** all variants degrade sharply — compression removes or alters forensic cues.  
3. **Cross-dataset (SDFVD):** AUCs ≈ 0.50 — near chance. The Phase-2 models did **not** generalize off FF++ with this small subset.  
4. **Capacity vs data:** fuller hybrids (LSTM / frequency) did not beat the spatial baseline under this data regime; they likely overfit or were under-trained (early stop around epoch 5, best epoch often 2).  
5. **Prototype fine-tune:** adapting the spatial checkpoint on SDFVD raised held-out SDFVD val AUC to **~0.98**, which is what the upload demo uses. That fine-tune is **separate** from the ablation table above.

---

## 8. System / repository layout

```
DL PROJECT/
  deepfake_hybrid/     # config, data_prep, model, train, evaluate, explain, prototype
  web/
    frontend/          # FrameForge UI
    backend/app.py     # FastAPI
    run.py             # python web/run.py → http://127.0.0.1:7860
  notebooks/           # Jupyter / Colab-oriented notebook
  user_media/          # demo uploads (not committed)
  docs/                # this report + viva Q&A
  README.md
  scope_implemented.md
  requirements.txt
```

Large artifacts (HF cache, face caches, checkpoints, videos) live under `D:\deepfake_phase2` on the development machine and are **gitignored**.

---

## 9. How to reproduce

```powershell
# 1) Install deps (use a CUDA PyTorch matching your GPU)
pip install -r requirements.txt
pip install facenet-pytorch --no-deps
pip install fastapi uvicorn python-multipart

# 2) Data + train + Grad-CAM (long)
python -m deepfake_hybrid.data_prep --step all
python -m deepfake_hybrid.train --all
python -m deepfake_hybrid.explain

# 3) Optional: demo fine-tune + web UI
python -m deepfake_hybrid.finetune_prototype
python web/run.py
```

---

## 10. Limitations and ethics

- Mini-project scale (40 FF++ videos) — not production-grade detection.  
- HF third-party mirror; simulated c40; SDFVD instead of Celeb-DF.  
- No face alignment; MTCNN box crop only.  
- Cross-dataset Phase-2 scores are weak without domain adaptation.  
- Deepfake detectors can be biased and can be wrong on celebrity / web photos.  
- Use only for education / research demos; do not treat outputs as legal evidence.

---

## 11. Conclusion

The project delivers a complete Phase-2 pipeline: hybrid modelling with clear ablations, three-way evaluation, explainability (Grad-CAM), and a FastAPI + browser prototype. Empirically, with a small FF++ subset the **spatial baseline** was strongest in-domain, while **cross-dataset** performance required an additional SDFVD fine-tune for a usable demo. Future work should scale training data (`POOL_SPLITS` including val), use official c40, and study calibrated thresholds and stronger backbones.

---

## 12. References (indicative)

1. Rössler et al., *FaceForensics++*, ICCV 2019.  
2. Tan & Le, *EfficientNet*, ICML 2019.  
3. Selvaraju et al., *Grad-CAM*, ICCV 2017.  
4. SDFVD dataset, Mendeley Data (CC BY 4.0).  
5. Project files: `README.md`, `scope_implemented.md`, `results/metrics.md`.
