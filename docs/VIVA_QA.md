# Likely viva / interview questions — Hybrid Deepfake Detection

Short model answers you can speak in 30–90 seconds. Numbers match the **actual run** on this project unless noted.

---

## A. Basics

### Q1. What is the problem you solved?
Detect whether a face video (or photo) is **real** or **deepfake/manipulated**, using a hybrid deep learning model, and wrap it in a small web demo.

### Q2. Why “hybrid”?
Deepfakes leave cues in different spaces: **pixels** (spatial), **spectrum** (frequency), and **over time** (temporal flicker / inconsistency). We combine EfficientNet features, an FFT CNN, and a Bi-LSTM.

### Q3. What is the positive class?
**Fake = 1**, real = 0. The model outputs **P(fake)** via a sigmoid.

### Q4. Datasets used?
- **Train/val/test:** FaceForensics++ c23 frames (HF mirror), reduced balanced subset.  
- **Compression test:** simulated c40 (CRF 40 re-encode).  
- **Cross-dataset:** SDFVD (53 real + 53 fake).  
- **Demo fine-tune:** SDFVD again (separate from the Phase-2 table).

---

## B. Pipeline & preprocessing

### Q5. Walk through the pipeline.
Frames → MTCNN face crop 224×224 → FFT frequency map → 10-frame sequences → model → P(fake) → video majority vote → metrics / Grad-CAM / web report.

### Q6. Why MTCNN without alignment?
Spec/scope: use box crop only. Alignment needs landmarks/warp; we expand the box with a margin, square it, and resize. Documented in `scope_implemented.md`.

### Q7. How are sequences built?
10 consecutive kept faces, stride 10 (no overlap), max 6 sequences per video. Short clips/images are padded by repeating the last face so the temporal head still runs.

### Q8. What is the frequency branch input?
Grayscale face → 2D FFT → fftshift → `log(1+|F|)` → normalize to [0,1] → small 3-layer CNN.

### Q9. Why cache faces as JPG + `.npy`?
MTCNN is slow. Training and evaluation load caches only; they never re-detect faces.

### Q10. How do you avoid identity leakage in splits?
Video IDs like `000_003` are grouped by person IDs (`IDENTITY_GROUPING`). Train/val/test must have **zero overlapping video IDs and person IDs**.

---

## C. Model & training

### Q11. Name the four ablations and why.
1. spatial — baseline CNN  
2. spatial_freq — add frequency  
3. spatial_temporal — add LSTM  
4. full — all three  
Ablations show **which cue helps** under the same data and protocol.

### Q12. Why EfficientNet-B0?
Strong ImageNet features, light enough for a 4 GB laptop GPU and Colab T4, and has a clear `conv_head` for Grad-CAM.

### Q13. Loss and optimizer?
BCE with **label smoothing**, Adam + weight decay, ReduceLROnPlateau, early stopping, FP16.

### Q14. How did you fit batch size 8 on 4 GB VRAM?
`MICRO_BATCH_SIZE=4` with gradient accumulation (2 steps), plus activation checkpointing on EfficientNet.

### Q15. How is the best checkpoint chosen?
Highest **validation sequence AUC** (more stable than video AUC on only 6 val videos). Reported test tables still use video-level metrics.

### Q16. Parameter counts?
Roughly 4.3M (spatial) up to 7.6M (full). Exact JSON is in `param_counts.json` from the run.

---

## D. Results & interpretation (expect this)

### Q17. What were your main results?
On the Phase-2 run, **spatial** had the best FF++ c23 video AUC (**0.80**). All models were weak on **c40-style** and near **chance (~0.50)** on SDFVD before demo fine-tuning.

### Q18. Why is SDFVD AUC ~0.5?
Domain shift: different cameras, face-swap tool (Remaker), and we trained on only **40 FF++ videos**. The model memorized FF++ cues that do not transfer.

### Q19. Why are precision/recall often 0 on FF++ / c40 tables?
The test split had **5 real videos and 1 fake**. If that one fake is missed, precision/recall for the fake class collapse even if AUC is nonzero.

### Q20. Did the full hybrid always win?
**No.** With little data, the simpler spatial model won in-domain. More parameters without more data often hurts.

### Q21. What is c40-style and why did scores drop?
Heavy H.264 compression (CRF 40) destroys high-frequency forensic traces. Detectors that rely on texture/spectrum usually degrade — our table shows that.

### Q22. How did you fix the web demo calling everything “FAKE”?
Fine-tuned the spatial checkpoint on SDFVD (`finetune_prototype`). Held-out SDFVD val AUC rose to **~0.98**. The web app loads `checkpoints/prototype/best.pt` when present. This is **domain adaptation for the demo**, not a rewrite of the Phase-2 ablation numbers.

---

## E. Explainability & web app

### Q23. What is Grad-CAM doing here?
It highlights which regions of the **EfficientNet** face feature map support a high fake score. Useful for demos and debugging, not a proof of authenticity.

### Q24. Frontend vs backend?
- Frontend: upload UI, shows verdict, bars, faces, Grad-CAM.  
- Backend: FastAPI receives file, runs prototype inference, returns JSON + report files.

### Q25. Where do uploads and reports go?
`user_media/uploads` (and copies), reports under `user_media/reports/`. Large media is gitignored.

---

## F. Scope / honesty questions (teachers love these)

### Q26. Deviations from the original Phase-1 plan?
HF mirror instead of raw FF++ videos; reduced subset; simulated c40; SDFVD instead of Celeb-DF; no face alignment; local outputs on `D:\`. All listed in `scope_implemented.md`.

### Q27. Is this ready for forensics / court use?
**No.** Educational prototype. Error rates, bias, and domain shift are too high for evidentiary claims.

### Q28. Ethical concerns?
Misuse of deepfakes; false accusations if a detector is wrong; privacy of faces in datasets. We use research/CC-licensed demo media and document limitations.

### Q29. What would you improve with more time?
Larger FF++ pool (`POOL_SPLITS` include val), longer training, official c40, stronger backbone / transformers, threshold calibration, multi-dataset training, better single-image handling.

### Q30. How do you ensure results are not made up?
Metrics are written by `evaluate.py` to CSV/MD from the run. Report cites those files; `scope_implemented.md` forbids inventing numbers.

---

## G. Quick “definition” prompts

| Term | One-liner |
| --- | --- |
| AUC-ROC | Ranking quality of scores; 0.5 = random, 1.0 = perfect separation |
| Label smoothing | Soften hard 0/1 labels to reduce overconfidence |
| Bi-LSTM | Reads the frame-feature sequence forward and backward |
| Majority vote | Video label = class predicted by most sequences |
| Ablation | Remove/add modules to see their contribution |
| Domain shift | Train and test distributions differ (FF++ → SDFVD) |

---

## H. Suggested 2-minute oral summary

> “We detect deepfakes with a hybrid of EfficientNet spatial features, FFT frequency features, and a Bi-LSTM. We trained four ablations on a small FaceForensics++ c23 subset with identity-aware splits, then tested on FF++ , a simulated compressed set, and SDFVD. Spatially alone worked best in-domain; cross-dataset was near chance until we fine-tuned for the web demo. The system includes Grad-CAM and a FastAPI upload UI called FrameForge. Limitations are data scale and generalization, which we document openly.”
