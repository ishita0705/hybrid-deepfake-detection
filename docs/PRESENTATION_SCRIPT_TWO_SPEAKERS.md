# Presentation speaking script (2 people)

**PPT:** `docs/Phase-2 Minor Specialization Project Presentation Template - FILLED.pptx` (27 slides)  
**Demo:** FrameForge at http://127.0.0.1:7860  
**Suggested pair:** **Speaker A** (intro → method) and **Speaker B** (experiments → demo → close).  
Swap names as you like (Ishita / Thoshitha / Annu / Maithili).

**Timing guide (~12–15 min + demo + QA)**  
- Speaker A: ~6–7 min (slides 1–15)  
- Speaker B: ~6–7 min (slides 16–27) + live prototype (~3 min)

---

## Before you start

1. Open the PPT in slide-show mode.  
2. Start the prototype: `python web/run.py` → browser http://127.0.0.1:7860  
3. Keep ready in `user_media`:
   - **Real photo/video:** e.g. `real/real_photo_v3.jpg` or `real/real_sdfvd_v1.mp4`  
   - **Fake photo/video:** e.g. `fake/fake_photo_vs3.jpg` or `fake/fake_sdfvd_vs1.mp4`  
4. Agree who clicks Next / who runs the upload.

---

# SPEAKER A

## Slide 1 — Title
**Say:**  
“Good morning. We are presenting **Hybrid Deepfake Video Detection using Spatial, Frequency and Temporal Forensic Analysis**. Our team is Thoshitha, Ishita, Annu, and Maithili, under Dr. Sabyasachi Kamila, for the Final Phase Review.”

## Slides 2–3 — Contents
**Say (short):**  
“I’ll cover the introduction, literature, proposed solution, and methodology. My co-speaker will cover datasets, results, Grad-CAM, the live prototype, conclusion, and references.”

## Slide 4 — Paper status
**Say:**  
“No paper is published yet. We plan a short communication on capacity-versus-data after we enlarge the FaceForensics++ subset. Code and figures are already public on GitHub: ishita0705/hybrid-deepfake-detection.”

## Slide 5 — (skip or hold for end)
If this “Thank you” slide appears early in the deck order, **skip** and go to Introduction. Use it only at the end if your PPT flow jumps here.

## Slide 6 — Introduction overview
**Say:**  
“Deepfakes swap or reenact faces with autoencoders or GANs. Quality is high enough that people cannot reliably spot them by eye, and the media can be used for fraud or misinformation. Manual checking does not scale, so we need automated forensics.

We look at three kinds of clues:  
- **Spatial** — blending seams, lighting mismatch  
- **Frequency** — generator artefacts in the Fourier spectrum  
- **Temporal** — inconsistency across frames”

## Slide 7 — Problem statement
**Say:**  
“Single-cue detectors break when generators improve, when video is re-compressed, or when the test set changes. Big transformer detectors exist, but they need heavy compute. Our question: can a **lightweight hybrid** that fuses these three cues, and explains itself, be built on a **4 GB laptop GPU** — and does fusion actually help?”

## Slide 8 — Objectives
**Say:**  
“Main goal: design, implement, and critically evaluate that hybrid detector.  
Secondary: reproducible preprocessing, four ablation variants, evaluate on in-domain, compression, and cross-dataset tests, Grad-CAM explanations, and a browser prototype that returns REAL or FAKE.”

## Slides 9–11 — Literature + gap
**Say:**  
“Foundational work includes MesoNet, Xception on FaceForensics++, FaceXRay, face-warping artefacts, CNN+LSTM, and Grad-CAM. Recent surveys stress that **cross-dataset** evaluation is often missing but most informative. Frequency methods like FreqBlender motivate an FFT branch.

**Gaps we target:** lightweight models are mostly spatial; temporal models often skip frequency; explainability is rare at small compute; cross-dataset is under-reported. Also: published fusion gains usually use **huge** FF++ — whether fusion helps at **small data** was untested. That is our ablation.”

## Slide 12 — Proposed solution
**Say:**  
“One model, parallel branches:  
- EfficientNet-B0 on each 224 face crop  
- Small CNN on the log FFT map  
- Optional Bi-LSTM over 10 frames  
- Head ends in a sigmoid → **P(fake)**  
Grad-CAM on EfficientNet’s last conv. Four variants differ only by config flags so comparisons are fair.”

## Slide 13 — Tools
**Say (quick):**  
“Python, PyTorch CUDA, timm EfficientNet, MTCNN, grad-cam, OpenCV, and a FastAPI web UI, on an RTX 3050 4 GB.”

## Slide 14 — Methodology workflow
**Say:**  
“Pipeline: sample frames → MTCNN crop → cache faces and FFT maps → build 10-frame sequences → spatial ± frequency → LSTM or mean pool → P(fake) → video verdict by majority vote, AUC from mean score.”

## Slide 15 — Ablations
**Say:**  
“Four nested variants: spatial; spatial+frequency; spatial+temporal; full hybrid. Parameter counts go from about 4.3 million to 7.6 million. Same data splits and hyperparameters, so differences come from the branches.”

**Hand over:** “I’ll pass to my co-speaker for experiments, results, and the live demo.”

---

# SPEAKER B

## Slide 16 — Datasets
**Say:**  
“Training uses a reduced FaceForensics++ c23 subset: 20 real and 20 fake videos, identity-aware 70/15/15 split — 148 train, 30 val, 30 test sequences.  
Compression test: same FF++ test frames re-encoded as **c40-style simulated** CRF 40.  
Cross-dataset: **SDFVD** — 53 real Pexels + 53 Remaker fakes, test only, no Phase-2 retraining on it for the ablation table.”

## Slide 17 — Training config
**Say:**  
“BCE with label smoothing, Adam, plateau LR, early stop, effective batch 8 via micro-batches because of 4 GB VRAM, FP16, checkpoint by best validation sequence AUC. We report video-level AUC.”

## Slide 18 — Results table
**Say slowly:**  
“In domain, **spatial-only is best: video AUC 0.80**. Adding frequency or LSTM made it worse on this small set.  
Under c40-style compression every variant drops hard.  
On SDFVD, all four sit at **about 0.50 — chance**.  
So fusion did **not** help at this data scale, and zero-shot transfer failed.”

## Slide 19 — Comparison chart
**Say:**  
“This bar chart is the same story visually: left group in-domain, middle compression collapse, right group chance on the unseen corpus.”

## Slide 20 — ROC
**Say:**  
“ROC plots true-positive rate vs false-positive rate. Diagonal is random.  
Spatial on FF++ c23 rises above the diagonal — AUC 0.80.  
On c40 it falls toward chance.  
On SDFVD it sits on the diagonal. Step shapes appear because the FF++ test set has only six videos.”

## Slide 21 — Confusion matrices
**Say:**  
“Hard decisions use threshold 0.5 and majority vote.  
On FF++ we often **miss the single fake video**, so fake recall and F1 go to zero even when AUC is 0.80 — that is class imbalance on a tiny test set, not a contradiction.  
On SDFVD the model leans toward predicting fake for both classes.”

## Slide 22 — Grad-CAM
**Say:**  
“Grad-CAM shows which face pixels push P(fake) up.  
On a correct Remaker fake, heat concentrates around the eyes — a common swap artefact region.  
On a missed Face2Face frame, heat sits on the forehead while P(fake) stays below 0.5.  
So explanations look face-focused, but they do **not** guarantee accuracy.”

## Slide 23 — Deviations
**Say:**  
“We expected full hybrid to win, frequency to help compression, temporal to help transfer. None held. Extra parameters overfit 148 sequences; CRF 40 removes the spectrum the FFT branch needs; SDFVD is a different domain with no adaptation.”

## Slide 24 — Diagnostic / prototype bridge
**Say:**  
“When we fine-tuned the **spatial** checkpoint on SDFVD, held-out val AUC rose to about **0.98**. Same code, better when given in-domain data. So the pipeline is sound; the ablation was data-limited.  
**That fine-tuned checkpoint is what FrameForge loads for the live demo.**”

---

# LIVE PROTOTYPE DEMO (Speaker B leads; A assists)

Open http://127.0.0.1:7860. Keep this section detailed — examiners ask this.

## What FrameForge is
**Say:**  
“FrameForge is our frontend plus FastAPI backend. You upload a photo or video. The server detects the face with MTCNN, builds frequency maps, runs the fine-tuned detector, and returns:

1. A **REAL / FAKE** verdict  
2. **P(fake)** — probability the content is manipulated  
3. Per-sequence score bars  
4. Face-crop thumbnails  
5. Optional **Grad-CAM** heatmap  
6. A downloadable HTML report”

## What happens step-by-step after Upload → Analyze

**Say while the spinner runs:**  
“Behind the scenes:  
1. File is saved under our media uploads folder.  
2. If it is a **photo**, we treat it as one frame and pad to a 10-frame window so the model can run.  
3. If it is a **video**, we sample every 5th frame, up to 60 frames, like SDFVD prep.  
4. MTCNN finds the largest confident face and crops to 224×224.  
5. We compute the FFT forensic map for each face.  
6. Faces are grouped into sequences of 10.  
7. The model outputs P(fake) per sequence.  
8. **Mean** of those scores is shown as overall P(fake).  
9. If mean ≥ **0.5**, verdict is **FAKE**; else **REAL**.  
10. Majority vote across sequences is also logged for the report.”

## Demo 1 — REAL image or video

**Do:** Upload a known-real file (e.g. `real_photo_v3.jpg` or `real_sdfvd_v1.mp4`). Click **Analyze**.

**Say when results appear:**  
“This file is a **real** face.  
Look at the big badge — it should show **REAL**.  
**P(fake)** should be **low**, typically well under 0.5 — often around 0.05 to 0.2 after our SDFVD fine-tune for similar clips.  
Sequence bars stay mostly green / below the threshold.  
Face crops show what the detector actually saw.  
Grad-CAM, if shown, highlights regions that still slightly influence the fake score, but the final decision stays real because the average score is low.

**How to explain a good real result to the panel:**  
‘Low P(fake) means the model assigns little probability of manipulation. The threshold 0.5 converts that into the REAL label.’”

**If a real file wrongly shows FAKE:**  
“Be honest: single web photos outside SDFVD can still false-alarm. Say we document domain shift; the fine-tune helps SDFVD-like faces most.”

## Demo 2 — FAKE image or video

**Do:** Upload a known-fake file (e.g. `fake_photo_vs3.jpg` or `fake_sdfvd_vs1.mp4`).

**Say:**  
“This is a **Remaker face-swap** deepfake from SDFVD.  
The badge should show **FAKE**.  
**P(fake)** should be **high**, often 0.65–0.97 on these clips after fine-tuning.  
Sequence bars sit above 0.5.  
Grad-CAM often lights up eyes or blended facial regions — places swaps leave artefacts.

**How to explain a good fake result:**  
‘High P(fake) means the network is confident the face is manipulated. Crossing 0.5 flips the UI to FAKE. The HTML report stores the same numbers for the submission folder.’”

## Demo 3 — Optional blind clip
Upload something from `to_check` without saying the label. Predict from the UI, then reveal.

**Say:**  
“This is how a user would use the tool — they do not tell us real or fake; the system decides from the file alone.”

## What the panel sees in the UI (point with mouse)

| UI element | Meaning |
| --- | --- |
| **REAL / FAKE badge** | Final verdict from mean P(fake) vs 0.5 |
| **P(fake) = …** | Continuous confidence (0–1) |
| **Stats row** | Frames read, faces kept, sequences, runtime |
| **Sequence bars** | Each 10-frame window’s score |
| **Face crops** | Evidence the model used |
| **Grad-CAM** | Where EfficientNet looked for the fake class |
| **Open full HTML report** | Offline report with the same analysis |

## Closing the demo
**Say:**  
“So the research ablation was honest: fusion failed at small FF++ scale and transfer was chance. The prototype still works as a **deployment demo** because we fine-tuned on SDFVD. Users get a clear REAL/FAKE report with scores and explanations.”

---

## Slides 25–26 — Conclusion + future (Speaker B)
**Say:**  
“We delivered the full pipeline on a 4 GB GPU. Central finding: **capacity must match data**. Cross-dataset evaluation is mandatory — the 0.80 in-domain number alone would have been misleading.  
Future work: more FF++ data, official c40, domain adaptation, calibration instead of raw 0.5, fairness breakdowns, stronger backbones.”

## Slide 27 — References
**Say:**  
“Key references: FaceForensics++, EfficientNet, Grad-CAM, MesoNet, FaceXRay, and our GitHub repo. Thank you — we are happy to take questions.”

---

# Quick QA cheat-sheet (either speaker)

| Question | Short answer |
| --- | --- |
| Why did hybrid lose to spatial? | Only 148 train sequences; extra params overfit. |
| Why F1 = 0 on FF++? | One fake video; missed at threshold 0.5; AUC can still be 0.80. |
| Why SDFVD ~0.5 in table? | Zero-shot transfer; different domain; no adaptation in ablation. |
| Why does the demo work then? | Separate SDFVD fine-tune of spatial → val AUC ~0.98. |
| Photo vs video? | Photo = one face padded to 10 frames; video = sampled frames → multiple sequences. |
| What is P(fake)? | Model probability the face is manipulated; ≥0.5 → FAKE. |

---

# Handoff lines (copy these)

**A → B after slide 15:**  
“That covers the design. Over to you for experiments and the live system.”

**B → A before demo (optional):**  
“While I upload, please watch the status line for backend ready.”

**Both at end:**  
“Thank you. Questions are welcome.”
