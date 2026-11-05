# Presentation speaking script (2 people) — full version

**PPT:** `docs/Phase-2 Minor Specialization Project Presentation Template - FILLED.pptx` (27 slides)  
**Demo:** FrameForge → http://127.0.0.1:7860  

| Role | Who | Slides |
| --- | --- | --- |
| **Speaker A** | (pick one teammate) | 1–15 |
| **Speaker B** | (pick one teammate) | 16–27 + live demo |

**Suggested total time:** 18–22 minutes speaking + 3 minutes demo + QA  
You can shorten by skipping the *Optional extra* paragraphs if time is tight.

---

## Before you start

1. Open the PPT in slide-show mode.  
2. Run `python web/run.py` and confirm the UI says **Ready**.  
3. Keep files ready:
   - Real: `user_media/real/real_photo_v3.jpg` or `real_sdfvd_v1.mp4`  
   - Fake: `user_media/fake/fake_photo_vs3.jpg` or `fake_sdfvd_vs1.mp4`  
4. Speaker A advances slides 1–15; Speaker B takes the clicker from slide 16.

---

# SPEAKER A — Slides 1 to 15

---

## Slide 1 — Title

**Say:**  
“Good morning everyone, and thank you for your time.  
We are here for our **Final Phase Review**. Our project title is **Hybrid Deepfake Video Detection using Spatial, Frequency and Temporal Forensic Analysis**.

Our team members are **Kolisetty Thoshitha**, **Ishita Sharma**, **Annu Mary Mahesh**, and **Maithili Sharma**, from the Department of Computer Science and Engineering, Data Science.  
This work was done under the supervision of our project guide, **Dr. Sabyasachi Kamila**.

In short, we built a lightweight deepfake detector that looks at a face video in three ways — how it looks in pixels, what its frequency spectrum shows, and how it changes over time — and we also built a small web app that can check a user’s own photo or video.”

**Optional extra:**  
“All of our code and result figures are open on GitHub under Ishita’s account, which we will mention again later.”

---

## Slide 2 — List of contents (part 1)

**Say:**  
“I will quickly walk through the agenda so the panel knows the flow.

First, the **introduction** — what deepfakes are, why they matter, our problem statement, and objectives.  
Second, a short **literature review** — important previous systems and the research gap we target.  
Third, our **proposed solution** — the hybrid spatial, frequency, and temporal design, and the tools we used.  
Fourth, the **methodology** — the full processing pipeline and the four ablation variants.

My co-speaker will then take over for experiments, results, explainability, conclusion, and the live prototype.”

---

## Slide 3 — List of contents (part 2)

**Say:**  
“Continuing the agenda:  
Section five is the **experimental setup** — FaceForensics++ c23, our simulated c40 compression set, and the SDFVD cross-dataset.  
Section six is **performance evaluation** — video-level AUC, ROC curves, confusion matrices, Grad-CAM, and where results differed from what we expected.  
Section seven is the **conclusion** and future scope.  
Section eight is **references**.

Please note that we will also give a short live demo of FrameForge, our upload-based detector UI.”

---

## Slide 4 — Published / communicated paper

**Say:**  
“On the publication status: we have **not yet published or communicated** a research paper from this work.

We are preparing a short paper around the ablation finding — that adding frequency and temporal capacity did **not** help when the training set is small. We are waiting to re-run the ablation on a larger FaceForensics++ subset so the claim is statistically stronger.

Meanwhile, the artefacts are already public: source code, configs, and result figures are on GitHub at **github.com/ishita0705/hybrid-deepfake-detection**. Examiners can open that link after the review.”

---

## Slide 5 — Thank you / QA (if it appears early)

**Do:** If this slide appears here in the deck, say:  
“We will keep the thank-you slide for the end. Moving to the introduction.”  
Then go to slide 6.

*(If your PPT order puts Thank You only at the end, ignore this and continue.)*

---

## Slide 6 — Introduction: Overview

**Say:**  
“Starting with the introduction.  
A **deepfake**, in our context, is a video where someone’s face has been synthetically swapped or reenacted — usually with autoencoders or GANs. Modern generators are good enough that casual viewing often cannot tell real from fake. That creates risks such as misinformation, non-consensual imagery, and fraud.

Checking every video by hand is impossible at internet scale, so we need **automated forensic detection**.

In the research literature, useful clues fall into three families:  
1. **Spatial clues** — blending boundaries, resolution mismatch, lighting that does not match the face.  
2. **Frequency clues** — periodic artefacts left by upsampling inside generators; these show up in the Fourier spectrum even when the face looks clean.  
3. **Temporal clues** — flicker or inconsistency across frames, because many generators work frame by frame and do not enforce perfect temporal coherence.

Our project tries to use all three families inside one lightweight system.”

---

## Slide 7 — Problem statement

**Say:**  
“The problem is that detectors based on only one cue are fragile.  
As generators improve, spatial artefacts become weaker. When videos are shared online they are **re-compressed**, which damages both spatial texture and frequency energy. And a model that scores above ninety-five percent on its training dataset often collapses on a new dataset.

The strongest published detectors now use transformers or multimodal large models. Those are powerful, but they are too heavy for a typical undergraduate lab GPU.

So the problem we addressed is this:  
**Can we build a lightweight hybrid detector that fuses spatial, frequency, and temporal cues, that can explain its decision, that runs on a 4 GB consumer GPU — and does that fusion actually help when we evaluate honestly?**”

---

## Slide 8 — Objectives

**Say:**  
“Our main objective was to **design, implement, and critically evaluate** a lightweight, explainable hybrid deepfake video detector that combines spatial, frequency-domain, and temporal analysis.

The secondary objectives were concrete engineering goals:  
- Build a reproducible preprocessing pipeline with face detection, FFT maps, caching, and identity-aware sequences.  
- Implement **four ablation variants** in one shared codebase.  
- Evaluate each variant once on an in-domain set, a high-compression set, and a cross-dataset set.  
- Generate Grad-CAM explanations on the spatial backbone.  
- Deliver a FastAPI plus browser prototype that returns a clear REAL or FAKE report.

So this is not only a model paper idea — it is a full mini-system from data to demo.”

---

## Slide 9 — Literature: foundational

**Say:**  
“In the foundational literature:  
**MesoNet** showed that a compact CNN can catch mesoscopic artefacts, but it is single-frame and has no spectral or temporal branch.  
**XceptionNet on FaceForensics++** became a strong benchmark on c23 compression, but accuracy drops sharply on heavier c40 compression.  
**FaceXRay** predicts blending boundaries and is interpretable, but it struggles when the whole face is generated.  
**Face-warping** methods do not need deepfake training data, but they are destroyed by resize, blur, and compression.  
**CNN plus LSTM** work showed that temporal inconsistency helps, but usually without a frequency branch or explainability.  
**Grad-CAM** gives cheap class heatmaps; we adopt it for our spatial backbone even though it was not designed only for forensics.

Together, these papers motivate combining cues and also motivate careful evaluation under compression.”

---

## Slide 10 — Literature: recent advances

**Say:**  
“More recent surveys say clearly that no single strategy generalises across all manipulation methods, datasets, and compression levels. They also say **cross-dataset evaluation** is often skipped — and that is exactly where many systems fail.

Comparative CNN–RNN studies show hybrids beat frame-only CNNs by using inter-frame inconsistency.  
Frequency-oriented work such as FreqBlender supports the idea that synthetic generation leaves spectral artefacts that are not obvious in pixels — that is the direct reason we built an FFT branch.  
There is also a push toward learning **generic forensic signals** rather than generator-specific fingerprints.  
At the top end, multimodal and LLM detectors can even write natural-language justifications, but they need transformer-scale compute, which we did not have.”

---

## Slide 11 — Research gap

**Say:**  
“From that reading we write four gaps.  
Gap one: lightweight systems still lean mainly on spatial artefacts and weaken as generators improve.  
Gap two: temporal models improve sequences but usually omit frequency forensics.  
Gap three: explainability is weak in the lightweight regime because the most interpretable systems are also the most expensive.  
Gap four: cross-dataset generalisation is rarely reported, so in-dataset scores look too optimistic.

We add one more gap that this project specifically measures:  
published evidence that ‘fusion helps’ usually comes from the **full FaceForensics++ corpus**. Whether fusion still helps at **small data scale** was untested. Our four-way ablation is designed to answer that.”

---

## Slide 12 — Proposed solution

**Say:**  
“Our proposed solution is one trainable model with parallel branches and a shared head.

The **spatial branch** is ImageNet-pretrained EfficientNet-B0. Each face crop is 224 by 224 RGB, and we get one descriptor per frame.  
The **frequency branch** is a small CNN on the log-magnitude 2D FFT of the grayscale face; that vector is concatenated with the spatial descriptor.  
The **temporal module** is a bidirectional LSTM over a 10-frame sequence. If temporal is turned off, we simply mean-pool the frame vectors.  
The **head** is Linear, ReLU, Dropout, Linear, Sigmoid — and the output is **P(fake)**, the probability that the clip is manipulated.  
For explainability we run Grad-CAM on EfficientNet’s convolution head.

Importantly, our four variants differ only by configuration flags. Training data, splits, and hyperparameters stay the same, so any performance difference can be attributed to which branches are on.”

---

## Slide 13 — Tools and technologies

**Say:**  
“On the engineering side we used Python with PyTorch 2.x and CUDA 11.8.  
Backbones come from timm — EfficientNet-B0 with ImageNet weights.  
Face detection uses facenet-pytorch MTCNN.  
Explainability uses pytorch-grad-cam.  
Metrics and plots use scikit-learn and matplotlib.  
Video and image I/O use OpenCV, Pillow, and ffmpeg with libx264 for the compression experiment.  
The web prototype is FastAPI and Uvicorn with a static HTML, CSS, and JavaScript frontend.  
All reported training ran on an NVIDIA RTX 3050 Laptop GPU with 4 GB of memory, which forced micro-batching and gradient checkpointing.”

---

## Slide 14 — Processing workflow

**Say:**  
“This slide is the full pipeline, top to bottom.

We start from a video or a frame set.  
We sample frames — for the FF++ mirror we keep the provided frame indices; for SDFVD we take every fifth frame, capped at sixty.  
Then MTCNN finds a face box; we add a margin, make it square, and resize to 224 by 224. We do **not** run landmark alignment.  
We cache the RGB face as JPEG and the FFT map as an NPY file so training never re-runs MTCNN.  
The frequency map is: grayscale, FFT2, fftshift, log of one plus magnitude, then per-image normalisation.  
Sequences are ten consecutive faces, stride ten, at most six sequences per video.  
At load time we apply ImageNet normalisation.  
Features go through spatial and optional frequency branches, then Bi-LSTM or mean pooling, then the classification head to P(fake).  
Finally, sequence scores become a video verdict: mean score for AUC, majority vote for accuracy-style metrics.

That same pipeline is what the web prototype uses when someone uploads a file.”

---

## Slide 15 — Ablation variants

**Say:**  
“Here are the four variants.

**spatial** — only EfficientNet, mean over frames, about 4.34 million trainable parameters. This is our baseline.  
**spatial_freq** — adds the frequency CNN, about 4.46 million parameters.  
**spatial_temporal** — spatial features through Bi-LSTM, about 7.29 million.  
**full hybrid** — spatial plus frequency plus Bi-LSTM, about 7.64 million.

Because they are nested, comparisons are clean:  
spatial_freq versus spatial isolates frequency;  
spatial_temporal versus spatial isolates time;  
full versus the others asks whether the two extras combine.

I will now hand over to my co-speaker for the experimental setup, the numbers, and the live demo.”

**Handoff line:**  
“Thank you. Over to you.”

---

# SPEAKER B — Slides 16 to 27 + demo

---

## Slide 16 — Datasets

**Say:**  
“Thank you. I will cover experiments, results, and the prototype.

We used three evaluation settings.  
First, **FaceForensics++ c23** from a Hugging Face frame mirror. After filtering and balancing we kept forty videos — twenty real and twenty fake, with fakes spread over Deepfakes, Face2Face, FaceSwap, and NeuralTextures. We split by identity so the same person never appears in two splits: 148 train sequences, 30 validation, 30 test.

Second, **c40-style simulated** — we re-encoded the FF++ test frames with libx264 at CRF 40, decoded them, and ran the same face pipeline. This tests compression robustness. It is labelled simulated because it is not the official FF++ c40 release. We have six videos and thirty sequences here.

Third, **SDFVD** — fifty-three real Pexels videos and fifty-three Remaker AI face-swaps. For the ablation table this set is **test only**; we did not retrain the four Phase-2 variants on SDFVD. That gives one hundred six videos and one hundred seventy-six sequences.

Each variant is scored exactly once per test set. Checkpoints were chosen only on validation sequence AUC, never by peeking at the test set.”

---

## Slide 17 — Training configuration

**Say:**  
“Training used binary cross-entropy with label smoothing of 0.1.  
Optimizer was Adam with learning rate 1e-4 and weight decay 1e-4.  
We used ReduceLROnPlateau on validation sequence AUC, maximum ten epochs, early stopping patience three.  
Because the GPU has only 4 GB, the effective batch size of eight is made from micro-batches of four with two accumulation steps. We also used FP16 autocast and gradient checkpointing.  
When real and fake sequence counts differed, a weighted sampler balanced them.  
The saved checkpoint is the epoch with the best validation sequence AUC.  
For the official table we mainly discuss **video-level AUC**, where each video’s score is the mean of its sequence scores.”

---

## Slide 18 — Video-level AUC table

**Say carefully — this is a key slide:**  
“These are the main numbers from our completed run.

On **FF++ c23**, spatial-only reaches **0.8000**, the best in-domain result.  
Spatial plus frequency drops to 0.60.  
Spatial plus temporal and the full hybrid both fall to 0.40.

On **c40-style**, every variant is weak: 0.40, 0.20, 0.00, and 0.20. Compression clearly hurts.

On **SDFVD**, all four are around **0.47 to 0.50** — essentially chance.

So the headline is: at this small data scale, the simple spatial baseline beat the fuller hybrids in domain, and **no variant transferred** to the unseen SDFVD corpus.

Two caveats we want to state honestly: the FF++ and c40 test columns use only six videos with one fake, so those AUCs are coarse. SDFVD with one hundred six videos is the statistically stronger cross-dataset column, and it shows chance performance.”

---

## Slide 19 — Comparison chart

**Say:**  
“This chart is the same table drawn as bars.  
Look left: spatial is tallest in domain.  
Look middle: every bar collapses under compression.  
Look right: every bar sits near the dashed chance line on SDFVD.

Visually, this is our negative result on cue fusion at small scale, and our negative result on zero-shot transfer.”

---

## Slide 20 — ROC curves (spatial baseline)

**Say:**  
“An ROC curve plots true-positive rate against false-positive rate as we slide the decision threshold. The diagonal is random guessing. AUC is the area under the curve.

For the spatial baseline:  
On FF++ c23 the curve rises above the diagonal and AUC is 0.80 — the model ranks the fake video above most reals.  
On c40-style the curve falls toward chance — AUC 0.40 — so compression damaged the cues.  
On SDFVD the curve hugs the diagonal — AUC about 0.50 — scores do not separate Remaker fakes from Pexels reals.

The step-like shape is normal when you only have six videos; there are only a few possible ranking positions.”

---

## Slide 21 — Confusion matrices (spatial baseline)

**Say:**  
“Confusion matrices use a hard threshold of 0.5 and majority vote over sequences.

On FF++ c23 we typically classify most reals correctly but **miss the single fake**. That is why precision, recall, and F1 for the fake class can be zero even though AUC is 0.80. Ranking can be decent while the fixed threshold still fails on one positive example.

On c40-style, decisions look closer to a biased coin flip.  
On SDFVD, with one hundred six videos, the model often predicts fake for both classes — so recall of fake looks high, but AUC stays near chance. That means the score distributions overlap; the model is not truly separating the two populations.”

---

## Slide 22 — Grad-CAM

**Say:**  
“Grad-CAM helps us see *where* the EfficientNet spatial branch looks when it increases P(fake). Red means high contribution; blue means low.

On the left example, a true Remaker fake with P(fake) about 0.69, the heatmap lights up around the eyes — a region where face swaps often leave blending artefacts. That is a **correct** detection with a sensible explanation.

On the right, a true Face2Face fake scored only about 0.42, so it is called real. The heat sits on the forehead and hairline instead of the mouth area where Face2Face often operates. So Grad-CAM can look plausible and still accompany a wrong decision.

Our takeaway: Grad-CAM is useful for demos and debugging, but it is **not** proof of authenticity.”

---

## Slide 23 — Deviations from expected results

**Say:**  
“We expected the full hybrid to beat the baseline, frequency to help under compression, and temporal modelling to help transfer. **None of those expectations held.**

Why?  
The full hybrid adds roughly 3.3 million parameters on top of spatial, but we only had 148 training sequences from 28 videos. Best validation AUC often appeared by epoch two and training stopped around epoch five — classic overfitting.

The frequency branch relies on high-frequency energy; CRF 40 quantization removes exactly that signal, so compression robustness did not improve.

Cross-dataset chance performance is expected when both the fake method and the real footage distribution change, and we applied no domain adaptation in the ablation.

AUCs below 0.5 on the six-video sets should be read as noise at the resolution limit, not as a deep claim that the model systematically inverted labels.”

---

## Slide 24 — Diagnostic: pipeline is sound

**Say:**  
“To check whether our code was simply broken, we ran a separate experiment: fine-tune the spatial checkpoint on SDFVD and evaluate on a held-out SDFVD validation split.  
Held-out validation AUC rose to about **0.98**.

That number is **not** part of the ablation table — it is in-domain SDFVD, whereas the table measures FF++ training with zero-shot SDFVD testing.

Why it matters: the same preprocessing, architecture, training loop, and evaluation code that look like chance under transfer become very strong when given modest in-domain data. So the limiter is **data volume and diversity**, not a broken implementation.

This fine-tuned checkpoint is exactly what our **FrameForge** web prototype loads.”

---

# LIVE PROTOTYPE DEMO (about 3 minutes)

**Open:** http://127.0.0.1:7860  
**Say while the page loads:**  
“I will now show the prototype. FrameForge is a simple browser UI talking to a FastAPI backend. The user does not need to know PyTorch. They upload a photo or video; the server returns REAL or FAKE with a confidence score and a report.”

## What happens inside after Analyze

**Say:**  
“When I click Analyze, the backend does the following.  
It saves the upload.  
For a **photo**, it detects one face and repeats it into a ten-frame window so the model can run.  
For a **video**, it samples frames, detects faces, builds frequency maps, and makes up to six sequences of ten faces.  
Each sequence gets a P(fake).  
We average those scores. If the average is **at least 0.5**, the UI shows **FAKE**; otherwise **REAL**.  
We also show per-sequence bars, face crops, Grad-CAM when available, and a full HTML report.”

## Demo A — Real media

**Do:** Upload a real photo or video → Analyze.

**Say:**  
“This file is known **real**.  
Watch the badge — it should say **REAL**.  
P(fake) should be **low**, clearly under 50 percent if the face matches the fine-tuned domain.  
The bars stay mostly below the threshold.  
Face crops show what was actually scored.  
So for a real input, low probability of manipulation leads to a REAL verdict.”

## Demo B — Fake media

**Do:** Upload a fake photo or video → Analyze.

**Say:**  
“This file is a known **face-swap deepfake**.  
The badge should say **FAKE**.  
P(fake) should be **high**, often well above 0.5 — sometimes 0.7 to 0.9 on these Remaker examples.  
Bars sit above the threshold.  
Grad-CAM may highlight eyes or blended face regions.  
So for a fake input, high P(fake) crosses the threshold and the UI reports FAKE, with a downloadable analysis report.”

## Bridge back to research honesty

**Say:**  
“Please note the distinction for the panel: the ablation table showed chance transfer from FF++ to SDFVD. The demo works better because we **fine-tuned** on SDFVD. We are not claiming the zero-shot hybrid solved cross-dataset detection; we are showing a working end-to-end product path once in-domain data is available.”

---

## Slide 25 — Conclusion: summary

**Say:**  
“To conclude the research side: we delivered the full pipeline — preprocessing, four-way ablation, three-way evaluation, Grad-CAM, and a browser prototype — all inside a 4 GB GPU budget.

The fusion hypothesis was **not** supported at this data scale. Spatial-only won in domain; compression hurt everyone; cross-dataset was chance.

Our central finding is that **capacity must be matched to available data**. An ablation run on too little data measures the data budget more than the architecture.

Methodologically, if we had reported only the best in-domain number 0.80, we could have claimed a working detector. Cross-dataset evaluation showed that claim would have been unfounded. We therefore treat cross-dataset testing as mandatory.”

---

## Slide 26 — Future scope

**Say:**  
“For future work we would first **scale the data** — include the mirror’s validation pool and raise the video cap — before drawing architectural conclusions. We would also keep a test split with enough fake videos and report confidence intervals.

On robustness and transfer: use official FF++ c40, sweep CRF values, and add domain adaptation or few-shot target fine-tuning, plus another corpus such as Celeb-DF.

On deployment: calibrate scores instead of a raw 0.5 threshold, abstain when uncertain, and break errors down by demographic attributes. Finally we can try stronger backbones and harden FrameForge to show uncertainty clearly to the user.”

---

## Slide 27 — References

**Say:**  
“Our main references include FaceForensics++ by Rössler and colleagues, EfficientNet by Tan and Le, Grad-CAM by Selvaraju and colleagues, MesoNet, FaceXRay, face-warping artefacts, CNN–RNN deepfake detection, MTCNN, LSTM, and Celeb-DF.  
Project code: github.com/ishita0705/hybrid-deepfake-detection.  
Data sources: the FF++ c23 Hugging Face mirror and SDFVD under CC BY 4.0.

Thank you for your attention. We are happy to take questions.”

**Closing together (both speakers):**  
“Thank you.”

---

# Extra Q&A lines (keep at the end of your printout)

| If they ask… | Answer like this… |
| --- | --- |
| Why not use a transformer? | Compute budget — 4 GB GPU; we chose a lightweight, explainable hybrid instead. |
| Is c40 official? | No — we say c40-style simulated CRF 40; documented as a deviation. |
| Why identity-aware split? | Same person must not appear in train and test; otherwise scores leak identity, not forgery. |
| Photo vs video in demo? | Photo → one face padded to 10 frames. Video → sampled frames → multiple sequences → average P(fake). |
| What does P(fake)=0.2 mean? | About 20% estimated chance of manipulation; below 0.5 → shown as REAL. |
| What does P(fake)=0.85 mean? | Strong evidence of manipulation; above 0.5 → shown as FAKE. |
| Did hybrid fail? | At this data scale, yes relative to spatial-only. The diagnostic fine-tune shows the code path is fine. |

---

# Print tip

Print or keep this Markdown open on a second laptop.  
Speaker A: pages for slides 1–15.  
Speaker B: pages for slides 16–27 + demo section.
