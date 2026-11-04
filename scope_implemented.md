# Scope implemented (Phase 2)

The package prepares the data, trains four ablation variants, evaluates each test set once, and writes Grad-CAM overlays. Dataset sizes and metric values are not typed into this file. They come from the prep and training runs:

- `OUTPUT_DIR/manifests/ffpp_balance_counts.json` — videos before and after balancing
- `OUTPUT_DIR/manifests/ffpp_split_counts.json` — sequences in train / val / test
- `OUTPUT_DIR/manifests/sdfvd_counts.json`
- `OUTPUT_DIR/manifests/c40_style_counts.json`
- `OUTPUT_DIR/results/param_counts.json` — trainable parameters per variant
- `OUTPUT_DIR/results/metrics.csv` and `metrics.md` — the 4×3 results table
- `OUTPUT_DIR/logs/train.log` — the same counts and the per-epoch validation AUC

## Deviations from the Phase 1 report

1. **FaceForensics++ frames come from a third-party Hugging Face mirror**, `TsienDragon/ffplusplus_c23_frames`. The provider already extracted the frames. Prep keeps each `frame_idx`. It does not open the original FF++ videos and it does not sample every 5th frame.

2. **Reduced video subset.** FaceShifter and DeepFakeDetection are dropped. Categories kept: original, Deepfakes, Face2Face, FaceSwap, NeuralTextures. The subset uses as many real videos as remain after the "at least 10 frames" filter, plus an equal number of fake videos spread evenly across the four manipulation types. Counts are whatever that run finds. There is no fixed video quota in the code.

3. **The mirror has a val split** as well as train and test. `POOL_SPLITS` in `config.py` defaults to train + test, then those videos are re-split 70/15/15. Set `POOL_SPLITS` to `("train", "val", "test")` to make the val rows eligible too.

4. **Simulated c40.** The FF++ test-split frames (before the face crop) are written to mp4 with ffmpeg `libx264 -crf 40`, read back, and cropped. Every output calls this set **c40-style (simulated)**. It is not the official FaceForensics++ c40 encode.

5. **SDFVD replaces Celeb-DF** as the cross-dataset test. The zip already on Drive (`DRIVE_ROOT/sdfvd.zip`) is unzipped and inspected. All discovered videos are test-only. Nothing is retrained on SDFVD.

6. **Face alignment is omitted.** MTCNN supplies a bounding box. The box is expanded by a small margin (`MTCNN_MARGIN`), squared, and resized to 224x224. Landmark alignment is not run. If the inspect step shows the mirror frames are already face crops, those frames are resized and MTCNN is not run on them.

7. **Checkpoint selection uses sequence-level validation AUC.** The validation split is a small share of the reduced subset, so a video-level AUC would rest on very few videos. The saved checkpoint is the epoch with the highest sequence AUC. The results table still reports video-level metrics: majority vote for accuracy, precision, recall, and F1, and the mean sequence score for video AUC. A tied vote uses that mean score.

8. **This PC has a 4GB GPU.** The optimizer still updates every 8 sequences. Those 8 are accumulated from smaller micro-batches so the card does not run out of memory. EfficientNet activations are recomputed on the backward pass for the same reason. On a 16GB Colab T4, `MICRO_BATCH_SIZE` can be set to 8.

9. **Local outputs go to `D:\deepfake_phase2`** when the D: drive exists, because the C: drive does not have room for the Hugging Face mirror plus the cached faces. The SDFVD zip already on disk is linked at `DL PROJECT\sdfvd.zip`.

## Preprocessing and model that match the spec

- ImageNet normalization is applied when a sequence is loaded. It is not baked into the saved jpgs.
- Sequences are 10 consecutive kept frames, stride 10, at most 6 sequences per video.
- The frequency map is `fftshift(FFT(gray))`, then `log(1+|F|)`, then per-image normalization.
- Face crops and frequency maps are cached under `OUTPUT_DIR`. Training does not call MTCNN.
- Variants: spatial-only baseline, spatial+frequency, spatial+temporal, full hybrid. Variants without a Bi-LSTM mean-pool the frame features.
- Loss is binary cross-entropy with label smoothing. Optimizer is Adam with weight decay. The schedule is ReduceLROnPlateau. Training stops early, runs at most 10 epochs, uses batch size 8, and uses FP16 autocast + GradScaler on CUDA.
- A weighted sampler balances real and fake sequences when those counts differ.
- Each variant is scored once on FF++ c23 test, c40-style (simulated), and SDFVD.
- Grad-CAM is taken from the EfficientNet-B0 `conv_head` of the full hybrid. The script saves four real and four fake overlays. Up to two videos of each class come from the FF++ c23 test split; the rest come from SDFVD. The Bi-LSTM is left in training mode for that backward pass because cuDNN refuses an LSTM backward while the module is in eval mode. The LSTM has no dropout, so its outputs stay the same.
