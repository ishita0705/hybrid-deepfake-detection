# Results snapshot (from the completed local Phase-2 run)

Copied from `D:\deepfake_phase2\results\metrics.md` so the GitHub repo documents the numbers without shipping checkpoints.

**Video-level AUC-ROC**

| Variant | FF++ c23 | c40-style | SDFVD |
| --- | ---: | ---: | ---: |
| spatial-only | 0.8000 | 0.4000 | 0.5041 |
| spatial+frequency | 0.6000 | 0.2000 | 0.5039 |
| spatial+temporal | 0.4000 | 0.0000 | 0.4745 |
| full hybrid | 0.4000 | 0.2000 | 0.4950 |

**Data sizes used**

- FF++ selected: 20 real + 20 fake videos  
- Sequences: train 148 · val 30 · test 30  
- SDFVD test: 106 videos, 176 sequences  
- c40-style: 6 videos, 30 sequences  

**Prototype fine-tune (web demo only)**

- Base: spatial `best.pt`  
- Fine-tuned on SDFVD; best held-out val AUC ≈ **0.9801**  
- Path on the training machine: `D:\deepfake_phase2\checkpoints\prototype\best.pt`
