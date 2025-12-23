# Current weak points / risks

- Model maturity: TinyDet is minimal, uses host-side CIoU, limited augmentation (flip/letterbox), and no robust target assignment or mAP computation.
- Data quality: synthetic-only captures; no real-data validation; limited augmentation diversity.
- Checkpointing & reproducibility: no bundled demo weights; splits not persisted (seeded shuffles exist but not enforced across runs).
- Validation metrics: only mean IoU + precision/recall; no mAP/PR curves; NMS/threshold assumptions may not match deployment.
- Runtime UX: HUD overlays are basic; thresholds require flags (no live toggle); heuristic fallback only logged.
- Performance: CPU NdArray backend and CPU NMS; no batching/GPU path, potential real-time bottleneck.
- Testing gaps: no end-to-end inference test in the sim; no HUD overlay tests; Burn training harness lacks a sanity test beyond a single-batch run.
- Docs: training doc lacks full workflow (data prep, expected outputs, sample checkpoint); deployment guidance for Burn model is minimal.
