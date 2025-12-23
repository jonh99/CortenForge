# Burn training harness

Basic usage (enable Burn runtime):
```bash
cargo run --features burn_runtime --bin train -- --help
```

Common flags:
- `--batch-size <N>`: training/val batch size (default 2)
- `--epochs <N>`: number of epochs (default 1)
- `--log-every <N>`: log loss/IoU every N steps (default 1)
- `--lr-start <f64>`, `--lr-end <f64>`: linear LR schedule across total steps
- `--scheduler <linear|cosine>`: pick LR schedule type
- `--ckpt-every-steps <usize>`: checkpoint cadence in steps (0 disables)
- `--ckpt-every-epochs <usize>`: checkpoint cadence in epochs
- `--val-ratio <f32>`: fraction of runs for validation split (default 0.2)
- `--seed <u64>`: deterministic shuffle/splits; omit for random
- `--ckpt-dir <path>`: where model/optim/scheduler checkpoints are read/written (default `checkpoints`)
- `--val-obj-thresh <f32>`: objectness threshold for val matching (default 0.3)
- `--val-iou-thresh <f32>`: IoU threshold for NMS/matching (default 0.5)
- `--patience <usize>` / `--patience-min-delta <f32>`: optional early stop on val IoU plateau

Sample run:
```bash
cargo run --features burn_runtime --bin train -- \
  --batch-size 4 \
  --epochs 5 \
  --scheduler cosine \
  --lr-start 1e-3 \
  --lr-end 1e-4 \
  --val-ratio 0.2 \
  --ckpt-every-epochs 1 \
  --val-obj-thresh 0.3 \
  --val-iou-thresh 0.5
```

What it does today:
- Loads capture runs via `BatchIter` (train with aug; val without), builds TinyDet, AdamW, and a linear LR scheduler.
- Runs epoch/batch loop with per-step optimizer updates; logs loss and mean IoU each log interval.
- Validation: decodes per-cell predictions, applies sigmoid + NMS, matches to GT boxes with IoU threshold, and reports mean IoU plus precision/recall (tp/fp/fn).
- Checkpoints: on start, loads model/optim/scheduler from `ckpt_dir` if present; saves them per configured cadence (steps/epochs). Optional early stop tracks best val IoU.

Notes:
- Requires `--features burn_runtime` to pull in Burn and the training harness.
- Val metric thresholds are tunable via CLI; adjust to trade off recall/precision during evaluation.
- Runtime inference will attempt to load `checkpoints/tinydet.bin`; if missing or failed, it logs a warning and falls back to the heuristic detector.

Next steps (nice-to-haves):
- Expose predicted boxes/confidence to HUD/`DetectionResult` so runtime shows actual detections, not just a bool. ✅
- Bundle a small demo checkpoint or fall back to the heuristic detector with a clear log when no Burn model is available. ✅ (warns + heuristic fallback when checkpoint missing)
- Tighten validation metrics with per-image precision/recall or mAP in addition to mean IoU (precision/recall now logged; mAP still TODO).
- Add a sample `train` command here with typical flags, and expose inference thresholds via CLI/env.
