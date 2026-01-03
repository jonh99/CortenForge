# training

Burn-based training/eval for TinyDet and the multibox BigDet model.

Contents
- `models`: TinyDet (single-logit) + BigDet (multibox) configs/constructors.
- `dataset`: DatasetConfig, RunSample loader; `collate` pads boxes to `max_boxes`, emits `gt_boxes`, `gt_mask`, and global features (mean/std RGB, aspect, box count).
- `util`: TrainArgs (model/backend/max-boxes/loss weights), run_train, eval helpers, checkpoint load helpers for TinyDet/BigDet, greedy IoU matcher, backend validation.
- `bin/train`: CLI for training with `--model {tiny,big}` (default tiny), `--max-boxes`, `--lambda-box`, `--lambda-obj`, `--backend {ndarray,wgpu}`.
- `bin/eval`: CLI to load a checkpoint (TinyDet/BigDet) and compute precision/recall at an IoU threshold.

Models
- TinyDet: single-logit detector, best for single-box targets.
- BigDet: multibox detector; config includes `max_boxes` (default 64) and optional `input_dim` (defaults to box-only; training sets 4+8 for box+global features).
  - `forward_multibox` returns `(boxes [B, max_boxes, 4], scores [B, max_boxes])`, normalized/clamped to [0,1].
  - TinyDet remains backward compatible for existing single-box flows.

Loss/matching
- Collate pads/truncates GT to `max_boxes` and provides a mask.
- Greedy IoU matching (GT -> best pred) builds objectness + box targets; unassigned preds are negative.
- Loss: masked L1 box regression on matched preds + BCE objectness for all preds; weighted by `--lambda-box`/`--lambda-obj` (optional IoU loss hook).

Backends/features
- Backends: NdArray by default; WGPU with `--features backend-wgpu`.
- CLI flags: `--backend`, `--model`, `--max-boxes`, `--lambda-box`, `--lambda-obj`, `--seed`, dataset roots.

Tests
- Collate test (padding/mask/features).
- BigDet smoke train/test (one step, save/load).
- BigDet forward-shape test (boxes/scores in expected shapes and [0,1] range).

## License
Apache-2.0 (see `LICENSE` in the repo root).
