# Burn training harness

## Commands

Basic (CPU) runtime:
```bash
cargo run --features burn_runtime --bin train -- --help
```

GPU (wgpu): add `burn_wgpu` alongside `burn_runtime`:
```bash
cargo run --features "burn_runtime,burn_wgpu" --bin train -- --help
```

High-performance preset (release, GPU, larger batch/epochs):
```bash
# From repo root
cargo train_hp
# Defaults: --input-root assets/datasets/captures_filtered --val-ratio 0.1 --batch-size 128 --epochs 20 --scheduler cosine --lr-start 3e-4 --lr-end 1e-5 --target-size 256x256 --val-target-size 256x256 --status-file logs/train_hp_status.json
```
Run from the repo root. Adjust adapter/backend in the shell if needed (`WGPU_POWER_PREF=high-performance`, `WGPU_BACKEND=dx12` or `vulkan`, `WGPU_ADAPTER_NAME=<name>`).

Full one-liner (Windows, repo root, forces NVIDIA DX12 and logs adapter):
<details>
<summary>NVIDIA (DX12)</summary>

```powershell
$env:WGPU_POWER_PREF="high-performance"; $env:WGPU_BACKEND="dx12"; $env:WGPU_ADAPTER_NAME="NVIDIA"; $env:RUST_LOG="info,wgpu_core=info"; cargo train_hp
```
If DX12 picks WARP/CPU, try `WGPU_BACKEND="vulkan"` and set `WGPU_ADAPTER_NAME` to the exact adapter name from Device Manager. Override flags by appending them without an extra `--`, e.g., `cargo train_hp --batch-size 128`.

</details>

<details>
<summary>AMD (DX12)</summary>

```powershell
$env:WGPU_POWER_PREF="high-performance"; $env:WGPU_BACKEND="dx12"; $env:WGPU_ADAPTER_NAME="AMD"; $env:RUST_LOG="info,wgpu_core=info"; cargo train_hp
```
If DX12 picks WARP/CPU, try `WGPU_BACKEND="vulkan"` and set `WGPU_ADAPTER_NAME` to the exact adapter name from Device Manager. Override flags by appending them without an extra `--`, e.g., `cargo train_hp --batch-size 128`.

</details>

<details>
<summary>Apple (Metal)</summary>

```bash
WGPU_POWER_PREF=high-performance WGPU_BACKEND=metal cargo train_hp
```
If Metal falls back or you want logging, add `RUST_LOG=info,wgpu_core=info`.

</details>

<details>
<summary>Linux (Vulkan)</summary>

```bash
WGPU_POWER_PREF=high-performance WGPU_BACKEND=vulkan WGPU_ADAPTER_NAME="NVIDIA" RUST_LOG=info,wgpu_core=info cargo train_hp
```
Adjust `WGPU_ADAPTER_NAME` to match your GPU vendor/name (e.g., "AMD", "Intel") or omit to auto-pick.

</details>

<details>
<summary>Linux (Mesa/GL fallback)</summary>

```bash
WGPU_POWER_PREF=high-performance WGPU_BACKEND=gl RUST_LOG=info,wgpu_core=info cargo train_hp
```
Use only if Vulkan isn’t available; performance may be lower.

</details>

## End-to-end workflow
- Capture/prune data: run `sim_view --mode datagen` to create runs under `assets/datasets/captures`, then prune empties with `cargo run --bin prune_empty -- --input assets/datasets/captures --output assets/datasets/captures_filtered`. Repeat for any real-val root (`assets/datasets/real_val_filtered`).
- Point training at filtered roots: e.g., `--input-root assets/datasets/captures_filtered` and `--real-val-dir assets/datasets/real_val_filtered`, or rely on the split manifest/seed for repeatable splits.
- Train: pick a batch/epoch schedule and run (see sample commands). Logs go to stdout; optional `--metrics-out` appends JSONL with IoU/PR/mAP per epoch.
- Outputs: model/optim/scheduler checkpoints land in `--ckpt-dir` (default `checkpoints`); metrics JSONL if requested. A demo checkpoint can be loaded via `--demo-checkpoint`.

## Common flags
- `--batch-size <N>`: train/val batch size (default 2); targets built per-image.
- `--epochs <N>`: number of epochs (default 1).
- `--log-every <N>`: log loss/IoU every N steps (default 1).
- `--lr-start <f64>`, `--lr-end <f64>`: linear LR schedule endpoints.
- `--scheduler <linear|cosine>`: LR schedule type.
- `--ckpt-every-steps <usize>` / `--ckpt-every-epochs <usize>`: checkpoint cadence (0 disables).
- `--val-ratio <f32>`: fraction of runs for validation split (default 0.2).
- `--seed <u64>`: deterministic shuffle/splits; default seed 42 if omitted.
- `--ckpt-dir <path>`: checkpoint in/out directory (default `checkpoints`).
- `--val-obj-thresh <f32>` / `--val-iou-thresh <f32>`: thresholds for val matching (mAP sweep uses 0.05–0.95 objectness).
- `--patience <usize>` / `--patience-min-delta <f32>`: optional early stop on val IoU plateau.
- `--real-val-dir <path>`: use all runs under this path for validation instead of a split.
- `--input-root <path>`: capture root to train from (default `assets/datasets/captures_filtered`); point to a filtered/pruned root if desired.
- `--stratify-split`: stratify train/val by box-count buckets (0/1/2+) instead of pure run-level split.
- `--split-manifest <path>`: optional JSON manifest; if present, load train/val label lists from it; if absent, save the current split for reuse.
- `--drop-last`: drop the last partial training batch (can help BN stability). Validation never drops.
- `--demo-checkpoint <path>`: optional model checkpoint to load at startup (model only; skips optimizer/scheduler). Useful for bundled/demo weights.
- `--metrics-out <path>`: append per-epoch val metrics (IoU/PR/mAP, tp/fp/fn) with seed/thresholds.
- `--tensor-warehouse <manifest>` (or env `TENSOR_WAREHOUSE_MANIFEST`): prefer precomputed tensor shards; falls back to live loader with a warning if unavailable.

Sample run (CPU backend):
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
  --val-iou-thresh 0.5 \
  --real-val-dir assets/datasets/real_val \
  --input-root assets/datasets/captures_filtered \
  --stratify-split \
  --demo-checkpoint assets/checkpoints/tinydet_demo.bin
```

Sample run (wgpu backend, warehouse preferred if set):
```pwsh
# Windows / NVIDIA (DX12), from repo root
$env:WGPU_POWER_PREF="high-performance"
$env:WGPU_BACKEND="dx12"
$env:WGPU_ADAPTER_NAME="NVIDIA"   # omit if single GPU
$env:RUST_LOG="info,wgpu_core=info"
$env:TENSOR_WAREHOUSE_MANIFEST="artifacts/tensor_warehouse/v<version>/manifest.json"  # required
$env:WAREHOUSE_STORE="memory"  # or mmap/stream; stream prefetch via WAREHOUSE_PREFETCH (default 2)
cargo run --features "burn_runtime,burn_wgpu" --bin train -- `
  --batch-size 64 `
  --epochs 20 `
  --scheduler cosine `
  --lr-start 3e-4 `
  --lr-end 1e-5 `
  --val-ratio 0.1
```
PowerShell one-liner variant:
```pwsh
$env:WGPU_POWER_PREF="high-performance"; $env:WGPU_BACKEND="dx12"; $env:WGPU_ADAPTER_NAME="NVIDIA"; $env:RUST_LOG="info,wgpu_core=info"; $env:TENSOR_WAREHOUSE_MANIFEST="artifacts/tensor_warehouse/v<version>/manifest.json"; $env:WAREHOUSE_STORE="memory"; cargo train_hp
```

```bash
# Linux / Vulkan (NVIDIA/AMD), from repo root
WGPU_POWER_PREF=high-performance \
WGPU_BACKEND=vulkan \
WGPU_ADAPTER_NAME="NVIDIA" \
RUST_LOG=info,wgpu_core=info \
TENSOR_WAREHOUSE_MANIFEST=artifacts/tensor_warehouse/v<version>/manifest.json \
cargo run --features "burn_runtime,burn_wgpu" --bin train -- \
  --batch-size 64 \
  --epochs 20 \
  --scheduler cosine \
  --lr-start 3e-4 \
  --lr-end 1e-5 \
  --val-ratio 0.1
```

Validation thresholds:
- `--val-obj-thresh <f32>`: objectness threshold for val matching (default 0.3).
- `--val-iou-thresh <f32>` / `--val-iou-sweep`: IoU threshold(s) for val matching/NMS (default 0.5).
Runtime inference thresholds are adjusted via hotkeys in the sim (`-`/`=` for objectness, `[`/`]` for IoU).

## What it does today
- Loads capture runs via `BatchIter` (train with aug; val without), builds TinyDet, AdamW, and a linear LR scheduler.
- Runs epoch/batch loop with per-step optimizer updates; logs loss and mean IoU each log interval.
- Validation: decodes per-cell predictions, applies sigmoid + NMS, matches to GT boxes with IoU threshold, and reports mean IoU plus precision/recall and an approximate mAP via a small PR sweep (tp/fp/fn).
- Checkpoints: on start, loads model/optim/scheduler from `ckpt_dir` if present; saves them per configured cadence (steps/epochs). Optional early stop tracks best val IoU.

## Notes
- Requires `--features burn_runtime` to pull in Burn and the training harness. Add `burn_wgpu` for GPU (wgpu); otherwise NdArray CPU is used.
- Val metric thresholds are tunable via CLI; adjust to trade off recall/precision during evaluation.
- Runtime inference loads `checkpoints/tinydet.bin`; if missing or failed, it logs a warning and falls back to the heuristic detector.
- Demo/bundled checkpoint: pass `--demo-checkpoint <path>` during training/eval; for runtime, place it at `checkpoints/tinydet.bin` (or update CLI flags) so the sim loads it automatically.
- Runtime knobs: during sim, adjust thresholds with `-`/`=` (objectness) and `[`/`]` (IoU); press `B` to toggle between Burn and heuristic detectors. The HUD shows mode, box count, inference latency, and fallback banners.
- Eval-only: `cargo run --features burn_runtime --bin eval -- --checkpoint <path> --input-root <val_root> [--val-iou-sweep ...] [--metrics-out ...]`.
