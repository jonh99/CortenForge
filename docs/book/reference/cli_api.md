# CLI/API reference for tools

## warehouse_etl
- Purpose: build the tensor warehouse (manifest + shards) from filtered captures.
- Key flags:
  - `--input-root <path>`
  - `--output-root <path>`
  - `--target-size <HxW>`
  - `--resize-mode <letterbox|...>`
  - `--max-boxes <N>`
  - `--shard-samples <N>`
  - `--skip-empty` (if available)
  - `--code-version <sha>` (or `CODE_VERSION` env)

## warehouse_cmd
- Purpose: emit one-liner training env/command based on shell/adapter/backend.
- Key flags:
  - `--shell <ps|sh>`
  - `--adapter <amd|nvidia>`
  - `--backend <dx12|vulkan>`
  - `--manifest <path>`
  - `--store <memory|mmap|stream>`
  - `--prefetch <N>`
  - `--batch-size <N>`
  - `--log-every <N>`
  - `--extra-args <string>`
  - Convenience subcommands (if kept): `amd-ps`, `amd-sh`, `nvidia-ps`, `nvidia-sh`

## train / train_hp variants
- Purpose: train models using the tensor warehouse.
- Key flags:
  - `--tensor-warehouse <path>`
  - `--warehouse-store <memory|mmap|stream>`
  - `--batch-size <N>`
  - `--epochs <N>`
  - `--log-every <N>`
  - `--status-file <path>`
  - Other model/task-specific flags (list here once finalized).

## inference_view
- Purpose: run the trained detector live and show boxes.
- Key flags:
  - `--output-root <path>` (recording output, if enabled)
  - `--infer-obj-thresh <float>` (default `0.3`)
  - `--infer-iou-thresh <float>` (default `0.5`)
  - `--detector-weights <path>` (optional; Burn checkpoint; defaults to `checkpoints/tinydet.bin` if unset)
  - `--headless <bool>` (hide window)
  - `--max-frames <N>` (optional cap)
  - `--seed <u64>` (optional)

## single_infer
- Purpose: run the detector on a single image and emit a boxed PNG.
- Key flags:
  - `--image <path>` (required)
  - `--out <path>` (optional; defaults to `<stem>_boxed.png` next to input)
  - `--infer-obj-thresh <float>` (default `0.3`)
  - `--infer-iou-thresh <float>` (default `0.5`)
- Notes: requires Burn features/weights to use the trained model; falls back to heuristic if weights are missing. Set WGPU envs if needed (`WGPU_BACKEND`, `WGPU_ADAPTER_NAME`, `WGPU_POWER_PREF`).

## Notes
- Keep CLI help in sync with docs; update here when flags change.
- Add examples per tool in their respective sections (Warehouse/Training).
