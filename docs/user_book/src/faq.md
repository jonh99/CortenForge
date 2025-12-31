# FAQ / Troubleshooting

- **Where are captures saved?**  
  Under `assets/datasets/captures/run_<timestamp>/` by default. Pruned copies go to `<output_root>_filtered` if you enable prune.

- **Detector weights missing?**  
  Real-time and single-image inference fall back to a heuristic if no checkpoint is provided.

- **How do I rerun capture headless?**  
  Use `cargo run -p colon_sim_tools --bin datagen` (defaults headless on). Outputs land under `assets/datasets/captures`.

- **ETL failing validation?**  
  Check input roots/paths; ensure labels/images exist; rerun `warehouse_etl` with defaults; see contributor docs for schema/validation details.

- **WGPU backend issues?**  
  Set `WGPU_BACKEND`/`WGPU_ADAPTER_NAME`/`WGPU_POWER_PREF` as needed; defaults to system choice.

- **Train command not matching my setup?**  
  Use `warehouse_cmd` to emit a tailored one-liner; override manifest/store/prefetch/batch/log flags as needed.

- **Boxes not showing?**  
  `sim_view` focuses on capture; real-time detector overlay is in `inference_view`. For offline overlays, run `overlay_labels` on a captured run.
- **Recording doesn’t start?**  
  Ensure HUD shows `REC` when you press `L`; autopilot start/stop (`O`) only works in the tunnel flow. Check disk space under `assets/datasets/captures`.
- **Store mode / version confusion?**  
  Each ETL run writes a new `v<timestamp>` under `artifacts/tensor_warehouse/`. Point training at the manifest inside the version you want.
- **Scheduler/TUI missing?**  
  Enable features: `cargo run -p colon_sim_tools --features scheduler --bin datagen_scheduler -- --help` (or `--features tui`).
- **Where do checkpoints live?**  
  By default under `checkpoints/`; TinyDet → `tinydet.bin`, BigDet → `bigdet.bin` unless you override `--checkpoint-out`.
- **Do I need to set seeds?**  
  For reproducibility across capture/train/infer, set seeds in capture (`--seed`), keep ETL defaults, and train with the same manifest.
