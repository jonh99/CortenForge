# Tools

Tools live in the `colon_sim_tools` crate. Most run with defaults; heavy ones are feature-gated.

Core (always available):
- `overlay_labels`: render boxes onto run images.
- `prune_empty`: filter runs to keep only labeled frames.
- `warehouse_etl`: build warehouse shards/manifest.
- `warehouse_export`: Parquet summary export.
- `warehouse_cmd`: emit a training command.
- `single_infer`: single-image detection → boxed PNG.
- `datagen`: headless capture wrapper (runs sim headless and writes a run dir).

Feature-gated:
- `datagen_scheduler` (`--features scheduler`): schedule headless datagen runs.
- `tui` (`--features tui`): terminal UI wrapper.
- `gpu_nvidia`: optional NVML telemetry for scheduler (pulls `nvml-wrapper`).

Examples:
```bash
cargo run -p colon_sim_tools --bin overlay_labels -- assets/datasets/captures/run_<ts>
cargo run -p colon_sim_tools --bin prune_empty -- --input assets/datasets/captures --output assets/datasets/captures_filtered
cargo run -p colon_sim_tools --features scheduler --bin datagen_scheduler -- --help
```

Defaults:
- Tools run with minimal flags; use defaults first, override only when needed (paths, thresholds, adapter/shell).
- Feature flags keep the default build lean; enable only what you need.
- Sane defaults: ETL targets `artifacts/tensor_warehouse`, prune defaults to keeping labeled frames, scheduler defaults to local machine (set shells/adapters if remote).

Quick reference (defaults):
- `overlay_labels <run_dir>` → writes overlays under the same run.
- `prune_empty --input captures --output captures_filtered` → keeps labeled frames only.
- `warehouse_etl` → reads `captures_filtered`, writes `artifacts/tensor_warehouse/v<ts>/`.
- `warehouse_cmd` → emits a one-liner train command tuned to the current defaults.
- `single_infer --image img.png` → writes `img_boxed.png` next to input.
- Screenshot marker: example overlay image (before/after) and a prune run folder view.
