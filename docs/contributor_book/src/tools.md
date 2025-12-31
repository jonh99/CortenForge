# Tools crate

- Crate: `colon_sim_tools`
- Bins:
  - Core: `overlay_labels`, `prune_empty`, `warehouse_etl`, `warehouse_export`, `warehouse_cmd`, `single_infer`.
  - Feature-gated: `datagen_scheduler` (`scheduler` feature), `tui` (`tui` feature), `gpu_nvidia` (optional NVML for scheduler).
- Shared helpers:
  - Uses `data_contracts`, `vision_core`/`capture_utils` for overlay/prune.
  - Imports common CLI args from `colon_sim::cli`.
  - `tools/src/services.rs` holds warehouse commands and shared helpers; prefer adding new helpers there instead of duplicating.
- Defaults:
  - Tools default to lean deps; heavy deps are gated.
- Testing:
  - NdArray backend for fast tests; gate WGPU-heavy paths behind features.
 - Adding a tool:
   - Put the bin under `tools/src/bin/your_tool.rs`.
   - Reuse CLI parsers from `colon_sim::cli` where possible.
   - Favor capture_utils/vision_core helpers for overlays/prune; avoid reimplementing.
