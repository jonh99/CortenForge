# Refactor targets (architecture/code cleanup)

- Detector/backend plumbing:
  1) Add a detector factory trait that returns the primary model or a heuristic fallback. ✅
  2) Move weight path selection into config (flag/env) so `vision.rs` doesn’t hardcode it. ✅
  3) Update `DetectorHandle` to use the factory; keep logging/fallback behavior consistent. ✅
- CLI/config reuse:
  1) Add a `common_cli` module with shared structs for output roots, thresholds, WGPU hints (and parsing helpers). ✅
  2) Wire `sim_view`, `inference_view`, and `single_infer` to use the shared parsing/validation (optionally `warehouse_cmd` if desired). ✅
  3) Update docs to point bins at the shared WGPU/env reference (already exists) and align flag descriptions. ✅
- Overlay drawing:
  1) Add a small helper module (e.g., `vision_overlay`) for box normalization/styling/drawing. ✅
  2) Switch `single_infer` to use it. ✅
  3) Switch `overlay_labels` and HUD overlay rendering to use it. ✅
  4) Add a minimal test/snapshot for the helper if practical. ✅
- Mode-specific systems:
  1) Define mode-specific Bevy schedules/system sets (sim/datagen vs inference). ✅
  2) Register only relevant systems/resources per mode (e.g., skip autopilot/recorder in inference). ✅
  3) Verify `sim_view` and `inference_view` still build/run as expected. ✅
- Docs dedupe:
  1) Centralize ETL/train command snippets in one include or reference page. ✅
  2) Link training/ingestion pages to the shared snippet; remove duplicates. ✅
- File/org:
  1) Confirm bin placement (`src/bin` for app entrypoints; tools under `tools/`). ✅
  2) Consider a `vision/` submodule to house detector/overlay code separate from sim logic. ✅

- Sim autopilot/recorder separation:
  1) Move autopilot/recorder systems into a `sim` submodule (e.g., `sim/autopilot.rs`, `sim/recorder.rs`). ✅
  2) Wire them via mode sets so `run_app` stays slim. ✅
  3) Recorder systems split (plan):
     - Extract recorder systems (auto_start_recording, auto_stop_recording_on_cecum, finalize_datagen_run, datagen_failsafe_recording, record_front_camera_metadata, recorder_toggle_hotkey) into `sim::recorder`. ✅ (full bodies moved; vision retains capture/inference only)
     - Rewire schedules to use recorder systems from `sim` in sim/datagen mode only. ✅
     - Verify `sim_view`/`inference_view` still build/run. ✅ (`cargo test` clean)

- CLI parsing reuse expansion:
  1) Extend `common_cli` to cover output roots/prune flags.
  2) Adopt it in `datagen`, `warehouse_etl/export`, `warehouse_cmd`.

- Tools module/crate:
  1) Group command-line tooling (e.g., `overlay_labels`, `prune_empty`) under a `tools` module/crate.
  2) Share drawing/schema helpers there to keep app runtime lean.

- Warehouse commands cleanup:
  1) Mirror the vision split for `tools/warehouse_commands` (common config + builders module).
  2) Keep bins as thin wrappers.

- Tests/docs organization:
  1) Group tests under `tests/` by domain (vision, warehouse, sim).
  2) Add module-level README/docs to new submodules to orient contributors.
