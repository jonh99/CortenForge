# Recorder systems migration plan

Goal: fully encapsulate recorder systems under `sim::recorder`, leaving capture/inference logic in `vision`, and keep schedules clean. Use these smaller prompts step by step.

- Prompt 1: Copy helpers into `src/sim/recorder.rs` under new names (no call-site changes yet). ✅
  - Copied as `recorder_init_run_dirs`, `recorder_generate_overlays`, `recorder_prune_run`, `recorder_draw_rect` (label-based overlays); imported needed structs/serde/image helpers.
  - Left `vision_core.rs` unchanged; `cargo check` passes with both versions present.
- Prompt 2: Move hotkey toggles. ✅
  - Moved `recorder_toggle_hotkey` and `auto_start_recording` into `sim::recorder`, wired to use `recorder_init_run_dirs`.
  - Updated `sim::recorder` exports/imports; call sites already use `sim::recorder` versions. `cargo check` passes.
- Prompt 3: Move stop/finalize/failsafe systems. ✅
  - Moved `auto_stop_recording_on_cecum`, `finalize_datagen_run`, `datagen_failsafe_recording` into `sim::recorder` using the staged helpers; `cargo check` passes.
  - Update imports/call sites accordingly.
- Prompt 4: Move metadata capture. ✅
  - Moved `record_front_camera_metadata` (and its `DiskRecorder`) into `sim::recorder`, keeping capture/inference types from `vision` passed as params; `cargo check` passes.
- Prompt 5: Clean up re-exports/imports. ✅
  - Stopped re-exporting recorder systems from `vision`; public re-exports now limited to non-recorder vision types. All call sites point to `sim::recorder`. `CaptureLimit`/camera readback stay in `vision`.
- Prompt 6: Rewire system registration in `lib.rs` to use recorder systems from `sim` in sim/datagen mode only (inference sets untouched). ✅
  - `lib.rs` already wires recorder systems from `sim::recorder` under `ModeSet::SimDatagen`; inference ModeSet remains detector-only.
- Prompt 7: Run `cargo check` and `cargo test` to verify; fix fallout. ✅ (`cargo test` clean; only dead_code warnings in legacy vision copy.)
- Prompt 8: Update `refactor_targets.md` to mark recorder encapsulation complete.
