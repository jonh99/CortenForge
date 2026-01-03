# vision_runtime

Bevy-facing vision runtime for capture and inference, built on `vision_core`.

Contents:
- Capture plugin: sets up a front capture camera, renders to an image target, enqueues GPU readbacks, and stores the latest frame/readback in resources.
- Inference plugin: runs detector inference asynchronously (Burn when available, heuristic fallback otherwise), updates overlay state, and exposes hotkeys for thresholds/detector switching.
- Overlay helper: `recorder_draw_rect` wraps the shared overlay helper for tools.

Runtime flags/backends:
- Burn runtime is controlled by the main crate features (`burn_runtime` / `burn_wgpu`); when Burn is unavailable, the detector kind is `Heuristic` and the overlay shows a fallback banner.
- No additional features are defined in this crate; it consumes whatever detector is provided by the inference crate via `DetectorHandle`.

Hooks / integration:
- Apps should add `CapturePlugin`/`InferencePlugin` and supply `SimRunMode` so capture/inference systems gate correctly.
- Recorder/world state is app-driven; use your own systems (e.g., `update_recorder_world_state` in the app crate) to feed recorder metadata/world state.

Smoke test guidance:
- Ensure capture readback wiring works: run the app in inference mode and confirm `FrontCaptureReadback` is populated (no panic).
- Threshold hotkeys: in inference mode, `-`/`=` adjust objectness and `[`/`]` adjust IoU; `0` forces heuristic detector. Overlay should reflect changes (fallback banner when heuristic).

## License
Apache-2.0 (see `LICENSE` in the repo root).
