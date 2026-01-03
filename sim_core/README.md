# sim_core

Sim runtime primitives shared by Bevy apps (sim, datagen, inference). It owns:
- ModeSet + SimRunMode: system sets for common vs sim/datagen vs inference paths.
- SimConfig: mode, headless, capture output/prune settings, max_frames, optional capture interval.
- Plugins: SimPlugin (mode sets), SimRuntimePlugin (default runtime resources); app crates wire their own systems (e.g., AppSystemsPlugin in `apps/colon_sim`).
- Modules: camera/controls, autopilot_types/recorder_types, probe_types.

How to use
1) Build the base app via `sim_core::build_app(SimConfig { .. })`.
2) Add plugins: `SimPlugin`, `SimRuntimePlugin`, plus your app systems plugin (e.g., `AppSystemsPlugin`), and any app-specific bootstrap (e.g., environment).
3) If you need inference, add detector resources/systems in an inference-only branch (ModeSet::Inference).
4) Bins set `SimRunMode` through CLI (sim/datagen/inference) and pass headless/output/prune/max_frames via `SimConfig`.

Adding systems
- Common systems: add to ModeSet::Common.
- Sim/datagen-only: add to ModeSet::SimDatagen.
- Inference-only: add to ModeSet::Inference.
Use `SimRuntimePlugin` to keep registration in one place; avoid detector wiring here to keep the core crate lean. Recorder metadata/sink/world-state live here (`recorder_meta`); apps provide world-state updates and can inject custom sinks.

Layout note
- Root crate is orchestration/CLI only (`src/cli/*`, `run_app`); domain systems live in app crates (e.g., `apps/colon_sim`).

## License
Apache-2.0 (see `LICENSE` in the repo root).
