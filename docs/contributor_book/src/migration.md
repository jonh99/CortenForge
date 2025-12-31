# Migration / roadmap

- Refactor summary:
  - Root crate is orchestration-only (`src/cli/*`, `run_app`); domain systems live in app crates under `apps/`.
  - Core crates are domain-agnostic (sim_core, vision_core/runtime, data_contracts, capture_utils, models, training, inference).
  - Tools moved into `colon_sim_tools`; bins import shared helpers via `colon_sim::cli` and `colon_sim_tools::services`.
  - Recorder defaults to `JsonRecorder`; apps supply metadata/world-state hooks and can inject sinks.
  - Bins for the reference app live under `apps/colon_sim/bin`; minimal demo at `apps/hello_substrate`.
- Branding: substrate is “ForgeLoop”; app crates consume it.
- See `MIGRATION.md` for more detail and contributor guidance.
- Porting a feature to the new layout:
  1) Decide if it belongs in substrate (generic) or app (domain-specific).
  2) If generic, add hooks or helpers to core crates; gate heavy deps with features.
  3) If app-only, place code under `apps/your_app/src` and register via hooks/plugins.
  4) Add a minimal doc blurb and a smoke test (NdArray) if applicable.
- Checklist before opening a PR:
  - Docs updated (user/contributor book).
  - Defaults documented; CLI examples verified.
  - Tests: `cargo check --workspace`; add feature-gated tests if new features introduced.
 - If adding a new app:
   - Copy `apps/hello_substrate` layout; wire hooks; add bins; add a short README and a smoke test.
 - If extending tools:
   - Put helpers in `tools/src/services.rs`; keep bins thin; gate heavy deps with features.
