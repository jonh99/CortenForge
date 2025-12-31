# App patterns

- Reference app: `apps/colon_sim`
  - Holds domain systems (world/entities, HUD), controls/autopilot hooks, recorder world-state updater.
  - Bins `sim_view`/`inference_view` live under `apps/colon_sim/bin` and call `run_app`.
- Minimal demo: `apps/hello_substrate`
  - Tiny plugin showing how to add systems to the substrate with no colon-specific code.
  - Useful as a template for building your own app crate.
- Building your own:
  - Add an app crate under `apps/your_app`.
  - Implement hooks (controls/autopilot) and recorder metadata/world-state updates.
  - Add bins (sim_view/inference_view) that parse args and call `run_app` or a thin wrapper.
  - Keep core crates detector-free and domain-agnostic; apps own domain logic.
  - Suggested layout:
    - `apps/your_app/src/lib.rs`: plugins + systems.
    - `apps/your_app/src/prelude.rs`: re-exports for bins/tests.
    - `apps/your_app/bin/sim_view.rs` / `inference_view.rs`: CLI → `run_app`.
  - Only touch core crates when adding generic hooks; all domain entities/systems stay in the app crate.
  - Minimal walkthrough:
    1) Copy `apps/hello_substrate` as a starter.
    2) Add your world/entities + systems; wire controls/autopilot via `SimHooks`.
    3) Add recorder world-state updater and optional meta provider.
    4) Add bins that parse args (reuse `colon_sim::cli` structs if helpful) and call `run_app`.
    5) Run `cargo check --workspace`; add a README describing your app’s systems and controls.
    6) Add a small smoke test or CLI example to ensure the app builds after changes.
