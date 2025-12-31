# Architecture: substrate + apps

Substrate (Pipelinea) vs. apps:
- Substrate crates: `sim_core`, `vision_core`, `vision_runtime`, `data_contracts`, `capture_utils`, `models`, `training`, `inference`, `colon_sim_tools`.
- Apps: `apps/colon_sim` (reference world) and `apps/hello_substrate` (minimal demo).
- Root crate is glue-only: `src/cli/*`, `run_app` wires `SimConfig` + core plugins + app hooks.

Runtime flow:
1) Parse `AppArgs` → set `RunMode` → `run_app`.
2) Build via `sim_core::build_app` + `SimPlugin`/`SimRuntimePlugin`.
3) Add capture/inference plugins from `vision_runtime`; add app systems/plugins.
4) Recorder runs in substrate with default `JsonRecorder`; apps provide metadata/world state and can inject sinks.
5) Tools consume shared schemas/helpers.

Data flow (happy path):
- Capture (`sim_view` or `datagen`) → run dirs under `assets/datasets/captures` with labels/overlays.
- Optional prune/overlays (capture_utils/tools) → `captures_filtered`.
- ETL (`warehouse_etl`) → warehouse shards + manifest under `artifacts/tensor_warehouse/v<ts>/`.
- Training consumes manifest → checkpoints in `checkpoints/`.
- Inference (real-time or single-image) loads checkpoint; falls back to heuristic if missing.

Call graph (high level):
- `run_app` → `sim_core::build_app` → add SimPlugin/SimRuntimePlugin (mode sets, hooks) → add vision_runtime plugins (capture/inference) → add app plugins (controls/autopilot/HUD/world) → run.
- Recorder sits in sim_core; sinks/meta/world state are provided by the app; outputs flow to disk, ETL then reads manifests → training → inference.

Repository layout at a glance:
```
src/              # glue: cli, run_app (orchestration only)
apps/colon_sim/   # reference app (world, HUD, autopilot, bins)
apps/hello_substrate/ # minimal demo app
sim_core/         # Bevy plumbing + hooks + recorder types
vision_core/      # detector interfaces + overlay math
vision_runtime/   # capture/inference plugins for Bevy
data_contracts/   # schemas + validation
capture_utils/    # recorder sinks + overlay/prune helpers
models/           # TinyDet/BigDet definitions
training/         # training CLI + dataset/loss
inference/        # detector factory (Burn + heuristic)
tools/            # CLI tools (overlay/prune/etl/cmd/scheduler/tui)
docs/user_book/   # user-facing guide
docs/contributor_book/ # contributor guide
```

Diagram marker: simple block diagram showing substrate crates feeding apps (colon_sim/hello_substrate), tools consuming shared crates, and docs split (user vs contributor).

Principles:
- Solve real flows; keep defaults sane and visible.
- Hide complexity; small surfaces (SimHooks + recorder meta/world state).
- Avoid gratuitous abstractions; gate heavy deps; NdArray default for tests.
- Pragmatic docs; no “tech priesthood.”
