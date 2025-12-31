# Documentation index

Two mdBooks replace the legacy docs:
- **User guide**: `docs/user_book` (how to run capture → ETL → train → infer, defaults, FAQs).
- **Contributor guide**: `docs/contributor_book` (architecture, crates, hooks, app patterns, testing, migration).

Build/read:
```bash
mdbook build docs/user_book
mdbook build docs/contributor_book
```

Repository map (docs-relevant bits):
- `src/`: root glue (`cli`, `run_app`).
- `apps/colon_sim/`: reference app (world/HUD/autopilot); bins under `apps/colon_sim/bin`.
- `apps/hello_substrate/`: minimal demo app.
- `sim_core`, `vision_core`, `vision_runtime`, `data_contracts`, `capture_utils`: substrate crates.
- `models`, `training`, `inference`: detectors, training/eval, detector factory.
- `tools`: CLI utilities (overlay/prune/etl/cmd/scheduler/tui/single_infer).

Quick doc pointers:
- Usage/default commands: see `docs/user_book/src/happy_path.md`.
- Detailed flows (capture/ETL/train/infer/tools): `docs/user_book/src/*.md`.
- Architecture/crate internals/hooks/testing: `docs/contributor_book/src/*.md`.
