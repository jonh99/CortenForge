# Testing / CI expectations

- Default backend: NdArray for fast, portable tests.
- WGPU-heavy tests: gate behind features; avoid forcing GPU/adapter setup in default runs.
- Smoke coverage:
  - Recorder sink (`capture_utils`): writes label JSON.
  - BigDet/TinyDet: forward-shape and smoke train tests (NdArray).
  - Inference factory: heuristic fallback smoke.
- Recommended commands:
  - `cargo check --workspace`
  - `cargo test --workspace --all-features` (when features are available)
- Keep test runtimes small; avoid heavy datasets in unit tests.
- When adding tests:
  - Prefer synthetic data; avoid touching large assets.
  - Add feature-gated GPU tests only if necessary; keep NdArray path as the default.
  - Keep bins/tools under test with minimal CLI args (defaults) to mirror user docs.
- Quick matrix:
  - Default CI/local: NdArray only (`cargo test --workspace`).
  - Full sweep (opt-in): `cargo test --workspace --all-features` (enables scheduler/tui/gpu_nvidia/burn_wgpu if available).
  - Manual smoke: run bins with defaults (`sim_view`, `warehouse_etl`, `train`, `inference_view`) on small inputs.
