# cortenforge

Umbrella crate for the CortenForge stack. Re-exports the app-agnostic crates with feature wiring so downstream users can opt into only what they need (sim runtime, vision core/runtime, inference, training, capture utilities, datasets, CLI helpers).

## Features
- `default`: `sim-core`, `vision-core`
- `sim-core`: re-exports `sim_core`
- `vision-core`: re-exports `vision_core`
- `vision-runtime`: re-exports `vision_runtime` (pulls vision-core)
- `capture-utils`: re-exports `capture_utils` (pulls vision-core, data-contracts)
- `data-contracts`: re-exports `data_contracts`
- `models`: re-exports `models`
- `inference`: re-exports `inference` (pulls sim-core, vision-core, models)
- `training`: re-exports `training` (pulls sim-core, vision-core, models)
- `burn-runtime`: enables burn runtime dependencies (ndarray); `burn-wgpu` adds the WGPU backend.
- `burn-dataset`: re-exports `burn_dataset`
- `cli-support`: re-exports `cli_support`

## License
Apache-2.0 (see `LICENSE` in the repo root).
