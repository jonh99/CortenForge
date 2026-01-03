# CortenForge

Modular simulation substrate for data capture, ETL, training, and inference. CortenForge bundles the common crates, runtime wiring, and tooling; apps plug in domain logic on top.

- Core crates: `sim_core`, `vision_core` / `vision_runtime`, `data_contracts`, `capture_utils`, `models`, `training`, `inference`, `colon_sim_tools`.
- Apps: `apps/colon_sim` (reference implementation, a.k.a. Deep Poo) and `apps/hello_substrate` (minimal demo). The root crate is orchestration/CLI glue only.
- Docs: mdBook under `docs/user_book` (user workflows) and `docs/contributor_book` (architecture, extension points).
- License: Apache-2.0 by default; see `LICENSE` and `COMMERCIAL_LICENSE.md` for terms.

## Commercial opportunities
- Via Balaena™ is offering a 50% profit split on commercial deals you source and help close. Reach out if you have leads or want to collaborate on deployments.

## Quick start (defaults)
- Interactive sim (reference app): `cargo run --bin sim_view`
- Headless capture: `cargo run -p colon_sim_tools --bin datagen`
- ETL: `cargo run -p colon_sim_tools --bin warehouse_etl`
- Train: `cargo run -p training --features burn_runtime --bin train -- --manifest artifacts/tensor_warehouse/v<version>/manifest.json`
- Inference (real-time): `cargo run --bin inference_view`

Release builds: add `--release` for smoother playback/throughput. The `datagen` wrapper shells out to the sibling `sim_view` in the same target profile; build it once (`cargo build --release --bin sim_view`) if missing.

## Apps
- Reference (Deep Poo / colon_sim): domain systems, HUD, controls/autopilot, capture settings. See `apps/colon_sim/README.md` for controls, recording shortcuts, and dataset details.
- hello_substrate: minimal app showing how to hook a custom plugin into the substrate without domain systems.

## Contributing
See `docs/contributor_book` for architecture, extension points, and testing notes.
