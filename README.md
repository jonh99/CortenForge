# Deep Poo

This is a simplified, non-commercial demonstration inspired by the locomotion principles described in United States Patent Application
[Pub. No.: US 2007/024.9906 A1](https://patentimages.storage.googleapis.com/6b/ce/a2/051225b0d2ff0a/US20070249906A1.pdf)

The implementation shown here does **not** replicate the full patented system.
Instead, it demonstrates the core inchworm-style anchoring and extension concept in a reduced, abstracted form suitable for simulation and education.

On top of this abstracted mechanism, an **original automated supervisory control layer** has been added. This control layer is not described in the patent and is introduced solely for the purposes of:
- enforcing safety interlocks,
- coordinating motion phases,
- and enabling higher-level autonomous or semi-autonomous operation in simulation.

No attempt is made to reproduce proprietary hardware, clinical configurations, or commercial embodiments described in the patent.

**Video of auto probe in action**
https://github.com/user-attachments/assets/cbf42edf-c61e-476c-b1e8-549b5f5b7580


## Controls
- `P` begin automated process
- `C` toggle camera between free-fly and probe POV
- `O` data recording run: enables autopilot + probe POV, auto-starts recording after a short delay, and auto-stops when reaching the tunnel end (no recording on the return leg)


## Binaries
- `sim_view` (apps/sim_view): interactive sim
- `inference_view` (apps/inference_view): inference mode
- `datagen_headless` (src/bin/datagen.rs): headless capture
- Training bins live under `training/` (train/eval)
- CLI tools are in the `tools/` crate (overlay_labels, prune_empty, warehouse_*; feature-gated: tui, datagen_scheduler)

## Running
- Interactive sim: `cargo run --bin sim_view --release`
- Inference view: `cargo run --bin inference_view --release`
- Tools: `cargo run -p colon_sim_tools --bin <tool> -- --help`

## Capturing + overlays
- Toggle to probe camera: press `C` until HUD shows `VISION :: cam=ON`.
- Start/stop recording: press `L` (HUD shows `REC :: on`). Frames + JSON labels saved under `assets/datasets/captures/run_<timestamp>/`.
- Data run shortcut: press `O` to enable autopilot + probe POV; recording auto-starts after ~8s and auto-stops when the probe reaches the tunnel end (no return leg captured).
- PNGs are raw; boxes live in the JSON (`labels/frame_XXXXX.json`).
- Render boxes onto PNGs (writes to `<run>/overlays` by default):
  ```bash
  cargo run -p colon_sim_tools --release --bin overlay_labels -- assets/datasets/captures/run_<timestamp>
  ```
  Or pick an output dir:
  ```bash
  cargo run -p colon_sim_tools --release --bin overlay_labels -- assets/datasets/captures/run_<timestamp> /tmp/overlays
  ```

## Polyp randomization / reproducibility
- Each run spawns polyps with randomized count, spacing, size/shape, color, and twist.
- Seed control: set `POLYP_SEED=<number>` before running to reproduce a layout; otherwise the seed comes from current time.
- The seed used for a run is stored in the capture JSON (`polyp_seed`), so datasets are traceable.

## Debug collider view
- Set `RAPIER_DEBUG_WIREFRAMES` in `src/lib.rs` to `true` to show collider wireframes (orange), or `false` to hide them. Rebuild/run after changing.

## Build your own sim on the core crates
`colon_sim_app` is the reference domain implementation built on the shared crates:
- `sim_core`: Bevy plumbing (mode sets, camera/controls hooks, autopilot/recorder scaffolding)
- `vision_core`: detector traits, overlay helpers, capture/readback types
- `vision_runtime`: Bevy capture/inference plugins on top of `vision_core`
- `inference`: Burn-backed detector factory

To build a custom sim, create your own app crate that:
1) Defines domain systems (world/entities, controls/autopilot, HUD) and re-exports them via a prelude.  
2) Registers those systems via `SimHooks` or your own plugins, keeping `sim_core` detector-free.  
3) Builds the app with `sim_core::build_app`/`SimPlugin`, and adds `vision_runtime`/`inference` plugins when you need capture + inference.

Workspace map (orchestrator + crates)
- Root crate: orchestration/CLI only (`src/cli/*`, `run_app`); domain systems live in `apps/colon_sim` (or your app crate).
- `apps/colon_sim`: reference app with world/entities, HUD, controls/autopilot hooks.
- `sim_core`: Bevy plumbing (mode sets, camera/controls hooks, recorder/autopilot scaffolding).
- `vision_core` / `vision_runtime`: detector interfaces and Bevy capture/inference plugins.
- `models`: TinyDet/BigDet model definitions.
- `training` / `inference`: training loop and Burn-backed detector factory.
- `tools`: CLI utilities (overlay, prune, warehouse commands, datagen scheduler, single_infer, gpu helper) and shared helpers.

## License
This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE` for full terms. For commercial licensing options, see `COMMERCIAL_LICENSE.md` (no patent license is granted under the default AGPL; commercial use that practices relevant patents requires a separate agreement).

## Documentation (mdBook)
Published docs: [https://via-balaena.github.io/Deep_Poo/](https://via-balaena.github.io/Deep_Poo/)

## Burn dataset loader
See the mdBook docs above (Burn Dataset section) for how to load capture runs, split train/val, and build Burn-ready batches with letterboxing and padded boxes.

## Burn training harness
Quick start:
```bash
cargo run --features burn_runtime --bin train -- --help
```
Key flags: `--batch-size`, `--epochs`, `--lr-start/--lr-end`, `--val-ratio`, `--seed`, `--ckpt-dir`, and val metric thresholds `--val-obj-thresh/--val-iou-thresh`. See the mdBook Training section for details.

## Warehouse command helper
Generate the training one-liner with a single CLI:
- PowerShell + AMD (DX12): `cargo run -p colon_sim_tools --bin warehouse_cmd -- --shell ps --adapter amd`
- PowerShell + NVIDIA (DX12): `cargo run -p colon_sim_tools --bin warehouse_cmd -- --shell ps --adapter nvidia`
- Bash + AMD (Vulkan): `cargo run -p colon_sim_tools --bin warehouse_cmd -- --shell sh --adapter amd`
- Bash + NVIDIA (Vulkan): `cargo run -p colon_sim_tools --bin warehouse_cmd -- --shell sh --adapter nvidia`

Defaults (manifest path, store/prefetch, batch/log cadence) live in `tools/warehouse_commands/lib/common.rs`. Useful overrides:
- `--manifest artifacts/tensor_warehouse/v2/manifest.json` to point at a specific version
- `--store stream --prefetch 4` to change backing and prefetch depth
- `--batch-size 64 --log-every 5` to tune training cadence
- `--backend metal --shell sh` to force a different WGPU backend
- `--extra-args "--lr-start 5e-4 --epochs 10"` to append training flags
