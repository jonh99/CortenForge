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


## Running
```bash
cargo run --release
```

## Capturing + overlays
- Toggle to probe camera: press `C` until HUD shows `VISION :: cam=ON`.
- Start/stop recording: press `L` (HUD shows `REC :: on`). Frames + JSON labels saved under `assets/datasets/captures/run_<timestamp>/`.
- Data run shortcut: press `O` to enable autopilot + probe POV; recording auto-starts after ~8s and auto-stops when the probe reaches the tunnel end (no return leg captured).
- PNGs are raw; boxes live in the JSON (`labels/frame_XXXXX.json`).
- Render boxes onto PNGs (writes to `<run>/overlays` by default):
  ```bash
  cargo run --release --bin overlay_labels -- assets/datasets/captures/run_<timestamp>
  ```
  Or pick an output dir:
  ```bash
  cargo run --release --bin overlay_labels -- assets/datasets/captures/run_<timestamp> /tmp/overlays
  ```

## Polyp randomization / reproducibility
- Each run spawns polyps with randomized count, spacing, size/shape, color, and twist.
- Seed control: set `POLYP_SEED=<number>` before running to reproduce a layout; otherwise the seed comes from current time.
- The seed used for a run is stored in the capture JSON (`polyp_seed`), so datasets are traceable.

## Debug collider view
- Set `RAPIER_DEBUG_WIREFRAMES` in `src/lib.rs` to `true` to show collider wireframes (orange), or `false` to hide them. Rebuild/run after changing.

## License
This project is licensed under the GNU Affero General Public License v3.0. See `LICENSE` for full terms. For commercial licensing options, see `COMMERCIAL_LICENSE.md`.

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
