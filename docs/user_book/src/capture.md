# Capture

Two primary ways to capture data:

## Interactive (sim_view)
Run the sim, drive manually, and record:
```bash
cargo run --bin sim_view
```
- Controls (HUD also shows them):
  - `WASD` + mouse: move/fly. `Shift/Ctrl`: speed up/down.
  - `C`: cycle cameras (first-person probe cam shows ground-truth boxes).
  - `P`: toggle autopilot/probe drive.
  - `L`: start/stop recording; HUD lights up while recording.
  - `H`: toggle HUD; `Esc` quits.
- Outputs: `assets/datasets/captures/run_<timestamp>/` with `run_manifest.json`, `images/`, `labels/`, `overlays/` (if you render them).
- Data run shortcut: `O` enables autopilot, switches to probe POV, and auto-starts/stops recording at the tunnel end.

## Headless wrapper (datagen)
Run headless capture with defaults:
```bash
cargo run -p colon_sim_tools --bin datagen
```
This creates a new `run_<timestamp>` under `assets/datasets/captures/`.

## Run layout
- `run_manifest.json`: metadata for the run (seed, timing, camera, version).
- `images/`: frames.
- `labels/`: `frame_XXXXX.json` (labels, polyp_seed, bbox_norm/bbox_px). Each JSON holds normalized and pixel boxes plus optional metadata (e.g., polyp seed).
- `overlays/`: optional rendered boxes (see below).
- Common manifest fields: run id, timestamp, seed, camera params, version, capture settings (resize/letterbox), checksum of frames.
- Screenshot marker: HUD with REC on and box overlays visible.
- Minimal manifest excerpt (example):
```json
{
  "run_id": "run_2024-05-01T12-00-00Z",
  "seed": 1234,
  "camera": { "fov": 90, "mode": "probe_fp" },
  "version": "v1",
  "frames": 240,
  "resize": { "target": [384, 384], "letterbox": true }
}
```

## Overlays and pruning
- Render boxes onto PNGs:
```bash
cargo run -p colon_sim_tools --bin overlay_labels -- assets/datasets/captures/run_<timestamp>
```
- Prune empty-label frames (recommended before ETL):
```bash
cargo run -p colon_sim_tools --bin prune_empty -- --input assets/datasets/captures --output assets/datasets/captures_filtered
```
- If you re-run overlays after pruning, point to the pruned root to keep paths consistent.

Quality tips:
- Keep HUD on while recording so you can confirm camera/record status.
- If you want deterministic runs, pass `--seed <n>` to `sim_view` or use headless `datagen` (defaults to headless + auto-run dir).
- Verify run completeness by checking `frames` count in `run_manifest.json` and matching `images/` + `labels/` counts.
