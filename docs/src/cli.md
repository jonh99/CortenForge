# CLI usage

The simulator ships multiple binaries (interactive, headless data-gen, overlay tool) driven by a Clap CLI. This page lists every flag, defaults, and ready-to-run examples.

## Flags (all binaries)
- `--mode <sim|datagen>`: run interactively (`sim`, default) or headless data generation (`datagen`).
- `--seed <u64>`: optional seed override for reproducible polyp layouts; default is time-based.
- `--output-root <path>`: root for run folders. Default: `assets/datasets/captures`.
- `--max-frames <N>`: optional frame cap for data runs (stop after N frames).
- `--headless`: hide the main window / offscreen rendering (for datagen).

## Binaries
- `sim_view`: interactive/visible sim (also usable for visible datagen with `--mode datagen`).
- `datagen_headless`: headless data-gen runner.
- `overlay_labels`: draw bounding boxes onto captured frames.

## Runtime hotkeys (vision)
- `-`/`=`: decrease/increase objectness threshold.
- `[`/`]`: decrease/increase IoU threshold.
- `B`: toggle between Burn and heuristic detectors; HUD shows the active mode/box stats.
- Burn checkpoint: place model at `checkpoints/tinydet.bin` (runtime loads automatically). If missing or load fails, sim falls back to the heuristic detector and shows a fallback banner in the HUD.

## Command gallery (covers every flag)
1) **Interactive sim (defaults)**
   - Command: `cargo run --release --bin sim_view`
   - Flags: none (mode=`sim`, time-based seed, output root `assets/datasets/captures`)

2) **Interactive sim with fixed seed**
   - Command: `cargo run --release --bin sim_view -- --seed 1234`
   - Flags: `--seed`

3) **Interactive datagen (visible) with frame cap**
   - Command: `cargo run --release --bin sim_view -- --mode datagen --max-frames 500`
   - Flags: `--mode datagen`, `--max-frames`

4) **Headless datagen with custom output + seed**
   - Command: `cargo run --release --bin datagen_headless -- --seed 42 --output-root /tmp/runs --max-frames 600`
   - Flags: `--headless` (implied by binary), `--seed`, `--output-root`, `--max-frames`

5) **Headless datagen using default output root**
   - Command: `cargo run --release --bin datagen_headless -- --mode datagen`
   - Flags: `--mode datagen` (explicit), other flags default

6) **Headless datagen with explicit headless flag**
   - Command: `cargo run --release --bin sim_view -- --mode datagen --headless --max-frames 300`
   - Flags: `--mode datagen`, `--headless`, `--max-frames`

7) **Overlay previously captured run**
   - Command: `cargo run --release --bin overlay_labels -- assets/datasets/captures/run_1234567890123`
   - Flags: positional path to run directory (no additional flags)

8) **Run with alternate output root (visible sim)**
   - Command: `cargo run --release --bin sim_view -- --output-root /tmp/captures`
   - Flags: `--output-root`

9) **Headless datagen with only a frame cap**
   - Command: `cargo run --release --bin datagen_headless -- --max-frames 1000`
   - Flags: `--max-frames`

10) **Visible datagen with max frames and seed**
    - Command: `cargo run --release --bin sim_view -- --mode datagen --seed 9876 --max-frames 750`
    - Flags: `--mode datagen`, `--seed`, `--max-frames`

11) **Headless datagen writing to default root (short form)**
    - Command: `cargo run --release --bin datagen_headless`
    - Flags: none (binary is headless; mode defaults to `sim` but headless path is implied)

12) **Headless datagen with custom output and headless flag (redundant but explicit)**
    - Command: `cargo run --release --bin datagen_headless -- --output-root /data/runs --headless`
    - Flags: `--output-root`, `--headless`
