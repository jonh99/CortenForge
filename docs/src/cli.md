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

### How to Run the Simulator (By Scenario)

1) **Interactive sim (defaults)**
   
   Launch the visible simulator for manual driving and HUD tuning; no capture unless you trigger it.

   - Command: `cargo run --release --bin sim_view`
   - Flags: none
     - mode=`sim`, time-based seed, output root `assets/datasets/captures`
     - Recording is manual in this mode:
       - `C` — toggle camera; switch to probe POV until HUD shows `VISION :: cam=ON` (needed for POV captures).
       - `L` — start/stop recording manually; HUD shows `REC :: on` and frames/labels are written under `assets/datasets/captures/run_<timestamp>/`.
       - `O` — data-run shortcut: enables autopilot + probe POV; recording auto-starts after ~8s and auto-stops at tunnel end (return leg not recorded).

2) **Interactive sim with fixed seed**

   Same as defaults, but deterministic polyp layout via seed.

   - Command: `cargo run --release --bin sim_view -- --seed 1234`
   - Flags: `--seed`

3) **Interactive datagen (visible) with frame cap**

   Visible data-gen session; you can watch and cap frames.

   - Command: `cargo run --release --bin sim_view -- --mode datagen --max-frames 500`
   - Flags: `--mode datagen`, `--max-frames`
     - Recording: same hotkeys as sim (`C` POV toggle, `L` start/stop, `O` auto-run with auto-stop).

4) **Headless datagen with custom output + seed**

   Fully automated headless data run to a custom folder with a fixed seed.

   - Command: `cargo run --release --bin datagen_headless -- --seed 42 --output-root /tmp/runs --max-frames 600`
   - Flags: `--headless` (implied by binary), `--seed`, `--output-root`, `--max-frames`
     - Recording: automatic in headless datagen; frames/labels written under the specified output root.

5) **Headless datagen using default output root**

   Automated headless run using defaults.

   - Command: `cargo run --release --bin datagen_headless -- --mode datagen`
   - Flags: `--mode datagen` (explicit), other flags default
     - Recording: automatic in headless datagen; output under `assets/datasets/captures`.

6) **Headless datagen with explicit headless flag**
   
   Force headless via `sim_view` with datagen mode and frame cap.

   - Command: `cargo run --release --bin sim_view -- --mode datagen --headless --max-frames 300`
   - Flags: `--mode datagen`, `--headless`, `--max-frames`
     - Recording: automatic in headless/datagen mode; respect frame cap.


7) **Run with alternate output root (visible sim)**
   
   Interactive sim writing captures to a custom root when you record.

   - Command: `cargo run --release --bin sim_view -- --output-root /tmp/captures`
   - Flags: `--output-root`
     - Recording: manual via `C`/`L`/`O`; writes under the custom root.

8) **Headless datagen with only a frame cap**
   
   Automated headless run with just a frame limit.

   - Command: `cargo run --release --bin datagen_headless -- --max-frames 1000`
   - Flags: `--max-frames`
     - Recording: automatic in headless; stops at cap.

9) **Visible datagen with max frames and seed**
     
     Visible datagen with deterministic layout and cap.

     - Command: `cargo run --release --bin sim_view -- --mode datagen --seed 9876 --max-frames 750`
     - Flags: `--mode datagen`, `--seed`, `--max-frames`
       - Recording: manual hotkeys; respects frame cap if recording is on.

10) **Headless datagen writing to default root (short form)**
     
     Shortest headless command; defaults everything else.

     - Command: `cargo run --release --bin datagen_headless`
     - Flags: none (binary is headless; mode defaults to `sim` but headless path is implied)
       - Recording: automatic to `assets/datasets/captures` with time-based seed.

11) **Headless datagen with custom output and headless flag (redundant but explicit)**
     
     Headless run to a specific output root with explicit headless flag.
     
     - Command: `cargo run --release --bin datagen_headless -- --output-root /data/runs --headless`
     - Flags: `--output-root`, `--headless`
       - Recording: automatic to `/data/runs` with time-based seed unless `--seed` is provided.




### Overlay previously captured run
   
   Render boxes onto stored frames for a completed run.

   *Data runs already auto-generate overlays at run end, so the entry isn’t strictly required for normal usage. It’s still useful if you want to regenerate overlays (e.g., after changing box-drawing logic)*

   - Command: `cargo run --release --bin overlay_labels -- assets/datasets/captures/run_1234567890123`
   - Flags: positional path to run directory (no additional flags)
     - Output: writes overlays under `<run>/overlays` (or optional custom output dir).
