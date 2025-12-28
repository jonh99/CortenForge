# Manual capture & controls

Run the simulator interactively to collect runs and inspect frames/labels before ETL.

## Launch interactive capture
```bash
cargo run --bin sim_view -- \
  --run-dir assets/datasets/captures_raw/<run_dir>
```

## Common controls
- Use the on-screen HUD/help overlay in `sim_view`—it lists the active keybindings for capture start/stop, pause/resume, camera switches, overlays/boxes/HUD toggles, stepping frames, snapshots, and quit.
- Keybindings can change between builds; prefer the HUD over stale docs. Capture a screenshot of the HUD if you need to share controls with teammates.
- If you customize bindings, note them here for your team.

## Probe control adjustments (from `src/controls.rs`)
- Tension: `[` / `]` to decrease/increase.
- Stiffness: `;` / `'` to decrease/increase.
- Damping: `,` / `.` to decrease/increase.
- Thrust: `1` / `2` to decrease/increase.
- Target speed: `3` / `4` to decrease/increase.
- Linear damping: `5` / `6` to decrease/increase.
- Friction: `7` / `8` to decrease/increase.
- These update spring motors and damping/friction on probe segments live. Use the HUD to confirm any additional bindings.

## HUD readout (what you see in `sim_view`)
- Displays live probe params: tension (TNS), stiffness (STF), damping (DMP), thrust (THR), target speed (SPD), linear damping (LIN), friction (FRI) with their hotkeys.
- Shows tip pressure (R/U/F), steer vector/strength, polyp counts and nearest distance.
- Vision status: camera on/off, detector kind (BURN/HEUR), confidence/boxes/latency, consensus state, recording status, removal progress.

## Notes on labels
- Keep overlays and boxes visible while capturing to spot misalignments early.
- If you see label jitter or offsets, pause and capture a snapshot, then record the run_dir and steps to reproduce.

## Workflow tips
- Keep notes on what changed during a run (settings, POV).
- After a manual session, prune/organize the run_dir before ETL.
- If you find label/render issues, capture a minimal repro and file it with run_dir and steps.
