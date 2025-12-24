# TUI (optional)

The TUI is an optional front-end kept separate from the core simulator. It is built only when the `tui` feature is enabled and lives in its own binary (`src/bin/tui.rs`) so core code stays free of UI dependencies.

## Running
- CPU: `cargo run --release --features tui --bin tui`
- (If you also need Burn GPU, combine features, e.g., `--features "tui,burn_runtime_wgpu"`)

## Design
- Separate bin: depends on the `colon_sim` library only; no TUI deps in core.
- Thin service surface: call into a small API for listing runs, starting datagen/train, and reading status/logs. Keep UI concerns out of core.
- UI stack: `ratatui` + `crossterm` for terminal rendering/input.
- Architecture: split state/render/handlers (e.g., `app.rs`, `ui.rs`, `handlers.rs`), drive via an event loop (input, tick, backend events).

## Keybindings
- `q` / `Esc`: quit
- `r`: refresh runs list
- `d`: start headless datagen (default output root)
- `m`: read last metrics entry (`checkpoints/metrics.jsonl`) and display (auto-refreshed on tick)
- `l`: tail last 5 lines of `logs/train.log` (auto-refreshed on tick)
- `t`: start training with defaults (writes `logs/train_status.json`)
- `↑` / `↓`: move selection in runs list (details shown in status pane)

## Status fields
- Selected run details: path, label/image/overlay counts, manifest (seed, max_frames, start time)
- Metrics: last entry from `checkpoints/metrics.jsonl` (auto)
- Logs: tail of `logs/train.log` (auto)
- Progress: if `max_frames` is in the manifest, show frames recorded vs. max
- PIDs: last launched datagen/train process IDs and whether they are still running (if started via TUI)
- Train status: contents of `logs/train_status.json` if present (status/epoch/epochs/seed/step/loss/lr)

## Next steps
- Optional: detect running datagen/train processes (and show progress) or surface error banners in the UI.
- Formatting: run `cargo fmt` locally if permissions were blocked to keep the TUI files tidy.
