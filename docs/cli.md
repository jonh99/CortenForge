# CLI Usage

The simulator includes a Clap-based CLI to control run mode, seeding, outputs, and headless operation.

## Flags
- `--mode <sim|datagen>`: run interactively (`sim`, default) or headless data generation (`datagen`).
- `--seed <u64>`: optional seed override for reproducible polyp layouts; if omitted, a time-based seed is used.
- `--output-root <path>`: where run folders are written. Default: `assets/datasets/captures`.
- `--max-frames <N>`: optional capture frame cap for data runs; stops recording after N frames.
- `--headless`: hide the main window/offscreen rendering (useful for datagen).

## Examples
- Interactive sim with default settings:
  ```bash
  cargo run --release
  ```
- Data-gen run with fixed seed, offscreen, capped frames, custom output:
  ```bash
  cargo run --release -- --mode datagen --seed 1234 --output-root /tmp/runs --max-frames 600 --headless
  ```
- Custom output root but still interactive:
  ```bash
  cargo run --release -- --output-root /tmp/runs
  ```
