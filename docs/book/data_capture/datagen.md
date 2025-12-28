# Headless datagen

Generate captures without the UI, useful for bulk data runs or CI smoke tests.

## Datagen (headless)
- Minimal smoke run (Bash):
```bash
cargo run --release --bin datagen_headless -- \
  --max-frames 20 \
  --output-root assets/datasets/captures_raw
```
- Tweak `--max-frames` and `--output-root` for real runs. Enable `RUN_DATAGEN_SMOKE=1` in CI to exercise the smoke test.
- Outputs: `images/`, `labels/`, `overlays/`, `run_manifest.json`. Prune/filter before ETL.

## Datagen scheduler (optional guards)
- Launch controlled runs with resource guards:
```bash
cargo run --bin datagen_scheduler -- \
  --runs 1 \
  --max-frames 200 \
  --output-root assets/datasets/captures_raw
```
- Add flags as needed (GPU selection, concurrency) per CLI help.

## After running
- Verify run contents, prune bad frames, and organize under `assets/datasets/captures_filtered` (or your filtered root) before ETL.
