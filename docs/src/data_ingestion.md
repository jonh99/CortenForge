# Data ingestion safeguards

## Processing order
1) **Decode labels** from JSON into in-memory structs.
2) **Prune/clip invalid boxes** (e.g., out-of-bounds, zero-area).
3) **Drop empty**: if `skip_empty_labels` is true (default) and no labels remain after pruning, skip the sample and log the label path.
4) **Tensorize & batch** the remaining samples for the model. If an entire split is empty, the iterator yields no batches; regenerate data or point at runs with labels.

## Controls and tooling
- `DatasetConfig::skip_empty_labels` (default true) enforces step 3 above to prevent zero-target batches.
- `prune_empty` CLI: copy runs into a new folder while omitting frames/overlays with empty labels, so you can keep originals intact and train against a filtered dataset.
  - Usage: `cargo run --release --bin prune_empty -- --input <in_dir> --output <out_dir>`
  - Copies `run_manifest.json`, labels/images/overlays for frames with at least one `polyp_labels` entry; skips empty-label frames (counts printed).
  - Example (captures): `cargo run --release --bin prune_empty -- --input assets/datasets/captures --output assets/datasets/captures_filtered`
  - Example (real val): `cargo run --release --bin prune_empty -- --input assets/datasets/real_val --output assets/datasets/real_val_filtered`
