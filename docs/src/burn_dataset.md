# Burn dataset loader usage

## Quick start
Load capture runs, split train/val by run, and iterate batches with letterboxed resizing and padded boxes:

```rust
use colon_sim::tools::burn_dataset::{BatchIter, DatasetConfig, ResizeMode, split_runs, index_runs};

let cfg = DatasetConfig {
    target_size: Some((512, 512)),
    resize_mode: ResizeMode::Letterbox,
    ..Default::default()
};

let all = index_runs(std::path::Path::new("assets/datasets/captures"))?;
let (train_idx, val_idx) = split_runs(all, 0.2);

let mut train = BatchIter::from_indices(train_idx, cfg.clone())?;
let mut val = BatchIter::from_indices(val_idx, cfg)?;

if let Some(batch) = train.next_batch::<burn_ndarray::NdArrayBackend<f32>>(8)? {
    // batch.images: [B,3,H,W], batch.boxes: [B,max_boxes,4], box_mask: [B,max_boxes]
}
```

## Feature flags
- `BatchIter`/`BurnBatch` require `--features burn_runtime` to pull in Burn (e.g., `cargo run --features burn_runtime --bin ...`).  
- Lower-level loaders (`load_run_dataset`, `DatasetSample`) work without that feature if you only need plain Rust structs.

## Augmentations
- `flip_horizontal_prob`: optional horizontal flip with box updates (after resize/letterbox).
- `scale_jitter_*`: optional zoom in/out with bbox-safe padding/cropping.
- `color_jitter_*`: optional brightness/contrast jitter.
- `noise_*`: optional uniform noise per channel.
- `blur_*`: optional blur.
- `skip_empty_labels` (default true): drop frames with no boxes; warns instead of emitting zero-target batches.

## Reproducibility
- `seed` controls shuffling in `BatchIter` so splits/batches are deterministic when set.
