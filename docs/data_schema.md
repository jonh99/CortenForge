# Data Schema and Run Layout

Each run is stored under `run_<timestamp>/` with three subdirectories:
- `images/` — raw PNG frames.
- `labels/` — JSON label files (`frame_XXXXX.json`).
- `overlays/` — PNGs with bounding boxes drawn (auto-generated at run end for data runs).

## run_manifest.json
Written once per run:
- `schema_version` (u32) — manifest schema version.
- `seed` (u64) — polyp generation seed.
- `output_root` (path) — root output directory.
- `run_dir` (path) — this run’s directory.
- `started_at_unix` (f64) — wall-clock time when run initialized (seconds).
- `max_frames` (optional u32) — capture frame cap, if provided.

## Label file (labels/frame_XXXXX.json)
- `frame_id` (u64)
- `sim_time` (f64)
- `unix_time` (f64)
- `image` (string) — path relative to run root, e.g., `images/frame_00000.png`
- `image_present` (bool)
- `camera_active` (bool)
- `polyp_seed` (u64)
- `polyp_labels` (array)
  - `center_world` ([f32; 3])
  - `bbox_px` (optional [f32; 4]) — min_x, min_y, max_x, max_y in pixels
  - `bbox_norm` (optional [f32; 4]) — normalized coords
