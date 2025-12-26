# Tensor Warehouse

A versioned, precomputed tensor store so training never decodes or resizes on the fly. ETL builds shards + manifest; training only reads the warehouse.

## What you get
- Versioned root (e.g., `artifacts/tensor_warehouse/v<sha>/`) with `manifest.json` and `shard_*.bin`.
- Deterministic transforms baked in (decode + resize/letterbox + box normalization); no augments cached.
- Strict validation (schema/bboxes/image presence) with summaries in the manifest.
- Observability: ETL timings/skip counts, shard load timings in training, optional trace, Parquet export of summaries.

## Layout
- `manifest.json`: dataset_root, cacheable transform config, version hash/recipe/code_version, thresholds, per-run + aggregate summaries, shard list with checksums/dtypes.
- `shard_*.bin`: header + contiguous CHW tensors (f32), padded boxes/masks. Default shard format: magic `TWH1`, little-endian, `shard_version=1`, dtype=f32, channels=3, offsets for images/boxes/masks. Size target ~128–256 MB/shard.

## Build (ETL)
- Base command (f32, 256x256 letterbox):  
  `CODE_VERSION=$(git rev-parse --short HEAD 2>/dev/null || echo "") cargo run --bin warehouse_etl -- --input-root assets/datasets/captures_filtered --output-root artifacts/tensor_warehouse --target-size 384x384 --resize-mode letterbox --max-boxes 16 --shard-samples 1024`
- PowerShell equivalent:  
  ```pwsh
  $env:CODE_VERSION = (git rev-parse --short HEAD 2>$null)
  cargo run --bin warehouse_etl -- --input-root assets/datasets/captures_filtered --output-root artifacts/tensor_warehouse --target-size 384x384 --resize-mode letterbox --max-boxes 16 --shard-samples 1024
  ```
  (Set once per PowerShell session; reuse for multiple runs unless you change commits—then re-run the assignment to keep the hash current.)
- Controls:  
  - `WAREHOUSE_CLEAR=1` to delete the computed version directory before writing.  
  - `WAREHOUSE_SKIP_IF_EXISTS=1` to exit early if the versioned manifest already exists.  
  - Validation env (fail/warn): e.g., `BURN_DATASET_MAX_INVALID=0`, `BURN_DATASET_MAX_MISSING=0`, `BURN_DATASET_MAX_EMPTY_RATIO=0.05`.
- Outputs to `artifacts/tensor_warehouse/v<version>/manifest.json` plus shards.

## Train from the warehouse
- Training requires a manifest; no live loader fallback.  
  `cargo train_hp -- --tensor-warehouse artifacts/tensor_warehouse/v<version>/manifest.json --batch-size 64 --epochs 20 --status-file logs/train_status.json`
- WGPU env (Windows NVIDIA example):  
  `$env:WGPU_POWER_PREF="high-performance"; $env:WGPU_BACKEND="dx12"; $env:WGPU_ADAPTER_NAME="NVIDIA"; $env:RUST_LOG="info,wgpu_core=info"`
- Training logs shard load times on startup and uses manifest counts for per-epoch math and progress.

## Analytics
- Export manifest summaries to Parquet:  
  `cargo run --bin warehouse_export -- --manifest artifacts/tensor_warehouse/v<version>/manifest.json --out logs/warehouse_summary.parquet`
- Quick peek with DuckDB:  
  `duckdb -c "SELECT * FROM 'logs/warehouse_summary.parquet' LIMIT 5"`

## Notes
- Version key = SHA256(source root + cacheable transform config + max_boxes + skip_empty + code_version), stored in manifest; shards live under `v<version>/`.
- Checksums per shard (SHA256) are recorded; f32 shards only for now.
- ETL trace: set `WAREHOUSE_TRACE=logs/warehouse_trace.jsonl` to emit per-shard timing.
- Warehouse loader store: choose via `WAREHOUSE_STORE` = `memory` (default), `mmap`, or `stream` (prefetch via `WAREHOUSE_PREFETCH`, default 2). Training also accepts `--warehouse-store` to override.
  - Trade-offs: `memory` loads all shards up front (fast per-batch, high RAM). `mmap` keeps shards on disk (lower RAM, depends on IO). `stream` reads via a bounded prefetch queue (lowest RAM, may increase IO contention; tune `WAREHOUSE_PREFETCH`).
