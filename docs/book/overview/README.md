# Overview

This page orients newcomers to the full data/training pipeline and gives a 10-minute quickstart. Replace the diagram placeholder once a visual is ready.

![Pipeline diagram — replace with final graphic](../media/pipeline_overview.png)

## Lifecycle at a glance

| Stage | What happens | Key artifacts/inputs | Primary commands |
| --- | --- | --- | --- |
| Ingest | Capture and prune raw data | Raw captures, filters | `cargo run --bin data_ingest …` |
| ETL | Validate, transform, shard | `manifest.json`, `shard_*.bin` | `cargo run --bin warehouse_etl …` |
| Warehouse | Store versioned tensors | `artifacts/tensor_warehouse/v<version>/` | n/a (consumed by training) |
| Train | Read warehouse, train model | checkpoints, logs | `cargo train_hp -- --tensor-warehouse …` |
| Evaluate | Analyze outputs/metrics | metrics, Parquet exports | `cargo run --bin warehouse_export …` |

## 10-minute quickstart

Follow these steps end-to-end; swap paths to match your machine.

1) Ingest or prepare filtered data roots (see ingestion chapter for capture/prune commands).  
2) Build the warehouse (see `reference/commands.md` for full snippet).
3) Train from the manifest (see `reference/commands.md` for full snippet). Set WGPU env vars if needed (see `reference/wgpu_envs.md`).
4) Inspect outputs: logs, checkpoints, and optional Parquet export:  
```bash
cargo run --bin warehouse_export -- \
  --manifest artifacts/tensor_warehouse/v<version>/manifest.json \
  --out logs/warehouse_summary.parquet
```
5) Troubleshoot with the FAQ if anything looks off, then iterate.
