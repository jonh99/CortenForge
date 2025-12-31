# ETL / Warehouse

Build a tensor warehouse (manifest + shards) from filtered captures with defaults:
```bash
cargo run -p colon_sim_tools --bin warehouse_etl
```
Defaults:
- Input root: `assets/datasets/captures_filtered`
- Output root: `artifacts/tensor_warehouse`
- Target size: 384x384, resize mode: letterbox, max boxes: 16

Optional summary/export:
```bash
cargo run -p colon_sim_tools --bin warehouse_export -- \
  --output-root artifacts/tensor_warehouse \
  --out warehouse_summary.parquet
```

If you need a canned training command, use:
```bash
cargo run -p colon_sim_tools --bin warehouse_cmd -- --shell sh --adapter nvidia
```

Notes:
- `warehouse_etl` validates inputs and produces `manifest.json` + shard files under the output root (by default under `artifacts/tensor_warehouse/v<version>/`).
- If you prune runs, point `--input-root` at the pruned path (e.g., `captures_filtered`).
- Defaults are sane; override only when needed (e.g., different resize, max boxes, store mode).
- Manifest highlights: lists runs, sample counts, shard paths/checksums; used by training.
- Store modes: defaults to a single local store; see contributor docs if you want alternate store modes or sharding strategies for scale.
- Validation: ETL fails fast if labels/images are missing or boxes are out of bounds. Re-run capture/prune or fix inputs, then rerun ETL.
- Versioning: each ETL run writes `v<timestamp>` under the output root; training consumes the chosen version’s manifest.

What the manifest looks like (simplified):
```json
{
  "version": "v2024-05-01T12-00-00Z",
  "runs": [
    { "run_id": "run_...", "frames": 240, "path": "captures_filtered/run_..."},
    ...
  ],
  "shards": [
    { "path": "shards/shard_000.parquet", "samples": 2048, "checksum": "..." }
  ],
  "resize": { "target": [384,384], "letterbox": true },
  "max_boxes": 16
}
```
- Diagram marker: simple flow “captures_filtered → ETL → warehouse manifest/shards” with defaults called out.
