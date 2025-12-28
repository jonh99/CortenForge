# Common commands (ETL + training)

## ETL (warehouse_etl)
```bash
cargo run --bin warehouse_etl -- \
  --input-root assets/datasets/captures_filtered \
  --output-root artifacts/tensor_warehouse \
  --target-size 384x384 \
  --resize-mode letterbox \
  --max-boxes 16 \
  --shard-samples 1024
```

## Training (train_hp)
```bash
cargo train_hp -- \
  --tensor-warehouse artifacts/tensor_warehouse/v<version>/manifest.json \
  --batch-size 64 \
  --epochs 20 \
  --status-file logs/train_status.json
```

See `reference/cli_api.md` for full flag details and `reference/wgpu_envs.md` for WGPU env vars.
