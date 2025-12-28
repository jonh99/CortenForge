# Pointing at a manifest (CLI/env)

Set the manifest path via env or CLI. Use the versioned manifest produced by ETL.

See `reference/commands.md` for common training commands.

## Batch/epoch math
- Epoch size = `manifest.sample_count / batch_size` (rounded up). Training uses manifest counts to compute progress.
- Adjust `batch_size` based on GPU memory; scale `log_every` to keep logs readable.

## Common flags
- `--tensor-warehouse <path>` — manifest path (required if not set via env).
- `--warehouse-store <memory|mmap|stream>` — shard loading mode (defaults to memory unless overridden by env).
- `--batch-size`, `--epochs`, `--log-every`, `--status-file` — core training controls.
