# Recipes, examples, and troubleshooting

## Known-good example run
See `reference/commands.md` for a baseline training invocation.

## Common flag sets
- Fast smoke test: `--batch-size 8 --epochs 1 --log-every 1`
- Lower RAM: use `--warehouse-store stream` and smaller `--batch-size`; set `WAREHOUSE_PREFETCH=2`
- Throughput profiling: `--warehouse-store memory --batch-size <max-that-fits> --log-every 10`

## Troubleshooting
- OOM (GPU): lower `--batch-size`, consider `--warehouse-store stream`, close other GPU workloads.
- OOM (host): prefer `mmap` or `stream`; reduce prefetch (`WAREHOUSE_PREFETCH=2`).
- Slow loads: check `WAREHOUSE_STORE`; for stream, ensure storage IO is sufficient; increase prefetch if RAM allows.
- Adapter mismatch: set `WGPU_ADAPTER_NAME` explicitly; confirm backend matches OS (DX12 on Windows, Vulkan on Linux).
- Validation errors: confirm manifest path/version and that it matches the ETL config used.
- Logging noise: adjust `--log-every`; set `RUST_LOG` to `info,wgpu_core=warn` if too verbose.
