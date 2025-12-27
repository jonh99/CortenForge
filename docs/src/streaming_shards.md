# Warehouse shard loading modes

We support three ways to load tensor warehouse shards. Pick based on memory/IO trade-offs and set via `WAREHOUSE_STORE` (or CLI `--warehouse-store`):

- `memory` (default): load all shards into RAM (`ShardBacking::Owned`).
- `mmap`: memory-map shard files (`ShardBacking::Mmap`), on-demand paging by the OS.
- `stream`: read per-sample slices from disk (`ShardBacking::Streamed`) with a bounded prefetch queue.

## How they work
- **memory**: reads shard payloads into `Vec<f32>` once at startup. Fast iteration; highest RAM use.
- **mmap**: maps shard files; `copy_sample` slices directly from the mapped region. Lower RAM; relies on OS page cache.
- **stream**: keeps only offsets/path in memory; a worker thread seeks/reads sample slices into temp buffers and sends them through a bounded channel (`WAREHOUSE_PREFETCH`, default 2). Back-pressure blocks the producer when the queue is full.

## Configuration
- Env/CLI: `WAREHOUSE_STORE={memory|mmap|stream}`; `WAREHOUSE_PREFETCH=<N>` for streaming prefetch depth (default 2).
- Training logs the selected mode; streaming logs prefetch depth and shard/sample counts.

## Command examples
Assuming a warehouse manifest at `artifacts/tensor_warehouse/v<version>/manifest.json`:
- **Env (memory, default)**:
```bash
TENSOR_WAREHOUSE_MANIFEST=artifacts/tensor_warehouse/v<version>/manifest.json \
WAREHOUSE_STORE=memory \
cargo train_hp
```
- **CLI flag (mmap)**:
```bash
TENSOR_WAREHOUSE_MANIFEST=artifacts/tensor_warehouse/v<version>/manifest.json \
cargo run --features "burn_runtime,burn_wgpu" --bin train -- \
  --tensor-warehouse artifacts/tensor_warehouse/v<version>/manifest.json \
  --warehouse-store mmap \
  --batch-size 64
```
- **Streaming with prefetch**:
```bash
TENSOR_WAREHOUSE_MANIFEST=artifacts/tensor_warehouse/v<version>/manifest.json \
WAREHOUSE_PREFETCH=4 \
cargo run --features "burn_runtime,burn_wgpu" --bin train -- \
  --tensor-warehouse artifacts/tensor_warehouse/v<version>/manifest.json \
  --warehouse-store stream \
  --batch-size 64 \
  --epochs 20
```

## Validation & tests
- Streamed backing validates shard header (magic/version/dtype/shape/offsets) like mmap.
- Tests:
  - Integration: streamed vs owned samples match on a tiny shard.
  - Optional bench (`STREAM_BENCH=1 cargo test --tests`): logs elapsed ms/img/s for memory/stream/mmap on a small shard.

## When to use which
- Use **memory** when RAM is plentiful and you want the fastest iteration.
- Use **mmap** when RAM is tighter but you still want OS-managed caching.
- Use **stream** when RAM is constrained or shards are large; expect more disk IO but bounded memory.

## Future considerations
- Per-device cache of recent batches if streaming re-reads the same shards frequently.
- Async IO only if profiling shows thread-based streaming is a bottleneck (current implementation is sync threads for portability).
