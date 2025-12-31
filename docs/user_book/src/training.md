# Training

Train using the warehouse manifest. Defaults use the NdArray backend unless you enable WGPU via features/env.

Minimal run:
```bash
cargo run -p training --features burn_runtime --bin train -- \
  --manifest artifacts/tensor_warehouse/v<version>/manifest.json
```

Key options you might override:
- `--model {tiny,big}` (default tiny; BigDet is multibox with `--max-boxes`)
- `--max-boxes` (for BigDet)
- `--batch-size`, `--epochs`, `--lr-*`
- `--checkpoint-out` (defaults per model)
- `--backend {ndarray,wgpu}` (NdArray by default for portability)

WGPU envs (if needed): see `reference/wgpu_envs.md` in the contributor docs.

Defaults are tuned for ease of use; start with the minimal command, then layer overrides as needed (e.g., batch size, epochs). For fast checks, keep NdArray; enable `--features burn_runtime_wgpu` if you want GPU acceleration and set WGPU envs.

Models:
- TinyDet: single-logit detector, best for single-box targets; fastest to train.
- BigDet: multibox detector; greedy IoU matching builds box/objectness targets; loss = masked L1 + BCE (optional IoU). `--max-boxes` sets the number of slots and must match ETL.

Checkpoints:
- Defaults: `checkpoints/tinydet.bin` or `checkpoints/bigdet.bin` unless you override.
- Use different filenames per model to avoid collisions; `--checkpoint-out` wins if set.

Data shapes (from ETL):
- Inputs: letterboxed RGB 384x384 (by default), normalized to [0,1].
- Targets: boxes are normalized x0,y0,x1,y1 with a per-frame mask to ignore padded slots.

Troubleshooting:
- Loss stuck near zero: ensure you passed the correct manifest; verify labels exist after prune.
- OOM: lower `--batch-size`; stick to NdArray for CPU runs.
- Slow runs: disable WGPU unless you need it; keep logging minimal (`--log-interval`).

Expected outputs (happy path):
- Training log prints loss per epoch and writes `checkpoints/<model>.bin`.
- A minimal 1-epoch sanity run should finish in minutes on CPU with NdArray.
- Reuse the checkpoint with `--detector-weights` in inference to verify end-to-end.

Worked example:
1) Run ETL: `cargo run -p colon_sim_tools --bin warehouse_etl`
2) Train TinyDet (1 epoch, small batch):  
```bash
cargo run -p training --features burn_runtime --bin train -- \
  --manifest artifacts/tensor_warehouse/v<ts>/manifest.json \
  --epochs 1 --batch-size 2 --checkpoint-out checkpoints/tinydet_demo.bin
```
3) Run inference with that checkpoint:  
```bash
cargo run --bin inference_view -- --detector-weights checkpoints/tinydet_demo.bin
```
You should see boxes overlaid in `inference_view`; adjust thresholds only if needed.
- Screenshot marker: tiny training log snippet and a checkpoints folder view showing tinydet/bigdet bins.
