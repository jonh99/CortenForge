# Happy path (defaults only)

Use these minimal commands to exercise the full pipeline with defaults:

1) Capture (interactive):
```bash
cargo run --bin sim_view
```
   Headless wrapper:
```bash
cargo run -p colon_sim_tools --bin datagen
```

2) ETL (build warehouse):
```bash
cargo run -p colon_sim_tools --bin warehouse_etl
```

3) Train (NdArray backend unless you enable WGPU):
```bash
cargo run -p training --features burn_runtime --bin train -- \
  --manifest artifacts/tensor_warehouse/v<version>/manifest.json
```

4) Inference (real-time):
```bash
cargo run --bin inference_view
```
   Single image:
```bash
cargo run -p colon_sim_tools --bin single_infer -- --image path/to/image.png
```

Expected artifacts by the end:
- `assets/datasets/captures/run_<ts>/` (raw capture) → optionally `captures_filtered/` after prune.
- `artifacts/tensor_warehouse/v<ts>/manifest.json` + shards (ETL).
- `checkpoints/tinydet.bin` or `checkpoints/bigdet.bin` (training).
- Overlays or boxed PNGs from inference.

Quick defaults table:
| Stage     | Command (defaults)                                                                 | Output                                      |
|-----------|------------------------------------------------------------------------------------|---------------------------------------------|
| Capture   | `cargo run --bin sim_view`                                                         | `assets/datasets/captures/run_<ts>/`        |
| Headless  | `cargo run -p colon_sim_tools --bin datagen`                                       | `assets/datasets/captures/run_<ts>/`        |
| ETL       | `cargo run -p colon_sim_tools --bin warehouse_etl`                                 | `artifacts/tensor_warehouse/v<ts>/manifest` |
| Train     | `cargo run -p training --features burn_runtime --bin train -- --manifest ...`      | `checkpoints/tinydet.bin` (or bigdet)       |
| Infer RT  | `cargo run --bin inference_view`                                                   | Live overlay; optional run dir if set       |
| Infer Img | `cargo run -p colon_sim_tools --bin single_infer -- --image img.png`               | `img_boxed.png`                             |
