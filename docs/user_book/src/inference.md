# Inference

## Real-time inference
Launch the inference view (defaults shown):
```bash
cargo run --bin inference_view
```
Flags:
- `--detector-weights <path>` (Burn checkpoint; optional, falls back to heuristic)
- `--infer-obj-thresh`, `--infer-iou-thresh` (defaults 0.3/0.5)
- `--headless`, `--max-frames`, `--output-root` (if you want to record)
- Overlay: bounding boxes are drawn in the viewport; set thresholds only if you need stricter filtering.
- Output: if `--output-root` is set, a `run_<timestamp>` is created with manifest/images/labels/overlays similar to capture.

## Single-image inference
Run the detector on one image and emit a boxed PNG:
```bash
cargo run -p colon_sim_tools --bin single_infer -- --image path/to/image.png
```
Options:
- `--out` (default `<stem>_boxed.png` next to input)
- `--infer-obj-thresh`, `--infer-iou-thresh`

Notes:
- If no weights are provided, a heuristic detector runs; with weights, Burn is used.
- Set WGPU envs if you need to force a backend/adapter.
- For predictable results, supply a checkpoint (`--detector-weights`) and keep defaults unless you need to tune thresholds.
- For repeatability, pin seeds in capture/training, then reuse the produced checkpoint here.
- If overlays are missing, confirm you are in `inference_view` (not `sim_view`) and that thresholds are not too strict.
- Expected outputs:
  - `inference_view` shows live boxes in the viewport; if `--output-root` is set, it writes a run dir with overlays.
  - `single_infer` writes `<stem>_boxed.png`; the log reports how many detections were drawn.

Backend tips:
- NdArray (default) keeps inference portable; enable `--features burn_runtime_wgpu` and set `WGPU_BACKEND`/`WGPU_ADAPTER_NAME` if you want GPU acceleration.
- If GPU startup fails, rerun with NdArray to confirm the checkpoint and thresholds are sound before debugging WGPU.
- Screenshot marker: inference_view viewport with boxes; single_infer boxed PNG output.
