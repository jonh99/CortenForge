# Real-time (sim_view)

Run the trained detector live with `inference_view` and show bounding boxes on the camera feed.

```bash
cargo run --bin inference_view -- \
  --output-root assets/datasets/captures \
  --infer-obj-thresh 0.3 \
  --infer-iou-thresh 0.5 \
  --max-frames <optional_cap> \
  --headless <true|false> \
  --seed <optional_seed>
```

<details>
<summary>Notes</summary>

- Requires burn features/artifacts present for your trained model (ensure the build includes `burn_wgpu` and your weights are available as expected by the app).
- WGPU envs: see `reference/wgpu_envs.md` for env vars and examples.
- Bounding boxes and scores render via the HUD overlay; the HUD also shows detector status, latency, and consensus.
- `--output-root` still writes a new `run_<timestamp>` with images/labels/overlays if recording is enabled; disable recording via HUD hotkeys if you only want live inference.
- Use `--headless true` for offscreen runs; `--max-frames` to auto-stop recording; `--seed` for reproducible scenes.
</details>
