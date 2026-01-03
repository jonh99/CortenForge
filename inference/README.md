# inference crate

- Purpose: provide the Burn-backed detector factory and inference plugin used by Bevy apps (sim_view/inference_view).
- Backend: defaults to `backend-ndarray`; enable `--features backend-wgpu` for WGPU. Needs `burn` features enabled in the root build if you want GPU.
- Model: loads `TinyDet` (default) or `BigDet` from the shared `models` crate via `BinFileRecorder` (full precision). Pass a weights path to the factory to load a checkpoint; otherwise it falls back to a heuristic detector.
- Use: `run_app` (root crate) inserts the detector built by `inference::InferenceFactory` when mode==Inference. Ensure the checkpoint exists and matches the model config.
- Smoke: unit test ensures fallback when no weights are provided. Add an integration test pointing at a real checkpoint once available.

## License
Apache-2.0 (see `LICENSE` in the repo root).
