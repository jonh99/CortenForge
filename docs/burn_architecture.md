# Vision/Detection Architecture (burn.rs)

- **burn.rs (vision module)**: Wrap a lightweight detector (e.g., tiny YOLO/UNet-lite) compiled for `burn`/`wgpu`. Expose a `BurnDetector` resource with async `infer(frame) -> Vec<Detection>` returning boxes/masks + confidence. Keep model weights under `assets/models/…`.

- **Front camera capture**: Add a low-res offscreen “endoscope” camera. A `capture_frame` system grabs RGB frames every few ticks into a buffer (avoid blocking the render thread) and hands them to the detector.

- **Inference scheduling**: A `DetectionJob` resource manages at most one in-flight inference, debounced to ~5–10 Hz. Use Bevy tasks to run `BurnDetector::infer` off the main thread, then a poller system writes results back when ready.

- **Telemetry + game loop integration**: On completion, update `PolypTelemetry` and/or attach a `PolypDetections` component with boxes/masks. HUD displays detections; removal logic can slow/stop and target detected polyps without changing the physics pipeline.

- **Data collection**: Add a `Recorder` system to save captured frames plus ground-truth polyp poses to `assets/datasets/captures/…` (PNG + JSON boxes/masks). Since the sim knows polyp transforms, you can auto-generate labels; allow manual correction via a small `tools/labeler` if needed.

- **Training pipeline (offline)**: In `burn_train/`, scripts load the captured dataset, augment, train, and export weights in the format burn.rs expects. This stays out of the runtime binary.

- **Performance/robustness**: Throttle resolution/FPS, reuse textures, and keep inference async. Provide a fallback “classical” proximity detection if the model is missing or slow.
