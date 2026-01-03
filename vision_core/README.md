# vision_core

Shared detector/capture/overlay interfaces for sims, tools, and inference.

Contents
- `interfaces`: Frame/DetectionResult/Label/FrameRecord; Detector/FrameSource/Recorder traits.
- `overlay`: box normalize + draw helpers.
- `capture`: CaptureLimit (max frames).
- `prelude`: re-exports interfaces, overlay helpers, CaptureLimit.

Usage
1) Add `vision_core` as a dependency.
2) Import via `vision_core::prelude::*` for interfaces/overlay helpers.
3) Implement `Detector`/`DetectorFactory` in your crate; keep heavy backends (Burn) behind feature flags outside vision_core.
4) Use `BurnDetectorFactory` only if you need a feature-gated Burn loader; keep the trait implementations out of vision_core to stay lean.

## License
Apache-2.0 (see `LICENSE` in the repo root).
