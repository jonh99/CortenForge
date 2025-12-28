pub mod interfaces;
pub mod overlay;

mod vision_core;

pub use vision_core::{
    BurnDetector, BurnInferenceState, CaptureLimit, DefaultDetectorFactory, DetectionOverlayState,
    DetectorFactory, DetectorHandle, DetectorKind, FrontCamera, FrontCameraFrame,
    FrontCameraFrameBuffer, FrontCameraState, FrontCaptureCamera, FrontCaptureReadback,
    FrontCaptureTarget, InferenceThresholds, capture_front_camera_frame,
    on_front_capture_readback, poll_burn_inference, schedule_burn_inference, setup_front_capture,
    threshold_hotkeys, track_front_camera_state,
};
