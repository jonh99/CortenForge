pub mod interfaces;
pub mod overlay;

pub use ::vision_core::prelude::{
    CaptureLimit, FrontCamera, FrontCaptureCamera, FrontCaptureReadback, FrontCaptureTarget,
};
pub use vision_runtime::{
    BurnDetectionResult, BurnDetector, BurnInferenceState, CapturePlugin, DetectionOverlayState,
    DetectorHandle, DetectorKind, FrontCameraFrame, FrontCameraFrameBuffer, FrontCameraState,
    InferencePlugin, InferenceThresholds, poll_inference_task, recorder_draw_rect,
    schedule_burn_inference, threshold_hotkeys,
};

pub mod prelude {
    pub use crate::vision::{
        CapturePlugin, DetectionOverlayState, DetectorHandle, DetectorKind, InferencePlugin,
        InferenceThresholds,
    };
    pub use ::vision_core::prelude::{
        CaptureLimit, DetectionResult, Detector, Frame, FrameRecord, FrameSource, Label, Recorder,
        draw_rect, normalize_box,
    };
    pub use ::vision_core::prelude::{
        FrontCamera, FrontCaptureCamera, FrontCaptureReadback, FrontCaptureTarget,
    };
}
