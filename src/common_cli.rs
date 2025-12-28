use std::path::PathBuf;

use crate::cli::AppArgs;
use crate::vision::InferenceThresholds;

/// Shared thresholds used by inference-related tools.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdOpts {
    pub obj_thresh: f32,
    pub iou_thresh: f32,
}

impl ThresholdOpts {
    pub fn new(obj_thresh: f32, iou_thresh: f32) -> Self {
        Self {
            obj_thresh,
            iou_thresh,
        }
    }

    pub fn to_inference_thresholds(self) -> InferenceThresholds {
        InferenceThresholds {
            obj_thresh: self.obj_thresh,
            iou_thresh: self.iou_thresh,
        }
    }
}

impl From<&AppArgs> for ThresholdOpts {
    fn from(args: &AppArgs) -> Self {
        ThresholdOpts::new(args.infer_obj_thresh, args.infer_iou_thresh)
    }
}

/// Optional detector weights path.
#[derive(Debug, Clone)]
pub struct WeightsOpts {
    pub detector_weights: Option<PathBuf>,
}

impl WeightsOpts {
    pub fn new(detector_weights: Option<PathBuf>) -> Self {
        Self { detector_weights }
    }
}

impl From<&AppArgs> for WeightsOpts {
    fn from(args: &AppArgs) -> Self {
        WeightsOpts::new(args.detector_weights.clone())
    }
}

/// Optional WGPU env hints for tooling; consumers can apply these to the environment or log them.
#[derive(Debug, Clone, Default)]
pub struct WgpuEnvHints {
    pub backend: Option<String>,
    pub adapter_name: Option<String>,
    pub power_pref: Option<String>,
    pub rust_log: Option<String>,
}

impl WgpuEnvHints {
    pub fn empty() -> Self {
        Self::default()
    }
}
