use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A frame of image data and associated metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub id: u64,
    /// Sim or capture timestamp (seconds).
    pub timestamp: f64,
    /// Optional raw RGBA8 data; can be `None` when operating on file-based frames.
    pub rgba: Option<Vec<u8>>,
    /// Image dimensions (width, height).
    pub size: (u32, u32),
    /// Optional on-disk location for lazy loading.
    pub path: Option<PathBuf>,
}

/// Result of running a detector on a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub frame_id: u64,
    pub positive: bool,
    pub confidence: f32,
    /// Normalized boxes \[x0,y0,x1,y1\] in 0..1.
    pub boxes: Vec<[f32; 4]>,
    /// Per-box scores aligned with `boxes`.
    pub scores: Vec<f32>,
}

/// Polyp label metadata for a frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    pub center_world: [f32; 3],
    pub bbox_px: Option<[f32; 4]>,
    pub bbox_norm: Option<[f32; 4]>,
}

/// Data passed to a recorder sink.
#[derive(Debug)]
pub struct FrameRecord<'a> {
    pub frame: Frame,
    pub labels: &'a [Label],
    pub camera_active: bool,
    pub polyp_seed: u64,
}

/// Pulls frames from some source (capture camera, file, test generator).
pub trait FrameSource {
    fn next_frame(&mut self) -> Option<Frame>;
}

/// Runs inference on a frame.
pub trait Detector {
    fn detect(&mut self, frame: &Frame) -> DetectionResult;
    /// Optional: adjust thresholds at runtime.
    fn set_thresholds(&mut self, _obj: f32, _iou: f32) {}
}

/// Persists frames/metadata to a sink (disk, stream, etc).
pub trait Recorder {
    fn record(&mut self, record: &FrameRecord) -> std::io::Result<()>;
}

/// Optional factory for feature-flagged Burn model runtime.
pub trait BurnDetectorFactory {
    type Detector: Detector;
    fn load(model_path: &std::path::Path) -> anyhow::Result<Self::Detector>;
}
