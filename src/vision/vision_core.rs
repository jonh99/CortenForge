use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::tasks::Task;
use std::path::Path;

use crate::cli::RunMode;
use crate::polyp::PolypDetectionVotes;
use crate::sim::recorder::RecorderConfig;
use crate::vision::interfaces::{self, DetectionResult, Frame};

#[cfg(feature = "burn_runtime")]
use crate::burn_model::{TinyDet, TinyDetConfig, nms};
#[cfg(all(feature = "burn_runtime", not(feature = "burn_wgpu")))]
use burn::backend::ndarray::NdArray;
#[cfg(feature = "burn_runtime")]
use burn::module::Module;
#[cfg(feature = "burn_runtime")]
use burn::record::{BinFileRecorder, FullPrecisionSettings};
#[cfg(feature = "burn_runtime")]
use burn::tensor::Tensor;
#[cfg(feature = "burn_runtime")]
use burn::tensor::backend::Backend;
#[cfg(all(feature = "burn_runtime", feature = "burn_wgpu"))]
use burn_wgpu::Wgpu;
#[cfg(feature = "burn_runtime")]
use std::sync::{Arc, Mutex};

#[derive(Component)]
pub struct FrontCamera;

#[derive(Component)]
pub struct FrontCaptureCamera;

#[derive(Clone)]
pub struct FrontCameraFrame {
    pub id: u64,
    pub transform: GlobalTransform,
    pub captured_at: f64,
}

#[derive(Resource, Default)]
pub struct FrontCameraState {
    pub active: bool,
    pub last_transform: Option<GlobalTransform>,
    pub frame_counter: u64,
}

#[derive(Resource, Default)]
pub struct FrontCameraFrameBuffer {
    pub latest: Option<FrontCameraFrame>,
}

#[derive(Resource, Default)]
pub struct BurnDetector {
    pub model_loaded: bool,
}

#[derive(Resource, Default, Clone)]
pub struct DetectionOverlayState {
    pub boxes: Vec<[f32; 4]>,
    pub scores: Vec<f32>,
    pub size: (u32, u32),
    pub fallback: Option<String>,
    pub inference_ms: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Resource)]
pub enum DetectorKind {
    Burn,
    Heuristic,
}

#[cfg(all(feature = "burn_runtime", feature = "burn_wgpu"))]
type RuntimeBackend = Wgpu<f32>;
#[cfg(all(feature = "burn_runtime", not(feature = "burn_wgpu")))]
type RuntimeBackend = NdArray<f32>;

#[derive(Resource, Debug, Clone, Copy)]
pub struct InferenceThresholds {
    pub obj_thresh: f32,
    pub iou_thresh: f32,
}

#[derive(Clone)]
pub struct BurnDetectionResult {
    pub frame_id: u64,
    pub positive: bool,
    pub confidence: f32,
    pub boxes: Vec<[f32; 4]>,
    pub scores: Vec<f32>,
}

struct HeuristicDetector;
impl interfaces::Detector for HeuristicDetector {
    fn detect(&mut self, frame: &Frame) -> DetectionResult {
        DetectionResult {
            frame_id: frame.id,
            positive: true,
            confidence: 0.8,
            boxes: Vec::new(),
            scores: Vec::new(),
        }
    }
}

#[derive(Resource)]
pub struct BurnInferenceState {
    pub pending: Option<Task<BurnDetectionResult>>,
    pub last_result: Option<BurnDetectionResult>,
    pub debounce: Timer,
}

impl Default for BurnInferenceState {
    fn default() -> Self {
        Self {
            pending: None,
            last_result: None,
            debounce: Timer::from_seconds(0.18, TimerMode::Repeating),
        }
    }
}

#[derive(Resource)]
pub struct DetectorHandle {
    pub detector: Box<dyn interfaces::Detector + Send + Sync>,
    pub kind: DetectorKind,
}

pub trait DetectorFactory: Send + Sync {
    fn build(&self, thresh: InferenceThresholds, weights: Option<&Path>) -> DetectorHandle;
}

pub struct DefaultDetectorFactory;

impl DetectorFactory for DefaultDetectorFactory {
    fn build(&self, _thresh: InferenceThresholds, _weights: Option<&Path>) -> DetectorHandle {
        #[cfg(feature = "burn_runtime")]
        {
            if let Some(det) = BurnTinyDetDetector::from_path_or_default(_thresh, _weights) {
                return DetectorHandle {
                    detector: Box::new(det),
                    kind: DetectorKind::Burn,
                };
            }
        }
        DetectorHandle {
            detector: Box::new(HeuristicDetector),
            kind: DetectorKind::Heuristic,
        }
    }
}

#[cfg(feature = "burn_runtime")]
struct BurnTinyDetDetector {
    model: Arc<Mutex<TinyDet<RuntimeBackend>>>,
    device: <RuntimeBackend as Backend>::Device,
    obj_thresh: f32,
    iou_thresh: f32,
}

#[cfg(feature = "burn_runtime")]
impl BurnTinyDetDetector {
    fn load_from_checkpoint(path: &Path, thresh: InferenceThresholds) -> anyhow::Result<Self> {
        let device = <RuntimeBackend as Backend>::Device::default();
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        let model = TinyDet::<RuntimeBackend>::new(TinyDetConfig::default(), &device)
            .load_file(path, &recorder, &device)?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            device,
            obj_thresh: thresh.obj_thresh,
            iou_thresh: thresh.iou_thresh,
        })
    }

    fn from_path_or_default(thresh: InferenceThresholds, weights: Option<&Path>) -> Option<Self> {
        let candidate = weights
            .map(|p| p.to_path_buf())
            .or_else(|| Some(Path::new("checkpoints").join("tinydet.bin")));
        let Some(path) = candidate else {
            return None;
        };
        if path.exists() {
            match Self::load_from_checkpoint(&path, thresh) {
                Ok(det) => return Some(det),
                Err(err) => {
                    warn!(
                        "Burn checkpoint load failed at {:?}: {:?}. Falling back to heuristic.",
                        path, err
                    );
                    return None;
                }
            }
        }
        warn!(
            "Burn checkpoint {:?} not found; using heuristic detector instead.",
            path
        );
        None
    }

    fn rgba_to_tensor(&self, rgba: &[u8], size: (u32, u32)) -> Tensor<RuntimeBackend, 4> {
        let (w, h) = size;
        let mut data = vec![0.0f32; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 4) as usize;
                let dst = ((y * w + x) * 3) as usize;
                data[dst] = rgba[idx] as f32 / 255.0;
                data[dst + 1] = rgba[idx + 1] as f32 / 255.0;
                data[dst + 2] = rgba[idx + 2] as f32 / 255.0;
                // alpha ignored
            }
        }
        Tensor::<RuntimeBackend, 4>::from_floats(data.as_slice(), &self.device)
            .reshape([1, 3, h as usize, w as usize])
    }
}

#[cfg(feature = "burn_runtime")]
impl interfaces::Detector for BurnTinyDetDetector {
    fn detect(&mut self, frame: &Frame) -> DetectionResult {
        let rgba = match &frame.rgba {
            Some(buf) => buf,
            None => {
                return DetectionResult {
                    frame_id: frame.id,
                    positive: false,
                    confidence: 0.0,
                    boxes: Vec::new(),
                    scores: Vec::new(),
                };
            }
        };
        let input = self.rgba_to_tensor(rgba, frame.size);
        let (obj_logits, box_logits) = {
            if let Ok(guard) = self.model.lock() {
                guard.forward(input)
            } else {
                return DetectionResult {
                    frame_id: frame.id,
                    positive: false,
                    confidence: 0.0,
                    boxes: Vec::new(),
                    scores: Vec::new(),
                };
            }
        };
        let obj = match obj_logits.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => {
                return DetectionResult {
                    frame_id: frame.id,
                    positive: false,
                    confidence: 0.0,
                    boxes: Vec::new(),
                    scores: Vec::new(),
                };
            }
        };
        let boxes = match box_logits.to_data().to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => {
                return DetectionResult {
                    frame_id: frame.id,
                    positive: false,
                    confidence: 0.0,
                    boxes: Vec::new(),
                    scores: Vec::new(),
                };
            }
        };
        let dims = obj_logits.dims();
        if dims.len() != 4 {
            return DetectionResult {
                frame_id: frame.id,
                positive: false,
                confidence: 0.0,
                boxes: Vec::new(),
                scores: Vec::new(),
            };
        }
        let (_b, _c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let hw = h * w;
        let mut preds = Vec::new();
        for yi in 0..h {
            for xi in 0..w {
                let idx = yi * w + xi;
                let score = 1.0 / (1.0 + (-obj[idx]).exp());
                if score < self.obj_thresh {
                    continue;
                }
                let base = yi * w + xi;
                let b0 = 1.0 / (1.0 + (-boxes[base]).exp());
                let b1 = 1.0 / (1.0 + (-boxes[base + hw]).exp());
                let b2 = 1.0 / (1.0 + (-boxes[base + 2 * hw]).exp());
                let b3 = 1.0 / (1.0 + (-boxes[base + 3 * hw]).exp());
                preds.push((score, [b0, b1, b2, b3]));
            }
        }
        if preds.is_empty() {
            return DetectionResult {
                frame_id: frame.id,
                positive: false,
                confidence: 0.0,
                boxes: Vec::new(),
                scores: Vec::new(),
            };
        }
        preds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let boxes_only: Vec<[f32; 4]> = preds.iter().map(|p| p.1).collect();
        let scores_only: Vec<f32> = preds.iter().map(|p| p.0).collect();
        let keep = nms(&boxes_only, &scores_only, self.iou_thresh);
        let best_idx = keep.first().copied().unwrap_or(0);
        let best = preds.get(best_idx).cloned().unwrap_or((0.0, [0.0; 4]));

        DetectionResult {
            frame_id: frame.id,
            positive: best.0 > self.obj_thresh,
            confidence: best.0 as f32,
            boxes: keep.iter().map(|&i| boxes_only[i]).collect(),
            scores: keep.iter().map(|&i| scores_only[i]).collect(),
        }
    }

    fn set_thresholds(&mut self, obj: f32, iou: f32) {
        self.obj_thresh = obj;
        self.iou_thresh = iou;
    }
}

#[derive(Resource, Default)]
pub struct CaptureLimit {
    pub max_frames: Option<u32>,
}

pub fn track_front_camera_state(
    mut state: ResMut<FrontCameraState>,
    mut votes: ResMut<PolypDetectionVotes>,
    cams: Query<(&Camera, &GlobalTransform), With<FrontCamera>>,
) {
    let mut active = false;
    let mut transform = None;
    for (cam, tf) in &cams {
        if cam.is_active {
            active = true;
            transform = Some(*tf);
            break;
        }
    }
    state.active = active;
    state.last_transform = transform;

    if !state.active {
        votes.vision = false;
    }
}

#[derive(Resource)]
pub struct FrontCaptureTarget {
    pub handle: Handle<Image>,
    pub size: UVec2,
    pub entity: Entity,
}

#[derive(Resource, Default, Clone)]
pub struct FrontCaptureReadback {
    pub latest: Option<Vec<u8>>,
}

pub fn setup_front_capture(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    config: Res<RecorderConfig>,
) {
    let size = config.resolution;
    let extent = Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: 1,
    };
    let mut image = Image::new_fill(
        extent,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::all(),
    );
    image.texture_descriptor.usage |= TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC;
    let handle = images.add(image);
    let entity = commands
        .spawn((
            Name::new("FrontCaptureTarget"),
            Readback::texture(handle.clone()),
        ))
        .id();
    commands.insert_resource(FrontCaptureTarget {
        handle: handle.clone(),
        size,
        entity,
    });
}

pub fn capture_front_camera_frame(
    time: Res<Time>,
    mut state: ResMut<FrontCameraState>,
    mut buffer: ResMut<FrontCameraFrameBuffer>,
) {
    if !state.active {
        buffer.latest = None;
        return;
    }
    let Some(transform) = state.last_transform else {
        return;
    };
    state.frame_counter = state.frame_counter.wrapping_add(1);
    buffer.latest = Some(FrontCameraFrame {
        id: state.frame_counter,
        transform,
        captured_at: time.elapsed_secs_f64(),
    });
}

pub fn on_front_capture_readback(
    ev: On<ReadbackComplete>,
    target: Res<FrontCaptureTarget>,
    mut readback: ResMut<FrontCaptureReadback>,
) {
    let expected_len = (target.size.x * target.size.y * 4) as usize;
    let ev = ev.event();
    if ev.entity != target.entity {
        return;
    }
    if ev.data.len() == expected_len {
        readback.latest = Some(ev.data.clone());
    }
}

pub fn schedule_burn_inference(
    mode: Res<RunMode>,
    time: Res<Time>,
    mut burn_detector: ResMut<BurnDetector>,
    mut jobs: ResMut<BurnInferenceState>,
    mut buffer: ResMut<FrontCameraFrameBuffer>,
    mut handle: ResMut<DetectorHandle>,
    capture: Res<FrontCaptureTarget>,
    mut readback: ResMut<FrontCaptureReadback>,
    mut overlay: ResMut<DetectionOverlayState>,
    _thresh: Res<InferenceThresholds>,
) {
    if *mode != RunMode::Inference {
        return;
    }
    jobs.debounce.tick(time.delta());
    if jobs.pending.is_some() || !jobs.debounce.is_finished() {
        return;
    }
    let Some(frame) = buffer.latest.take() else {
        return;
    };

    // Run detection via the vision interface (sync for now).
    let rgba = readback.latest.take();
    let start = std::time::Instant::now();
    let f = Frame {
        id: frame.id,
        timestamp: frame.captured_at,
        rgba,
        size: (capture.size.x, capture.size.y),
        path: None,
    };
    let result = handle.detector.detect(&f);
    let infer_ms = start.elapsed().as_secs_f32() * 1000.0;
    burn_detector.model_loaded = handle.kind == DetectorKind::Burn;
    if handle.kind == DetectorKind::Heuristic {
        overlay.fallback = Some("Heuristic detector active (Burn unavailable)".into());
    } else {
        overlay.fallback = None;
    }
    overlay.inference_ms = Some(infer_ms);
    overlay.boxes = result.boxes.clone();
    overlay.scores = result.scores.clone();
    overlay.size = (capture.size.x, capture.size.y);
    jobs.last_result = Some(BurnDetectionResult {
        frame_id: result.frame_id,
        positive: result.positive,
        confidence: result.confidence,
        boxes: result.boxes,
        scores: result.scores,
    });
}

pub fn poll_burn_inference(jobs: Res<BurnInferenceState>, mut votes: ResMut<PolypDetectionVotes>) {
    if let Some(result) = jobs.last_result.as_ref() {
        votes.vision = result.positive;
    }
}

pub fn threshold_hotkeys(
    mode: Res<RunMode>,
    keys: Res<ButtonInput<KeyCode>>,
    thresh: Option<ResMut<InferenceThresholds>>,
    handle: Option<ResMut<DetectorHandle>>,
    #[cfg(feature = "burn_runtime")] burn_loaded: Option<ResMut<BurnDetector>>,
) {
    if *mode != RunMode::Inference {
        return;
    }
    let (Some(mut thresh), Some(mut handle)) = (thresh, handle) else {
        return;
    };
    #[cfg(feature = "burn_runtime")]
    let Some(mut burn_loaded) = burn_loaded else {
        return;
    };

    let mut changed = false;
    if keys.just_pressed(KeyCode::Minus) {
        thresh.obj_thresh = (thresh.obj_thresh - 0.05).clamp(0.0, 1.0);
        changed = true;
    }
    if keys.just_pressed(KeyCode::Equal) {
        thresh.obj_thresh = (thresh.obj_thresh + 0.05).clamp(0.0, 1.0);
        changed = true;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        thresh.iou_thresh = (thresh.iou_thresh - 0.05).clamp(0.1, 0.95);
        changed = true;
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        thresh.iou_thresh = (thresh.iou_thresh + 0.05).clamp(0.1, 0.95);
        changed = true;
    }
    if changed {
        handle
            .detector
            .set_thresholds(thresh.obj_thresh, thresh.iou_thresh);
        info!(
            "Thresholds updated: obj={:.2}, iou={:.2}",
            thresh.obj_thresh, thresh.iou_thresh
        );
    }

    if keys.just_pressed(KeyCode::KeyB) {
        #[cfg(feature = "burn_runtime")]
        {
            if handle.kind == DetectorKind::Burn {
                handle.detector = Box::new(HeuristicDetector);
                handle.kind = DetectorKind::Heuristic;
                info!("Switched to heuristic detector");
            } else if let Some(det) = BurnTinyDetDetector::from_default_or_fallback(*thresh) {
                handle.detector = Box::new(det);
                handle.kind = DetectorKind::Burn;
                burn_loaded.model_loaded = true;
                info!("Switched to Burn detector");
            } else {
                info!("Burn detector unavailable; staying on heuristic");
            }
        }
        #[cfg(not(feature = "burn_runtime"))]
        {
            handle.detector = Box::new(HeuristicDetector);
            handle.kind = DetectorKind::Heuristic;
            info!("Burn runtime disabled; heuristic only");
        }
    }
}

// Adapters for the vision interfaces can be introduced here when swapping
// detectors/sources; left uninstantiated for now to avoid unused-code noise.
// (e.g., a CaptureSource wrapping FrontCameraFrameBuffer/FrontCaptureReadback,
// and a DiskRecorder wrapping RecorderState/Config).
