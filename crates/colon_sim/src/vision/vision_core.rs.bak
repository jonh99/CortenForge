use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::tasks::Task;
use image::{ImageFormat, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::sim::autopilot::AutoDrive;
use crate::sim::recorder::{AutoRecordTimer, RecorderConfig, RecorderMotion, RecorderState};
use crate::balloon_control::BalloonControl;
use crate::camera::PovState;
use crate::cli::RunMode;
use crate::polyp::PolypDetectionVotes;
use crate::polyp::PolypTelemetry;
use crate::tunnel::CecumState;
use crate::vision::interfaces::{self, DetectionResult, Frame, FrameRecord, Label, Recorder};

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

const MAX_LABEL_DEPTH: f32 = 8.0;
const IMAGES_DIR: &str = "images";
const LABELS_DIR: &str = "labels";
const OVERLAYS_DIR: &str = "overlays";

#[derive(Resource, Default)]
pub struct CaptureLimit {
    pub max_frames: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct PolypLabel {
    center_world: [f32; 3],
    bbox_px: Option<[f32; 4]>,
    bbox_norm: Option<[f32; 4]>,
}

#[derive(Serialize, Deserialize)]
struct CaptureMetadata {
    frame_id: u64,
    sim_time: f64,
    unix_time: f64,
    image: String,
    image_present: bool,
    camera_active: bool,
    polyp_seed: u64,
    polyp_labels: Vec<PolypLabel>,
}

#[derive(Serialize)]
struct RunManifest {
    schema_version: u32,
    seed: u64,
    output_root: PathBuf,
    run_dir: PathBuf,
    started_at_unix: f64,
    max_frames: Option<u32>,
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

pub fn recorder_toggle_hotkey(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    config: ResMut<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    polyp_meta: Res<crate::polyp::PolypSpawnMeta>,
    cap_limit: Res<CaptureLimit>,
) {
    if !keys.just_pressed(KeyCode::KeyL) {
        return;
    }
    let _now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    state.enabled = !state.enabled;
    state.last_toggle = time.elapsed_secs_f64();
    if state.enabled {
        if !state.initialized {
            init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);
        }
        state.paused = false;
        state.overlays_done = false;
    } else {
        state.paused = false;
        state.overlays_done = false;
    }
}

pub fn auto_start_recording(
    time: Res<Time>,
    auto: Res<AutoDrive>,
    pov: Res<PovState>,
    config: ResMut<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    mut motion: ResMut<RecorderMotion>,
    polyp_meta: Res<crate::polyp::PolypSpawnMeta>,
    cap_limit: Res<CaptureLimit>,
    head_q: Query<&GlobalTransform, With<crate::probe::ProbeHead>>,
) {
    if !auto.enabled || !pov.use_probe {
        motion.last_head_z = None;
        motion.cumulative_forward = 0.0;
        motion.started = false;
        return;
    }
    if state.enabled {
        return;
    }
    let Ok(head_tf) = head_q.single() else {
        return;
    };
    let z = head_tf.translation().z;
    if let Some(last) = motion.last_head_z {
        let dz = z - last;
        if dz > 0.0 {
            motion.cumulative_forward += dz;
        }
    }
    motion.last_head_z = Some(z);
    motion.started = motion.cumulative_forward >= 0.25;
    if !motion.started {
        return;
    }

    if !state.initialized {
        init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    state.overlays_done = false;
}

pub fn auto_stop_recording_on_cecum(
    cecum: Res<CecumState>,
    mut data_run: ResMut<crate::sim::autopilot::DataRun>,
    mut auto: ResMut<AutoDrive>,
    mut state: ResMut<RecorderState>,
    mut auto_timer: ResMut<AutoRecordTimer>,
    mut motion: ResMut<RecorderMotion>,
    _run_mode: Option<Res<RunMode>>,
) {
    if !state.enabled {
        return;
    }
    if !data_run.active {
        return;
    }
    if cecum.reached {
        if !state.overlays_done {
            generate_overlays(&state.session_dir);
            state.overlays_done = true;
        }
        state.enabled = false;
        auto_timer.timer.reset();
        state.paused = false;
        motion.last_head_z = None;
        motion.cumulative_forward = 0.0;
        motion.started = false;
        data_run.active = false;
        auto.enabled = false;
        // Let finalize_datagen_run handle overlays/pruning/exit in Datagen mode.
    }
}

fn generate_overlays(run_dir: &Path) {
    let labels_dir = run_dir.join(LABELS_DIR);
    let out_dir = run_dir.join(OVERLAYS_DIR);
    if fs::create_dir_all(&out_dir).is_err() {
        return;
    }

    for entry in fs::read_dir(&labels_dir).into_iter().flatten() {
        let Ok(path) = entry.map(|e| e.path()) else {
            continue;
        };
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(meta) = fs::read(&path).and_then(|bytes| {
            serde_json::from_slice::<CaptureMetadata>(&bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }) else {
            continue;
        };
        if !meta.image_present {
            continue;
        }
        let img_path = run_dir.join(&meta.image);
        if !img_path.exists() {
            continue;
        }
        let Ok(mut img) = image::open(&img_path).map(|im| im.into_rgba8()) else {
            continue;
        };
        for label in meta.polyp_labels.iter().filter_map(|l| l.bbox_px) {
            draw_rect(&mut img, label, Rgba([255, 64, 192, 255]), 2);
        }
        let filename = Path::new(&meta.image)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(meta.image);
        let _ = img.save(out_dir.join(filename));
    }
}

fn prune_run(input_run: &Path, output_root: &Path) -> std::io::Result<(usize, usize)> {
    let run_name = input_run
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "invalid run dir"))?;
    let out_run = output_root.join(run_name);
    fs::create_dir_all(out_run.join(LABELS_DIR))?;
    fs::create_dir_all(out_run.join(IMAGES_DIR))?;
    fs::create_dir_all(out_run.join(OVERLAYS_DIR))?;

    // Copy manifest if present.
    let manifest_in = input_run.join("run_manifest.json");
    if manifest_in.exists() {
        let manifest_out = out_run.join("run_manifest.json");
        let _ = fs::copy(&manifest_in, &manifest_out);
    }

    let labels_dir = input_run.join(LABELS_DIR);
    let mut kept = 0usize;
    let mut skipped = 0usize;
    for lbl in fs::read_dir(&labels_dir)? {
        let lbl = lbl?;
        if lbl.path().extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let raw = fs::read(lbl.path())?;
        let meta: CaptureMetadata = match serde_json::from_slice(&raw) {
            Ok(m) => m,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        if !meta.image_present || meta.polyp_labels.is_empty() {
            skipped += 1;
            continue;
        }
        kept += 1;
        // Copy label
        let out_label = out_run.join(LABELS_DIR).join(lbl.file_name());
        fs::write(&out_label, &raw)?;
        // Copy image
        let in_img = input_run.join(&meta.image);
        let out_img = out_run.join(&meta.image);
        if let Some(parent) = out_img.parent() {
            fs::create_dir_all(parent)?;
        }
        let _ = fs::copy(&in_img, &out_img);
        // Copy overlay if present
        if let Some(fname) = Path::new(&meta.image).file_name() {
            let overlay_in = input_run.join(OVERLAYS_DIR).join(fname);
            if overlay_in.exists() {
                let overlay_out = out_run.join(OVERLAYS_DIR).join(fname);
                let _ = fs::copy(&overlay_in, &overlay_out);
            }
        }
    }

    Ok((kept, skipped))
}

pub(crate) fn init_run_dirs(
    state: &mut RecorderState,
    config: &RecorderConfig,
    polyp_meta: &crate::polyp::PolypSpawnMeta,
    cap_limit: &CaptureLimit,
) {
    // Reset per-run flags before creating a new session directory so overlays,
    // manifests, and pruning run for every capture session.
    state.overlays_done = false;
    state.prune_done = false;
    state.manifest_written = false;

    let started_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let started_ms = (started_unix * 1000.0).round() as u128;
    let session = format!("run_{}", started_ms);
    state.session_dir = config.output_root.join(session);
    state.frame_idx = 0;
    let _ = fs::create_dir_all(&state.session_dir);
    let _ = fs::create_dir_all(state.session_dir.join(IMAGES_DIR));
    let _ = fs::create_dir_all(state.session_dir.join(LABELS_DIR));
    let _ = fs::create_dir_all(state.session_dir.join(OVERLAYS_DIR));
    if !state.manifest_written {
        let manifest = RunManifest {
            schema_version: 1,
            seed: polyp_meta.seed,
            output_root: config.output_root.clone(),
            run_dir: state.session_dir.clone(),
            started_at_unix: started_unix,
            max_frames: cap_limit.max_frames,
        };
        let manifest_path = state.session_dir.join("run_manifest.json");
        if let Ok(serialized) = serde_json::to_string_pretty(&manifest) {
            let _ = fs::write(manifest_path, serialized);
            state.manifest_written = true;
        }
    }
    state.initialized = true;
}

fn draw_rect(img: &mut RgbaImage, bbox: [f32; 4], color: Rgba<u8>, thickness: u32) {
    let (w, h) = img.dimensions();
    let clamp = |v: f32, max: u32| -> u32 { v.max(0.0).min((max as i32 - 1) as f32) as u32 };
    let x0 = clamp(bbox[0], w);
    let y0 = clamp(bbox[1], h);
    let x1 = clamp(bbox[2], w);
    let y1 = clamp(bbox[3], h);
    if x0 >= w || y0 >= h || x1 >= w || y1 >= h {
        return;
    }
    for t in 0..thickness {
        let xx0 = x0 + t;
        let yy0 = y0 + t;
        let xx1 = x1.saturating_sub(t);
        let yy1 = y1.saturating_sub(t);
        if xx0 >= w || yy0 >= h || xx1 >= w || yy1 >= h || xx0 > xx1 || yy0 > yy1 {
            continue;
        }
        for x in xx0..=xx1 {
            if yy0 < h {
                img.put_pixel(x, yy0, color);
            }
            if yy1 < h {
                img.put_pixel(x, yy1, color);
            }
        }
        for y in yy0..=yy1 {
            if xx0 < w {
                img.put_pixel(xx0, y, color);
            }
            if xx1 < w {
                img.put_pixel(xx1, y, color);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tempfile::tempdir;

    #[test]
    fn init_run_dirs_writes_manifest_and_dirs() {
        let dir = tempdir().unwrap();
        let output_root = dir.path().to_path_buf();

        let mut state = RecorderState::default();
        let config = RecorderConfig {
            output_root: output_root.clone(),
            ..Default::default()
        };
        let polyp_meta = crate::polyp::PolypSpawnMeta { seed: 999 };
        let cap_limit = CaptureLimit {
            max_frames: Some(5),
        };

        init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);

        assert!(state.initialized);
        assert!(state.session_dir.starts_with(&output_root));
        assert!(state.session_dir.join(IMAGES_DIR).is_dir());
        assert!(state.session_dir.join(LABELS_DIR).is_dir());
        assert!(state.session_dir.join(OVERLAYS_DIR).is_dir());

        let manifest_path = state.session_dir.join("run_manifest.json");
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["seed"], 999);
        assert_eq!(manifest["max_frames"], 5);
        assert!(manifest["started_at_unix"].as_f64().unwrap() > 0.0);
    }
}

pub fn record_front_camera_metadata(
    time: Res<Time>,
    mut config: ResMut<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    buffer: Res<FrontCameraFrameBuffer>,
    front_state: Res<FrontCameraState>,
    balloon: Res<BalloonControl>,
    removal: Res<PolypTelemetry>,
    cams: Query<(&Camera, &GlobalTransform), With<FrontCamera>>,
    capture_cams: Query<(&Camera, &GlobalTransform), With<FrontCaptureCamera>>,
    capture: Res<FrontCaptureTarget>,
    readback: Res<FrontCaptureReadback>,
    spawn_meta: Res<crate::polyp::PolypSpawnMeta>,
    polyp_telemetry: Res<PolypTelemetry>,
    polyps: Query<&GlobalTransform, With<crate::polyp::Polyp>>,
    cap_limit: Res<CaptureLimit>,
) {
    if !state.enabled {
        return;
    }
    // Pause capture while front balloon/vacuum is engaged or during polyp removal dwell.
    state.paused = balloon.head_inflated || removal.removing;
    if state.paused {
        return;
    }
    {
        let interval = &mut config.capture_interval;
        let mut delta = time.delta();
        if let Some(d) = polyp_telemetry.nearest_distance {
            if d <= 4.0 && d > 2.0 {
                delta *= 2;
            }
        }
        interval.tick(delta);
        if !interval.just_finished() {
            return;
        }
    }
    let Some(frame) = buffer.latest.as_ref() else {
        return;
    };
    // Prefer the capture camera (renders the PNGs) for projection to keep boxes aligned.
    let (cam, cam_tf, viewport) = if let Ok((cap_cam, cap_tf)) = capture_cams.single() {
        (
            cap_cam,
            cap_tf,
            Vec2::new(capture.size.x as f32, capture.size.y as f32),
        )
    } else if let Ok((cam, tf)) = cams.single() {
        let Some(vp) = cam.logical_viewport_size() else {
            return;
        };
        (cam, tf, vp)
    } else {
        return;
    };

    let right = cam_tf.right();
    let up = cam_tf.up();
    let bbox_radius = 0.28;

    let mut labels = Vec::new();
    for tf in polyps.iter() {
        let center = tf.translation();
        let to_polyp = center - cam_tf.translation();
        let forward = (cam_tf.rotation() * -Vec3::Z).normalize_or_zero();
        let depth = forward.dot(to_polyp);
        if depth <= 0.0 || depth > MAX_LABEL_DEPTH {
            continue;
        }
        let offsets = [
            Vec3::ZERO,
            right * bbox_radius,
            -right * bbox_radius,
            up * bbox_radius,
            -up * bbox_radius,
        ];
        let mut min = Vec2::new(f32::INFINITY, f32::INFINITY);
        let mut max = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut all_projected = true;
        for off in offsets {
            let world = center + off;
            if let Ok(p) = cam.world_to_viewport(cam_tf, world) {
                min = min.min(p);
                max = max.max(p);
            } else {
                all_projected = false;
                break;
            }
        }
        let bbox_px = if all_projected {
            Some([min.x, min.y, max.x, max.y])
        } else {
            None
        };
        let bbox_norm = bbox_px.map(|b| {
            [
                b[0] / viewport.x,
                b[1] / viewport.y,
                b[2] / viewport.x,
                b[3] / viewport.y,
            ]
        });
        labels.push(PolypLabel {
            center_world: [center.x, center.y, center.z],
            bbox_px,
            bbox_norm,
        });
    }

    let frame_image = readback.latest.clone();
    let file_labels: Vec<Label> = labels
        .iter()
        .map(|l| Label {
            center_world: l.center_world,
            bbox_px: l.bbox_px,
            bbox_norm: l.bbox_norm,
        })
        .collect();
    let frame = Frame {
        id: frame.id,
        timestamp: frame.captured_at,
        rgba: frame_image,
        size: (capture.size.x, capture.size.y),
        path: None,
    };
    let record = FrameRecord {
        frame,
        labels: &file_labels,
        camera_active: front_state.active,
        polyp_seed: spawn_meta.seed,
    };
    let mut recorder = DiskRecorder {
        state: &mut state,
        config: &config,
        polyp_labels: labels,
    };
    let _ = recorder.record(&record);

    if let Some(max) = cap_limit.max_frames {
        if state.frame_idx >= max as u64 {
            state.enabled = false;
            // Keep data_run.active true so finalize_datagen_run can cleanly exit and write overlays.
        }
    }
}

struct DiskRecorder<'a> {
    state: &'a mut RecorderState,
    config: &'a RecorderConfig,
    polyp_labels: Vec<PolypLabel>,
}

impl<'a> Recorder for DiskRecorder<'a> {
    fn record(&mut self, record: &FrameRecord) -> std::io::Result<()> {
        let image_name = format!("frame_{:05}.png", self.state.frame_idx);
        let images_dir = self.state.session_dir.join(IMAGES_DIR);
        fs::create_dir_all(&images_dir)?;
        let image_path = images_dir.join(&image_name);
        let mut image_present = false;
        if let Some(data) = record.frame.rgba.as_ref() {
            let expected_len = (self.config.resolution.x * self.config.resolution.y * 4) as usize;
            if data.len() == expected_len {
                image::save_buffer_with_format(
                    &image_path,
                    data,
                    self.config.resolution.x,
                    self.config.resolution.y,
                    image::ColorType::Rgba8,
                    ImageFormat::Png,
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                image_present = true;
                self.state.last_image_ok = true;
            } else {
                self.state.last_image_ok = false;
            }
        }

        let meta = CaptureMetadata {
            frame_id: record.frame.id,
            sim_time: record.frame.timestamp,
            unix_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            image: format!("{}/{}", IMAGES_DIR, image_name),
            image_present,
            camera_active: record.camera_active,
            polyp_seed: record.polyp_seed,
            polyp_labels: self.polyp_labels.clone(),
        };

        let out_dir = self.state.session_dir.join(LABELS_DIR);
        fs::create_dir_all(&out_dir)?;
        let meta_path = out_dir.join(format!("frame_{:05}.json", self.state.frame_idx));
        fs::write(meta_path, serde_json::to_string_pretty(&meta)?)?;
        self.state.frame_idx += 1;
        Ok(())
    }
}

pub fn finalize_datagen_run(
    mode: Res<RunMode>,
    config: Res<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    mut data_run: ResMut<crate::sim::autopilot::DataRun>,
    mut exit: MessageWriter<AppExit>,
) {
    if *mode != RunMode::Datagen {
        return;
    }
    if state.enabled || !state.initialized {
        return;
    }
    if !state.overlays_done && state.initialized {
        generate_overlays(&state.session_dir);
        state.overlays_done = true;
    }
    if config.prune_empty && !state.prune_done && state.initialized {
        let out_root = config
            .prune_output_root
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                let mut base = config.output_root.clone();
                let suffix = base
                    .file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| format!("{s}_filtered"))
                    .unwrap_or_else(|| "captures_filtered".to_string());
                base.set_file_name(suffix);
                base
            });
        match prune_run(&state.session_dir, &out_root) {
            Ok((kept, skipped)) => {
                state.prune_done = true;
                println!(
                    "Pruned run {} -> {} (kept {}, skipped {})",
                    state.session_dir.display(),
                    out_root.display(),
                    kept,
                    skipped
                );
            }
            Err(err) => {
                eprintln!(
                    "Prune failed for {} -> {}: {:?}",
                    state.session_dir.display(),
                    out_root.display(),
                    err
                );
            }
        }
    }
    data_run.active = false;
    exit.write(AppExit::Success);
}

pub fn datagen_failsafe_recording(
    time: Res<Time>,
    mode: Res<RunMode>,
    mut init: ResMut<crate::sim::autopilot::DatagenInit>,
    mut state: ResMut<RecorderState>,
    mut motion: ResMut<RecorderMotion>,
    config: ResMut<RecorderConfig>,
    polyp_meta: Res<crate::polyp::PolypSpawnMeta>,
    cap_limit: Res<CaptureLimit>,
    head_q: Query<&GlobalTransform, With<crate::probe::ProbeHead>>,
) {
    if *mode != RunMode::Datagen {
        return;
    }
    if !init.started || state.enabled {
        return;
    }
    let Ok(head_tf) = head_q.single() else {
        return;
    };
    let z = head_tf.translation().z;
    if let Some(last) = motion.last_head_z {
        let dz = z - last;
        if dz > 0.0 {
            motion.cumulative_forward += dz;
        }
    }
    motion.last_head_z = Some(z);

    init.elapsed += time.delta_secs();
    // Only start after forward motion begins; timer is just a guard against never starting.
    if motion.cumulative_forward < 0.25 {
        return;
    }

    if !state.initialized {
        init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    init.elapsed = 0.0;
}
// Adapters for the vision interfaces can be introduced here when swapping
// detectors/sources; left uninstantiated for now to avoid unused-code noise.
// (e.g., a CaptureSource wrapping FrontCameraFrameBuffer/FrontCaptureReadback,
// and a DiskRecorder wrapping RecorderState/Config).
