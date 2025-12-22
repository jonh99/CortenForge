use bevy::app::AppExit;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages};
use bevy::tasks::{AsyncComputeTaskPool, Task};
use futures_lite::future;
use image::{ImageFormat, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::polyp::PolypDetectionVotes;
use crate::balloon_control::BalloonControl;
use crate::polyp::PolypTelemetry;
use crate::autopilot::AutoDrive;
use crate::camera::PovState;
use crate::tunnel::CecumState;
use crate::cli::RunMode;

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

#[derive(Clone)]
pub struct BurnDetectionResult {
    pub frame_id: u64,
    pub positive: bool,
    pub confidence: f32,
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
pub struct RecorderConfig {
    pub output_root: PathBuf,
    pub capture_interval: Timer,
    pub resolution: UVec2,
}

const MAX_LABEL_DEPTH: f32 = 8.0;
const IMAGES_DIR: &str = "images";
const LABELS_DIR: &str = "labels";
const OVERLAYS_DIR: &str = "overlays";

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            output_root: PathBuf::from("assets/datasets/captures"),
            capture_interval: Timer::from_seconds(0.33, TimerMode::Repeating),
            resolution: UVec2::new(640, 360),
        }
    }
}

#[derive(Resource)]
pub struct RecorderState {
    pub enabled: bool,
    pub session_dir: PathBuf,
    pub frame_idx: u64,
    pub last_toggle: f64,
    pub last_image_ok: bool,
    pub paused: bool,
    pub overlays_done: bool,
    pub initialized: bool,
    pub manifest_written: bool,
}

impl Default for RecorderState {
    fn default() -> Self {
        Self {
            enabled: false,
            session_dir: PathBuf::from("assets/datasets/captures/unsynced"),
            frame_idx: 0,
            last_toggle: 0.0,
            last_image_ok: false,
            paused: false,
            overlays_done: false,
            initialized: false,
            manifest_written: false,
        }
    }
}

#[derive(Resource)]
pub struct AutoRecordTimer {
    pub timer: Timer,
}

impl Default for AutoRecordTimer {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(30.0, TimerMode::Once),
        }
    }
}

#[derive(Resource, Default)]
pub struct RecorderMotion {
    pub last_head_z: Option<f32>,
    pub cumulative_forward: f32,
    pub started: bool,
}

#[derive(Resource, Default)]
pub struct CaptureLimit {
    pub max_frames: Option<u32>,
}

#[derive(Serialize, Deserialize)]
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
        .spawn((Name::new("FrontCaptureTarget"), Readback::texture(handle.clone())))
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
    time: Res<Time>,
    mut detector: ResMut<BurnDetector>,
    mut jobs: ResMut<BurnInferenceState>,
    mut buffer: ResMut<FrontCameraFrameBuffer>,
) {
    jobs.debounce.tick(time.delta());
    if jobs.pending.is_some() || !jobs.debounce.is_finished() {
        return;
    }
    let Some(frame) = buffer.latest.take() else {
        return;
    };

    // Placeholder inference off the main thread; replace with real burn model.
    let task = AsyncComputeTaskPool::get().spawn(async move {
        let confidence = 0.8;
        let positive = true;
        BurnDetectionResult {
            frame_id: frame.id,
            positive,
            confidence,
        }
    });
    detector.model_loaded = true;
    jobs.pending = Some(task);
}

pub fn poll_burn_inference(
    mut jobs: ResMut<BurnInferenceState>,
    mut votes: ResMut<PolypDetectionVotes>,
) {
    if let Some(task) = jobs.pending.as_mut() {
        if let Some(result) = future::block_on(future::poll_once(task)) {
            votes.vision = result.positive;
            jobs.last_result = Some(result);
            jobs.pending = None;
        }
    } else if let Some(result) = jobs.last_result.as_ref() {
        votes.vision = result.positive;
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
            init_run_dirs(
                &mut state,
                &config,
                time.elapsed_secs_f64(),
                &polyp_meta,
                &cap_limit,
            );
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
        init_run_dirs(&mut state, &config, time.elapsed_secs_f64(), &polyp_meta, &cap_limit);
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    state.overlays_done = false;
}

pub fn auto_stop_recording_on_cecum(
    cecum: Res<CecumState>,
    mut data_run: ResMut<crate::autopilot::DataRun>,
    mut auto: ResMut<AutoDrive>,
    mut state: ResMut<RecorderState>,
    mut auto_timer: ResMut<AutoRecordTimer>,
    mut motion: ResMut<RecorderMotion>,
    run_mode: Option<Res<RunMode>>,
    mut exit: MessageWriter<AppExit>,
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
        state.frame_idx = 0;
        auto_timer.timer.reset();
        state.paused = false;
        motion.last_head_z = None;
        motion.cumulative_forward = 0.0;
        motion.started = false;
        state.initialized = false;
        state.manifest_written = false;
        data_run.active = false;
        auto.enabled = false;
        if run_mode.map_or(false, |m| *m == RunMode::Datagen) {
            exit.write(AppExit::Success);
        }
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
        let Ok(meta) = fs::read(&path).and_then(|bytes| serde_json::from_slice::<CaptureMetadata>(&bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))) else {
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

pub(crate) fn init_run_dirs(
    state: &mut RecorderState,
    config: &RecorderConfig,
    started_at: f64,
    polyp_meta: &crate::polyp::PolypSpawnMeta,
    cap_limit: &CaptureLimit,
) {
    let session = format!("run_{}", started_at as u64);
    let dir = config.output_root.join(session);
    state.session_dir = dir;
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
            started_at_unix: started_at,
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
    let clamp = |v: f32, max: u32| -> u32 {
        v.max(0.0).min((max as i32 - 1) as f32) as u32
    };
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
        (cap_cam, cap_tf, Vec2::new(capture.size.x as f32, capture.size.y as f32))
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

    let image_name = format!("frame_{:05}.png", state.frame_idx);
    let images_dir = state.session_dir.join(IMAGES_DIR);
    let _ = fs::create_dir_all(&images_dir);
    let image_path = images_dir.join(&image_name);
    let mut image_present = false;
    if let Some(data) = readback.latest.as_ref() {
        let expected_len = (capture.size.x * capture.size.y * 4) as usize;
        if data.len() == expected_len
            && image::save_buffer_with_format(
                &image_path,
                data,
                capture.size.x,
                capture.size.y,
                image::ColorType::Rgba8,
                ImageFormat::Png,
            )
            .is_ok()
        {
            image_present = true;
            state.last_image_ok = true;
        } else {
            state.last_image_ok = false;
        }
    }

    let meta = CaptureMetadata {
        frame_id: frame.id,
        sim_time: frame.captured_at,
        unix_time: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0),
        image: format!("{}/{}", IMAGES_DIR, image_name),
        image_present,
        camera_active: front_state.active,
        polyp_seed: spawn_meta.seed,
        polyp_labels: labels,
    };

    let out_dir = state.session_dir.join(LABELS_DIR);
    let _ = fs::create_dir_all(&out_dir);
    let meta_path = out_dir.join(format!("frame_{:05}.json", state.frame_idx));
    if let Ok(serialized) = serde_json::to_string_pretty(&meta) {
        let _ = fs::write(meta_path, serialized);
    }
    state.frame_idx += 1;

    if let Some(max) = cap_limit.max_frames {
        if state.frame_idx >= max as u64 {
            state.enabled = false;
            // Keep data_run.active true so finalize_datagen_run can cleanly exit and write overlays.
        }
    }
}

pub fn finalize_datagen_run(
    mode: Res<RunMode>,
    mut state: ResMut<RecorderState>,
    mut data_run: ResMut<crate::autopilot::DataRun>,
    mut exit: MessageWriter<AppExit>,
) {
    if *mode != RunMode::Datagen {
        return;
    }
    if !data_run.active {
        return;
    }
    if state.enabled || !state.initialized {
        return;
    }
    if !state.overlays_done && state.initialized {
        generate_overlays(&state.session_dir);
        state.overlays_done = true;
    }
    data_run.active = false;
    exit.write(AppExit::Success);
}

pub fn datagen_failsafe_recording(
    time: Res<Time>,
    mode: Res<RunMode>,
    mut init: ResMut<crate::autopilot::DatagenInit>,
    mut state: ResMut<RecorderState>,
    mut motion: ResMut<RecorderMotion>,
    config: ResMut<RecorderConfig>,
    polyp_meta: Res<crate::polyp::PolypSpawnMeta>,
    cap_limit: Res<CaptureLimit>,
) {
    if *mode != RunMode::Datagen {
        return;
    }
    if !init.started || state.enabled {
        return;
    }
    init.elapsed += time.delta_secs();
    if init.elapsed < 5.0 {
        return;
    }
    if !state.initialized {
        init_run_dirs(
            &mut state,
            &config,
            time.elapsed_secs_f64(),
            &polyp_meta,
            &cap_limit,
        );
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    init.elapsed = 0.0;
}
