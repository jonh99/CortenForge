use std::path::PathBuf;

use bevy::math::UVec2;
use bevy::app::AppExit;
use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, KeyCode, MessageWriter, Query, Res, ResMut, Resource,
    Time, Timer, TimerMode, Vec2, Vec3, With,
};
use image::{ColorType, ImageFormat, Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::camera::PovState;
use crate::cli::RunMode;
use crate::polyp::PolypSpawnMeta;
use crate::polyp::PolypTelemetry;
use crate::probe::ProbeHead;
use crate::sim::autopilot::AutoDrive;
use crate::vision::interfaces::{Frame, FrameRecord, Label, Recorder};
use crate::vision::{
    FrontCamera, FrontCameraFrameBuffer, FrontCameraState, FrontCaptureCamera,
    FrontCaptureReadback, FrontCaptureTarget,
};
use crate::tunnel::CecumState;
use crate::vision::CaptureLimit;
use crate::BalloonControl;

#[derive(Resource)]
pub struct RecorderConfig {
    pub output_root: PathBuf,
    pub capture_interval: Timer,
    pub resolution: UVec2,
    pub prune_empty: bool,
    pub prune_output_root: Option<PathBuf>,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            output_root: PathBuf::from("assets/datasets/captures"),
            capture_interval: Timer::from_seconds(0.33, TimerMode::Repeating),
            resolution: UVec2::new(640, 360),
            prune_empty: false,
            prune_output_root: None,
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
    pub prune_done: bool,
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
            prune_done: false,
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

// Temporary duplicated helpers to stage recorder migration; call sites still use the
// originals in vision_core. These versions allow us to move system bodies later
// without breaking the build while both copies coexist.
const IMAGES_DIR: &str = "images";
const LABELS_DIR: &str = "labels";
const OVERLAYS_DIR: &str = "overlays";

#[derive(Serialize)]
struct SimRunManifest {
    schema_version: u32,
    seed: u64,
    output_root: PathBuf,
    run_dir: PathBuf,
    started_at_unix: f64,
    max_frames: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone)]
struct SimPolypLabel {
    center_world: [f32; 3],
    bbox_px: Option<[f32; 4]>,
    bbox_norm: Option<[f32; 4]>,
}

#[derive(Serialize, Deserialize)]
struct SimCaptureMetadata {
    frame_id: u64,
    sim_time: f64,
    unix_time: f64,
    image: String,
    image_present: bool,
    camera_active: bool,
    polyp_seed: u64,
    polyp_labels: Vec<SimPolypLabel>,
}

#[allow(dead_code)]
pub(crate) fn recorder_init_run_dirs(
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
        let manifest = SimRunManifest {
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

#[allow(dead_code)]
pub(crate) fn recorder_generate_overlays(run_dir: &Path) {
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
            serde_json::from_slice::<SimCaptureMetadata>(&bytes)
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
            recorder_draw_rect(&mut img, label, Rgba([255, 64, 192, 255]), 2);
        }
        let filename = Path::new(&meta.image)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(meta.image);
        let _ = img.save(out_dir.join(filename));
    }
}

#[allow(dead_code)]
pub(crate) fn recorder_prune_run(
    input_run: &Path,
    output_root: &Path,
) -> std::io::Result<(usize, usize)> {
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
        let meta: SimCaptureMetadata = match serde_json::from_slice(&raw) {
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

#[allow(dead_code)]
pub(crate) fn recorder_draw_rect(
    img: &mut RgbaImage,
    bbox: [f32; 4],
    color: Rgba<u8>,
    thickness: u32,
) {
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

pub fn recorder_toggle_hotkey(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    polyp_meta: Res<PolypSpawnMeta>,
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
            recorder_init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);
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
    config: Res<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    mut motion: ResMut<RecorderMotion>,
    polyp_meta: Res<PolypSpawnMeta>,
    cap_limit: Res<CaptureLimit>,
    head_q: Query<&GlobalTransform, With<ProbeHead>>,
    _run_mode: Option<Res<RunMode>>,
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
        recorder_init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);
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
            recorder_generate_overlays(&state.session_dir);
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
        recorder_generate_overlays(&state.session_dir);
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
        match recorder_prune_run(&state.session_dir, &out_root) {
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
    config: Res<RecorderConfig>,
    polyp_meta: Res<crate::polyp::PolypSpawnMeta>,
    cap_limit: Res<CaptureLimit>,
    head_q: Query<&GlobalTransform, With<ProbeHead>>,
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
        recorder_init_run_dirs(&mut state, &config, &polyp_meta, &cap_limit);
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    init.elapsed = 0.0;
}

const RECORDER_MAX_LABEL_DEPTH: f32 = 8.0;

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
    spawn_meta: Res<PolypSpawnMeta>,
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
        if depth <= 0.0 || depth > RECORDER_MAX_LABEL_DEPTH {
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
        labels.push(SimPolypLabel {
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
    polyp_labels: Vec<SimPolypLabel>,
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
                    ColorType::Rgba8,
                    ImageFormat::Png,
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                image_present = true;
                self.state.last_image_ok = true;
            } else {
                self.state.last_image_ok = false;
            }
        }

        let meta = SimCaptureMetadata {
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
