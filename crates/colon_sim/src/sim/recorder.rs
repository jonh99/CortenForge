use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::app::AppExit;
use bevy::prelude::{
    ButtonInput, Camera, GlobalTransform, KeyCode, MessageWriter, Query, Res, ResMut, Time, Transform,
    With,
};
use data_contracts::RunManifestSchemaVersion;
use serde_json;
use capture_utils::{generate_overlays, prune_run};
use sim_core::autopilot_types::{AutoDrive, DataRun, DatagenInit};
use sim_core::camera::PovState;
use sim_core::recorder_types::{AutoRecordTimer, RecorderConfig, RecorderMotion, RecorderState};
use sim_core::recorder_meta::{RecorderMetaProvider, RecorderSink, RecorderWorldState};
use sim_core::SimRunMode;
use vision_core::prelude::{
    CaptureLimit, Frame, FrameRecord, FrontCaptureCamera, FrontCaptureTarget,
};
use vision_runtime::{FrontCameraFrame, FrontCameraFrameBuffer, FrontCameraState};

use colon_sim_app::prelude::{CecumState, ProbeHead, PolypTelemetry};

const IMAGES_DIR: &str = "images";
const LABELS_DIR: &str = "labels";
const OVERLAYS_DIR: &str = "overlays";

type SimRunManifest = data_contracts::RunManifest;

pub fn recorder_init_run_dirs(
    state: &mut RecorderState,
    config: &RecorderConfig,
    meta: &RecorderMetaProvider,
    cap_limit: &CaptureLimit,
    sink: &mut RecorderSink,
) {
    state.overlays_done = false;
    state.prune_done = false;
    state.manifest_written = false;

    let started_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let started_ms = (started_unix * 1000.0).round() as u128;
    let session = format!("run_{started_ms}");
    state.session_dir = config.output_root.join(session);
    state.frame_idx = 0;
    let _ = fs::create_dir_all(&state.session_dir);
    let _ = fs::create_dir_all(state.session_dir.join(IMAGES_DIR));
    let _ = fs::create_dir_all(state.session_dir.join(LABELS_DIR));
    let _ = fs::create_dir_all(state.session_dir.join(OVERLAYS_DIR));
    if !state.manifest_written {
        let manifest = SimRunManifest {
            schema_version: RunManifestSchemaVersion::V1,
            seed: Some(meta.provider.polyp_seed()),
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
    if sink.writer.is_none() {
        sink.writer = Some(Box::new(capture_utils::JsonRecorder {
            run_dir: state.session_dir.clone(),
        }));
    }
    state.initialized = true;
}

pub fn recorder_generate_overlays(run_dir: &Path) {
    let _ = generate_overlays(run_dir);
}

pub(crate) fn recorder_prune_run(
    input_run: &Path,
    output_root: &Path,
) -> std::io::Result<(usize, usize)> {
    prune_run(input_run, output_root)
}

pub fn recorder_toggle_hotkey(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    config: Res<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    meta: Res<RecorderMetaProvider>,
    cap_limit: Res<CaptureLimit>,
    mut sink: ResMut<RecorderSink>,
) {
    if !keys.just_pressed(KeyCode::KeyL) {
        return;
    }
    state.enabled = !state.enabled;
    state.last_toggle = time.elapsed_secs_f64();
    if state.enabled {
        if !state.initialized {
            recorder_init_run_dirs(&mut state, &config, &meta, &cap_limit, &mut sink);
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
    meta: Res<RecorderMetaProvider>,
    world_state: Res<RecorderWorldState>,
    cap_limit: Res<CaptureLimit>,
    mut sink: ResMut<RecorderSink>,
    _run_mode: Option<Res<SimRunMode>>,
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
    let Some(head_z) = world_state.head_z else {
        return;
    };
    if let Some(last) = motion.last_head_z {
        let dz = head_z - last;
        if dz > 0.0 {
            motion.cumulative_forward += dz;
        }
    }
    motion.last_head_z = Some(head_z);
    motion.started = motion.cumulative_forward >= 0.25;
    if !motion.started {
        return;
    }

    if !state.initialized {
        recorder_init_run_dirs(&mut state, &config, &meta, &cap_limit, &mut sink);
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    state.overlays_done = false;
}

pub fn auto_stop_recording_on_cecum(
    world_state: Res<RecorderWorldState>,
    mut data_run: ResMut<DataRun>,
    mut auto: ResMut<AutoDrive>,
    mut state: ResMut<RecorderState>,
    mut auto_timer: ResMut<AutoRecordTimer>,
    mut motion: ResMut<RecorderMotion>,
    _run_mode: Option<Res<SimRunMode>>,
) {
    if !state.enabled || !data_run.active {
        return;
    }
    if world_state.stop_flag {
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
    }
}

pub fn finalize_datagen_run(
    mode: Res<SimRunMode>,
    config: Res<RecorderConfig>,
    mut state: ResMut<RecorderState>,
    mut data_run: ResMut<DataRun>,
    mut exit: MessageWriter<AppExit>,
) {
    if *mode != SimRunMode::Datagen {
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
    mode: Res<SimRunMode>,
    mut init: ResMut<DatagenInit>,
    mut state: ResMut<RecorderState>,
    mut motion: ResMut<RecorderMotion>,
    config: Res<RecorderConfig>,
    meta: Res<RecorderMetaProvider>,
    world_state: Res<RecorderWorldState>,
    cap_limit: Res<CaptureLimit>,
    mut sink: ResMut<RecorderSink>,
) {
    if *mode != SimRunMode::Datagen || !init.started || state.enabled {
        return;
    }
    let Some(head_z) = world_state.head_z else {
        return;
    };
    if let Some(last) = motion.last_head_z {
        let dz = head_z - last;
        if dz > 0.0 {
            motion.cumulative_forward += dz;
        }
    }
    motion.last_head_z = Some(head_z);

    init.elapsed += time.delta_secs();
    if motion.cumulative_forward < 0.25 {
        return;
    }

    if !state.initialized {
        recorder_init_run_dirs(&mut state, &config, &meta, &cap_limit, &mut sink);
    }
    state.enabled = true;
    state.last_toggle = time.elapsed_secs_f64();
    state.paused = false;
    motion.started = true;
    init.elapsed = 0.0;
}

pub fn record_front_camera_metadata(
    time: Res<Time>,
    camera: Query<(&Camera, &GlobalTransform), With<FrontCaptureCamera>>,
    target: Res<FrontCaptureTarget>,
    mut sink: ResMut<RecorderSink>,
    mut state: ResMut<RecorderState>,
    _motion: Res<RecorderMotion>,
    meta: Res<RecorderMetaProvider>,
    _polyp_telemetry: Res<PolypTelemetry>,
    _pov: Res<PovState>,
    _auto: Res<AutoDrive>,
    _cecum: Res<CecumState>,
    cap_limit: Res<CaptureLimit>,
    probes: Query<(&Transform, &ProbeHead), With<ProbeHead>>,
    mut buffer: ResMut<FrontCameraFrameBuffer>,
    cap_state: ResMut<FrontCameraState>,
) {
    if !state.enabled || state.paused {
        return;
    }

    if let Some(limit) = cap_limit.max_frames {
        if state.frame_idx >= limit as u64 {
            return;
        }
    }

    // Early exit if no camera.
    let Ok((cam, cam_tf)) = camera.get(target.entity) else {
        return;
    };
    if !cam.is_active {
        return;
    }

    // Push metadata.
    let Some((_probe_tf, _)) = probes.get(target.entity).ok() else {
        // No probe for this camera; skip.
        return;
    };

    let frame_id = state.frame_idx + 1;
    let now = time.elapsed_secs_f64();
    let metadata = FrameRecord {
        frame: Frame {
            id: frame_id,
            timestamp: now,
            rgba: None,
            size: (target.size.x, target.size.y),
            path: Some(Path::new(IMAGES_DIR).join(format!("frame_{frame_id:05}.png"))),
        },
        labels: &[],
        camera_active: cap_state.active,
        polyp_seed: meta.provider.polyp_seed(),
    };
    if let Some(writer) = sink.writer.as_mut() {
        let _ = writer.record(&metadata).ok();
    }

    // Track buffer for inference. Capture plugin already populates Readback;
    // here we only tag frame id/transform/time.
    buffer.latest = Some(FrontCameraFrame {
        id: frame_id,
        transform: cam_tf.compute_transform().into(),
        captured_at: now,
    });

    state.frame_idx += 1;
}
