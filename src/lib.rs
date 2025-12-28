pub mod autopilot;
pub mod balloon_control;
#[cfg(feature = "burn_runtime")]
pub mod burn_model;
pub mod camera;
pub mod cli;
pub mod common_cli;
pub mod controls;
pub mod hud;
pub mod polyp;
pub mod probe;
pub mod seed;
pub mod service;
pub mod tools;
#[cfg(feature = "burn_runtime")]
pub mod tools_postprocess {
    pub use crate::tools::postprocess::*;
}
pub mod tunnel;
pub mod vision;

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

const RAPIER_DEBUG_WIREFRAMES: bool = true;

use autopilot::{
    AutoDrive, DataRun, DatagenInit, auto_inchworm, auto_toggle, data_run_toggle, datagen_autostart,
};
use balloon_control::{
    BalloonControl, balloon_body_update, balloon_control_input, balloon_marker_update,
    spawn_balloon_body, spawn_balloon_marker,
};
use camera::{PovState, camera_controller, pov_toggle_system, setup_camera};
use controls::{ControlParams, control_inputs_and_apply};
use crate::cli::RunMode;
use hud::{spawn_controls_ui, spawn_detection_overlay, update_controls_ui};
use polyp::{
    PolypDetectionVotes, PolypRandom, PolypRemoval, PolypSpawnMeta, PolypTelemetry,
    apply_detection_votes, polyp_detection_system, polyp_removal_system, spawn_polyps,
};
use probe::{StretchState, TipSense, distributed_thrust, peristaltic_drive, spawn_probe};
use seed::{SeedState, resolve_seed};
use tunnel::{CecumState, cecum_detection, setup_tunnel, start_detection, tunnel_expansion_system};
use vision::{
    AutoRecordTimer, BurnDetector, BurnInferenceState, DetectionOverlayState, DefaultDetectorFactory,
    DetectorFactory, FrontCameraFrameBuffer, FrontCameraState, FrontCaptureReadback,
    InferenceThresholds, RecorderConfig, RecorderMotion, RecorderState, auto_start_recording,
    auto_stop_recording_on_cecum, capture_front_camera_frame, datagen_failsafe_recording,
    finalize_datagen_run, on_front_capture_readback, poll_burn_inference,
    record_front_camera_metadata, recorder_toggle_hotkey, schedule_burn_inference,
    setup_front_capture, threshold_hotkeys, track_front_camera_state,
};

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModeSet {
    Common,
    SimDatagen,
    Inference,
}

pub fn run_app(args: crate::cli::AppArgs) {
    let polyp_seed = resolve_seed(args.seed);
    let headless = args.headless;
    let thresh_opts: common_cli::ThresholdOpts = (&args).into();
    let infer_thresh = thresh_opts.to_inference_thresholds();
    let weights_opts: common_cli::WeightsOpts = (&args).into();
    let weights_path = weights_opts.detector_weights.as_deref();
    let mut app = App::new();

    if args.mode == RunMode::Inference {
        let factory = DefaultDetectorFactory;
        app.insert_resource(factory.build(infer_thresh, weights_path));
    }

    app
        .insert_resource(SeedState { value: polyp_seed })
        .insert_resource(args.mode)
        .insert_resource(AmbientLight {
            color: Color::srgb(1.0, 1.0, 1.0),
            brightness: 0.4,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(BalloonControl::default())
        .insert_resource(StretchState::default())
        .insert_resource(TipSense::default())
        .insert_resource(PolypSpawnMeta { seed: polyp_seed })
        .insert_resource(PolypRandom::new(polyp_seed))
        .insert_resource(PolypTelemetry::default())
        .insert_resource(PolypDetectionVotes::default())
        .insert_resource(PolypRemoval::default())
        .insert_resource(AutoDrive::default())
        .insert_resource(DataRun::default())
        .insert_resource(DatagenInit::default())
        .insert_resource(CecumState::default())
        .insert_resource(PovState::default())
        .insert_resource(FrontCameraState::default())
        .insert_resource(FrontCameraFrameBuffer::default())
        .insert_resource(FrontCaptureReadback::default())
        .insert_resource(BurnDetector::default())
        .insert_resource(BurnInferenceState::default())
        .insert_resource(InferenceThresholds {
            obj_thresh: args.infer_obj_thresh,
            iou_thresh: args.infer_iou_thresh,
        })
        .insert_resource(DetectionOverlayState::default())
        .insert_resource(RecorderConfig {
            output_root: args.output_root.clone(),
            prune_empty: args.prune_empty,
            prune_output_root: args.prune_output_root.clone(),
            ..default()
        })
        .insert_resource(RecorderState::default())
        .insert_resource(RecorderMotion::default())
        .insert_resource(vision::CaptureLimit {
            max_frames: args.max_frames,
        })
        .insert_resource(AutoRecordTimer::default())
        .insert_resource(ControlParams {
            tension: 0.5,
            stiffness: 500.0,
            damping: 20.0,
            thrust: 40.0,
            target_speed: 1.2,
            linear_damping: 0.2,
            friction: 1.2,
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                visible: !headless,
                fit_canvas_to_parent: true,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(ConditionalRapierDebug)
        .configure_sets(Update, (ModeSet::Common, ModeSet::SimDatagen, ModeSet::Inference))
        .add_systems(
            Startup,
            (
                setup_camera,
                setup_front_capture,
                spawn_environment,
                disable_gravity,
                setup_tunnel,
                spawn_probe,
                spawn_balloon_body,
                spawn_balloon_marker,
                spawn_polyps,
                spawn_controls_ui,
                spawn_detection_overlay,
            )
                .chain(),
        )
        .add_observer(on_front_capture_readback)
        .add_systems(
            Update,
            (
                balloon_control_input,
                balloon_body_update,
                camera_controller,
                pov_toggle_system,
                track_front_camera_state,
                capture_front_camera_frame.after(track_front_camera_state),
                apply_detection_votes
                    .after(polyp_detection_system)
                    .after(poll_burn_inference),
            )
                .in_set(ModeSet::Common),
        )
        .add_systems(
            Update,
            (
                datagen_autostart,
                data_run_toggle,
                auto_toggle,
                auto_inchworm,
                balloon_marker_update,
                recorder_toggle_hotkey,
                threshold_hotkeys,
                auto_start_recording,
                auto_stop_recording_on_cecum,
                finalize_datagen_run.after(auto_stop_recording_on_cecum),
                datagen_failsafe_recording,
                record_front_camera_metadata.after(capture_front_camera_frame),
                control_inputs_and_apply,
                update_controls_ui,
                cecum_detection,
                start_detection,
                tunnel_expansion_system,
                polyp_detection_system,
                polyp_removal_system
                    .after(polyp_detection_system)
                    .after(apply_detection_votes),
            )
                .in_set(ModeSet::SimDatagen),
        )
        .add_systems(
            FixedUpdate,
            (
                peristaltic_drive,
                distributed_thrust.before(PhysicsSet::SyncBackend),
            ),
        );

    if args.mode == RunMode::Inference {
        app.add_systems(
            Update,
            (
                schedule_burn_inference.after(capture_front_camera_frame),
                poll_burn_inference.after(schedule_burn_inference),
                threshold_hotkeys,
                hud::update_detection_overlay_ui.after(poll_burn_inference),
            )
                .in_set(ModeSet::Inference),
        );
    }

    app.run();
}

fn spawn_environment(mut commands: Commands) {
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 15_000.0,
            ..default()
        },
        Transform::from_xyz(5.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn disable_gravity(mut configs: Query<&mut RapierConfiguration, With<DefaultRapierContext>>) {
    for mut config in &mut configs {
        config.gravity = Vec3::new(0.0, -0.5, 0.0);
    }
}

struct ConditionalRapierDebug;
impl Plugin for ConditionalRapierDebug {
    fn build(&self, app: &mut App) {
        if RAPIER_DEBUG_WIREFRAMES {
            app.add_plugins(RapierDebugRenderPlugin::default());
        }
    }
}
