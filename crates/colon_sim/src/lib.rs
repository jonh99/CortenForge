pub mod cli;
pub mod vision;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
pub use colon_sim_app::prelude::*;

const RAPIER_DEBUG_WIREFRAMES: bool = true;

use crate::cli::RunMode;
use crate::cli::seed::{SeedState, resolve_seed};
use inference::prelude::{InferenceFactory, InferenceThresholds as InferenceFactoryThresholds};
use sim_core::camera::PovState;
use sim_core::hooks::SimHooks;
use sim_core::recorder_meta::{
    BasicRecorderMeta, RecorderMetaProvider, RecorderSink, RecorderWorldState,
};
use sim_core::recorder_types::{AutoRecordTimer, RecorderConfig, RecorderMotion, RecorderState};
use sim_core::{ModeSet, SimConfig, SimPlugin, SimRunMode, build_app};
use vision::{
    BurnDetector, BurnInferenceState, DetectionOverlayState, DetectorHandle, DetectorKind,
    FrontCameraFrameBuffer, FrontCameraState, FrontCaptureReadback, InferencePlugin,
    InferenceThresholds, schedule_burn_inference,
};

pub fn run_app(args: crate::cli::AppArgs) {
    let polyp_seed = resolve_seed(args.seed);
    let headless = args.headless;
    let thresh_opts: crate::cli::common::ThresholdOpts = (&args).into();
    let infer_thresh = thresh_opts.to_inference_thresholds();
    let factory_thresh = InferenceFactoryThresholds {
        obj_thresh: infer_thresh.obj_thresh,
        iou_thresh: infer_thresh.iou_thresh,
    };
    let weights_opts: crate::cli::common::WeightsOpts = (&args).into();
    let weights_path = weights_opts.detector_weights.as_deref();
    let capture_opts: crate::cli::common::CaptureOutputOpts = (&args).into();
    let sim_mode = match args.mode {
        RunMode::Datagen => SimRunMode::Datagen,
        RunMode::Inference => SimRunMode::Inference,
        RunMode::Sim => SimRunMode::Sim,
    };

    let sim_config = SimConfig {
        mode: sim_mode,
        headless,
        capture_output_root: capture_opts.output_root.clone(),
        prune_empty: capture_opts.prune_empty,
        prune_output_root: capture_opts.prune_output_root.clone(),
        max_frames: args.max_frames,
        capture_interval_secs: None,
    };

    let mut app = build_app(sim_config.clone());

    if args.mode == RunMode::Inference {
        let factory = InferenceFactory;
        let detector = factory.build(factory_thresh, weights_path);
        app.insert_resource(DetectorHandle {
            detector,
            // InferenceFactory currently returns a heuristic detector; mark as such.
            kind: DetectorKind::Heuristic,
        });
    }

    app.insert_resource(SeedState { value: polyp_seed })
        .insert_resource(sim_mode)
        .insert_resource(RecorderMetaProvider {
            provider: Box::new(BasicRecorderMeta { seed: polyp_seed }),
        })
        .insert_resource(SimHooks {
            controls: Some(Box::new(colon_sim_app::controls::ControlsHookImpl)),
            autopilot: Some(Box::new(colon_sim_app::autopilot::AutopilotHookImpl)),
        })
        .insert_resource(AmbientLight {
            color: Color::srgb(1.0, 1.0, 1.0),
            brightness: 0.4,
            affects_lightmapped_meshes: true,
        })
        .insert_resource(PovState::default())
        .insert_resource(FrontCameraState::default())
        .insert_resource(FrontCameraFrameBuffer::default())
        .insert_resource(FrontCaptureReadback::default())
        .insert_resource(DetectionOverlayState::default())
        .insert_resource({
            let mut cfg = RecorderConfig {
                output_root: sim_config.capture_output_root.clone(),
                prune_empty: sim_config.prune_empty,
                prune_output_root: sim_config.prune_output_root.clone(),
                ..default()
            };
            if let Some(interval) = sim_config.capture_interval_secs {
                cfg.capture_interval = bevy::prelude::Timer::from_seconds(
                    interval,
                    bevy::prelude::TimerMode::Repeating,
                );
            }
            cfg
        })
        .insert_resource(RecorderState::default())
        .insert_resource(RecorderMotion::default())
        .insert_resource(RecorderSink::default())
        .insert_resource(RecorderWorldState::default())
        .insert_resource(vision::CaptureLimit {
            max_frames: sim_config.max_frames,
        })
        .insert_resource(AutoRecordTimer::default());

    insert_domain_resources(&mut app, polyp_seed);

    app.add_plugins(ConditionalRapierDebug)
        .add_plugins(SimPlugin)
        .add_plugins(sim_core::runtime::SimRuntimePlugin)
        .add_plugins(vision::CapturePlugin)
        .add_plugins(AppSystemsPlugin)
        .add_plugins(AppBootstrapPlugin);

    if args.mode == RunMode::Inference {
        app.insert_resource(BurnDetector::default())
            .insert_resource(BurnInferenceState::default())
            .insert_resource(InferenceThresholds {
                obj_thresh: args.infer_obj_thresh,
                iou_thresh: args.infer_iou_thresh,
            });

        app.add_plugins(InferencePlugin).add_systems(
            Update,
            (colon_sim_app::hud::update_detection_overlay_ui.after(schedule_burn_inference),)
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

struct AppBootstrapPlugin;
impl Plugin for AppBootstrapPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_environment, disable_gravity));
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
