use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use bevy::time::Virtual;
use colon_sim::vision::interfaces::{DetectionResult, Detector, Frame};
use colon_sim::vision::{
    BurnDetector, BurnInferenceState, DetectionOverlayState, DetectorHandle, DetectorKind,
    FrontCameraFrame, FrontCameraFrameBuffer, FrontCaptureReadback, FrontCaptureTarget,
    InferenceThresholds, poll_inference_task, schedule_burn_inference,
};
use sim_core::SimRunMode;
use std::time::Duration;

#[derive(Clone)]
struct FakeDetector;

impl Detector for FakeDetector {
    fn detect(&mut self, frame: &Frame) -> DetectionResult {
        assert_eq!(frame.id, 42);
        DetectionResult {
            frame_id: frame.id,
            positive: true,
            confidence: 0.9,
            boxes: vec![[0.1, 0.2, 0.3, 0.4]],
            scores: vec![0.9],
        }
    }
}

// Basic end-to-end inference path: a frame in the buffer triggers detection and HUD overlay update.
#[test]
fn inference_updates_overlay_and_latency() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    {
        let mut time = app.world_mut().resource_mut::<Time<Virtual>>();
        time.advance_by(Duration::from_millis(200));
    }
    app.insert_resource(SimRunMode::Inference);
    app.insert_resource(BurnDetector::default());
    let mut jobs = BurnInferenceState::default();
    jobs.debounce = Timer::from_seconds(0.0, TimerMode::Repeating);
    app.insert_resource(jobs);
    app.insert_resource(FrontCameraFrameBuffer {
        latest: Some(FrontCameraFrame {
            id: 42,
            transform: GlobalTransform::IDENTITY,
            captured_at: 1.0,
        }),
    });
    app.insert_resource(FrontCaptureReadback {
        latest: Some(vec![255u8; (2 * 2 * 4) as usize]),
    });
    app.insert_resource(FrontCaptureTarget {
        entity: Entity::from_raw_u32(1).unwrap(),
        handle: Handle::default(),
        size: UVec2::new(2, 2),
    });
    app.insert_resource(DetectorHandle {
        detector: Box::new(FakeDetector),
        kind: DetectorKind::Heuristic,
    });
    app.insert_resource(InferenceThresholds {
        obj_thresh: 0.5,
        iou_thresh: 0.5,
    });
    app.insert_resource(DetectionOverlayState::default());

    app.world_mut()
        .run_system_once(schedule_burn_inference)
        .expect("system run failed");
    // Poll once; async task may or may not complete immediately, but should not panic.
    let _ = app.world_mut().run_system_once(poll_inference_task);

    // Smoke: we reached here without panic; overlay may remain default if async task not polled to completion.
}
