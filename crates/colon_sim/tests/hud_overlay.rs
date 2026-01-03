use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use colon_sim_app::hud::{
    DetectionBoxUI, FallbackBanner, spawn_detection_overlay, update_detection_overlay_ui,
};
use vision_runtime::DetectionOverlayState;

#[test]
fn overlay_spawns_boxes_and_toggles_fallback_banner() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.insert_resource(DetectionOverlayState {
        boxes: vec![[0.1, 0.2, 0.4, 0.5], [0.6, 0.1, 0.8, 0.3]],
        scores: vec![0.9, 0.7],
        size: (128, 128),
        fallback: None,
        inference_ms: Some(1.0),
    });

    // Spawn root and fallback banner nodes.
    app.world_mut()
        .run_system_once(spawn_detection_overlay)
        .expect("spawn_detection_overlay");

    // First pass: boxes rendered, banner hidden.
    app.world_mut()
        .run_system_once(update_detection_overlay_ui)
        .expect("update_detection_overlay_ui");

    let box_count = app
        .world_mut()
        .query::<&DetectionBoxUI>()
        .iter(app.world())
        .count();
    assert_eq!(box_count, 2);

    let mut banner_q = app.world_mut().query::<(&Node, &FallbackBanner)>();
    let Ok((node, _)) = banner_q.single(app.world_mut()) else {
        panic!("fallback banner not found");
    };
    assert_eq!(node.display, Display::None);

    // Second pass: turn on fallback message and ensure banner shows.
    {
        let mut overlay = app.world_mut().resource_mut::<DetectionOverlayState>();
        overlay.fallback = Some("Heuristic detector active".into());
    }
    app.world_mut()
        .run_system_once(update_detection_overlay_ui)
        .expect("update_detection_overlay_ui");

    let Ok((node, _)) = banner_q.single(app.world_mut()) else {
        panic!("fallback banner not found");
    };
    assert_eq!(node.display, Display::Flex);
}
