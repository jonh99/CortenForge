use bevy::ecs::system::RunSystemOnce;
use bevy::prelude::*;
use colon_sim_app::polyp::{Polyp, PolypRandom, PolypSpawnMeta, spawn_polyps};

fn run_polyp_layout(seed: u64) -> Vec<Vec3> {
    let mut app = App::new();

    // Minimal resources required by spawn_polyps.
    app.insert_resource(Assets::<Mesh>::default());
    app.insert_resource(Assets::<StandardMaterial>::default());
    app.insert_resource(PolypRandom::new(seed));
    app.insert_resource(PolypSpawnMeta { seed });

    // Run the spawn system once.
    let _ = app.world_mut().run_system_once(spawn_polyps);

    // Collect positions of spawned polyps.
    let mut positions: Vec<Vec3> = app
        .world_mut()
        .query_filtered::<&Transform, With<Polyp>>()
        .iter(&app.world())
        .map(|tf| tf.translation)
        .collect();

    // Order for deterministic comparison.
    positions.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    positions
}

#[test]
fn same_seed_produces_same_polyp_positions() {
    let a = run_polyp_layout(777);
    let b = run_polyp_layout(777);
    assert_eq!(a.len(), b.len(), "polyp counts should match");
    for (pa, pb) in a.iter().zip(b.iter()) {
        let delta = (*pa - *pb).length();
        assert!(
            delta < 1e-5,
            "positions should match for identical seeds (delta {delta})"
        );
    }
}
