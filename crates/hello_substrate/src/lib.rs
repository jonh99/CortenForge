use bevy::prelude::*;
use sim_core::ModeSet;

/// Minimal app plugin showing how to register systems against the substrate.
pub struct HelloAppPlugin;

impl Plugin for HelloAppPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, tick.in_set(ModeSet::Common));
    }
}

fn setup() {
    info!("hello_substrate: startup");
}

fn tick(time: Res<Time>) {
    // Simple heartbeat to prove the app is running.
    let _ = time.delta_secs();
}
