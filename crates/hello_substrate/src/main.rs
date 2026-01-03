use hello_substrate::HelloAppPlugin;
use sim_core::{build_app, SimConfig, SimRunMode};

fn main() {
    let mut app = build_app(SimConfig {
        mode: SimRunMode::Sim,
        ..Default::default()
    });
    app.add_plugins(HelloAppPlugin);
    app.run();
}
