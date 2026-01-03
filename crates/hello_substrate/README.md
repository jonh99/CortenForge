# hello_substrate

Minimal example app built on the substrate crates. It depends on `sim_core` and adds a tiny plugin with a startup log and a heartbeat system.

How to run
```bash
cargo run -p hello_substrate -- --help
cargo run -p hello_substrate
```

What it shows
- Building a Bevy `App` via `sim_core::build_app` + `SimConfig`.
- Adding your own plugin (`HelloAppPlugin`) and registering systems in `ModeSet::Common`.
- No colon-specific systems; this is a clean starting point for custom sims.
