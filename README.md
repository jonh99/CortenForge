# Mission Impossibowel

![demo](media/ball_jointed_probe.gif)

Simple Bevy + Rapier sandbox with a capsule “probe” navigating a snug tunnel. Arrow keys (Up/Down) or `I/K` apply thrust; right mouse + WASD/space/shift moves the fly camera. A small gravity keeps the probe settled.

## Controls
- Arrow Up / `I`: thrust forward (along +Z).
- Arrow Down / `K`: thrust backward.
- Right Mouse + Move: look around.
- W/A/S/D: strafe camera; Space/Shift: move camera up/down.


## Running
```bash
cargo run --release
```

## Notes
- Physics: Rapier 3D with gentle gravity and high friction for a tight fit.
- Assets: currently uses a generated capsule mesh (no GLB import (yet)).
