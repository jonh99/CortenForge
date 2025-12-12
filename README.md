# Mission Impossibowel

This is a simplified demonstration of United States patent: [Pub. No.: US 2007/024.9906 A1](https://patentimages.storage.googleapis.com/6b/ce/a2/051225b0d2ff0a/US20070249906A1.pdf)

*you can legally **demonstrate or prototype** patented technologies in a **non-commercial, research or educational context**, including within a physics engine like Bevy + Rapier, as long as you don’t monetize, distribute, or sell the resulting product.*

![demo](media/Endotics_Probe_Demo1.gif)
![demo](media/Endotics_Probe_Demo2.gif)

Simple Bevy + Rapier sandbox with a soft, pneumatic “probe” navigating a snug tunnel.

## Controls
- Tail balloon on/off: `N` (anchors rear, collapses tunnel at tail)
- Head balloon on/off: `B` (anchors front, collapses tunnel at nose)
- Extend (pneumatic): hold Arrow Up / `I` while tail balloon is on and head balloon is off. Slow, capped at ~172% length.
- Retract/deflate: hold Arrow Down / `K` (works in any anchor state; with head balloon on, the rear slides forward).
- Right Mouse + Move: look around.
- W/A/S/D: strafe camera; Space/Shift: move camera up/down.


## Running
```bash
cargo run --release
```

## Notes
- Physics: Rapier 3D with gentle gravity and friction that ramps up where the tunnel is collapsed.
- Probe: elastic tube, stretch limited to ~172% of deflated length; tail/front anchoring via balloons.
- Tunnel: long, ring-based shell with localized contraction around balloons.
