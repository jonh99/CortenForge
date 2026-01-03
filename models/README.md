# models

Burn-based model definitions for the CortenForge stack.

## Contents
- `TinyDet` / `TinyDetConfig`: small detector MLP.
- `BigDet` / `BigDetConfig`: configurable multibox MLP (depth/hidden/max_boxes/input_dim) with helper to clamp boxes to \[0,1\].
- `prelude`: re-export of configs and models.

## Features
- `tinydet` (default): includes TinyDet.
- `bigdet`: includes BigDet.

## License
Apache-2.0 (see `LICENSE` in the repo root).
