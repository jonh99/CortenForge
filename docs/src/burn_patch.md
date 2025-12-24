# Burn patch note

## What’s patched
- Patched source: `vendor/burn-core-0.14.0/src/record/memory.rs` (swapped `decode_borrowed_from_slice` for `decode_from_slice` with error mapping).
- Cargo override: `[patch.crates-io] burn-core = { path = "vendor/burn-core-0.14.0" }`.
- Rationale: upstream `burn-core` calls `bincode::serde::decode_borrowed_from_slice`, which isn’t available in the published bincode versions we resolve here, breaking `--features burn_runtime` builds.
- Upstream PR filed: https://github.com/tracel-ai/burn/pull/4247

## Future cleanup
- Drop the patch when Burn publishes a version compatible with current bincode (or bincode restores the missing API).
- Remove `vendor/burn-core-0.14.0` and the `[patch.crates-io]` override in `Cargo.toml`.
- Re-run `cargo update` and verify `--features burn_runtime` builds without overrides.
