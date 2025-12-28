# ETL commands, envs, and debugging

## Command matrix (Bash / PowerShell)

- See `reference/commands.md` for Bash/PowerShell snippets.

## Env toggles
- `WAREHOUSE_CLEAR=1` — delete the computed version dir before writing.
- `WAREHOUSE_SKIP_IF_EXISTS=1` — exit early if the versioned manifest already exists.
- `CODE_VERSION` — embed the code commit hash (short SHA) into the manifest.
- Validation envs (fail/warn): e.g., `BURN_DATASET_MAX_INVALID=0`, `BURN_DATASET_MAX_MISSING=0`, `BURN_DATASET_MAX_EMPTY_RATIO=0.05`.

## Outputs
- Versioned directory under `artifacts/tensor_warehouse/v<version>/`.
- `manifest.json` with schema, transforms, checksums, and run summaries.
- `shard_*.bin` files containing contiguous tensors.

## Debug ETL failures (checklist)
- Verify input roots and globs resolve to non-empty datasets.
- Check env vars: are `WAREHOUSE_CLEAR`/`WAREHOUSE_SKIP_IF_EXISTS` set as intended?
- Confirm thresholds: invalid/missing/empty ratios—reduce strictness only if you understand the fallout.
- Inspect logs for the first failing sample; re-run on a tiny subset to isolate.
- Validate disk space and permissions on the output root.
- Regenerate `CODE_VERSION` if switching branches/commits mid-run.
