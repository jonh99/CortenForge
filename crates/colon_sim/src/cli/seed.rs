use bevy::prelude::Resource;
use std::time::{SystemTime, UNIX_EPOCH};

/// Stored run seed for deterministic layouts.
#[derive(Resource, Clone, Copy, Debug)]
pub struct SeedState {
    pub value: u64,
}

/// Resolve seed from CLI, then env (`POLYP_SEED`), else time.
pub fn resolve_seed(cli_seed: Option<u64>) -> u64 {
    if let Some(s) = cli_seed {
        return s;
    }
    if let Ok(env_seed) = std::env::var("POLYP_SEED") {
        if let Ok(parsed) = env_seed.parse::<u64>() {
            return parsed;
        }
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1)
}
