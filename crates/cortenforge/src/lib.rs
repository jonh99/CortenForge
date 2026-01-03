//! CortenForge umbrella crate: re-export app-agnostic crates with feature wiring.

#[cfg(feature = "sim-core")]
pub use sim_core;

#[cfg(feature = "vision-core")]
pub use vision_core;

#[cfg(feature = "vision-runtime")]
pub use vision_runtime;

#[cfg(feature = "data-contracts")]
pub use data_contracts;

#[cfg(feature = "capture-utils")]
pub use capture_utils;

#[cfg(feature = "models")]
pub use models;

#[cfg(feature = "inference")]
pub use inference;

#[cfg(feature = "training")]
pub use training;

#[cfg(feature = "burn-dataset")]
pub use burn_dataset;

#[cfg(feature = "cli-support")]
pub use cli_support;
