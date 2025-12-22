use std::path::PathBuf;

use bevy::prelude::Resource;
use clap::{Parser, ValueEnum};

/// Command line configuration for the simulator / data generator.
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct AppArgs {
    /// Run mode: interactive sim or headless data generation.
    #[arg(long, value_enum, default_value_t = RunMode::Sim)]
    pub mode: RunMode,
    /// Optional seed override for polyp layout / reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Directory to write captures into.
    #[arg(long, default_value = "assets/datasets/captures")]
    pub output_root: PathBuf,
    /// Optional frame cap for data runs (stops recording after this many frames).
    #[arg(long)]
    pub max_frames: Option<u32>,
    /// Hide the main window (offscreen/headless).
    #[arg(long, default_value_t = false)]
    pub headless: bool,
}

#[derive(Resource, ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Sim,
    Datagen,
}
