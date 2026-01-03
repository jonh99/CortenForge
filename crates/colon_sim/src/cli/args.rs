use std::path::PathBuf;

use bevy::prelude::Resource;
use clap::{Parser, ValueEnum};

/// Command line configuration for the simulator / data generator.
#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about,
    long_about = "Deep Poo simulator / data generator — licensed under Apache-2.0. Source: https://github.com/via-balaena/Deep_Poo"
)]
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
    /// Burn inference objectness threshold (runtime).
    #[arg(long, default_value_t = 0.3)]
    pub infer_obj_thresh: f32,
    /// Burn inference IoU threshold for NMS (runtime).
    #[arg(long, default_value_t = 0.5)]
    pub infer_iou_thresh: f32,
    /// Optionally prune empty-label frames after datagen (writes filtered copy).
    #[arg(long, default_value_t = false)]
    pub prune_empty: bool,
    /// Optional output root for pruned runs (defaults to "<output_root>_filtered").
    #[arg(long)]
    pub prune_output_root: Option<PathBuf>,
    /// Optional path to detector weights (Burn checkpoint).
    #[arg(long)]
    pub detector_weights: Option<PathBuf>,
}

#[derive(Resource, ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Sim,
    Datagen,
    Inference,
}
