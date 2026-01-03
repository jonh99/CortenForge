use clap::Args;
use std::path::PathBuf;

/// Shared thresholds used by inference-related tools.
#[derive(Debug, Clone, Copy)]
pub struct ThresholdOpts {
    pub obj_thresh: f32,
    pub iou_thresh: f32,
}

impl ThresholdOpts {
    pub fn new(obj_thresh: f32, iou_thresh: f32) -> Self {
        Self {
            obj_thresh,
            iou_thresh,
        }
    }
}

/// Optional detector weights path.
#[derive(Debug, Clone)]
pub struct WeightsOpts {
    pub detector_weights: Option<PathBuf>,
}

impl WeightsOpts {
    pub fn new(detector_weights: Option<PathBuf>) -> Self {
        Self { detector_weights }
    }
}

/// Capture output/prune options shared across capture-related binaries.
#[derive(Debug, Clone, Args)]
pub struct CaptureOutputArgs {
    /// Directory to write captures into.
    #[arg(long, default_value = "assets/datasets/captures")]
    pub output_root: PathBuf,
    /// Optionally prune empty-label frames after datagen (writes filtered copy).
    #[arg(long, default_value_t = false)]
    pub prune_empty: bool,
    /// Optional output root for pruned runs (defaults to "<output_root>_filtered").
    #[arg(long)]
    pub prune_output_root: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct CaptureOutputOpts {
    pub output_root: PathBuf,
    pub prune_empty: bool,
    pub prune_output_root: Option<PathBuf>,
}

impl CaptureOutputOpts {
    pub fn new(
        output_root: PathBuf,
        prune_empty: bool,
        prune_output_root: Option<PathBuf>,
    ) -> Self {
        Self {
            output_root,
            prune_empty,
            prune_output_root,
        }
    }

    /// Resolve the destination root for pruned runs, defaulting to "<output_root>_filtered".
    pub fn resolve_prune_output_root(&self) -> PathBuf {
        if let Some(root) = &self.prune_output_root {
            return root.clone();
        }
        let mut base = self.output_root.clone();
        let suffix = base
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}_filtered"))
            .unwrap_or_else(|| "captures_filtered".to_string());
        base.set_file_name(suffix);
        base
    }
}

impl From<&CaptureOutputArgs> for CaptureOutputOpts {
    fn from(args: &CaptureOutputArgs) -> Self {
        CaptureOutputOpts::new(
            args.output_root.clone(),
            args.prune_empty,
            args.prune_output_root.clone(),
        )
    }
}

/// Warehouse output root shared across warehouse_* tooling.
#[derive(Debug, Clone, Args)]
pub struct WarehouseOutputArgs {
    /// Output root for warehouse artifacts.
    #[arg(long, default_value = "artifacts/tensor_warehouse")]
    pub output_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct WarehouseOutputOpts {
    pub output_root: PathBuf,
}

impl From<&WarehouseOutputArgs> for WarehouseOutputOpts {
    fn from(args: &WarehouseOutputArgs) -> Self {
        WarehouseOutputOpts {
            output_root: args.output_root.clone(),
        }
    }
}

/// Optional WGPU env hints for tooling; consumers can apply these to the environment or log them.
#[derive(Debug, Clone, Default)]
pub struct WgpuEnvHints {
    pub backend: Option<String>,
    pub adapter_name: Option<String>,
    pub power_pref: Option<String>,
    pub rust_log: Option<String>,
}

impl WgpuEnvHints {
    pub fn empty() -> Self {
        Self::default()
    }
}
