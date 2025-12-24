use std::env;
use std::fs;
use std::io;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};

use serde::Deserialize;
#[cfg(feature = "tui")]
use sysinfo::{Pid, ProcessRefreshKind, RefreshKind, System};
use thiserror::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct RunManifestSummary {
    pub schema_version: Option<u32>,
    pub seed: Option<u64>,
    pub output_root: Option<PathBuf>,
    pub run_dir: Option<PathBuf>,
    pub started_at_unix: Option<f64>,
    pub max_frames: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct RunInfo {
    pub path: PathBuf,
    pub manifest: Option<RunManifestSummary>,
    pub label_count: usize,
    pub image_count: usize,
    pub overlay_count: usize,
}

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// List run directories under a root, with manifest and counts if available.
pub fn list_runs(root: &Path) -> Result<Vec<RunInfo>, ServiceError> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name_ok = path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.starts_with("run_"))
            .unwrap_or(false);
        if !name_ok {
            continue;
        }
        let manifest_path = path.join("run_manifest.json");
        let manifest = if manifest_path.exists() {
            let data = fs::read(&manifest_path)?;
            serde_json::from_slice::<RunManifestSummary>(&data).ok()
        } else {
            None
        };
        let counts = count_artifacts(&path);
        out.push(RunInfo {
            path: path.clone(),
            manifest,
            label_count: counts.0,
            image_count: counts.1,
            overlay_count: counts.2,
        });
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

fn count_artifacts(run_dir: &Path) -> (usize, usize, usize) {
    let labels = run_dir.join("labels");
    let images = run_dir.join("images");
    let overlays = run_dir.join("overlays");
    let count_ext = |dir: &Path, ext: &str| -> usize {
        fs::read_dir(dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some(ext))
            .count()
    };
    (
        count_ext(&labels, "json"),
        count_ext(&images, "png"),
        count_ext(&overlays, "png"),
    )
}

#[derive(Debug, Clone)]
pub struct ServiceCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
}

pub fn spawn(cmd: &ServiceCommand) -> io::Result<Child> {
    Command::new(&cmd.program).args(&cmd.args).spawn()
}

fn bin_path(bin: &str) -> io::Result<PathBuf> {
    let mut exe = env::current_exe()?;
    exe.pop(); // drop current binary name
    exe.push(bin);
    Ok(exe)
}

#[derive(Debug, Clone)]
pub struct DatagenOptions {
    pub output_root: PathBuf,
    pub seed: Option<u64>,
    pub max_frames: Option<u32>,
    pub headless: bool,
    pub prune_empty: bool,
    pub prune_output_root: Option<PathBuf>,
}

/// Build a command to launch datagen (headless by default).
pub fn datagen_command(opts: &DatagenOptions) -> io::Result<ServiceCommand> {
    let bin = if opts.headless {
        bin_path("datagen_headless")?
    } else {
        bin_path("sim_view")?
    };
    let mut args = Vec::new();
    if !opts.headless {
        args.push("--mode".into());
        args.push("datagen".into());
    }
    if let Some(seed) = opts.seed {
        args.push("--seed".into());
        args.push(seed.to_string());
    }
    args.push("--output-root".into());
    args.push(opts.output_root.display().to_string());
    if let Some(max) = opts.max_frames {
        args.push("--max-frames".into());
        args.push(max.to_string());
    }
    if opts.headless && !opts.prune_empty && opts.prune_output_root.is_none() {
        // no-op
    }
    if opts.prune_empty {
        args.push("--prune-empty".into());
        if let Some(root) = &opts.prune_output_root {
            args.push("--prune-output-root".into());
            args.push(root.display().to_string());
        }
    }
    Ok(ServiceCommand { program: bin, args })
}

#[derive(Debug, Clone)]
pub struct TrainOptions {
    pub input_root: PathBuf,
    pub val_ratio: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub seed: Option<u64>,
    pub drop_last: bool,
    pub real_val_dir: Option<PathBuf>,
    pub status_file: Option<PathBuf>,
}

/// Build a command to launch the training binary with common options.
pub fn train_command(opts: &TrainOptions) -> io::Result<ServiceCommand> {
    let bin = bin_path("train")?;
    let mut args = Vec::new();
    args.push("--input-root".into());
    args.push(opts.input_root.display().to_string());
    args.push("--val-ratio".into());
    args.push(opts.val_ratio.to_string());
    args.push("--batch-size".into());
    args.push(opts.batch_size.to_string());
    args.push("--epochs".into());
    args.push(opts.epochs.to_string());
    if let Some(seed) = opts.seed {
        args.push("--seed".into());
        args.push(seed.to_string());
    }
    if opts.drop_last {
        args.push("--drop-last".into());
    }
    if let Some(val_dir) = &opts.real_val_dir {
        args.push("--real-val-dir".into());
        args.push(val_dir.display().to_string());
    }
    if let Some(status) = &opts.status_file {
        args.push("--status-file".into());
        args.push(status.display().to_string());
    }
    Ok(ServiceCommand { program: bin, args })
}

/// Read JSONL metrics (e.g., from `--metrics-out`) and return parsed entries.
pub fn read_metrics(
    path: &Path,
    limit: Option<usize>,
) -> Result<Vec<serde_json::Value>, ServiceError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut rows: Vec<serde_json::Value> = reader
        .lines()
        .filter_map(|line| line.ok())
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect();
    if let Some(n) = limit {
        if rows.len() > n {
            rows.drain(0..rows.len().saturating_sub(n));
        }
    }
    Ok(rows)
}

/// Read the last N lines of a text log.
pub fn read_log_tail(path: &Path, limit: usize) -> Result<Vec<String>, ServiceError> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
    if lines.len() > limit {
        lines.drain(0..lines.len().saturating_sub(limit));
    }
    Ok(lines)
}

#[cfg(feature = "tui")]
pub fn is_process_running(pid: u32) -> bool {
    let mut sys = System::new_with_specifics(
        RefreshKind::nothing().with_processes(ProcessRefreshKind::nothing()),
    );
    let pid = Pid::from_u32(pid);
    sys.refresh_process(pid)
}

#[cfg(feature = "tui")]
pub fn read_status(path: &Path) -> Option<serde_json::Value> {
    let data = std::fs::read(path).ok()?;
    serde_json::from_slice(&data).ok()
}
