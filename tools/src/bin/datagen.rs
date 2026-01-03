use clap::Parser;
use cli_support::common::CaptureOutputArgs;
use cli_support::seed::resolve_seed;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Headless datagen launcher (wrapper over sim_view)"
)]
struct Args {
    /// Seed for the run (defaults to randomized in resolve_seed).
    #[arg(long)]
    seed: Option<u64>,
    /// Override max frames for the capture run.
    #[arg(long)]
    max_frames: Option<u32>,
    #[command(flatten)]
    capture: CaptureOutputArgs,
    /// Run headless (default true for datagen).
    #[arg(long, default_value_t = true)]
    headless: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let seed = resolve_seed(args.seed);
    // Prefer the sibling sim_view binary in the same target dir as this tool.
    let sim_view_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .map(|dir| dir.join("sim_view"))
        .filter(|p| p.exists());

    let mut cmd = Command::new(
        sim_view_path
            .as_deref()
            .unwrap_or_else(|| std::path::Path::new("sim_view")),
    );
    cmd.arg("--mode")
        .arg("datagen")
        .arg("--seed")
        .arg(seed.to_string());
    if args.headless {
        cmd.arg("--headless");
    }
    if let Some(max) = args.max_frames {
        cmd.arg("--max-frames").arg(max.to_string());
    }
    cmd.arg("--output-root")
        .arg(args.capture.output_root.display().to_string());
    if args.capture.prune_empty {
        cmd.arg("--prune-empty");
    }
    if let Some(root) = &args.capture.prune_output_root {
        cmd.arg("--prune-output-root")
            .arg(root.display().to_string());
    }
    let status = cmd.status()?;
    if !status.success() {
        anyhow::bail!("sim_view datagen exited with status {:?}", status);
    }
    Ok(())
}
