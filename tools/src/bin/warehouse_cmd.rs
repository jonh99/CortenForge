//! Unified warehouse command generator CLI.

use clap::{Parser, Subcommand, ValueEnum};
use colon_sim_tools::warehouse_commands::{
    builder::{build_command, Shell},
    common::CmdConfig,
    common::{ModelKind, WarehouseStore},
};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ShellArg {
    Ps,
    Sh,
}

impl From<ShellArg> for Shell {
    fn from(value: ShellArg) -> Self {
        match value {
            ShellArg::Ps => Shell::PowerShell,
            ShellArg::Sh => Shell::Bash,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum StoreArg {
    Memory,
    Mmap,
    Stream,
}

impl From<StoreArg> for WarehouseStore {
    fn from(value: StoreArg) -> Self {
        match value {
            StoreArg::Memory => WarehouseStore::Memory,
            StoreArg::Mmap => WarehouseStore::Mmap,
            StoreArg::Stream => WarehouseStore::Stream,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModelArg {
    Tiny,
    Big,
}

impl From<ModelArg> for ModelKind {
    fn from(value: ModelArg) -> Self {
        match value {
            ModelArg::Tiny => ModelKind::Tiny,
            ModelArg::Big => ModelKind::Big,
        }
    }
}

#[derive(Subcommand, Debug)]
enum Preset {
    /// AMD + PowerShell (DX12)
    AmdPs,
    /// AMD + Bash (Vulkan)
    AmdSh,
    /// NVIDIA + PowerShell (DX12)
    NvidiaPs,
    /// NVIDIA + Bash (Vulkan)
    NvidiaSh,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Generate warehouse one-liner commands")]
struct Cli {
    #[command(flatten)]
    output: cli_support::common::WarehouseOutputArgs,

    #[arg(
        long,
        value_enum,
        default_value = "sh",
        help = "Shell to target (ps|sh)"
    )]
    shell: ShellArg,

    #[arg(long, help = "GPU adapter name (e.g., AMD, NVIDIA)")]
    adapter: Option<String>,

    #[arg(long, help = "WGPU backend (dx12, vulkan, metal, etc.)")]
    backend: Option<String>,

    #[arg(long, help = "Path to manifest JSON")]
    manifest: Option<PathBuf>,

    #[arg(long, value_enum, help = "Warehouse store mode")]
    store: Option<StoreArg>,

    #[arg(long, help = "Prefetch depth (used when store=stream)")]
    prefetch: Option<usize>,

    #[arg(long, help = "Batch size")]
    batch_size: Option<usize>,

    #[arg(long, help = "Log every N batches")]
    log_every: Option<usize>,

    #[arg(long, value_enum, help = "Model size")]
    model: Option<ModelArg>,

    #[arg(
        long,
        default_value = "",
        help = "Extra args appended to cargo command"
    )]
    extra_args: String,

    #[command(subcommand)]
    preset: Option<Preset>,
}

fn apply_preset(cli: &Cli, cfg: &mut CmdConfig<'_>, shell: &mut Shell) {
    if let Some(preset) = &cli.preset {
        match preset {
            Preset::AmdPs => {
                *shell = Shell::PowerShell;
                cfg.wgpu_backend = "dx12".into();
                cfg.wgpu_adapter = Some("AMD".into());
            }
            Preset::AmdSh => {
                *shell = Shell::Bash;
                cfg.wgpu_backend = "vulkan".into();
                cfg.wgpu_adapter = Some("AMD".into());
            }
            Preset::NvidiaPs => {
                *shell = Shell::PowerShell;
                cfg.wgpu_backend = "dx12".into();
                cfg.wgpu_adapter = Some("NVIDIA".into());
            }
            Preset::NvidiaSh => {
                *shell = Shell::Bash;
                cfg.wgpu_backend = "vulkan".into();
                cfg.wgpu_adapter = Some("NVIDIA".into());
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();

    let mut cfg = CmdConfig::default();
    let mut shell: Shell = cli.shell.into();

    apply_preset(&cli, &mut cfg, &mut shell);

    let manifest = cli
        .manifest
        .unwrap_or_else(|| cli.output.output_root.join("manifest.json"));
    cfg = cfg.with_manifest(manifest.display().to_string());

    if let Some(store) = cli.store {
        cfg = cfg.with_store(store.into());
    }

    if let Some(prefetch) = cli.prefetch {
        cfg = cfg.with_prefetch(Some(prefetch));
    }

    if let Some(model) = cli.model {
        cfg = cfg.with_model(model.into());
    }

    if let Some(batch_size) = cli.batch_size {
        cfg = cfg.with_batch_size(batch_size);
    }

    if let Some(log_every) = cli.log_every {
        cfg = cfg.with_log_every(log_every);
    }

    if let Some(backend) = cli.backend {
        cfg = cfg.with_backend(backend);
    } else if matches!(shell, Shell::PowerShell) {
        cfg = cfg.with_backend("dx12");
    }

    if let Some(adapter) = cli.adapter {
        cfg = cfg.with_adapter(adapter);
    }

    if !cli.extra_args.trim().is_empty() {
        cfg = cfg.with_extra_args(cli.extra_args);
    }

    let cmd = build_command(&cfg, shell);
    println!("{cmd}");
}
