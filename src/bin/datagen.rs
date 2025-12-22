use clap::Parser;

use colon_sim::cli::{AppArgs, RunMode};
use colon_sim::run_app;

fn main() {
    let mut args = AppArgs::parse();
    args.mode = RunMode::Datagen;
    args.headless = true;
    run_app(args);
}
