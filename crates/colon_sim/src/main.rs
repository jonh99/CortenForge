use clap::Parser;

use colon_sim::cli::AppArgs;
use colon_sim::run_app;

fn main() {
    let args = AppArgs::parse();
    run_app(args);
}
