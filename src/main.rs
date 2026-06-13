mod benchmark;
mod cli;
mod common;
mod continue_mod;
mod diag;
mod kv_io;
mod kv_quant;
mod model;
mod plot;
mod prefill;
mod quant;
mod qwen3_model;
mod verify;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "kvcdn")]
#[command(about = "KV-cache reuse toolkit (Rust)")]
enum Cli {
    Verify(cli::VerifyArgs),
    Diag(cli::DiagArgs),
    Benchmark(cli::BenchmarkArgs),
    Plot(cli::PlotArgs),
    Quant(cli::QuantArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Verify(args) => verify::run(args),
        Cli::Diag(args) => diag::run(args),
        Cli::Benchmark(args) => benchmark::run(args),
        Cli::Plot(args) => plot::run(&args.csv_path),
        Cli::Quant(args) => quant::run(args),
    }
}
