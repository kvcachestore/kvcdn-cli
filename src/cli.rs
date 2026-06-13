use clap::Parser;

#[derive(Parser)]
pub struct VerifyArgs {
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    #[arg(long, default_value_t = 32)]
    pub n: usize,
    #[arg(long, default_value = "results/sample.kv")]
    pub kv_path: String,
}

#[derive(Parser)]
pub struct DiagArgs {
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
}

#[derive(Parser)]
pub struct BenchmarkArgs {
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    #[arg(long, value_delimiter = ',')]
    pub lengths: Option<Vec<usize>>,
}

#[derive(Parser)]
pub struct PlotArgs {
    #[arg(long, default_value = "results/bench.csv")]
    pub csv_path: String,
}

#[derive(Parser)]
pub struct QuantArgs {
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    #[arg(long, default_value = "results/sample.kv")]
    pub input: String,
    #[arg(long, default_value = "results/sample.q8.kv")]
    pub output: String,
    #[arg(long, default_value_t = 64)]
    pub context_tokens: usize,
    #[arg(long, default_value_t = 8)]
    pub n: usize,
    #[arg(long)]
    pub verify: bool,
}
