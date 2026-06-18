mod admin;
mod api;
mod api_key;
mod benchmark;
mod callback_server;
mod cli;
mod common;
mod config;
mod continue_mod;
mod credential_store;
mod crypto;
mod delete;
mod diag;
mod download;
mod http;
mod kv_io;
mod kv_quant;
mod list;
mod login;
mod logout;
mod model;
mod models;
mod oidc;
mod output;
mod plot;
mod prefill;
mod quant;
mod quota;
mod token_store;
mod tokenize;
mod transfer;
mod upload;
mod verify;
mod whoami;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "kvcdn")]
#[command(
    about = "KVCDN CLI: create, verify, quantize, benchmark, and share LLM KV caches for any model or quant type"
)]
#[command(version)]
enum Cli {
    /// Verify a saved KV cache produces token-exact output.
    Verify(cli::VerifyArgs),
    /// Compare scratch-prefill vs. KV-cache logits.
    Diag(cli::DiagArgs),
    /// Measure prefill vs. continuation speedup.
    Benchmark(cli::BenchmarkArgs),
    /// Plot benchmark results from a CSV.
    Plot(cli::PlotArgs),
    /// Quantize a KV artifact and optionally verify accuracy.
    Quant(cli::QuantArgs),
    /// Authenticate with the hosted KVCDN service.
    Login(cli::LoginArgs),
    /// Remove stored OIDC tokens and API key.
    Logout(cli::LogoutArgs),
    /// Manage the stored KVCDN API key.
    ApiKey(cli::ApiKeyArgs),
    /// Upload a KV artifact to KVCDN.
    Upload(cli::UploadArgs),
    /// List remote KV artifacts in a KVCDN project.
    List(cli::ListArgs),
    /// Download a remote KV artifact from KVCDN.
    Download(cli::DownloadArgs),
    /// Delete a remote KV artifact from KVCDN.
    Delete(cli::DeleteArgs),
    /// Show account quota amount and utilization.
    Quota(cli::QuotaArgs),
    /// Show the current user and active org/project.
    Whoami(cli::WhoamiArgs),
    /// Operator-only administration commands.
    Admin(cli::AdminArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Verify(args) => verify::run(args),
        Cli::Diag(args) => diag::run(args),
        Cli::Benchmark(args) => benchmark::run(args),
        Cli::Plot(args) => plot::run(
            args.csv_path.as_deref(),
            args.out.as_deref(),
            args.max_n,
            &args.model,
        ),
        Cli::Quant(args) => quant::run(args),
        Cli::Login(args) => login::run(args),
        Cli::Logout(args) => logout::run(args),
        Cli::ApiKey(args) => match args.command {
            cli::ApiKeyCommand::Set { key } => api_key::set(key),
            cli::ApiKeyCommand::Verify {
                api_key,
                api_url,
                org,
                project,
            } => api_key::verify(api_key, api_url, org, project),
            cli::ApiKeyCommand::Clear => api_key::clear(),
        },
        Cli::Upload(args) => upload::run(args),
        Cli::List(args) => list::run(args),
        Cli::Download(args) => download::run(args),
        Cli::Delete(args) => delete::run(args),
        Cli::Quota(args) => quota::run(args),
        Cli::Whoami(args) => whoami::run(args),
        Cli::Admin(args) => match args.command {
            cli::AdminCommand::MintApiKey {
                org,
                api_url,
                admin_secret,
            } => admin::mint_api_key(org, api_url, admin_secret),
        },
    }
}
