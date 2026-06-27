use anyhow::Result;
use clap::Parser;
use kvcdn::cli::{
    AdminArgs, ApiKeyArgs, BenchmarkArgs, DeleteArgs, DiagArgs, DownloadArgs, InferArgs, ListArgs,
    LoginArgs, LogoutArgs, PlotArgs, QuantArgs, QuotaArgs, UploadArgs, VerifyArgs, WhoamiArgs,
};
use kvcdn::cli::{AdminCommand, ApiKeyCommand};
use kvcdn::{hosted, local};

#[derive(Parser)]
#[command(name = "kvcdn")]
#[command(
    about = "KVCDN CLI: create, verify, quantize, benchmark, and share LLM KV caches for any model or quant type"
)]
#[command(version)]
enum Cli {
    /// Verify a saved KV cache produces token-exact output.
    Verify(VerifyArgs),
    /// Compare scratch-prefill vs. KV-cache logits.
    Diag(DiagArgs),
    /// Measure prefill vs. continuation speedup.
    Benchmark(BenchmarkArgs),
    /// Plot benchmark results from a CSV.
    Plot(PlotArgs),
    /// Quantize a KV artifact and optionally verify accuracy.
    Quant(QuantArgs),
    /// Authenticate with the hosted KVCDN service.
    Login(LoginArgs),
    /// Remove stored OIDC tokens and API key.
    Logout(LogoutArgs),
    /// Manage the stored KVCDN API key.
    ApiKey(ApiKeyArgs),
    /// Upload a KV artifact to KVCDN.
    Upload(UploadArgs),
    /// List remote KV artifacts in a KVCDN project.
    List(ListArgs),
    /// Download a remote KV artifact from KVCDN.
    Download(DownloadArgs),
    /// Delete a remote KV artifact from KVCDN.
    Delete(DeleteArgs),
    /// Show account quota amount and utilization.
    Quota(QuotaArgs),
    /// Show the current user and active org/project.
    Whoami(WhoamiArgs),
    /// Internal: run continuation generation against a loaded KV artifact.
    Infer(InferArgs),
    /// Operator-only administration commands.
    Admin(AdminArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli {
        Cli::Verify(args) => local::verify::run(args),
        Cli::Diag(args) => local::diag::run(args),
        Cli::Benchmark(args) => local::benchmark::run(args),
        Cli::Plot(args) => local::plot::run(
            args.csv_path.as_deref(),
            args.out.as_deref(),
            args.max_n,
            &args.model,
        ),
        Cli::Quant(args) => local::quant::run(args),
        Cli::Login(args) => hosted::login::run(args),
        Cli::Logout(args) => hosted::logout::run(args),
        Cli::ApiKey(args) => match args.command {
            ApiKeyCommand::Set { key } => hosted::api_key::set(key),
            ApiKeyCommand::Verify {
                api_key,
                api_url,
                org,
                project,
            } => hosted::api_key::verify(api_key, api_url, org, project),
            ApiKeyCommand::Clear => hosted::api_key::clear(),
        },
        Cli::Upload(args) => hosted::upload::run(args),
        Cli::List(args) => hosted::list::run(args),
        Cli::Download(args) => hosted::download::run(args),
        Cli::Delete(args) => hosted::delete::run(args),
        Cli::Quota(args) => hosted::quota::run(args),
        Cli::Whoami(args) => hosted::whoami::run(args),
        Cli::Infer(args) => {
            let tokens = local::infer::run(args)?;
            for token in tokens {
                println!("{token}");
            }
            Ok(())
        }
        Cli::Admin(args) => match args.command {
            AdminCommand::MintApiKey {
                org,
                api_url,
                admin_secret,
            } => hosted::admin::mint_api_key(org, api_url, admin_secret),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn quant_args_dtype_alias() {
        let cli = Cli::try_parse_from(["kvcdn", "quant", "--dtype", "F16"]).unwrap();
        match cli {
            Cli::Quant(args) => assert_eq!(args.target_dtype, "F16"),
            _ => panic!("expected Quant subcommand"),
        }
    }

    #[test]
    fn quant_args_target_dtype_still_works() {
        let cli = Cli::try_parse_from(["kvcdn", "quant", "--target-dtype", "F16"]).unwrap();
        match cli {
            Cli::Quant(args) => assert_eq!(args.target_dtype, "F16"),
            _ => panic!("expected Quant subcommand"),
        }
    }
}
