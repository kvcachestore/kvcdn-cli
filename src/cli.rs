use clap::Parser;

/// Verify that loading a saved KV cache produces token-exact output.
#[derive(Parser)]
pub struct VerifyArgs {
    /// Hugging Face model identifier (e.g. Qwen/Qwen3-0.6B).
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    /// Hugging Face model revision (commit/branch/tag). Falls back to HF_REVISION, then the pinned default.
    #[arg(long)]
    pub revision: Option<String>,
    /// Number of continuation tokens to generate and compare.
    #[arg(long, default_value_t = 32)]
    pub n: usize,
    /// Path to read or write the KV artifact. If omitted, a temporary file is used.
    #[arg(long)]
    pub kv_path: Option<String>,
    /// Path to a text file containing the long context to prefill.
    #[arg(long)]
    pub context_file: Option<String>,
    /// Question appended to the context to trigger continuation generation.
    #[arg(
        long,
        default_value = "\n\nQ: Summarize the key claim in one sentence.\nA:"
    )]
    pub question: String,
}

/// Logits-level diagnostic comparing scratch-prefill vs. KV-cache continuation paths.
#[derive(Parser)]
pub struct DiagArgs {
    /// Hugging Face model identifier (e.g. Qwen/Qwen3-0.6B).
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    /// Hugging Face model revision (commit/branch/tag). Falls back to HF_REVISION, then the pinned default.
    #[arg(long)]
    pub revision: Option<String>,
    /// Path to write the diagnostic report. Prints to stdout if omitted.
    #[arg(long)]
    pub output: Option<String>,
    /// Path to a text file containing the long context to prefill.
    #[arg(long)]
    pub context_file: Option<String>,
    /// Question appended to the context to trigger continuation generation.
    #[arg(
        long,
        default_value = "\n\nQ: Summarize the key claim in one sentence.\nA:"
    )]
    pub question: String,
}

/// Measure full-prefill cost against resident-KV continuation speedup.
#[derive(Parser)]
pub struct BenchmarkArgs {
    /// Hugging Face model identifier (e.g. Qwen/Qwen3-0.6B).
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    /// Hugging Face model revision (commit/branch/tag). Falls back to HF_REVISION, then the pinned default.
    #[arg(long)]
    pub revision: Option<String>,
    /// Comma-separated list of context lengths (tokens) to benchmark.
    #[arg(long, value_delimiter = ',')]
    pub lengths: Option<Vec<usize>>,
    /// Path to write the CSV results. Prints summary tables if omitted.
    #[arg(long)]
    pub output: Option<String>,
    /// Number of repetitions per context length.
    #[arg(long, default_value_t = 5)]
    pub reps: usize,
    /// Path to a text file containing the long context to prefill.
    #[arg(long)]
    pub context_file: Option<String>,
    /// Question appended to the context to trigger continuation generation.
    #[arg(
        long,
        default_value = "\n\nQ: Summarize the key claim in one sentence.\nA:"
    )]
    pub question: String,
    /// Device to run on: cpu, cuda, or metal. Auto-selected when omitted.
    #[arg(long, value_parser = crate::models::engine::parse_device)]
    pub device: Option<candle_core::Device>,
}

/// Plot benchmark results from a CSV file.
#[derive(Parser)]
pub struct PlotArgs {
    /// Hugging Face model identifier used in the plot title and output filename.
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    /// Path to the CSV produced by `kvcdn benchmark --output`.
    #[arg(long)]
    pub csv_path: Option<String>,
    /// Path to write the PNG plot. If omitted, a timestamped PNG is created under the data dir.
    #[arg(long)]
    pub out: Option<String>,
    /// Maximum reuse count N to plot on the x-axis.
    #[arg(long, default_value_t = 1000)]
    pub max_n: usize,
}

/// Quantize a KV artifact and run optional token-exact verification.
#[derive(Parser)]
pub struct QuantArgs {
    /// Hugging Face model identifier (e.g. Qwen/Qwen3-0.6B).
    #[arg(long, default_value = "Qwen/Qwen3-0.6B")]
    pub model: String,
    /// Hugging Face model revision (commit/branch/tag). Falls back to HF_REVISION, then the pinned default.
    #[arg(long)]
    pub revision: Option<String>,
    /// Path to the input KV artifact. If omitted, one is generated from the context.
    #[arg(long)]
    pub input: Option<String>,
    /// Path to write the quantized KV artifact.
    #[arg(long)]
    pub output: Option<String>,
    /// Number of context tokens to prefill before quantizing.
    #[arg(long, default_value_t = 64)]
    pub context_tokens: usize,
    /// Number of continuation tokens to generate when verifying.
    #[arg(long, default_value_t = 8)]
    pub n: usize,
    /// Also run token-exact verification after quantization.
    #[arg(long)]
    pub verify: bool,
    /// Path to a text file containing the long context to prefill.
    #[arg(long)]
    pub context_file: Option<String>,
    /// Question appended to the context to trigger continuation generation.
    #[arg(
        long,
        default_value = "\n\nQ: Summarize the key claim in one sentence.\nA:"
    )]
    pub question: String,
    /// Target dequantized dtype: F32, F16, BF16, or FP8.
    #[arg(long, default_value = "F16")]
    pub target_dtype: String,
    /// Device to run on: cpu, cuda, or metal. Auto-selected when omitted.
    #[arg(long, value_parser = crate::models::engine::parse_device)]
    pub device: Option<candle_core::Device>,
}

/// Authenticate with the hosted KVCDN service via OIDC.
///
/// Uses the OIDC device-code flow when the issuer supports it, falling back
/// to a local callback-based authorization-code flow otherwise. The device
/// flow prints a URL and a short user code; open the URL and enter the code
/// to complete login.
#[derive(Parser)]
pub struct LoginArgs {
    /// Do not open a browser; print the URL or device-code instructions.
    #[arg(long)]
    pub no_browser: bool,
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// Organization slug to use for this session. Defaults to the configured default org.
    #[arg(long)]
    pub org: Option<String>,
    /// Project slug to use for this session. Defaults to the configured default project.
    #[arg(long)]
    pub project: Option<String>,
}

/// Remove stored OIDC tokens and API key from this machine.
#[derive(Parser)]
pub struct LogoutArgs {}

/// Manage the stored KVCDN API key.
#[derive(Parser)]
pub struct ApiKeyArgs {
    #[command(subcommand)]
    pub command: ApiKeyCommand,
}

#[derive(Parser, Clone)]
pub enum ApiKeyCommand {
    /// Save a KVCDN API key (kv_<hex>) encrypted on disk.
    Set {
        /// The API key copied from the KVCDN dashboard.
        #[arg(value_name = "KEY")]
        key: String,
    },
    /// Check the stored (or provided) API key against the portal.
    Verify {
        /// API key to verify. Uses the stored key if omitted.
        #[arg(long)]
        api_key: Option<String>,
        /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
        #[arg(long)]
        api_url: Option<String>,
        /// Organization slug for the verification context. Defaults to the configured default org.
        #[arg(long)]
        org: Option<String>,
        /// Project slug for the verification context. Defaults to the configured default project.
        #[arg(long)]
        project: Option<String>,
    },
    /// Remove the stored API key.
    Clear,
}

/// Upload a KV artifact to the hosted KVCDN endpoint.
#[derive(Parser)]
pub struct UploadArgs {
    /// Path to the KV artifact file produced by `kvcdn verify` or `kvcdn quant`.
    pub path: String,
    /// Name to assign to the uploaded KV artifact.
    #[arg(long)]
    pub name: String,
    /// Project slug to upload into. Defaults to the configured default project.
    #[arg(long)]
    pub project: Option<String>,
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// API key for non-interactive uploads. Falls back to KVCDN_API_KEY or the stored key.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Artifact visibility: public (fetchable without auth) or private.
    #[arg(long, default_value = "private")]
    pub visibility: String,
    /// Organization slug to upload under. Defaults to the configured default org.
    #[arg(long)]
    pub org: Option<String>,
}

/// Delete a remote KV artifact from the hosted KVCDN endpoint.
#[derive(Parser)]
pub struct DeleteArgs {
    /// Artifact ID to delete.
    pub artifact_id: String,
    /// Project slug the artifact belongs to. Defaults to the configured default project.
    #[arg(long)]
    pub project: Option<String>,
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// API key for non-interactive deletes. Falls back to KVCDN_API_KEY or the stored key.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Skip the interactive confirmation prompt.
    #[arg(long, short = 'y')]
    pub yes: bool,
    /// Organization slug the artifact belongs to. Defaults to the configured default org.
    #[arg(long)]
    pub org: Option<String>,
}

/// List remote KV artifacts in a hosted KVCDN project.
#[derive(Parser)]
pub struct ListArgs {
    /// Project slug to list. Defaults to the configured default project.
    #[arg(long)]
    pub project: Option<String>,
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// API key for non-interactive listing. Falls back to KVCDN_API_KEY or the stored key.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Output format: table, json, ids.
    #[arg(long, default_value = "table")]
    pub format: String,
    /// Show extended metadata: created_at, dtype, storage_dtype, sha256.
    #[arg(long)]
    pub long: bool,
    /// Organization slug to list. Defaults to the configured default org.
    #[arg(long)]
    pub org: Option<String>,
}

/// Download a remote KV artifact from the hosted KVCDN endpoint.
#[derive(Parser)]
pub struct DownloadArgs {
    /// Artifact ID to download.
    pub artifact_id: String,
    /// Path to write the downloaded KV artifact. Defaults to the artifact name in the current directory.
    #[arg(long)]
    pub output: Option<String>,
    /// Project slug the artifact belongs to. Defaults to the configured default project.
    #[arg(long)]
    pub project: Option<String>,
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// API key for non-interactive downloads. Falls back to KVCDN_API_KEY or the stored key.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Organization slug the artifact belongs to. Defaults to the configured default org.
    #[arg(long)]
    pub org: Option<String>,
}

/// Show account quota amount and utilization.
#[derive(Parser)]
pub struct QuotaArgs {
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// API key for non-interactive lookup. Falls back to KVCDN_API_KEY or the stored key.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Output format: table, json.
    #[arg(long, default_value = "table")]
    pub format: String,
}

/// Show the current authenticated user and active org/project.
#[derive(Parser)]
pub struct WhoamiArgs {
    /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
    #[arg(long)]
    pub api_url: Option<String>,
    /// OIDC issuer URL. Falls back to KVCDN_ISSUER_URL, then config, then default.
    #[arg(long)]
    pub issuer_url: Option<String>,
    /// OIDC client ID. Falls back to KVCDN_CLIENT_ID, then config, then default.
    #[arg(long)]
    pub client_id: Option<String>,
    /// API key for non-interactive lookup. Falls back to KVCDN_API_KEY or the stored key.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Organization slug to use. Defaults to the configured default org.
    #[arg(long)]
    pub org: Option<String>,
    /// Project slug to use. Defaults to the configured default project.
    #[arg(long)]
    pub project: Option<String>,
}

/// Operator-only administration commands.
#[derive(Parser)]
pub struct AdminArgs {
    #[command(subcommand)]
    pub command: AdminCommand,
}

#[derive(Parser)]
pub enum AdminCommand {
    /// Mint a deterministic API key for an org using the admin secret.
    MintApiKey {
        /// Organization slug to mint the key for.
        org: String,
        /// KVCDN API base URL. Falls back to KVCDN_API_URL, then config, then default.
        #[arg(long)]
        api_url: Option<String>,
        /// Admin secret bearer token. Falls back to KVCDN_ADMIN_SECRET.
        #[arg(long, env = "KVCDN_ADMIN_SECRET")]
        admin_secret: Option<String>,
    },
}
