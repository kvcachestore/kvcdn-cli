use crate::config::Config;
use crate::hosted::api::ApiClient;
use crate::hosted::transfer::{HttpTransfer, Transfer};
use anyhow::{Context, Result};

pub fn run(args: crate::cli::DownloadArgs) -> Result<()> {
    let cfg = Config::load(
        args.api_url,
        args.issuer_url,
        args.client_id,
        args.org,
        args.project,
        args.api_key,
    )?;

    cfg.require_hosted()?;
    let org = cfg.default_org.clone();
    let project = cfg.default_project.clone();

    let mut client = ApiClient::new(cfg)?;
    let init = client
        .get_download_url(&args.artifact_id)
        .context("failed to initiate download")?;
    if init.artifact_id != args.artifact_id {
        anyhow::bail!(
            "backend returned unexpected artifact id {}",
            init.artifact_id
        );
    }

    let output_path = args.output.unwrap_or_else(|| args.artifact_id.clone());
    if std::path::Path::new(&output_path)
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        anyhow::bail!(
            "output path must not contain parent-directory references (..), got: {output_path}"
        );
    }

    println!(
        "downloading {} from {} (org: {}, project: {})",
        args.artifact_id, init.download_url, org, project
    );

    let total_size = 0u64;
    HttpTransfer::new()
        .download(&init.download_url, &output_path, total_size)
        .context("failed to download artifact")?;

    println!("Downloaded {} to {}", args.artifact_id, output_path);
    Ok(())
}
