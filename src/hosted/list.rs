use crate::config::Config;
use crate::core::common::format_size;
use crate::hosted::api::ApiClient;
use anyhow::{Context, Result};

pub fn run(args: crate::cli::ListArgs) -> Result<()> {
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
    let artifacts = client
        .list_artifacts(&org, &project)
        .context("failed to list artifacts")?;

    match args.format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&artifacts)?);
        }
        "ids" => {
            for artifact in artifacts {
                println!("{}", artifact.artifact_id);
            }
        }
        _ => {
            println!("Artifacts for org: {}, project: {}", org, project);
            if artifacts.is_empty() {
                println!("No artifacts in project {}.", project);
                return Ok(());
            }
            if args.long {
                println!(
                    "{:<36} {:<8} {:>12} {:<8} {:<10} {:<64} NAME",
                    "ARTIFACT ID", "TOKENS", "SIZE", "DTYPE", "CREATED", "SHA256"
                );
                for artifact in artifacts {
                    println!(
                        "{:<36} {:<8} {:>12} {:<8} {:<10} {:<64} {}",
                        artifact.artifact_id,
                        artifact.num_tokens,
                        format_size(artifact.size_bytes),
                        artifact.dtype,
                        artifact
                            .created_at
                            .split('T')
                            .next()
                            .unwrap_or(&artifact.created_at),
                        artifact.sha256,
                        artifact.name
                    );
                }
            } else {
                println!("{:<36} {:<8} {:>12} NAME", "ARTIFACT ID", "TOKENS", "SIZE");
                for artifact in artifacts {
                    println!(
                        "{:<36} {:<8} {:>12} {}",
                        artifact.artifact_id,
                        artifact.num_tokens,
                        format_size(artifact.size_bytes),
                        artifact.name
                    );
                }
            }
        }
    }

    Ok(())
}
