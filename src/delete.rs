use crate::api::ApiClient;
use crate::config::Config;
use anyhow::{Context, Result};

pub fn run(args: crate::cli::DeleteArgs) -> Result<()> {
    let cfg = Config::load(
        args.api_url,
        args.issuer_url,
        args.client_id,
        args.org,
        args.project,
        args.api_key,
    )?;
    let org = cfg.default_org.clone();
    let project = cfg.default_project.clone();

    if !args.yes {
        eprintln!(
            "Delete artifact {} from project {} (org: {}) at {}? [y/N]",
            args.artifact_id, project, org, cfg.api_url
        );
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .context("failed to read confirmation")?;
        let input = input.trim().to_lowercase();
        if input != "y" && input != "yes" {
            println!("Delete cancelled.");
            return Ok(());
        }
    }

    let mut client = ApiClient::new(cfg)?;
    client
        .delete_artifact(&args.artifact_id, &org, &project)
        .context("failed to delete artifact")?;
    println!(
        "Deleted artifact {} from project {}.",
        args.artifact_id, project
    );
    Ok(())
}
