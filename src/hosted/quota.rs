use crate::config::Config;
use crate::core::common::format_size;
use crate::hosted::api::{ApiClient, QuotaResponse};
use anyhow::{Context, Result};

pub fn run(args: crate::cli::QuotaArgs) -> Result<()> {
    let cfg = Config::load(
        args.api_url,
        args.issuer_url,
        args.client_id,
        None,
        None,
        args.api_key,
    )?;

    cfg.require_hosted()?;
    let mut client = ApiClient::new(cfg)?;
    let quota = client.get_quota().context("failed to fetch quota")?;

    match args.format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&quota)?);
        }
        _ => print_table(&quota),
    }

    Ok(())
}

fn print_table(quota: &QuotaResponse) {
    println!("customer: {}", quota.customer_id);
    println!();
    println!(
        "{:<16} {:>16} {:>16} {:>14}",
        "RESOURCE", "LIMIT", "USED", "UTILIZATION"
    );

    print_row(
        "Organizations",
        quota.quota.organizations,
        quota.used.organizations,
        false,
    );
    print_row("Projects", quota.quota.projects, quota.used.projects, false);
    print_row(
        "Artifacts",
        quota.quota.artifacts,
        quota.used.artifacts,
        false,
    );
    print_row(
        "Storage",
        quota.quota.storage_bytes,
        quota.used.storage_bytes,
        true,
    );
}

fn print_row(label: &str, limit: u64, used: u64, human_readable: bool) {
    let limit_str = if human_readable {
        format_size(limit)
    } else {
        format!("{}", limit)
    };
    let used_str = if human_readable {
        format_size(used)
    } else {
        format!("{}", used)
    };
    let utilization = if limit == 0 {
        "-".to_string()
    } else {
        format!("{:.2}%", (used as f64 / limit as f64) * 100.0)
    };

    println!(
        "{:<16} {:>16} {:>16} {:>14}",
        label, limit_str, used_str, utilization
    );
}
