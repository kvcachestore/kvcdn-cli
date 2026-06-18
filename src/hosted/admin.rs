use crate::config::Config;
use crate::hosted::http::Client;
use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Clone, Deserialize)]
struct MintApiKeyResponse {
    api_key: String,
    customer_id: String,
    org_slug: String,
}

pub fn mint_api_key(
    org: String,
    api_url: Option<String>,
    admin_secret: Option<String>,
) -> Result<()> {
    let cfg = Config::load(api_url, None, None, None, None, None)
        .context("loading kvcdn configuration")?;

    let admin_secret = admin_secret.ok_or_else(|| {
        anyhow!("missing admin secret; pass --admin-secret or set KVCDN_ADMIN_SECRET")
    })?;

    let mut url = url::Url::parse(&cfg.api_url)
        .with_context(|| format!("parsing API URL {}", cfg.api_url))?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("admin")
        .push("api-keys");
    let url = url.to_string();

    let client = Client::default_timeout()?;
    let resp = client
        .post_json(&url, &admin_secret, &json!({ "org_slug": org }))
        .with_context(|| format!("POST {url}"))?;
    let resp = Client::require_success(resp, &url)?;

    let body: MintApiKeyResponse = resp.json().context("parsing mint-api-key response")?;

    println!("org:       {}", body.org_slug);
    println!("customer:  {}", body.customer_id);
    println!("api_key:   {}", body.api_key);

    Ok(())
}
