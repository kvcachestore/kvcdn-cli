use crate::token_store::Tokens;
use anyhow::{Context, Result};
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use chrono::{Duration as ChronoDuration, Utc};
use rand::RngCore;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use url::Url;

const DEVICE_CODE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";

#[derive(Debug, Clone, Deserialize)]
pub struct OidcConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    #[serde(default)]
    pub device_authorization_endpoint: Option<String>,
    #[serde(default)]
    pub grant_types_supported: Vec<String>,
}

impl OidcConfig {
    pub fn supports_device_code(&self) -> bool {
        self.device_authorization_endpoint.is_some()
            && self
                .grant_types_supported
                .iter()
                .any(|g| g == DEVICE_CODE_GRANT)
    }
}

pub fn discover(issuer_url: &str) -> Result<OidcConfig> {
    let url = format!(
        "{}/.well-known/openid-configuration",
        issuer_url.trim_end_matches('/')
    );
    let client = crate::http::Client::default_timeout()?;
    let resp = client
        .get(&url)
        .with_context(|| format!("fetching OIDC discovery from {url}"))?;
    let resp = crate::http::Client::require_success(resp, &url)?;
    resp.json().context("parsing OIDC discovery document")
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeviceAuthorizationResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    pub expires_in: u64,
    #[serde(default = "default_interval")]
    pub interval: u64,
}

fn default_interval() -> u64 {
    5
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}

pub fn initiate_device_auth(
    cfg: &OidcConfig,
    client_id: &str,
) -> Result<DeviceAuthorizationResponse> {
    let endpoint = cfg
        .device_authorization_endpoint
        .as_ref()
        .context("issuer does not advertise a device authorization endpoint")?;

    let client = crate::http::Client::default_timeout()?;
    let resp = client
        .post_form(
            endpoint,
            &[("client_id", client_id), ("scope", "openid profile email")]
                .into_iter()
                .collect(),
        )
        .context("POST to device authorization endpoint")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp
            .text()
            .context("reading device authorization error response")?;
        anyhow::bail!("device authorization failed ({status}): {body}");
    }

    resp.json().context("parsing device authorization response")
}

pub fn poll_device_token(
    cfg: &OidcConfig,
    client_id: &str,
    device_code: &str,
    interval_secs: u64,
    expires_in_secs: u64,
) -> Result<Tokens> {
    let client = crate::http::Client::default_timeout()?;
    let params = [
        ("grant_type", DEVICE_CODE_GRANT),
        ("device_code", device_code),
        ("client_id", client_id),
    ]
    .into_iter()
    .collect();

    let started = std::time::Instant::now();
    let expires = std::time::Duration::from_secs(expires_in_secs);
    let interval = std::time::Duration::from_secs(interval_secs.max(1));

    loop {
        std::thread::sleep(interval);

        let resp = client
            .post_form(&cfg.token_endpoint, &params)
            .context("POST to token endpoint")?;
        if resp.status().is_success() {
            let token_resp: TokenResponse = resp.json().context("parsing token response")?;
            let expires_at = token_resp
                .expires_in
                .map(|s| Utc::now() + ChronoDuration::seconds(s as i64));
            return Ok(Tokens {
                access_token: token_resp.access_token,
                refresh_token: token_resp.refresh_token,
                expires_at,
            });
        }

        let status = resp.status();
        let body = resp.text().context("reading token error response")?;

        #[derive(Debug, Deserialize)]
        struct OidcError {
            error: String,
            #[serde(default)]
            error_description: Option<String>,
        }

        let parsed_err = serde_json::from_str::<OidcError>(&body).ok();
        let error_code = parsed_err.as_ref().map(|e| e.error.as_str());

        match error_code {
            Some("authorization_pending") => {
                if started.elapsed() >= expires {
                    anyhow::bail!("device login timed out");
                }
                continue;
            }
            Some("slow_down") => {
                if started.elapsed() >= expires {
                    anyhow::bail!("device login timed out");
                }
                std::thread::sleep(interval);
                continue;
            }
            Some("expired_token") => anyhow::bail!("device code expired; please retry login"),
            _ => {
                let err = parsed_err
                    .map(|e| {
                        let desc = e.error_description.as_deref().unwrap_or("no description");
                        format!("{}: {}", e.error, desc)
                    })
                    .unwrap_or(body);
                anyhow::bail!("token exchange failed ({status}): {err}")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pkce {
    pub verifier: String,
    pub challenge: String,
}

impl Pkce {
    pub fn generate() -> Self {
        let mut bytes = [0u8; 96];
        rand::rng().fill_bytes(&mut bytes);
        let verifier = URL_SAFE_NO_PAD.encode(bytes);
        let hash = Sha256::digest(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(hash);
        Self {
            verifier,
            challenge,
        }
    }
}

pub fn random_state() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn build_authorization_url(
    cfg: &OidcConfig,
    client_id: &str,
    redirect_uri: &str,
    pkce: &Pkce,
    state: &str,
) -> Result<String> {
    let mut url = Url::parse(&cfg.authorization_endpoint).with_context(|| {
        format!(
            "parsing authorization endpoint {}",
            cfg.authorization_endpoint
        )
    })?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("scope", "openid profile email")
        .append_pair("state", state)
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(url.to_string())
}

pub fn exchange_code(
    cfg: &OidcConfig,
    client_id: &str,
    redirect_uri: &str,
    code: &str,
    pkce: &Pkce,
) -> Result<Tokens> {
    let mut params = HashMap::new();
    params.insert("grant_type", "authorization_code");
    params.insert("client_id", client_id);
    params.insert("code", code);
    params.insert("redirect_uri", redirect_uri);
    params.insert("code_verifier", &pkce.verifier);

    let client = crate::http::Client::default_timeout()?;
    let resp = client
        .post_form(&cfg.token_endpoint, &params)
        .context("POST to token endpoint")?;

    if !resp.status().is_success() {
        #[derive(Debug, Deserialize)]
        struct OidcError {
            error: String,
            #[serde(default)]
            error_description: Option<String>,
        }

        let status = resp.status();
        let body = resp.text().context("reading token error response")?;
        let msg = serde_json::from_str::<OidcError>(&body)
            .map(|e| {
                let desc = e.error_description.as_deref().unwrap_or("no description");
                format!("{}: {}", e.error, desc)
            })
            .unwrap_or(body);
        anyhow::bail!("token exchange failed ({}): {msg}", status);
    }

    let token_resp: TokenResponse = resp.json().context("parsing token response")?;
    let expires_at = token_resp
        .expires_in
        .map(|s| Utc::now() + ChronoDuration::seconds(s as i64));

    Ok(Tokens {
        access_token: token_resp.access_token,
        refresh_token: token_resp.refresh_token,
        expires_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_verifier() {
        let pkce = Pkce::generate();
        assert_eq!(pkce.verifier.len(), 128);
        assert!(
            pkce.verifier
                .chars()
                .all(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '.' | '_' | '~')),
            "verifier contains characters outside the PKCE allowed set"
        );
        let expected_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(pkce.verifier.as_bytes()));
        assert_eq!(pkce.challenge, expected_challenge);
    }

    #[test]
    fn pkce_generators_are_unique() {
        let a = Pkce::generate();
        let b = Pkce::generate();
        assert_ne!(a.verifier, b.verifier, "PKCE verifiers should be unique");
        assert_ne!(a.challenge, b.challenge, "PKCE challenges should be unique");
    }

    #[test]
    fn authorization_url_contains_required_params() {
        let cfg = OidcConfig {
            authorization_endpoint: "https://issuer.example.com/authorize".to_string(),
            token_endpoint: "https://issuer.example.com/token".to_string(),
            device_authorization_endpoint: None,
            grant_types_supported: Vec::new(),
        };
        let pkce = Pkce::generate();
        let state = random_state();
        let url = build_authorization_url(
            &cfg,
            "test-client-id",
            "http://localhost:8080/callback",
            &pkce,
            &state,
        )
        .expect("build_authorization_url should succeed");

        assert!(url.contains("response_type=code"), "missing response_type");
        assert!(
            url.contains("client_id=test-client-id"),
            "missing client_id"
        );
        assert!(url.contains("code_challenge="), "missing code_challenge");
        assert!(
            url.contains("code_challenge_method=S256"),
            "missing code_challenge_method"
        );
        assert!(url.contains(&format!("state={}", state)), "missing state");
    }

    #[test]
    fn random_state_is_unique_and_url_safe() {
        let a = random_state();
        let b = random_state();
        assert_ne!(a, b, "random states should be unique");
        assert!(
            a.chars()
                .all(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_')),
            "state contains non-URL-safe characters"
        );
        assert!(
            b.chars()
                .all(|c| matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_')),
            "state contains non-URL-safe characters"
        );
    }
}
