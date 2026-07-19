use anyhow::{Context, Result};
use reqwest::blocking::Response;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;

/// A small wrapper around [`reqwest::blocking::Client`] that centralizes the
/// timeout, bearer-auth, and error-context conventions used by the CLI when
/// talking to the Portal and OIDC providers.
#[derive(Debug, Clone)]
pub struct Client {
    inner: reqwest::blocking::Client,
}

impl Client {
    /// Create a client with the given request timeout in seconds.
    pub fn new(timeout_secs: u64) -> Result<Self> {
        let inner = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self { inner })
    }

    /// Shorthand for the standard 30-second Portal/OIDC client.
    pub fn default_timeout() -> Result<Self> {
        Self::new(30)
    }

    /// Perform a `GET` and return the raw response with request context.
    pub fn get(&self, url: &str) -> Result<Response> {
        self.inner
            .get(url)
            .send()
            .with_context(|| format!("GET {url}"))
    }

    /// Perform a bearer-authenticated `GET`.
    pub fn get_bearer(&self, url: &str, token: &str) -> Result<Response> {
        self.inner
            .get(url)
            .bearer_auth(token)
            .send()
            .with_context(|| format!("GET {url}"))
    }

    /// Perform a bearer-authenticated `POST` with a JSON body.
    pub fn post_json<B: Serialize>(&self, url: &str, token: &str, body: &B) -> Result<Response> {
        self.inner
            .post(url)
            .bearer_auth(token)
            .json(body)
            .send()
            .with_context(|| format!("POST {url}"))
    }

    /// Perform a `POST` with a `application/x-www-form-urlencoded` body.
    /// No bearer token is attached; this is intended for OIDC token endpoints.
    pub fn post_form(&self, url: &str, params: &HashMap<&str, &str>) -> Result<Response> {
        self.inner
            .post(url)
            .form(params)
            .send()
            .with_context(|| format!("POST {url}"))
    }

    /// Perform a bearer-authenticated `POST` with no body.
    pub fn post_empty_bearer(&self, url: &str, token: &str) -> Result<Response> {
        self.inner
            .post(url)
            .bearer_auth(token)
            .send()
            .with_context(|| format!("POST {url}"))
    }

    pub fn delete_bearer(&self, url: &str, token: &str) -> Result<Response> {
        self.inner
            .delete(url)
            .bearer_auth(token)
            .send()
            .with_context(|| format!("DELETE {url}"))
    }

    /// Map a Portal response to an error when it is not successful.
    ///
    /// `401 Unauthorized` is treated as an expired OIDC session. Callers that
    /// need different semantics (e.g. API-key verification) should inspect the
    /// status before calling this.
    pub fn require_auth_success(resp: Response, url: &str) -> Result<Response> {
        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::bail!("unauthorized; check your API key or run `kvcdn login`");
        }
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("request failed ({url}): {text}");
        }
        Ok(resp)
    }

    /// Map any non-success response to an error, including its body text.
    pub fn require_success(resp: Response, url: &str) -> Result<Response> {
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("request failed ({url}): {text}");
        }
        Ok(resp)
    }
}
