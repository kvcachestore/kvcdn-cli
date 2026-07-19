use crate::config::Config;
use crate::hosted::credential_store::{CredentialStore, Credentials, FileCredentialStore};
use crate::hosted::token_store::Tokens;
use anyhow::{Context, Result, anyhow};
use chrono::{Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadMeta {
    pub name: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub dtype: String,
    pub storage_dtype: Option<String>,
    pub num_tokens: usize,
    pub num_layers: usize,
    pub quantized: bool,
    pub visibility: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMeta {
    /// The production API returns `id`; older backends returned `artifact_id`.
    #[serde(alias = "id")]
    pub artifact_id: String,
    /// The production API returns `model_name`; older backends returned `name`.
    #[serde(alias = "model_name")]
    pub name: String,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub storage_dtype: Option<String>,
    #[serde(default)]
    pub num_tokens: usize,
    #[serde(default)]
    pub num_layers: usize,
    #[serde(default)]
    pub quantized: bool,
    #[serde(default)]
    pub visibility: String,
    #[serde(default)]
    pub created_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListArtifactsResponse {
    pub artifacts: Vec<ArtifactMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaLimits {
    pub organizations: u64,
    pub projects: u64,
    pub artifacts: u64,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaUsage {
    pub organizations: u64,
    pub projects: u64,
    pub artifacts: u64,
    pub storage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotaResponse {
    pub customer_id: String,
    pub quota: QuotaLimits,
    pub used: QuotaUsage,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DownloadInitResponse {
    pub artifact_id: String,
    pub download_url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UploadInitResponse {
    pub artifact_id: String,
    /// The production API returns `url`; older backends returned `upload_url`.
    #[serde(alias = "url")]
    pub upload_url: String,
}

/// Request body for `POST /api/v1/artifacts/upload-url`. The portal reads the
/// top-level fields; everything under `metadata` is stored verbatim and
/// returned on artifact reads.
#[derive(Debug, Serialize)]
struct UploadInitRequest<'a> {
    model_name: &'a str,
    dtype: &'a str,
    num_tokens: usize,
    visibility: &'a str,
    size_bytes: u64,
    checksum: &'a str,
    metadata: UploadExtraMetadata<'a>,
}

#[derive(Debug, Serialize)]
struct UploadExtraMetadata<'a> {
    name: &'a str,
    sha256: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_dtype: Option<&'a str>,
    num_layers: usize,
    quantized: bool,
}

impl<'a> From<&'a UploadMeta> for UploadInitRequest<'a> {
    fn from(meta: &'a UploadMeta) -> Self {
        Self {
            model_name: &meta.name,
            dtype: &meta.dtype,
            num_tokens: meta.num_tokens,
            visibility: &meta.visibility,
            size_bytes: meta.size_bytes,
            checksum: &meta.sha256,
            metadata: UploadExtraMetadata {
                name: &meta.name,
                sha256: &meta.sha256,
                storage_dtype: meta.storage_dtype.as_deref(),
                num_layers: meta.num_layers,
                quantized: meta.quantized,
            },
        }
    }
}

/// Request body for `POST /api/v1/artifacts/{id}/confirm-upload`.
#[derive(Debug, Serialize)]
struct ConfirmUploadRequest<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    digest: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
}

pub struct ApiClient {
    cfg: Config,
    credentials: Box<dyn CredentialStore>,
    client: crate::hosted::http::Client,
}

impl ApiClient {
    pub fn new(cfg: Config) -> Result<Self> {
        let credentials = FileCredentialStore::new(&cfg)?;
        Self::with_credential_store(cfg, Box::new(credentials))
    }

    pub fn with_credential_store(
        cfg: Config,
        credentials: Box<dyn CredentialStore>,
    ) -> Result<Self> {
        Ok(Self {
            cfg,
            credentials,
            client: crate::hosted::http::Client::default_timeout()?,
        })
    }

    #[cfg(test)]
    pub fn new_testable(cfg: Config, credentials: Box<dyn CredentialStore>) -> Result<Self> {
        Self::with_credential_store(cfg, credentials)
    }

    fn bearer_token(&mut self) -> Result<String> {
        match self.credentials.load()? {
            Credentials::ApiKey(k) => Ok(k),
            Credentials::Oidc(tokens) => {
                let fresh = self.ensure_fresh_token(tokens)?;
                Ok(fresh.access_token)
            }
        }
    }

    pub fn init_upload(&mut self, meta: &UploadMeta) -> Result<UploadInitResponse> {
        let token = self.bearer_token()?;
        let url = upload_init_url(&self.cfg.api_url)?;
        let body = UploadInitRequest::from(meta);
        let resp = self
            .client
            .post_json(&url, &token, &body)
            .with_context(|| format!("POST {url}"))?;
        let resp = crate::hosted::http::Client::require_auth_success(resp, &url)?;
        resp.json().context("parsing upload init response")
    }

    pub fn complete_upload(
        &mut self,
        artifact_id: &str,
        digest: Option<&str>,
        size_bytes: Option<u64>,
    ) -> Result<()> {
        let token = self.bearer_token()?;
        let url = complete_upload_url(&self.cfg.api_url, artifact_id)?;
        let body = ConfirmUploadRequest { digest, size_bytes };
        let resp = self
            .client
            .post_json(&url, &token, &body)
            .with_context(|| format!("POST {url}"))?;
        crate::hosted::http::Client::require_auth_success(resp, &url)?;
        Ok(())
    }

    pub fn delete_artifact(&mut self, artifact_id: &str) -> Result<()> {
        let token = self.bearer_token()?;
        let url = delete_artifact_url(&self.cfg.api_url, artifact_id)?;
        let resp = self
            .client
            .delete_bearer(&url, &token)
            .with_context(|| format!("DELETE {url}"))?;
        crate::hosted::http::Client::require_auth_success(resp, &url)?;
        Ok(())
    }

    pub fn list_artifacts(&mut self) -> Result<Vec<ArtifactMeta>> {
        let token = self.bearer_token()?;
        let url = list_artifacts_url(&self.cfg.api_url)?;
        let resp = self
            .client
            .get_bearer(&url, &token)
            .with_context(|| format!("GET {url}"))?;
        let resp = crate::hosted::http::Client::require_auth_success(resp, &url)?;
        let body: ListArtifactsResponse = resp.json().context("parsing artifact list response")?;
        Ok(body.artifacts)
    }

    pub fn get_download_url(&mut self, artifact_id: &str) -> Result<DownloadInitResponse> {
        let token = self.bearer_token()?;
        let url = download_artifact_url(&self.cfg.api_url, artifact_id)?;
        let resp = self
            .client
            .post_empty_bearer(&url, &token)
            .with_context(|| format!("POST {url}"))?;
        let resp = crate::hosted::http::Client::require_auth_success(resp, &url)?;
        let body: DownloadInitResponse = resp.json().context("parsing download init response")?;
        Ok(body)
    }

    /// Verify the current API key and return the customer_id returned by the backend.
    pub fn get_quota(&mut self) -> Result<QuotaResponse> {
        let token = self.bearer_token()?;
        let url = quota_url(&self.cfg.api_url)?;
        let resp = self
            .client
            .get_bearer(&url, &token)
            .with_context(|| format!("GET {url}"))?;
        let resp = crate::hosted::http::Client::require_auth_success(resp, &url)?;
        resp.json().context("parsing quota response")
    }

    pub fn whoami_customer_id(&mut self) -> Result<String> {
        use crate::hosted::http::Client;
        let token = self.bearer_token()?;
        let mut url = url::Url::parse(&self.cfg.api_url)?;
        url.path_segments_mut()
            .map_err(|_| anyhow!("cannot modify URL path"))?
            .push("api")
            .push("v1")
            .push("api-keys")
            .push("verify");
        let url = url.to_string();
        let resp = self
            .client
            .post_empty_bearer(&url, &token)
            .with_context(|| format!("POST {url}"))?;
        let resp = Client::require_success(resp, &url)?;
        #[derive(Deserialize)]
        struct VerifyResponse {
            customer_id: String,
        }
        let body: VerifyResponse = resp.json().context("parsing verify response")?;
        Ok(body.customer_id)
    }

    fn ensure_fresh_token(&mut self, mut tokens: Tokens) -> Result<Tokens> {
        let needs_refresh = tokens
            .expires_at
            .map(|t| Utc::now() + ChronoDuration::seconds(60) >= t)
            .unwrap_or(false);
        if !needs_refresh {
            return Ok(tokens);
        }
        let refresh = tokens
            .refresh_token
            .as_ref()
            .ok_or_else(|| anyhow!("session expired; run `kvcdn login`"))?;

        let oidc = crate::hosted::oidc::discover(&self.cfg.issuer_url)?;
        let mut params = std::collections::HashMap::new();
        params.insert("grant_type", "refresh_token");
        params.insert("client_id", self.cfg.client_id.as_str());
        params.insert("refresh_token", refresh.as_str());

        let resp = self
            .client
            .post_form(&oidc.token_endpoint, &params)
            .context("refresh token request")?;
        let resp = crate::hosted::http::Client::require_success(resp, &oidc.token_endpoint)?;

        #[derive(serde::Deserialize)]
        struct RefreshResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: Option<u64>,
        }
        let r: RefreshResponse = resp.json().context("parsing refresh response")?;
        tokens.access_token = r.access_token;
        if let Some(rt) = r.refresh_token {
            tokens.refresh_token = Some(rt);
        }
        tokens.expires_at = r
            .expires_in
            .map(|s| Utc::now() + ChronoDuration::seconds(s as i64));

        self.credentials.save_tokens(&tokens)?;
        Ok(tokens)
    }
}

/// Build a URL under the fixed `/api/v1` prefix. Artifacts are scoped by the
/// API key (or login session), so the routes do not carry org/project segments.
fn api_v1_url(api_url: &str, segments: &[&str]) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| anyhow!("cannot modify URL path"))?;
        path.push("api").push("v1").extend(segments.iter().copied());
    }
    Ok(url.to_string())
}

fn upload_init_url(api_url: &str) -> Result<String> {
    api_v1_url(api_url, &["artifacts", "upload-url"])
}

fn delete_artifact_url(api_url: &str, artifact_id: &str) -> Result<String> {
    api_v1_url(api_url, &["artifacts", artifact_id])
}

fn list_artifacts_url(api_url: &str) -> Result<String> {
    api_v1_url(api_url, &["artifacts"])
}

fn download_artifact_url(api_url: &str, artifact_id: &str) -> Result<String> {
    api_v1_url(api_url, &["artifacts", artifact_id, "download-url"])
}

fn complete_upload_url(api_url: &str, artifact_id: &str) -> Result<String> {
    api_v1_url(api_url, &["artifacts", artifact_id, "confirm-upload"])
}

fn quota_url(api_url: &str) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("api")
        .push("v1")
        .push("quota");
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::hosted::credential_store::StaticCredentialStore;
    use crate::hosted::token_store::Tokens;
    use chrono::{Duration as ChronoDuration, Utc};

    #[test]
    fn api_client_refreshes_expired_token_and_initiates_upload() {
        let tokens = Tokens {
            access_token: "old_token".to_string(),
            refresh_token: Some("refresh_token_123".to_string()),
            expires_at: Some(Utc::now() - ChronoDuration::seconds(10)),
        };
        let credentials = StaticCredentialStore::tokens(tokens);

        let mut server = mockito::Server::new();
        let base = server.url();

        let _discovery = server
            .mock("GET", "/.well-known/openid-configuration")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"authorization_endpoint":"{base}/auth","token_endpoint":"{base}/token"}}"#
            ))
            .create();

        let _refresh = server
            .mock("POST", "/token")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"access_token":"new_token","expires_in":3600}"#)
            .create();

        let body = r#"{"artifact_id":"abc","url":"https://example.com/upload"}"#;
        let _init = server
            .mock("POST", "/api/v1/artifacts/upload-url")
            .match_header("authorization", "Bearer new_token")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            None,
        )
        .unwrap();

        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        let meta = UploadMeta {
            name: "test".to_string(),
            size_bytes: 100,
            sha256: "deadbeef".repeat(8),
            dtype: "F16".to_string(),
            storage_dtype: Some("F16".to_string()),
            num_tokens: 32,
            num_layers: 28,
            quantized: false,
            visibility: "private".to_string(),
        };
        let resp = client.init_upload(&meta).unwrap();

        assert_eq!(resp.artifact_id, "abc");
        assert_eq!(resp.upload_url, "https://example.com/upload");
    }

    #[test]
    fn api_client_uses_api_key_and_skips_oidc_refresh() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"artifact_id":"artifact-1","url":"https://example.com/upload"}"#;
        let _init = server
            .mock("POST", "/api/v1/artifacts/upload-url")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_test_key_123".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_test_key_123".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        let meta = UploadMeta {
            name: "api-key-model".to_string(),
            size_bytes: 200,
            sha256: "deadbeef".repeat(8),
            dtype: "F16".to_string(),
            storage_dtype: Some("F16".to_string()),
            num_tokens: 64,
            num_layers: 28,
            quantized: false,
            visibility: "private".to_string(),
        };
        let resp = client.init_upload(&meta).unwrap();
        assert_eq!(resp.artifact_id, "artifact-1");
        assert_eq!(resp.upload_url, "https://example.com/upload");
    }

    #[test]
    fn init_upload_url_uses_flat_artifacts_upload_url() {
        let url = upload_init_url("https://api.example.com").unwrap();
        assert_eq!(url, "https://api.example.com/api/v1/artifacts/upload-url");
    }

    #[test]
    fn delete_artifact_url_uses_flat_artifacts_id() {
        let url = delete_artifact_url("https://api.example.com", "abc").unwrap();
        assert_eq!(url, "https://api.example.com/api/v1/artifacts/abc");
    }

    #[test]
    fn list_artifacts_url_uses_flat_artifacts() {
        let url = list_artifacts_url("https://api.example.com").unwrap();
        assert_eq!(url, "https://api.example.com/api/v1/artifacts");
    }

    #[test]
    fn download_artifact_url_uses_flat_artifacts_id_download_url() {
        let url = download_artifact_url("https://api.example.com", "abc").unwrap();
        assert_eq!(
            url,
            "https://api.example.com/api/v1/artifacts/abc/download-url"
        );
    }

    #[test]
    fn quota_url_uses_v1_quota() {
        let url = quota_url("https://api.example.com").unwrap();
        assert!(url.ends_with("/api/v1/quota"));
    }

    #[test]
    fn api_client_gets_quota() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"customer_id":"cust-1","quota":{"organizations":10,"projects":100,"artifacts":1000,"storage_bytes":1099511627776},"used":{"organizations":1,"projects":2,"artifacts":3,"storage_bytes":1234}}"#;
        let _quota = server
            .mock("GET", "/api/v1/quota")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_test_key_123".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_test_key_123".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        let resp = client.get_quota().unwrap();
        assert_eq!(resp.customer_id, "cust-1");
        assert_eq!(resp.quota.organizations, 10);
        assert_eq!(resp.used.artifacts, 3);
        assert_eq!(resp.used.storage_bytes, 1234);
    }

    #[test]
    fn api_client_uses_api_key_to_delete_artifact() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let _delete = server
            .mock("DELETE", "/api/v1/artifacts/artifact-1")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(204)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_test_key_123".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_test_key_123".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        client.delete_artifact("artifact-1").unwrap();
    }

    #[test]
    fn api_client_completes_upload() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let _complete = server
            .mock("POST", "/api/v1/artifacts/artifact-2/confirm-upload")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(200)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_test_key_123".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_test_key_123".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        client
            .complete_upload("artifact-2", Some(&"ab".repeat(32)), Some(200))
            .unwrap();
    }

    #[test]
    fn api_client_lists_artifacts() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"artifacts":[{"id":"a1","model_name":"first","size_bytes":100,"dtype":"F16","num_tokens":8,"visibility":"private","public_url":null}],"total":1,"limit":50,"offset":0}"#;
        let _list = server
            .mock("GET", "/api/v1/artifacts")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_test_key_123".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_test_key_123".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        let artifacts = client.list_artifacts().unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].artifact_id, "a1");
        assert_eq!(artifacts[0].name, "first");
        // Fields the flat list response does not carry fall back to defaults.
        assert_eq!(artifacts[0].sha256, "");
        assert_eq!(artifacts[0].num_layers, 0);
    }

    #[test]
    fn api_client_gets_download_url() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"artifact_id":"a1","download_url":"https://example.com/download"}"#;
        let _download = server
            .mock("POST", "/api/v1/artifacts/a1/download-url")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(body)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_test_key_123".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_test_key_123".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();
        let resp = client.get_download_url("a1").unwrap();
        assert_eq!(resp.artifact_id, "a1");
        assert_eq!(resp.download_url, "https://example.com/download");
    }

    #[test]
    fn api_client_full_remote_round_trip_with_api_key() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let _init = server
            .mock("POST", "/api/v1/artifacts/upload-url")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"artifact_id":"rt1","url":"https://example.com/upload"}"#)
            .create();

        let _complete = server
            .mock("POST", "/api/v1/artifacts/rt1/confirm-upload")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .create();

        let _list = server
            .mock("GET", "/api/v1/artifacts")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"artifacts":[{"id":"rt1","model_name":"rt","size_bytes":1,"dtype":"F16","num_tokens":1,"visibility":"private","public_url":null}],"total":1,"limit":50,"offset":0}"#)
            .create();

        let _download = server
            .mock("POST", "/api/v1/artifacts/rt1/download-url")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"artifact_id":"rt1","download_url":"https://example.com/dl"}"#)
            .create();

        let _delete = server
            .mock("DELETE", "/api/v1/artifacts/rt1")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(204)
            .create();

        let cfg = Config::load(
            Some(base.clone()),
            Some(base.clone()),
            Some("test-client".to_string()),
            None,
            None,
            Some("kv_round_trip_key".to_string()),
        )
        .unwrap();

        let credentials = StaticCredentialStore::api_key("kv_round_trip_key".to_string());
        let mut client = ApiClient::new_testable(cfg, Box::new(credentials)).unwrap();

        let meta = UploadMeta {
            name: "rt".to_string(),
            size_bytes: 1,
            sha256: "aa".to_string(),
            dtype: "F16".to_string(),
            storage_dtype: None,
            num_tokens: 1,
            num_layers: 1,
            quantized: false,
            visibility: "private".to_string(),
        };
        let init = client.init_upload(&meta).unwrap();
        assert_eq!(init.artifact_id, "rt1");

        client
            .complete_upload(&init.artifact_id, Some("aa"), Some(1))
            .unwrap();

        let artifacts = client.list_artifacts().unwrap();
        assert!(artifacts.iter().any(|a| a.artifact_id == "rt1"));

        let download = client.get_download_url("rt1").unwrap();
        assert_eq!(download.download_url, "https://example.com/dl");

        client.delete_artifact("rt1").unwrap();
    }
}
