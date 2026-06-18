use crate::config::Config;
use crate::credential_store::{CredentialStore, Credentials, FileCredentialStore};
use crate::token_store::Tokens;
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
    pub artifact_id: String,
    pub name: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub dtype: String,
    pub storage_dtype: Option<String>,
    pub num_tokens: usize,
    pub num_layers: usize,
    pub quantized: bool,
    pub visibility: String,
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
    pub upload_url: String,
}

pub struct ApiClient {
    cfg: Config,
    credentials: Box<dyn CredentialStore>,
    client: crate::http::Client,
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
            client: crate::http::Client::default_timeout()?,
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

    pub fn init_upload(
        &mut self,
        org: &str,
        project: &str,
        meta: &UploadMeta,
    ) -> Result<UploadInitResponse> {
        let token = self.bearer_token()?;
        let url = upload_init_url(&self.cfg.api_url, org, project)?;
        let resp = self
            .client
            .post_json(&url, &token, meta)
            .with_context(|| format!("POST {url}"))?;
        let resp = crate::http::Client::require_auth_success(resp, &url)?;
        resp.json().context("parsing upload init response")
    }

    pub fn complete_upload(&mut self, org: &str, project: &str, artifact_id: &str) -> Result<()> {
        let token = self.bearer_token()?;
        let url = complete_upload_url(&self.cfg.api_url, org, project, artifact_id)?;
        let resp = self
            .client
            .post_empty_bearer(&url, &token)
            .with_context(|| format!("POST {url}"))?;
        crate::http::Client::require_auth_success(resp, &url)?;
        Ok(())
    }

    pub fn delete_artifact(&mut self, artifact_id: &str, org: &str, project: &str) -> Result<()> {
        let token = self.bearer_token()?;
        let url = delete_artifact_url(&self.cfg.api_url, org, project, artifact_id)?;
        let resp = self
            .client
            .delete_bearer(&url, &token)
            .with_context(|| format!("DELETE {url}"))?;
        crate::http::Client::require_auth_success(resp, &url)?;
        Ok(())
    }

    pub fn list_artifacts(&mut self, org: &str, project: &str) -> Result<Vec<ArtifactMeta>> {
        let token = self.bearer_token()?;
        let url = list_artifacts_url(&self.cfg.api_url, org, project)?;
        let resp = self
            .client
            .get_bearer(&url, &token)
            .with_context(|| format!("GET {url}"))?;
        let resp = crate::http::Client::require_auth_success(resp, &url)?;
        let body: ListArtifactsResponse = resp.json().context("parsing artifact list response")?;
        Ok(body.artifacts)
    }

    pub fn get_download_url(
        &mut self,
        artifact_id: &str,
        org: &str,
        project: &str,
    ) -> Result<DownloadInitResponse> {
        let token = self.bearer_token()?;
        let url = download_artifact_url(&self.cfg.api_url, org, project, artifact_id)?;
        let resp = self
            .client
            .get_bearer(&url, &token)
            .with_context(|| format!("GET {url}"))?;
        let resp = crate::http::Client::require_auth_success(resp, &url)?;
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
        let resp = crate::http::Client::require_auth_success(resp, &url)?;
        resp.json().context("parsing quota response")
    }

    pub fn whoami_customer_id(&mut self) -> Result<String> {
        use crate::http::Client;
        let token = self.bearer_token()?;
        let mut url = url::Url::parse(&self.cfg.api_url)?;
        url.path_segments_mut()
            .map_err(|_| anyhow!("cannot modify URL path"))?
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

        let oidc = crate::oidc::discover(&self.cfg.issuer_url)?;
        let mut params = std::collections::HashMap::new();
        params.insert("grant_type", "refresh_token");
        params.insert("client_id", self.cfg.client_id.as_str());
        params.insert("refresh_token", refresh.as_str());

        let resp = self
            .client
            .post_form(&oidc.token_endpoint, &params)
            .context("refresh token request")?;
        let resp = crate::http::Client::require_success(resp, &oidc.token_endpoint)?;

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

fn upload_init_url(api_url: &str, org: &str, project: &str) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("orgs")
        .push(org)
        .push("projects")
        .push(project)
        .push("artifacts");
    Ok(url.to_string())
}

fn delete_artifact_url(
    api_url: &str,
    org: &str,
    project: &str,
    artifact_id: &str,
) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("orgs")
        .push(org)
        .push("projects")
        .push(project)
        .push("artifacts")
        .push(artifact_id);
    Ok(url.to_string())
}

fn list_artifacts_url(api_url: &str, org: &str, project: &str) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("orgs")
        .push(org)
        .push("projects")
        .push(project)
        .push("artifacts");
    Ok(url.to_string())
}

fn download_artifact_url(
    api_url: &str,
    org: &str,
    project: &str,
    artifact_id: &str,
) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("orgs")
        .push(org)
        .push("projects")
        .push(project)
        .push("artifacts")
        .push(artifact_id)
        .push("download");
    Ok(url.to_string())
}

fn complete_upload_url(
    api_url: &str,
    org: &str,
    project: &str,
    artifact_id: &str,
) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("orgs")
        .push(org)
        .push("projects")
        .push(project)
        .push("artifacts")
        .push(artifact_id)
        .push("complete");
    Ok(url.to_string())
}

fn quota_url(api_url: &str) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("quota");
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::credential_store::StaticCredentialStore;
    use crate::token_store::Tokens;
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

        let body = r#"{"artifact_id":"abc","upload_url":"https://example.com/upload"}"#;
        let _init = server
            .mock("POST", "/v1/orgs/default/projects/default/artifacts")
            .match_header("authorization", "Bearer new_token")
            .with_status(201)
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
        let resp = client.init_upload("default", "default", &meta).unwrap();

        assert_eq!(resp.artifact_id, "abc");
        assert_eq!(resp.upload_url, "https://example.com/upload");
    }

    #[test]
    fn api_client_uses_api_key_and_skips_oidc_refresh() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"artifact_id":"artifact-1","upload_url":"https://example.com/upload"}"#;
        let _init = server
            .mock("POST", "/v1/orgs/default/projects/default/artifacts")
            .match_header("authorization", "Bearer kv_test_key_123")
            .with_status(201)
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
        let resp = client.init_upload("default", "default", &meta).unwrap();
        assert_eq!(resp.artifact_id, "artifact-1");
        assert_eq!(resp.upload_url, "https://example.com/upload");
    }

    #[test]
    fn init_upload_url_uses_v1_orgs_projects_artifacts() {
        let url = upload_init_url("https://api.example.com", "default", "default").unwrap();
        assert!(url.contains("/v1/orgs/default/projects/default/artifacts"));
    }

    #[test]
    fn delete_artifact_url_uses_v1_orgs_projects_artifacts_id() {
        let url = delete_artifact_url("https://api.example.com", "default", "acme", "abc").unwrap();
        assert!(url.contains("/v1/orgs/default/projects/acme/artifacts/abc"));
    }

    #[test]
    fn list_artifacts_url_uses_v1_orgs_projects_artifacts() {
        let url = list_artifacts_url("https://api.example.com", "default", "acme").unwrap();
        assert!(url.contains("/v1/orgs/default/projects/acme/artifacts"));
    }

    #[test]
    fn download_artifact_url_uses_v1_orgs_projects_artifacts_id_download() {
        let url =
            download_artifact_url("https://api.example.com", "default", "acme", "abc").unwrap();
        assert!(url.contains("/v1/orgs/default/projects/acme/artifacts/abc/download"));
    }

    #[test]
    fn quota_url_uses_v1_quota() {
        let url = quota_url("https://api.example.com").unwrap();
        assert!(url.ends_with("/v1/quota"));
    }

    #[test]
    fn api_client_gets_quota() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"customer_id":"cust-1","quota":{"organizations":10,"projects":100,"artifacts":1000,"storage_bytes":1099511627776},"used":{"organizations":1,"projects":2,"artifacts":3,"storage_bytes":1234}}"#;
        let _quota = server
            .mock("GET", "/v1/quota")
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
            .mock(
                "DELETE",
                "/v1/orgs/default/projects/acme/artifacts/artifact-1",
            )
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
        client
            .delete_artifact("artifact-1", "default", "acme")
            .unwrap();
    }

    #[test]
    fn api_client_completes_upload() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let _complete = server
            .mock(
                "POST",
                "/v1/orgs/acme/projects/widgets/artifacts/artifact-2/complete",
            )
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
            .complete_upload("acme", "widgets", "artifact-2")
            .unwrap();
    }

    #[test]
    fn api_client_lists_artifacts() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"artifacts":[{"artifact_id":"a1","name":"first","size_bytes":100,"sha256":"deadbeef","dtype":"F16","storage_dtype":null,"num_tokens":8,"num_layers":2,"quantized":false,"visibility":"private","created_at":"2024-01-01T00:00:00Z"}]}"#;
        let _list = server
            .mock("GET", "/v1/orgs/acme/projects/widgets/artifacts")
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
        let artifacts = client.list_artifacts("acme", "widgets").unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].artifact_id, "a1");
        assert_eq!(artifacts[0].name, "first");
    }

    #[test]
    fn api_client_gets_download_url() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let body = r#"{"artifact_id":"a1","download_url":"https://example.com/download"}"#;
        let _download = server
            .mock(
                "GET",
                "/v1/orgs/acme/projects/widgets/artifacts/a1/download",
            )
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
        let resp = client.get_download_url("a1", "acme", "widgets").unwrap();
        assert_eq!(resp.artifact_id, "a1");
        assert_eq!(resp.download_url, "https://example.com/download");
    }

    #[test]
    fn api_client_full_remote_round_trip_with_api_key() {
        let mut server = mockito::Server::new();
        let base = server.url();

        let _init = server
            .mock("POST", "/v1/orgs/acme/projects/widgets/artifacts")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(201)
            .with_header("content-type", "application/json")
            .with_body(r#"{"artifact_id":"rt1","upload_url":"https://example.com/upload"}"#)
            .create();

        let _complete = server
            .mock(
                "POST",
                "/v1/orgs/acme/projects/widgets/artifacts/rt1/complete",
            )
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .create();

        let _list = server
            .mock("GET", "/v1/orgs/acme/projects/widgets/artifacts")
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"artifacts":[{"artifact_id":"rt1","name":"rt","size_bytes":1,"sha256":"aa","dtype":"F16","storage_dtype":null,"num_tokens":1,"num_layers":1,"quantized":false,"visibility":"private","created_at":"2024-01-01T00:00:00Z"}]}"#)
            .create();

        let _download = server
            .mock(
                "GET",
                "/v1/orgs/acme/projects/widgets/artifacts/rt1/download",
            )
            .match_header("authorization", "Bearer kv_round_trip_key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"artifact_id":"rt1","download_url":"https://example.com/dl"}"#)
            .create();

        let _delete = server
            .mock("DELETE", "/v1/orgs/acme/projects/widgets/artifacts/rt1")
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
        let init = client.init_upload("acme", "widgets", &meta).unwrap();
        assert_eq!(init.artifact_id, "rt1");

        client
            .complete_upload("acme", "widgets", &init.artifact_id)
            .unwrap();

        let artifacts = client.list_artifacts("acme", "widgets").unwrap();
        assert!(artifacts.iter().any(|a| a.artifact_id == "rt1"));

        let download = client.get_download_url("rt1", "acme", "widgets").unwrap();
        assert_eq!(download.download_url, "https://example.com/dl");

        client.delete_artifact("rt1", "acme", "widgets").unwrap();
    }
}
