use crate::config::Config;
use crate::crypto::{self, Key};
use crate::token_store::TokenStore;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

const API_KEY_FILE: &str = "api_key.enc";
const API_KEY_PREFIX: &str = "kv_";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredApiKey {
    pub key: String,
}

pub struct ApiKeyStore {
    base_dir: PathBuf,
}

impl ApiKeyStore {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    pub fn load(&self, key: &Key) -> Result<Option<String>> {
        let path = self.base_dir.join(API_KEY_FILE);
        if !path.exists() {
            return Ok(None);
        }
        let ciphertext = fs::read(&path)
            .with_context(|| format!("failed to read API key: {}", path.display()))?;
        let plaintext = crypto::decrypt(&ciphertext, key).context("failed to decrypt API key")?;
        let stored: StoredApiKey =
            serde_json::from_slice(&plaintext).context("failed to parse stored API key")?;
        Ok(Some(stored.key))
    }

    pub fn save(&self, api_key: &str, key: &Key) -> Result<()> {
        fs::create_dir_all(&self.base_dir).context("failed to create API key directory")?;
        let plaintext = serde_json::to_vec(&StoredApiKey {
            key: api_key.to_string(),
        })
        .context("failed to serialize API key")?;
        let ciphertext = crypto::encrypt(&plaintext, key).context("failed to encrypt API key")?;
        let path = self.base_dir.join(API_KEY_FILE);
        fs::write(&path, ciphertext)
            .with_context(|| format!("failed to write API key: {}", path.display()))?;
        Ok(())
    }

    pub fn delete(&self) -> Result<()> {
        let path = self.base_dir.join(API_KEY_FILE);
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove API key: {}", path.display()))?;
        }
        Ok(())
    }
}

fn looks_like_api_key(key: &str) -> bool {
    // Keep this in sync with the backend regex in
    // backend/src/stores/identity-verifier.ts.
    let Some(suffix) = key.strip_prefix(API_KEY_PREFIX) else {
        return false;
    };
    suffix.len() >= 32 && suffix.chars().all(|c| c.is_ascii_hexdigit())
}

pub fn set(api_key: String) -> Result<()> {
    if !looks_like_api_key(&api_key) {
        anyhow::bail!("API key must start with '{}'", API_KEY_PREFIX);
    }
    let key = crypto::load_or_create_key().context("failed to load encryption key")?;
    let dir = TokenStore::default_dir().context("failed to determine config directory")?;
    let store = ApiKeyStore::new(&dir);
    store
        .save(&api_key, &key)
        .context("failed to save API key")?;
    println!("API key saved.");
    Ok(())
}

pub fn verify(
    api_key: Option<String>,
    api_url: Option<String>,
    org: Option<String>,
    project: Option<String>,
) -> Result<()> {
    let key = api_key.or_else(|| load_stored_api_key().ok().flatten());
    let key = key.context("no API key provided; run `kvcdn api-key set <key>`")?;

    let cfg = Config::load(api_url, None, None, org, project, Some(key.clone()))?;
    let org = cfg.default_org.clone();
    let project = cfg.default_project.clone();
    let url = verify_url(&cfg.api_url)?;

    let client = crate::http::Client::default_timeout()?;
    let resp = client
        .post_empty_bearer(&url, &key)
        .with_context(|| format!("POST {url}"))?;

    if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        println!("API key is invalid or revoked.");
        return Ok(());
    }
    let resp = crate::http::Client::require_success(resp, &url)?;

    #[derive(serde::Deserialize)]
    struct VerifyResponse {
        customer_id: String,
    }
    let body: VerifyResponse = resp.json().context("parsing verify response")?;
    println!(
        "API key valid (customer_id: {}, org: {}, project: {})",
        body.customer_id, org, project
    );
    Ok(())
}

pub fn clear() -> Result<()> {
    let dir = TokenStore::default_dir().context("failed to determine config directory")?;
    let store = ApiKeyStore::new(&dir);
    store.delete()?;
    println!("API key removed.");
    Ok(())
}

pub fn load_stored_api_key() -> Result<Option<String>> {
    let key = crypto::load_or_create_key().context("failed to load encryption key")?;
    let dir = TokenStore::default_dir().context("failed to determine config directory")?;
    let store = ApiKeyStore::new(&dir);
    store.load(&key)
}

fn verify_url(api_url: &str) -> Result<String> {
    let mut url = url::Url::parse(api_url)?;
    url.path_segments_mut()
        .map_err(|_| anyhow::anyhow!("cannot modify URL path"))?
        .push("v1")
        .push("api-keys")
        .push("verify");
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_store_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let store = ApiKeyStore::new(dir.path());
        store
            .save("kv_a1b2c3d4e5f6789012345678a1b2c3d4", &key)
            .unwrap();
        let loaded = store.load(&key).unwrap().unwrap();
        assert_eq!(loaded, "kv_a1b2c3d4e5f6789012345678a1b2c3d4");
    }

    #[test]
    fn api_key_store_load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let store = ApiKeyStore::new(dir.path());
        assert!(store.load(&key).unwrap().is_none());
    }

    #[test]
    fn api_key_store_delete_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = ApiKeyStore::new(dir.path());
        store.delete().unwrap();
        assert!(store.load(&crate::crypto::random_key()).unwrap().is_none());
    }

    #[test]
    fn looks_like_api_key_requires_prefix_and_hex() {
        assert!(looks_like_api_key("kv_1234567890abcdef1234567890abcdef"));
        assert!(!looks_like_api_key("kv_123"));
        assert!(!looks_like_api_key("not-a-key"));
        assert!(!looks_like_api_key("kv_"));
        assert!(!looks_like_api_key("kv_ghijklmnopqrstuvwxyz1234567890"));
    }
}
