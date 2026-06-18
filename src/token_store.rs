use crate::crypto::{self, Key};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

pub struct TokenStore {
    base_dir: PathBuf,
}

impl TokenStore {
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
        }
    }

    pub fn default_dir() -> Result<PathBuf> {
        let config_dir = dirs::config_dir().context("failed to determine config directory")?;
        Ok(config_dir.join("kvcdn"))
    }

    pub fn load(&self, key: &Key) -> Result<Option<Tokens>> {
        let path = self.base_dir.join("credentials.enc");
        if !path.exists() {
            return Ok(None);
        }
        let ciphertext = fs::read(&path)
            .with_context(|| format!("failed to read credentials: {}", path.display()))?;
        let plaintext =
            crypto::decrypt(&ciphertext, key).context("failed to decrypt credentials")?;
        let tokens: Tokens =
            serde_json::from_slice(&plaintext).context("failed to parse credentials")?;
        Ok(Some(tokens))
    }

    pub fn save(&self, tokens: &Tokens, key: &Key) -> Result<()> {
        fs::create_dir_all(&self.base_dir).context("failed to create credentials directory")?;
        let plaintext = serde_json::to_vec(tokens).context("failed to serialize tokens")?;
        let ciphertext = crypto::encrypt(&plaintext, key).context("failed to encrypt tokens")?;
        let path = self.base_dir.join("credentials.enc");
        fs::write(&path, ciphertext)
            .with_context(|| format!("failed to write credentials: {}", path.display()))?;
        Ok(())
    }

    pub fn delete(&self) -> Result<()> {
        let path = self.base_dir.join("credentials.enc");
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove credentials: {}", path.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let tokens = Tokens {
            access_token: "acc".to_string(),
            refresh_token: Some("ref".to_string()),
            expires_at: Some(Utc::now() + chrono::Duration::seconds(3600)),
        };
        let store = TokenStore::new(dir.path());
        store.save(&tokens, &key).unwrap();
        let loaded = store.load(&key).unwrap().unwrap();
        assert_eq!(loaded.access_token, "acc");
        assert_eq!(loaded.refresh_token, Some("ref".to_string()));
        assert_eq!(loaded.expires_at, tokens.expires_at);
    }

    #[test]
    fn load_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let store = TokenStore::new(dir.path());
        assert!(store.load(&key).unwrap().is_none());
    }

    #[test]
    fn delete_removes_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let tokens = Tokens {
            access_token: "acc".to_string(),
            refresh_token: None,
            expires_at: None,
        };
        let store = TokenStore::new(dir.path());
        store.save(&tokens, &key).unwrap();
        assert!(store.load(&key).unwrap().is_some());
        store.delete().unwrap();
        assert!(store.load(&key).unwrap().is_none());
    }

    #[test]
    fn delete_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let store = TokenStore::new(dir.path());
        store.delete().unwrap();
        assert!(store.load(&crate::crypto::random_key()).unwrap().is_none());
    }

    #[test]
    fn load_with_wrong_key_fails() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let wrong_key = crate::crypto::random_key();
        let tokens = Tokens {
            access_token: "acc".to_string(),
            refresh_token: None,
            expires_at: None,
        };
        let store = TokenStore::new(dir.path());
        store.save(&tokens, &key).unwrap();
        assert!(store.load(&wrong_key).is_err());
    }

    #[test]
    fn load_corrupted_file_fails() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let store = TokenStore::new(dir.path());
        fs::create_dir_all(dir.path()).unwrap();
        let path = dir.path().join("credentials.enc");
        fs::write(&path, b"not valid encrypted data").unwrap();
        assert!(store.load(&key).is_err());
    }

    #[test]
    fn load_malformed_json_after_decrypt_fails() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let store = TokenStore::new(dir.path());
        fs::create_dir_all(dir.path()).unwrap();
        let plaintext = b"this is not valid json";
        let ciphertext = crate::crypto::encrypt(plaintext, &key).unwrap();
        let path = dir.path().join("credentials.enc");
        fs::write(&path, ciphertext).unwrap();
        assert!(store.load(&key).is_err());
    }
}
