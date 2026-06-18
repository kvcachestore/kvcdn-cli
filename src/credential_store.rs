use crate::config::Config;
use crate::crypto::{self, Key};
use crate::token_store::{TokenStore, Tokens};
use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum Credentials {
    ApiKey(String),
    Oidc(Tokens),
}

pub trait CredentialStore: Send + Sync {
    fn load(&self) -> Result<Credentials>;
    fn save_tokens(&self, tokens: &Tokens) -> Result<()>;
}

pub struct FileCredentialStore {
    api_key: Option<String>,
    key: Key,
    store_dir: PathBuf,
}

impl FileCredentialStore {
    pub fn new(cfg: &Config) -> Result<Self> {
        let store_dir = TokenStore::default_dir()?;
        let key = crypto::load_or_create_key()?;
        let api_key = cfg
            .api_key
            .clone()
            .or_else(|| crate::api_key::load_stored_api_key().ok().flatten());
        Ok(Self {
            api_key,
            key,
            store_dir,
        })
    }
}

impl CredentialStore for FileCredentialStore {
    fn load(&self) -> Result<Credentials> {
        if let Some(api_key) = self.api_key.clone() {
            return Ok(Credentials::ApiKey(api_key));
        }
        let store = TokenStore::new(&self.store_dir);
        let tokens = store.load(&self.key)?.ok_or_else(|| {
            anyhow!("not logged in; run `kvcdn login` or `kvcdn api-key set <key>`")
        })?;
        Ok(Credentials::Oidc(tokens))
    }

    fn save_tokens(&self, tokens: &Tokens) -> Result<()> {
        let store = TokenStore::new(&self.store_dir);
        store
            .save(tokens, &self.key)
            .context("failed to save refreshed tokens")
    }
}

#[cfg(test)]
pub struct StaticCredentialStore {
    credentials: Credentials,
}

#[cfg(test)]
impl StaticCredentialStore {
    pub fn api_key(api_key: String) -> Self {
        Self {
            credentials: Credentials::ApiKey(api_key),
        }
    }

    pub fn tokens(tokens: Tokens) -> Self {
        Self {
            credentials: Credentials::Oidc(tokens),
        }
    }
}

#[cfg(test)]
impl CredentialStore for StaticCredentialStore {
    fn load(&self) -> Result<Credentials> {
        Ok(self.credentials.clone())
    }

    fn save_tokens(&self, _tokens: &Tokens) -> Result<()> {
        Ok(())
    }
}
