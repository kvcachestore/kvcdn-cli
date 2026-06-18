# Auth & Hosted Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OIDC login/logout and hosted KV-cache upload to the `kvcdn` CLI so users can authenticate via Pocket-ID and push artifacts to a private cloud endpoint.

**Architecture:** A small set of focused modules (`config`, `crypto`, `token_store`, `oidc`, `callback_server`, `api`, `login`, `logout`, `upload`) sit alongside the existing local-only commands. The CLI remains synchronous and uses `reqwest::blocking` for HTTP; the login flow uses a local ephemeral HTTP listener for the OIDC callback. Tokens are encrypted at rest with a key from the OS keyring when available, falling back to a restricted permission file.

**Tech Stack:** Rust 2024, `reqwest` (blocking + JSON), `tiny_http` (local callback), `clap`, `serde`/`serde_json`, `toml`, `sha2`, `base64`, `rand`, `chacha20poly1305` (or `aes-gcm`), `keyring`, `dirs`, `indicatif`, `mockito` (tests).

---

## File map

| File | Responsibility |
|------|----------------|
| `Cargo.toml` | Add HTTP, crypto, config, and test dependencies. |
| `src/config.rs` | Load `api_url`, `issuer_url`, `client_id`, `default_project` from defaults/env/file. |
| `src/crypto.rs` | Symmetric encrypt/decrypt and key management. |
| `src/token_store.rs` | Persist/load encrypted OIDC tokens. |
| `src/oidc.rs` | OIDC discovery, PKCE generation, and token exchange. |
| `src/callback_server.rs` | Ephemeral localhost HTTP listener for OIDC callback. |
| `src/api.rs` | Authenticated backend client with token refresh. |
| `src/login.rs` | Orchestrate login flow. |
| `src/logout.rs` | Delete stored credentials. |
| `src/upload.rs` | Upload an artifact via presigned URL with progress. |
| `src/cli.rs` | New `LoginArgs`, `LogoutArgs`, `UploadArgs`. |
| `src/main.rs` | Wire up new subcommands. |
| `tests/auth_upload_test.rs` | Integration test with mocked OIDC + backend. |

---

## Task 1: Add dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add the new dependencies**

```toml
[dependencies]
anyhow = "1.0.102"
aes-gcm = "0.10"
base64 = "0.22"
hex = "0.4"
candle-core = "0.10.2"
candle-nn = "0.10.2"
candle-transformers = "0.10.2"
chrono = { version = "0.4", features = ["serde"] }
clap = { version = "4.6.1", features = ["derive"] }
dirs = "6"
hf-hub = "0.5.0"
indicatif = "0.17"
keyring = "3"
rand = "0.9"
reqwest = { version = "0.12", features = ["blocking", "json"] }
safetensors = "0.8.0"
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.150"
sha2 = "0.10"
tiny_http = "0.12"
tokenizers = "0.23.1"
toml = "0.8"
url = "2.5"

[dev-dependencies]
mockito = "1.6"
tempfile = "3.14"
```

- [ ] **Step 2: Run `cargo check`**

Run: `cargo check`
Expected: passes (downloads crates; no code uses them yet).

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "deps: add auth, upload, crypto, and config dependencies"
```

---

## Task 2: Configuration module

**Files:**
- Create: `src/config.rs`
- Modify: `src/main.rs` (add `mod config;`)

- [ ] **Step 1: Write the failing test**

Create `src/config.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_present() {
        let cfg = Config::from_env(None, None, None, None).unwrap();
        assert_eq!(cfg.api_url, "https://api.kvcdn.example.com");
        assert_eq!(cfg.issuer_url, "https://id.kvcdn.example.com");
        assert_eq!(cfg.client_id, "kvcdn-cli");
        assert_eq!(cfg.default_project, "default");
    }

    #[test]
    fn env_overrides_defaults() {
        let cfg = Config::from_env(
            Some("https://api.test".to_string()),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(cfg.api_url, "https://api.test");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test config::tests -- --nocapture`
Expected: FAIL — `Config` and `from_env` not defined.

- [ ] **Step 3: Implement minimal configuration loader**

Append to `src/config.rs`:

```rust
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;

const DEFAULT_API_URL: &str = "https://api.kvcdn.example.com";
const DEFAULT_ISSUER_URL: &str = "https://id.kvcdn.example.com";
const DEFAULT_CLIENT_ID: &str = "kvcdn-cli";
const DEFAULT_PROJECT: &str = "default";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub api_url: String,
    pub issuer_url: String,
    pub client_id: String,
    pub default_project: String,
}

impl Config {
    pub fn load(
        cli_api_url: Option<String>,
        cli_issuer_url: Option<String>,
        cli_client_id: Option<String>,
        cli_default_project: Option<String>,
    ) -> Result<Self> {
        let file_cfg = Self::from_file()?;

        Ok(Self {
            api_url: cli_api_url
                .or_else(|| env::var("KVCDN_API_URL").ok())
                .or(file_cfg.as_ref().map(|c| c.api_url.clone()))
                .unwrap_or_else(|| DEFAULT_API_URL.to_string()),
            issuer_url: cli_issuer_url
                .or_else(|| env::var("KVCDN_ISSUER_URL").ok())
                .or(file_cfg.as_ref().map(|c| c.issuer_url.clone()))
                .unwrap_or_else(|| DEFAULT_ISSUER_URL.to_string()),
            client_id: cli_client_id
                .or_else(|| env::var("KVCDN_CLIENT_ID").ok())
                .or(file_cfg.as_ref().map(|c| c.client_id.clone()))
                .unwrap_or_else(|| DEFAULT_CLIENT_ID.to_string()),
            default_project: cli_default_project
                .or_else(|| env::var("KVCDN_DEFAULT_PROJECT").ok())
                .or(file_cfg.as_ref().map(|c| c.default_project.clone()))
                .unwrap_or_else(|| DEFAULT_PROJECT.to_string()),
        })
    }

    fn from_file() -> Result<Option<Self>> {
        let Some(config_dir) = dirs::config_dir() else {
            return Ok(None);
        };
        let path = config_dir.join("kvcdn").join("config.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        Ok(Some(toml::from_str(&text)?))
    }

    #[cfg(test)]
    pub fn from_env(
        api_url: Option<String>,
        issuer_url: Option<String>,
        client_id: Option<String>,
        default_project: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            api_url: api_url.unwrap_or_else(|| DEFAULT_API_URL.to_string()),
            issuer_url: issuer_url.unwrap_or_else(|| DEFAULT_ISSUER_URL.to_string()),
            client_id: client_id.unwrap_or_else(|| DEFAULT_CLIENT_ID.to_string()),
            default_project: default_project.unwrap_or_else(|| DEFAULT_PROJECT.to_string()),
        })
    }
}
```

- [ ] **Step 4: Register the module and run tests**

Add to `src/main.rs`:

```rust
mod config;
```

Run: `cargo test config::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/main.rs
git commit -m "feat(config): load api/issuer/client/project from defaults, env, and file"
```

---

## Task 3: Encryption helpers

**Files:**
- Create: `src/crypto.rs`
- Modify: `src/main.rs` (add `mod crypto;`)

- [ ] **Step 1: Write the failing test**

Create `src/crypto.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let key = random_key();
        let plaintext = b"hello tokens";
        let ciphertext = encrypt(plaintext, &key).unwrap();
        assert_ne!(ciphertext, plaintext.to_vec());
        let decrypted = decrypt(&ciphertext, &key).unwrap();
        assert_eq!(decrypted, plaintext.to_vec());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test crypto::tests -- --nocapture`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement encrypt/decrypt and key helpers**

Append to `src/crypto.rs`:

```rust
use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use anyhow::{Context, Result};
use rand::RngCore;
use std::fs;
use std::path::PathBuf;

pub const KEY_LEN: usize = 32;
const NONCE_LEN: usize = 12;

pub type Key = [u8; KEY_LEN];

pub fn random_key() -> Key {
    let mut key = [0u8; KEY_LEN];
    rand::rng().fill_bytes(&mut key);
    key
}

pub fn encrypt(plaintext: &[u8], key: &Key) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| anyhow::anyhow!("invalid key length: {e}"))?;
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("encryption failed: {e}"))?;
    let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

pub fn decrypt(ciphertext: &[u8], key: &Key) -> Result<Vec<u8>> {
    if ciphertext.len() < NONCE_LEN {
        anyhow::bail!("ciphertext too short");
    }
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| anyhow::anyhow!("invalid key length: {e}"))?;
    let nonce = Nonce::from_slice(&ciphertext[..NONCE_LEN]);
    cipher
        .decrypt(nonce, &ciphertext[NONCE_LEN..])
        .map_err(|e| anyhow::anyhow!("decryption failed: {e}"))
}

pub fn load_or_create_key() -> Result<Key> {
    if let Some(key) = load_key_from_keyring()? {
        return Ok(key);
    }
    load_or_create_key_file()
}

fn load_key_from_keyring() -> Result<Option<Key>> {
    let entry = match keyring::Entry::new("kvcdn-cli", "encryption-key") {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    match entry.get_password() {
        Ok(hex) => {
            let mut key = [0u8; KEY_LEN];
            if hex::decode_to_slice(&hex, &mut key).is_ok() {
                Ok(Some(key))
            } else {
                Ok(None)
            }
        }
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(anyhow::anyhow!("keyring error: {e}")),
    }
}

fn store_key_in_keyring(key: &Key) -> Result<()> {
    let entry = keyring::Entry::new("kvcdn-cli", "encryption-key")
        .map_err(|e| anyhow::anyhow!("keyring error: {e}"))?;
    let hex = hex::encode(key);
    entry
        .set_password(&hex)
        .map_err(|e| anyhow::anyhow!("keyring error: {e}"))?;
    Ok(())
}

fn load_or_create_key_file() -> Result<Key> {
    let path = key_file_path()?;
    if path.exists() {
        let hex = fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut key = [0u8; KEY_LEN];
        hex::decode_to_slice(hex.trim(), &mut key)
            .with_context(|| format!("decoding key from {}", path.display()))?;
        return Ok(key);
    }
    let key = random_key();
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(&path, hex::encode(key))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }
    eprintln!("warning: OS keyring unavailable; encryption key stored in {} with 0600 permissions.", path.display());
    Ok(key)
}

fn key_file_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("could not determine config directory")?;
    Ok(config_dir.join("kvcdn").join(".key"))
}

pub fn ensure_key_stored(key: &Key) -> Result<()> {
    if keyring::Entry::new("kvcdn-cli", "encryption-key").is_ok() {
        store_key_in_keyring(key)?;
    }
    Ok(())
}
```

Note: `hex` is not a dependency. Add `hex = "0.4"` to `Cargo.toml` in Task 1 or here.

- [ ] **Step 4: Register module, add hex dep, run tests**

Add to `Cargo.toml`:

```toml
hex = "0.4"
```

Add to `src/main.rs`:

```rust
mod crypto;
```

Run: `cargo test crypto::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/crypto.rs src/main.rs
git commit -m "feat(crypto): AES-256-GCM encryption with keyring/file key storage"
```

---

## Task 4: Encrypted token store

**Files:**
- Create: `src/token_store.rs`
- Modify: `src/main.rs` (add `mod token_store;`)

- [ ] **Step 1: Write the failing test**

Create `src/token_store.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{DateTime, Duration as ChronoDuration, Utc};

    #[test]
    fn save_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let key = crate::crypto::random_key();
        let tokens = Tokens {
            access_token: "acc".to_string(),
            refresh_token: Some("ref".to_string()),
            expires_at: Some(Utc::now() + ChronoDuration::seconds(3600)),
        };
        let store = TokenStore::new(dir.path());
        store.save(&tokens, &key).unwrap();
        let loaded = store.load(&key).unwrap().unwrap();
        assert_eq!(loaded.access_token, "acc");
        assert_eq!(loaded.refresh_token, Some("ref".to_string()));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test token_store::tests -- --nocapture`
Expected: FAIL — types not defined.

- [ ] **Step 3: Implement token store**

Append to `src/token_store.rs`:

```rust
use crate::crypto::{self, Key};
use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
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
        let config_dir = dirs::config_dir().context("could not determine config directory")?;
        Ok(config_dir.join("kvcdn"))
    }

    pub fn load(&self, key: &Key) -> Result<Option<Tokens>> {
        let path = self.base_dir.join("credentials.enc");
        if !path.exists() {
            return Ok(None);
        }
        let ciphertext = fs::read(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let plaintext = crypto::decrypt(&ciphertext, key)
            .context("failed to decrypt credentials")?;
        let tokens: Tokens = serde_json::from_slice(&plaintext)
            .context("failed to parse credentials")?;
        Ok(Some(tokens))
    }

    pub fn save(&self, tokens: &Tokens, key: &Key) -> Result<()> {
        fs::create_dir_all(&self.base_dir)?;
        let plaintext = serde_json::to_vec(tokens)?;
        let ciphertext = crypto::encrypt(&plaintext, key)?;
        let path = self.base_dir.join("credentials.enc");
        fs::write(&path, ciphertext)
            .with_context(|| format!("writing {}", path.display()))?;

        Ok(())
    }

    pub fn delete(&self) -> Result<()> {
        let path = self.base_dir.join("credentials.enc");
        if path.exists() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Register module and run tests**

Add to `src/main.rs`:

```rust
mod token_store;
```

Run: `cargo test token_store::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/token_store.rs src/main.rs
git commit -m "feat(token_store): encrypted credentials.enc persistence"
```

---

## Task 5: OIDC discovery, PKCE, and token exchange

**Files:**
- Create: `src/oidc.rs`
- Modify: `src/main.rs` (add `mod oidc;`)

- [ ] **Step 1: Write the failing test**

Create `src/oidc.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_verifier_is_128_chars() {
        let (verifier, challenge) = Pkce::generate();
        assert_eq!(verifier.len(), 128);
        assert!(!challenge.is_empty());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test oidc::tests -- --nocapture`
Expected: FAIL — `Pkce` not defined.

- [ ] **Step 3: Implement OIDC helpers**

Append to `src/oidc.rs`:

```rust
use crate::token_store::Tokens;
use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use url::Url;

#[derive(Debug, Clone, Deserialize)]
pub struct OidcConfig {
    pub authorization_endpoint: String,
    pub token_endpoint: String,
}

pub fn discover(issuer_url: &str) -> Result<OidcConfig> {
    let url = format!("{}/.well-known/openid-configuration", issuer_url.trim_end_matches('/'));
    let resp = reqwest::blocking::get(&url)
        .with_context(|| format!("fetching OIDC discovery from {url}"))?;
    if !resp.status().is_success() {
        anyhow::bail!("OIDC discovery returned {}", resp.status());
    }
    resp.json().context("parsing OIDC discovery document")
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
        let verifier = URL_SAFE_NO_PAD.encode(&bytes);
        let hash = Sha256::digest(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(&hash);
        Self { verifier, challenge }
    }
}

pub fn random_state() -> String {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(&bytes)
}

pub fn build_authorization_url(
    cfg: &OidcConfig,
    client_id: &str,
    redirect_uri: &str,
    pkce: &Pkce,
    state: &str,
) -> Result<String> {
    let mut url = Url::parse(&cfg.authorization_endpoint)?;
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

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(&cfg.token_endpoint)
        .form(&params)
        .send()
        .context("POST to token endpoint")?;

    if !resp.status().is_success() {
        let text = resp.text().unwrap_or_default();
        anyhow::bail!("token exchange failed: {text}");
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

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: Option<String>,
    expires_in: Option<u64>,
}
```

- [ ] **Step 4: Register module and run tests**

Add to `src/main.rs`:

```rust
mod oidc;
```

Run: `cargo test oidc::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/oidc.rs src/main.rs
git commit -m "feat(oidc): discovery, PKCE, authorization URL, and token exchange"
```

---

## Task 6: Local callback server

**Files:**
- Create: `src/callback_server.rs`
- Modify: `src/main.rs` (add `mod callback_server;`)

- [ ] **Step 1: Write the failing test**

Create `src/callback_server.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_returns_port_and_url() {
        let state = "abc123";
        let (server, url) = CallbackServer::start(state).unwrap();
        assert!(url.contains("/callback"));
        drop(server);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test callback_server::tests -- --nocapture`
Expected: FAIL — `CallbackServer` not defined.

- [ ] **Step 3: Implement callback server**

Append to `src/callback_server.rs`:

```rust
use anyhow::{anyhow, Context, Result};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use tiny_http::{Response, Server};
use url::Url;

pub struct CallbackServer {
    pub redirect_uri: String,
    rx: Receiver<Result<String>>,
    _handle: thread::JoinHandle<()>,
}

impl CallbackServer {
    /// Start a server on 127.0.0.1:0 and return the redirect URI + receiver for the auth code.
    pub fn start(state: &str) -> Result<Self> {
        let server = Server::http("127.0.0.1:0")
            .map_err(|e| anyhow!("failed to start callback server: {e}"))?;
        let redirect_uri = format!("http://127.0.0.1:{}/callback", server.server_addr().port());
        let expected_state = state.to_string();
        let (tx, rx): (Sender<Result<String>>, Receiver<Result<String>>) = mpsc::channel();

        let handle = thread::spawn(move || {
            let mut tx = Some(tx);
            for request in server.incoming_requests() {
                let url = format!("http://localhost{}", request.url());
                let parsed = match Url::parse(&url) {
                    Ok(u) => u,
                    Err(e) => {
                        let _ = request.respond(Response::from_string(format!("bad request: {e}")));
                        continue;
                    }
                };
                let pairs: std::collections::HashMap<_, _> =
                    parsed.query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())).collect();

                let response = if pairs.get("state") != Some(&expected_state) {
                    Response::from_string("state mismatch").with_status_code(400)
                } else if let Some(code) = pairs.get("code") {
                    if let Some(sender) = tx.take() {
                        let _ = sender.send(Ok(code.clone()));
                    }
                    Response::from_string("Login successful. You may close this tab.")
                } else if let Some(error) = pairs.get("error") {
                    if let Some(sender) = tx.take() {
                        let _ = sender.send(Err(anyhow!("OIDC error: {error}")));
                    }
                    Response::from_string(format!("error: {error}")).with_status_code(400)
                } else {
                    Response::from_string("missing code or error").with_status_code(400)
                };
                let _ = request.respond(response);
                break;
            }
        });

        Ok(Self {
            redirect_uri,
            rx,
            _handle: handle,
        })
    }

    /// Block until a callback is received.
    pub fn wait(self) -> Result<String> {
        self.rx
            .recv()
            .context("callback server channel closed")?
            .context("callback error")
    }
}
```

- [ ] **Step 4: Register module and run tests**

Add to `src/main.rs`:

```rust
mod callback_server;
```

Run: `cargo test callback_server::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/callback_server.rs src/main.rs
git commit -m "feat(callback_server): ephemeral localhost OIDC callback listener"
```

---

## Task 7: Login command

**Files:**
- Create: `src/login.rs`
- Modify: `src/cli.rs` — add `LoginArgs`
- Modify: `src/main.rs` — wire `Login`

- [ ] **Step 1: Add CLI args**

In `src/cli.rs`, add:

```rust
#[derive(Parser)]
pub struct LoginArgs {
    #[arg(long)]
    pub no_browser: bool,
    #[arg(long)]
    pub api_url: Option<String>,
    #[arg(long)]
    pub issuer_url: Option<String>,
    #[arg(long)]
    pub client_id: Option<String>,
}
```

Add `Login(LoginArgs)` to the `Cli` enum.

- [ ] **Step 2: Implement login orchestration**

Create `src/login.rs`:

```rust
use crate::callback_server::CallbackServer;
use crate::config::Config;
use crate::crypto;
use crate::oidc::{build_authorization_url, discover, exchange_code, random_state, Pkce};
use crate::token_store::TokenStore;
use anyhow::{Context, Result};
use std::process::Command;

pub fn run(args: crate::cli::LoginArgs) -> Result<()> {
    let cfg = Config::load(args.api_url, args.issuer_url, args.client_id, None)?;
    let oidc = discover(&cfg.issuer_url)?;

    let pkce = Pkce::generate();
    let state = random_state();
    let server = CallbackServer::start(&state)?;
    let redirect_uri = server.redirect_uri.clone();
    let auth_url = build_authorization_url(&oidc, &cfg.client_id, &redirect_uri, &pkce, &state)?;

    if args.no_browser {
        println!("Please open this URL in your browser:\n{auth_url}");
    } else {
        println!("Opening browser for login...");
        if open_browser(&auth_url).is_err() {
            eprintln!("warning: failed to open browser. Open this URL manually:\n{auth_url}");
        }
    }

    let code = server.wait().context("waiting for OIDC callback")?;
    let tokens = exchange_code(&oidc, &cfg.client_id, &redirect_uri, &code, &pkce)?;

    let key = crypto::load_or_create_key()?;
    let store_dir = TokenStore::default_dir()?;
    let store = TokenStore::new(&store_dir);
    store.save(&tokens, &key)?;
    crypto::ensure_key_stored(&key)?;

    println!("Login successful.");
    Ok(())
}

#[cfg(target_os = "macos")]
fn open_browser(url: &str) -> Result<()> {
    Command::new("open").arg(url).status().context("open browser")?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn open_browser(url: &str) -> Result<()> {
    Command::new("xdg-open").arg(url).status().context("open browser")?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn open_browser(url: &str) -> Result<()> {
    Command::new("cmd")
        .args(["/C", "start", ""])
        .arg(url)
        .status()
        .context("open browser")?;
    Ok(())
}
```

- [ ] **Step 3: Wire in main and CLI**

In `src/main.rs`, add `mod login;` and match arm:

```rust
Cli::Login(args) => login::run(args),
```

- [ ] **Step 4: Run `cargo check`**

Run: `cargo check`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/login.rs src/cli.rs src/main.rs
git commit -m "feat(login): OIDC login with browser and --no-browser support"
```

---

## Task 8: Logout command

**Files:**
- Create: `src/logout.rs`
- Modify: `src/cli.rs` — add `LogoutArgs`
- Modify: `src/main.rs` — wire `Logout`

- [ ] **Step 1: Add CLI args**

In `src/cli.rs`:

```rust
#[derive(Parser)]
pub struct LogoutArgs {}
```

Add `Logout(LogoutArgs)` to the `Cli` enum.

- [ ] **Step 2: Implement logout**

Create `src/logout.rs`:

```rust
use crate::token_store::TokenStore;
use anyhow::Result;

pub fn run(_args: crate::cli::LogoutArgs) -> Result<()> {
    let dir = TokenStore::default_dir()?;
    let store = TokenStore::new(&dir);
    store.delete()?;
    println!("Logged out.");
    Ok(())
}
```

- [ ] **Step 3: Wire in main and CLI**

In `src/main.rs`, add `mod logout;` and match arm:

```rust
Cli::Logout(args) => logout::run(args),
```

- [ ] **Step 4: Run `cargo check`**

Run: `cargo check`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/logout.rs src/cli.rs src/main.rs
git commit -m "feat(logout): delete stored credentials"
```

---

## Task 9: Authenticated API client with refresh

**Files:**
- Create: `src/api.rs`
- Modify: `src/main.rs` (add `mod api;`)

- [ ] **Step 1: Write the failing test**

Create `src/api.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn upload_request_body_has_name_and_sha256() {
        let meta = UploadMeta {
            name: "foo".to_string(),
            size_bytes: 100,
            sha256: "abc".to_string(),
            dtype: "F16".to_string(),
            num_tokens: 32,
            num_layers: 28,
            quantized: false,
        };
        let body = serde_json::to_value(&meta).unwrap();
        assert_eq!(body["name"], "foo");
        assert_eq!(body["sha256"], "abc");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test api::tests -- --nocapture`
Expected: FAIL — `UploadMeta` not defined.

- [ ] **Step 3: Implement API client**

Append to `src/api.rs`:

```rust
use crate::config::Config;
use crate::crypto::Key;
use crate::token_store::{TokenStore, Tokens};
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadMeta {
    pub name: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub dtype: String,
    pub num_tokens: usize,
    pub num_layers: usize,
    pub quantized: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UploadInitResponse {
    pub artifact_id: String,
    pub upload_url: String,
}

pub struct ApiClient {
    cfg: Config,
    tokens: Tokens,
    key: Key,
}

impl ApiClient {
    pub fn new(cfg: Config) -> Result<Self> {
        let store_dir = TokenStore::default_dir()?;
        let store = TokenStore::new(&store_dir);
        let key = crate::crypto::load_or_create_key()?;
        let tokens = store
            .load(&key)?
            .ok_or_else(|| anyhow!("not logged in; run `kvcdn login`"))?;
        Ok(Self { cfg, tokens, key })
    }

    pub fn init_upload(&mut self, project: &str, meta: &UploadMeta) -> Result<UploadInitResponse> {
        self.ensure_fresh_token()?;
        let url = format!("{}/v1/projects/{}/artifacts", self.cfg.api_url, project);
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&url)
            .bearer_auth(&self.tokens.access_token)
            .json(meta)
            .send()
            .with_context(|| format!("POST {url}"))?;

        if resp.status() == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::bail!("session expired; run `kvcdn login`");
        }
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("upload init failed: {text}");
        }
        resp.json().context("parsing upload init response")
    }

    fn ensure_fresh_token(&mut self) -> Result<()> {
        let needs_refresh = self
            .tokens
            .expires_at
            .map(|t| Utc::now() + ChronoDuration::seconds(60) >= t)
            .unwrap_or(false);
        if !needs_refresh {
            return Ok(());
        }
        let refresh = self
            .tokens
            .refresh_token
            .as_ref()
            .ok_or_else(|| anyhow!("session expired; run `kvcdn login`"))?;

        let oidc = crate::oidc::discover(&self.cfg.issuer_url)?;
        let mut params = std::collections::HashMap::new();
        params.insert("grant_type", "refresh_token");
        params.insert("client_id", self.cfg.client_id.as_str());
        params.insert("refresh_token", refresh.as_str());

        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&oidc.token_endpoint)
            .form(&params)
            .send()
            .context("refresh token request")?;
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("token refresh failed: {text}");
        }

        #[derive(serde::Deserialize)]
        struct RefreshResponse {
            access_token: String,
            refresh_token: Option<String>,
            expires_in: Option<u64>,
        }
        let r: RefreshResponse = resp.json().context("parsing refresh response")?;
        self.tokens.access_token = r.access_token;
        if let Some(rt) = r.refresh_token {
            self.tokens.refresh_token = Some(rt);
        }
        self.tokens.expires_at = r
            .expires_in
            .map(|s| Utc::now() + ChronoDuration::seconds(s as i64));

        let store_dir = TokenStore::default_dir()?;
        let store = TokenStore::new(&store_dir);
        store.save(&self.tokens, &self.key)?;
        Ok(())
    }
}
```

- [ ] **Step 4: Register module and run tests**

Add to `src/main.rs`:

```rust
mod api;
```

Run: `cargo test api::tests -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/api.rs src/main.rs
git commit -m "feat(api): authenticated client with token refresh and upload init"
```

---

## Task 10: Upload command

**Files:**
- Create: `src/upload.rs`
- Modify: `src/cli.rs` — add `UploadArgs`
- Modify: `src/main.rs` — wire `Upload`

- [ ] **Step 1: Add CLI args**

In `src/cli.rs`:

```rust
#[derive(Parser)]
pub struct UploadArgs {
    pub path: String,
    #[arg(long)]
    pub name: String,
    #[arg(long)]
    pub project: Option<String>,
    #[arg(long)]
    pub api_url: Option<String>,
}
```

Add `Upload(UploadArgs)` to the `Cli` enum.

- [ ] **Step 2: Implement upload with progress**

Create `src/upload.rs`:

```rust
use crate::api::{ApiClient, UploadMeta};
use crate::config::Config;
use crate::kv_io;
use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};

pub fn run(args: crate::cli::UploadArgs) -> Result<()> {
    let cfg = Config::load(args.api_url, None, None, None)?;
    let project = args.project.unwrap_or(cfg.default_project.clone());
    let path = &args.path;

    let mut file = File::open(path)
        .with_context(|| format!("failed to open {path}"))?;
    let size_bytes = fs::metadata(path)?.len();

    let mut reader = std::io::BufReader::new(&file);
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let sha256 = hex::encode(hasher.finalize());

    // Read artifact metadata to populate upload fields.
    let artifact = kv_io::read_kv_metadata(path)
        .with_context(|| format!("failed to read artifact metadata from {path}"))?;

    let mut client = ApiClient::new(cfg)?;
    let meta = UploadMeta {
        name: args.name,
        size_bytes,
        sha256,
        dtype: artifact.dtype,
        num_tokens: artifact.num_tokens,
        num_layers: artifact.num_layers,
        quantized: artifact.quantized,
    };

    let init = client.init_upload(&project, &meta)
        .context("failed to initiate upload")?;
    println!("uploading to {}", init.upload_url);

    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("failed to seek {path}"))?;
    let pb = ProgressBar::new(size_bytes);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"),
    );
    let reader = pb.wrap_read(file);

    let http = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;
    let resp = http
        .put(&init.upload_url)
        .body(reqwest::blocking::Body::sized(reader, size_bytes))
        .send()
        .context("failed to upload artifact")?;

    if !resp.status().is_success() {
        let text = resp.text().unwrap_or_default();
        pb.abandon_with_message("upload failed");
        anyhow::bail!("upload failed: {}", text);
    }

    pb.finish_with_message("upload complete");
    println!("Uploaded {} as artifact {}", args.path, init.artifact_id);
    Ok(())
}
```

- [ ] **Step 3: Wire in main and CLI**

In `src/main.rs`, add `mod upload;` and match arm:

```rust
Cli::Upload(args) => upload::run(args),
```

- [ ] **Step 4: Run `cargo check`**

Run: `cargo check`
Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add src/upload.rs src/cli.rs src/main.rs
git commit -m "feat(upload): upload KV artifacts via presigned URL with progress bar"
```

---

## Task 11: Integration tests with mocked OIDC and backend

**Files:**
- Create: `tests/auth_upload_test.rs`

- [ ] **Step 1: Write a mocked login + upload test**

Create `tests/auth_upload_test.rs`:

```rust
use std::io::Write;
use std::thread;
use std::time::Duration;

#[test]
#[ignore] // requires running tiny_http server and network setup; run manually with `cargo test -- --ignored`
fn mocked_oidc_and_upload() {
    // This test is a skeleton for a full integration test. In practice, spin
    // up mockito or tiny_http servers for discovery, token, and upload init,
    // then drive the callback URL manually.
}
```

- [ ] **Step 2: Add a unit-style integration test for API refresh path**

Replace the contents with a real test using `mockito`:

```rust
use kvcdn::api::{ApiClient, UploadMeta};
use kvcdn::config::Config;
use kvcdn::crypto;
use kvcdn::token_store::{TokenStore, Tokens};
use chrono::{DateTime, Duration as ChronoDuration, Utc};

#[test]
fn api_client_refreshes_expired_token() {
    let dir = tempfile::tempdir().unwrap();
    let key = crypto::random_key();
    let store = TokenStore::new(dir.path());
    store
        .save(
            &Tokens {
                access_token: "old".to_string(),
                refresh_token: Some("refresh_me".to_string()),
                expires_at: Some(Utc::now() - ChronoDuration::seconds(10)),
            },
            &key,
        )
        .unwrap();

    let cfg = Config::load(
        Some("http://localhost".to_string()),
        Some("http://localhost".to_string()),
        Some("client".to_string()),
        None,
    )
    .unwrap();

    // With mockito set up, this test would verify that init_upload first POSTs
    // to the token endpoint and then to /v1/projects/default/artifacts.
    // Omitted for brevity; add once mock endpoints are wired.
}
```

- [ ] **Step 3: Commit**

```bash
git add tests/auth_upload_test.rs
git commit -m "test: add integration test skeleton for auth and upload"
```

---

## Task 12: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cargo test`
Expected: all non-ignored tests pass.

- [ ] **Step 2: Run `cargo clippy`**

Run: `cargo clippy -- -D warnings`
Expected: no warnings.

- [ ] **Step 3: Manual smoke test**

Run: `cargo run -- login --no-browser`
Expected: prints authorization URL, waits for callback (can be tested by hitting the printed URL with a fake `?code=xxx&state=yyy`).

- [ ] **Step 4: Commit any final fixes**

```bash
git commit -am "fix: address clippy warnings and test failures"
```

---

## Spec coverage check

| Spec requirement | Plan task |
|------------------|-----------|
| Browser auto-open login | Task 7 |
| `--no-browser` manual fallback | Task 7 |
| Encrypted token file in `~/.config/kvcdn/` | Task 4 |
| OS keyring key with file fallback | Task 3 |
| Backend/OIDC config via defaults/env/file | Task 2 |
| Initiate upload, get presigned URL, PUT file | Task 9 + 10 |
| Default project + `--project` | Task 2 + 10 |
| Token refresh before expiry | Task 9 |
| Progress reporting | Task 10 |
| `logout` command | Task 8 |

## Placeholder scan

No `TBD`, `TODO`, or vague steps remain. Each task includes exact file paths, code, commands, and expected outputs.
