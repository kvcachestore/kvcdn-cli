use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;

const DEFAULT_API_URL: &str = "https://api.kvcachestore.com";
const DEFAULT_ISSUER_URL: &str = "https://pocketid.kvcachestore.com";
const DEFAULT_CLIENT_ID: &str = "kvcdn-cli";
const DEFAULT_ORG: &str = "default";
const DEFAULT_PROJECT: &str = "default";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    #[serde(default = "default_api_url")]
    pub api_url: String,
    #[serde(default = "default_issuer_url")]
    pub issuer_url: String,
    #[serde(default = "default_client_id")]
    pub client_id: String,
    #[serde(default = "default_org")]
    pub default_org: String,
    #[serde(default = "default_project")]
    pub default_project: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

fn default_api_url() -> String {
    DEFAULT_API_URL.to_string()
}

fn default_issuer_url() -> String {
    DEFAULT_ISSUER_URL.to_string()
}

fn default_client_id() -> String {
    DEFAULT_CLIENT_ID.to_string()
}

fn default_org() -> String {
    DEFAULT_ORG.to_string()
}

fn default_project() -> String {
    DEFAULT_PROJECT.to_string()
}

fn pick(cli: Option<String>, env: Option<String>, file: Option<String>, default: &str) -> String {
    cli.or(env).or(file).unwrap_or_else(|| default.to_string())
}

fn pick_optional(cli: Option<String>, env: Option<String>, file: Option<String>) -> Option<String> {
    cli.or(env).or(file)
}

impl Config {
    pub fn load(
        cli_api_url: Option<String>,
        cli_issuer_url: Option<String>,
        cli_client_id: Option<String>,
        cli_default_org: Option<String>,
        cli_default_project: Option<String>,
        cli_api_key: Option<String>,
    ) -> Result<Self> {
        Self::load_with_config_dir(
            dirs::config_dir().as_deref(),
            cli_api_url,
            cli_issuer_url,
            cli_client_id,
            cli_default_org,
            cli_default_project,
            cli_api_key,
        )
    }

    pub(crate) fn load_with_config_dir(
        config_dir: Option<&Path>,
        cli_api_url: Option<String>,
        cli_issuer_url: Option<String>,
        cli_client_id: Option<String>,
        cli_default_org: Option<String>,
        cli_default_project: Option<String>,
        cli_api_key: Option<String>,
    ) -> Result<Self> {
        let file_cfg = match config_dir {
            Some(dir) => Self::from_dir(dir)?,
            None => Self::from_file()?,
        };

        Ok(Self {
            api_url: pick(
                cli_api_url,
                env::var("KVCDN_API_URL").ok(),
                file_cfg.as_ref().map(|c| c.api_url.clone()),
                DEFAULT_API_URL,
            ),
            issuer_url: pick(
                cli_issuer_url,
                env::var("KVCDN_ISSUER_URL").ok(),
                file_cfg.as_ref().map(|c| c.issuer_url.clone()),
                DEFAULT_ISSUER_URL,
            ),
            client_id: pick(
                cli_client_id,
                env::var("KVCDN_CLIENT_ID").ok(),
                file_cfg.as_ref().map(|c| c.client_id.clone()),
                DEFAULT_CLIENT_ID,
            ),
            default_org: pick(
                cli_default_org,
                env::var("KVCDN_DEFAULT_ORG").ok(),
                file_cfg.as_ref().map(|c| c.default_org.clone()),
                DEFAULT_ORG,
            ),
            default_project: pick(
                cli_default_project,
                env::var("KVCDN_DEFAULT_PROJECT").ok(),
                file_cfg.as_ref().map(|c| c.default_project.clone()),
                DEFAULT_PROJECT,
            ),
            api_key: pick_optional(
                cli_api_key,
                env::var("KVCDN_API_KEY").ok(),
                file_cfg.as_ref().and_then(|c| c.api_key.clone()),
            ),
        })
    }

    fn from_file() -> Result<Option<Self>> {
        match dirs::config_dir() {
            Some(config_dir) => Self::from_dir(&config_dir),
            None => Ok(None),
        }
    }

    fn from_dir(config_dir: &Path) -> Result<Option<Self>> {
        let path = config_dir.join("kvcdn").join("config.toml");
        if !path.exists() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&path)?;
        Ok(Some(toml::from_str(&text)?))
    }

    pub fn require_hosted(&self) -> Result<()> {
        if self.api_url.is_empty() {
            anyhow::bail!(
                "api_url is not configured. Set KVCDN_API_URL or add api_url to ~/.config/kvcdn/config.toml."
            );
        }
        if self.issuer_url.is_empty() {
            anyhow::bail!(
                "issuer_url is not configured. Set KVCDN_ISSUER_URL or add issuer_url to ~/.config/kvcdn/config.toml."
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const KVCDN_ENV_VARS: &[&str] = &[
        "KVCDN_API_URL",
        "KVCDN_ISSUER_URL",
        "KVCDN_CLIENT_ID",
        "KVCDN_DEFAULT_ORG",
        "KVCDN_DEFAULT_PROJECT",
        "KVCDN_API_KEY",
    ];

    /// Run a closure with all KVCDN_* environment variables cleared, restoring
    /// them afterwards. Config tests assume a clean environment so that
    /// assertions about file/env/cli precedence are not affected by the calling
    /// process's environment (e.g. a locally set KVCDN_API_KEY).
    fn with_kvcdn_env_cleared<F: FnOnce()>(f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let saved: Vec<(&str, Option<String>)> = KVCDN_ENV_VARS
            .iter()
            .map(|&v| (v, env::var(v).ok()))
            .collect();
        for v in KVCDN_ENV_VARS {
            // SAFETY: tests hold ENV_LOCK while mutating the environment and
            // only config tests inspect these variables.
            unsafe { env::remove_var(v) };
        }
        f();
        for (v, val) in saved {
            match val {
                Some(val) => unsafe { env::set_var(v, val) },
                None => unsafe { env::remove_var(v) },
            }
        }
    }

    #[test]
    fn pick_uses_default_when_no_override() {
        assert_eq!(pick(None, None, None, DEFAULT_API_URL), DEFAULT_API_URL);
    }

    #[test]
    fn pick_uses_file_over_default() {
        assert_eq!(
            pick(
                None,
                None,
                Some("https://api.file.test".to_string()),
                DEFAULT_API_URL
            ),
            "https://api.file.test"
        );
    }

    #[test]
    fn pick_uses_env_over_file() {
        assert_eq!(
            pick(
                None,
                Some("https://api.env.test".to_string()),
                Some("https://api.file.test".to_string()),
                DEFAULT_API_URL,
            ),
            "https://api.env.test"
        );
    }

    #[test]
    fn pick_uses_cli_over_env() {
        assert_eq!(
            pick(
                Some("https://api.cli.test".to_string()),
                Some("https://api.env.test".to_string()),
                None,
                DEFAULT_API_URL,
            ),
            "https://api.cli.test"
        );
    }

    #[test]
    fn pick_uses_cli_over_env_over_file_over_default() {
        assert_eq!(
            pick(
                Some("https://api.cli.test".to_string()),
                Some("https://api.env.test".to_string()),
                Some("https://api.file.test".to_string()),
                DEFAULT_API_URL,
            ),
            "https://api.cli.test"
        );
        assert_eq!(
            pick(
                None,
                Some("https://api.env.test".to_string()),
                Some("https://api.file.test".to_string()),
                DEFAULT_API_URL,
            ),
            "https://api.env.test"
        );
        assert_eq!(
            pick(
                None,
                None,
                Some("https://api.file.test".to_string()),
                DEFAULT_API_URL
            ),
            "https://api.file.test"
        );
        assert_eq!(pick(None, None, None, DEFAULT_API_URL), DEFAULT_API_URL);
    }

    #[test]
    fn load_uses_config_dir() {
        with_kvcdn_env_cleared(|| {
            let dir = tempfile::tempdir().unwrap();
            let kvcdn_dir = dir.path().join("kvcdn");
            std::fs::create_dir(&kvcdn_dir).unwrap();
            std::fs::write(
                kvcdn_dir.join("config.toml"),
                r#"api_url = "https://api.custom.test"
issuer_url = "https://issuer.custom.test"
client_id = "custom-client"
default_org = "custom-org"
default_project = "custom-project"
"#,
            )
            .unwrap();

            let cfg =
                Config::load_with_config_dir(Some(dir.path()), None, None, None, None, None, None)
                    .unwrap();
            assert_eq!(cfg.api_url, "https://api.custom.test");
            assert_eq!(cfg.issuer_url, "https://issuer.custom.test");
            assert_eq!(cfg.client_id, "custom-client");
            assert_eq!(cfg.default_org, "custom-org");
            assert_eq!(cfg.default_project, "custom-project");
            assert_eq!(cfg.api_key, None);
        });
    }

    #[test]
    fn load_errors_for_malformed_config() {
        with_kvcdn_env_cleared(|| {
            let dir = tempfile::tempdir().unwrap();
            let kvcdn_dir = dir.path().join("kvcdn");
            std::fs::create_dir(&kvcdn_dir).unwrap();
            std::fs::write(kvcdn_dir.join("config.toml"), "this is not valid toml").unwrap();

            // from_dir returns an error for malformed TOML.
            assert!(Config::from_dir(dir.path()).is_err());

            // Config::load now surfaces the error so users notice typos.
            assert!(
                Config::load_with_config_dir(Some(dir.path()), None, None, None, None, None, None)
                    .is_err()
            );
        });
    }

    #[test]
    fn load_uses_api_key_env() {
        let dir = tempfile::tempdir().unwrap();
        let kvcdn_dir = dir.path().join("kvcdn");
        std::fs::create_dir(&kvcdn_dir).unwrap();
        std::fs::write(
            kvcdn_dir.join("config.toml"),
            r#"api_url = "https://api.custom.test"
api_key = "kv_file"
"#,
        )
        .unwrap();

        let cfg = Config::load_with_config_dir(
            Some(dir.path()),
            None,
            None,
            None,
            None,
            None,
            Some("kv_cli".to_string()),
        )
        .unwrap();
        assert_eq!(cfg.api_key, Some("kv_cli".to_string()));
    }

    #[test]
    fn load_uses_default_org() {
        let cfg = Config::load_with_config_dir(None, None, None, None, None, None, None).unwrap();
        assert_eq!(cfg.default_org, DEFAULT_ORG);
    }

    #[test]
    fn load_uses_production_defaults() {
        with_kvcdn_env_cleared(|| {
            let dir = tempfile::tempdir().unwrap();
            let cfg =
                Config::load_with_config_dir(Some(dir.path()), None, None, None, None, None, None)
                    .unwrap();
            assert_eq!(cfg.api_url, DEFAULT_API_URL);
            assert_eq!(cfg.issuer_url, DEFAULT_ISSUER_URL);
        });
    }
}
