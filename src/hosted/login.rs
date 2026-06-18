use crate::config::Config;
use crate::core::crypto;
use crate::hosted::callback_server::CallbackServer;
use crate::hosted::oidc::{
    Pkce, build_authorization_url, discover, exchange_code, initiate_device_auth,
    poll_device_token, random_state,
};
use crate::hosted::token_store::{TokenStore, Tokens};
use anyhow::{Context, Result};
use std::process::Command;

pub fn run(args: crate::cli::LoginArgs) -> Result<()> {
    let cfg = Config::load(
        args.api_url,
        args.issuer_url,
        args.client_id,
        args.org,
        args.project,
        None,
    )?;
    let oidc = discover(&cfg.issuer_url).context("OIDC discovery failed")?;

    let tokens = if oidc.supports_device_code() {
        login_device_code(&oidc, &cfg.client_id, args.no_browser)?
    } else {
        login_authorization_code(&oidc, &cfg.client_id, args.no_browser)?
    };

    let key = crypto::load_or_create_key().context("failed to load or create encryption key")?;
    let store_dir =
        TokenStore::default_dir().context("failed to determine credentials directory")?;
    let store = TokenStore::new(&store_dir);
    store
        .save(&tokens, &key)
        .context("failed to save credentials")?;
    crypto::ensure_key_stored(&key).context("failed to store encryption key")?;

    println!("Login successful.");
    Ok(())
}

fn login_device_code(
    oidc: &crate::hosted::oidc::OidcConfig,
    client_id: &str,
    no_browser: bool,
) -> Result<Tokens> {
    let auth =
        initiate_device_auth(oidc, client_id).context("initiating device authorization failed")?;

    let url = auth
        .verification_uri_complete
        .as_ref()
        .unwrap_or(&auth.verification_uri);

    if no_browser {
        println!(
            "To sign in, open:\n  {}\nand enter code:\n  {}",
            auth.verification_uri, auth.user_code
        );
    } else {
        println!("Opening browser for device login...");
        if open_browser(url).is_err() {
            eprintln!(
                "warning: failed to open browser. Open\n  {}\nand enter code:\n  {}",
                auth.verification_uri, auth.user_code
            );
        }
    }

    poll_device_token(
        oidc,
        client_id,
        &auth.device_code,
        auth.interval,
        auth.expires_in,
    )
    .context("waiting for device authorization failed")
}

fn login_authorization_code(
    oidc: &crate::hosted::oidc::OidcConfig,
    client_id: &str,
    no_browser: bool,
) -> Result<Tokens> {
    let pkce = Pkce::generate();
    let state = random_state();
    let server = CallbackServer::start(&state)?;
    let redirect_uri = server.redirect_uri.clone();
    let auth_url = build_authorization_url(oidc, client_id, &redirect_uri, &pkce, &state)
        .context("failed to build authorization URL")?;

    if no_browser {
        println!("Please open this URL in your browser:\n{auth_url}");
    } else {
        println!("Opening browser for login...");
        if open_browser(&auth_url).is_err() {
            eprintln!("warning: failed to open browser. Open this URL manually:\n{auth_url}");
        }
    }

    let code = server.wait().context("waiting for OIDC callback")?;
    exchange_code(oidc, client_id, &redirect_uri, &code, &pkce).context("token exchange failed")
}

#[cfg(target_os = "macos")]
fn open_browser(url: &str) -> Result<()> {
    let status = Command::new("open")
        .arg(url)
        .status()
        .context("failed to run open")?;
    if !status.success() {
        anyhow::bail!("open command exited with status {status}");
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn open_browser(url: &str) -> Result<()> {
    let status = Command::new("xdg-open")
        .arg(url)
        .status()
        .context("failed to run xdg-open")?;
    if !status.success() {
        anyhow::bail!("xdg-open command exited with status {status}");
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn open_browser(url: &str) -> Result<()> {
    let status = Command::new("cmd")
        .args(["/C", "start", ""])
        .arg(url)
        .status()
        .context("failed to run start")?;
    if !status.success() {
        anyhow::bail!("start command exited with status {status}");
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn open_browser(_url: &str) -> Result<()> {
    anyhow::bail!("automatic browser open not supported on this platform")
}
