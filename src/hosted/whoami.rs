use crate::config::Config;
use crate::hosted::api::ApiClient;
use crate::hosted::api_key::load_stored_api_key;
use crate::hosted::token_store::{TokenStore, Tokens};
use anyhow::Result;
use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
use serde::Deserialize;

pub fn run(args: crate::cli::WhoamiArgs) -> Result<()> {
    let cfg = Config::load(
        args.api_url,
        args.issuer_url,
        args.client_id,
        args.org,
        args.project,
        args.api_key,
    )?;

    cfg.require_hosted()?;
    let org = cfg.default_org.clone();
    let project = cfg.default_project.clone();

    println!("api_url:  {}", cfg.api_url);
    println!("issuer:   {}", cfg.issuer_url);
    println!("client:   {}", cfg.client_id);
    println!("org:      {}", org);
    println!("project:  {}", project);

    let auth = resolve_auth(&cfg)?;
    match auth {
        Auth::ApiKey { customer_id } => {
            println!("method:   api_key");
            println!("customer: {}", customer_id);
        }
        Auth::Oidc { sub, email, name } => {
            println!("method:   oidc");
            println!("subject:  {}", sub);
            if let Some(email) = email {
                println!("email:    {}", email);
            }
            if let Some(name) = name {
                println!("name:     {}", name);
            }
        }
        Auth::None => {
            println!("method:   none");
            println!("Run `kvcdn login` or `kvcdn api-key set <key>` to authenticate.");
        }
    }

    Ok(())
}

enum Auth {
    ApiKey {
        customer_id: String,
    },
    Oidc {
        sub: String,
        email: Option<String>,
        name: Option<String>,
    },
    None,
}

fn resolve_auth(cfg: &Config) -> Result<Auth> {
    let api_key = cfg
        .api_key
        .clone()
        .or_else(|| load_stored_api_key().ok().flatten());
    if api_key.is_some() {
        let mut client = ApiClient::new(cfg.clone())?;
        let customer_id = client.whoami_customer_id()?;
        return Ok(Auth::ApiKey { customer_id });
    }

    let Some(tokens) = load_stored_oidc_tokens()? else {
        return Ok(Auth::None);
    };
    let Some(claims) = decode_oidc_access_token(&tokens.access_token)? else {
        return Ok(Auth::None);
    };
    Ok(Auth::Oidc {
        sub: claims.sub,
        email: claims.email,
        name: claims.name,
    })
}

fn load_stored_oidc_tokens() -> Result<Option<Tokens>> {
    let key = crate::core::crypto::load_or_create_key()?;
    let dir = TokenStore::default_dir()?;
    TokenStore::new(&dir).load(&key)
}

#[derive(Debug, Deserialize)]
struct OidcClaims {
    sub: String,
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

fn decode_oidc_access_token(token: &str) -> Result<Option<OidcClaims>> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return Ok(None);
    }
    let payload = URL_SAFE_NO_PAD
        .decode(parts[1])
        .map_err(|e| anyhow::anyhow!("failed to base64-decode access token payload: {e}"))?;
    let claims: OidcClaims = serde_json::from_slice(&payload)
        .map_err(|e| anyhow::anyhow!("invalid access token claims: {e}"))?;
    Ok(Some(claims))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_oidc_access_token_extracts_claims() {
        let payload = URL_SAFE_NO_PAD
            .encode(br#"{"sub":"user-123","email":"a@example.com","name":"A User"}"#);
        let token = format!("header.{payload}.signature");
        let claims = decode_oidc_access_token(&token).unwrap().unwrap();
        assert_eq!(claims.sub, "user-123");
        assert_eq!(claims.email, Some("a@example.com".to_string()));
        assert_eq!(claims.name, Some("A User".to_string()));
    }

    #[test]
    fn decode_oidc_access_token_returns_none_for_non_jwt() {
        assert!(decode_oidc_access_token("not-a-jwt").unwrap().is_none());
    }

    #[test]
    fn decode_oidc_access_token_allows_missing_optional_claims() {
        let payload = URL_SAFE_NO_PAD.encode(br#"{"sub":"user-456"}"#);
        let token = format!("header.{payload}.signature");
        let claims = decode_oidc_access_token(&token).unwrap().unwrap();
        assert_eq!(claims.sub, "user-456");
        assert_eq!(claims.email, None);
        assert_eq!(claims.name, None);
    }
}
