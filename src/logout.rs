use crate::api_key::ApiKeyStore;
use crate::token_store::TokenStore;
use anyhow::Result;

pub fn run(_args: crate::cli::LogoutArgs) -> Result<()> {
    let dir = TokenStore::default_dir()?;
    TokenStore::new(&dir).delete()?;
    ApiKeyStore::new(&dir).delete()?;
    println!("Logged out.");
    Ok(())
}
