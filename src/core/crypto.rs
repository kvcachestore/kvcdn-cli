use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, KeyInit},
};
use anyhow::{Context, Result};
use rand::RngCore;
use std::fs;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

pub const KEY_LEN: usize = 32;
const NONCE_LEN: usize = 12;

pub type Key = [u8; KEY_LEN];

pub fn random_key() -> Key {
    let mut key = [0u8; KEY_LEN];
    rand::rng().fill_bytes(&mut key);
    key
}

pub fn encrypt(plaintext: &[u8], key: &Key) -> Result<Vec<u8>> {
    let cipher =
        Aes256Gcm::new_from_slice(key).map_err(|e| anyhow::anyhow!("invalid key length: {e}"))?;
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
    let cipher =
        Aes256Gcm::new_from_slice(key).map_err(|e| anyhow::anyhow!("invalid key length: {e}"))?;
    let nonce = Nonce::from_slice(&ciphertext[..NONCE_LEN]);
    cipher
        .decrypt(nonce, &ciphertext[NONCE_LEN..])
        .map_err(|e| anyhow::anyhow!("decryption failed: {e}"))
}

pub fn load_or_create_key() -> Result<Key> {
    match load_key_from_keyring() {
        Ok(Some(key)) => return Ok(key),
        Ok(None) => {}
        Err(e) => {
            eprintln!(
                "warning: OS keyring unavailable ({e}); falling back to file-based key storage."
            );
        }
    }
    load_or_create_key_file()
}

fn load_key_from_keyring() -> Result<Option<Key>> {
    let entry = match keyring::Entry::new("kvcdn-cli", "encryption-key") {
        Ok(e) => e,
        Err(keyring::Error::NoEntry) => return Ok(None),
        Err(e) => return Err(anyhow::anyhow!("keyring error: {e}")),
    };
    match entry.get_password() {
        Ok(hex) => {
            let mut key = [0u8; KEY_LEN];
            hex::decode_to_slice(&hex, &mut key)
                .with_context(|| "keyring entry is not valid hex")?;
            Ok(Some(key))
        }
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(anyhow::anyhow!("keyring error: {e}")),
    }
}

fn load_or_create_key_file() -> Result<Key> {
    let path = key_file_path()?;
    load_or_create_key_file_at_path(&path)
}

#[cfg(test)]
fn load_or_create_key_file_in_dir(base: &Path) -> Result<Key> {
    let path = key_file_path_in(base);
    load_or_create_key_file_at_path(&path)
}

fn load_or_create_key_file_at_path(path: &Path) -> Result<Key> {
    loop {
        match OpenOptions::new().read(true).open(path) {
            Ok(mut f) => {
                let mut hex = String::new();
                f.read_to_string(&mut hex)
                    .with_context(|| format!("reading {}", path.display()))?;
                let mut key = [0u8; KEY_LEN];
                hex::decode_to_slice(hex.trim(), &mut key)
                    .with_context(|| format!("decoding key from {}", path.display()))?;
                return Ok(key);
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }

                let mut opts = OpenOptions::new();
                opts.create_new(true).write(true);
                #[cfg(unix)]
                opts.mode(0o600);
                match opts.open(path) {
                    Ok(mut f) => {
                        let key = random_key();
                        f.write_all(hex::encode(key).as_bytes())
                            .with_context(|| format!("writing {}", path.display()))?;
                        eprintln!(
                            "warning: OS keyring unavailable; encryption key stored in {} with 0600 permissions.",
                            path.display()
                        );
                        return Ok(key);
                    }
                    Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                        // Another process created the file between the open and create_new
                        // attempts. Loop around and try to read it.
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
}

fn key_file_path() -> Result<PathBuf> {
    let config_dir = dirs::config_dir().context("could not determine config directory")?;
    Ok(key_file_path_in(&config_dir))
}

fn key_file_path_in(base: &Path) -> PathBuf {
    base.join("kvcdn").join(".key")
}

pub fn ensure_key_stored(key: &Key) -> Result<()> {
    match keyring::Entry::new("kvcdn-cli", "encryption-key") {
        Ok(entry) => {
            let hex = hex::encode(key);
            entry
                .set_password(&hex)
                .map_err(|e| anyhow::anyhow!("keyring error: {e}"))?;
            Ok(())
        }
        Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("keyring error: {e}")),
    }
}

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

    #[test]
    fn decrypt_with_wrong_key_fails() {
        let key = random_key();
        let wrong_key = random_key();
        let plaintext = b"secret data";
        let ciphertext = encrypt(plaintext, &key).unwrap();
        assert!(decrypt(&ciphertext, &wrong_key).is_err());
    }

    #[test]
    fn decrypt_tampered_ciphertext_fails() {
        let key = random_key();
        let plaintext = b"secret data";
        let mut ciphertext = encrypt(plaintext, &key).unwrap();
        let last = ciphertext.len() - 1;
        ciphertext[last] ^= 0xff;
        assert!(decrypt(&ciphertext, &key).is_err());
    }

    #[test]
    fn round_trip_empty_plaintext() {
        let key = random_key();
        let plaintext: &[u8] = b"";
        let ciphertext = encrypt(plaintext, &key).unwrap();
        let decrypted = decrypt(&ciphertext, &key).unwrap();
        assert_eq!(decrypted, plaintext.to_vec());
    }

    #[test]
    fn key_file_is_created_with_restricted_permissions() {
        let temp_dir = std::env::temp_dir().join(format!(
            "kvcdn-key-test-{}-{}",
            std::process::id(),
            rand::random::<u32>()
        ));
        fs::create_dir_all(&temp_dir).unwrap();

        let key = load_or_create_key_file_in_dir(&temp_dir).unwrap();
        assert_ne!(key, [0u8; KEY_LEN]);

        let path = key_file_path_in(&temp_dir);
        assert!(path.exists(), "key file should exist");
        let contents = fs::read_to_string(&path).unwrap();
        assert!(!contents.is_empty(), "key file should not be empty");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = path.metadata().unwrap().permissions().mode() & 0o777;
            assert_eq!(
                mode, 0o600,
                "key file should be created with 0600 permissions"
            );
        }

        fs::remove_dir_all(&temp_dir).unwrap();
    }
}
