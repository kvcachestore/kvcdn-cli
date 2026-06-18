use crate::api::{ApiClient, UploadMeta};
use crate::config::Config;
use crate::kv_io;
use crate::transfer::{HttpTransfer, Transfer};
use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek};

pub fn run(args: crate::cli::UploadArgs) -> Result<()> {
    let cfg = Config::load(
        args.api_url,
        args.issuer_url,
        args.client_id,
        args.org,
        args.project,
        args.api_key,
    )?;
    let org = cfg.default_org.clone();
    let project = cfg.default_project.clone();
    let path = &args.path;

    // Read artifact metadata, then hash and upload the file using a single handle.
    let artifact = kv_io::read_kv_metadata(path)
        .with_context(|| format!("failed to read artifact metadata from {path}"))?;

    let mut file = File::open(path).with_context(|| format!("failed to open {path}"))?;
    let size_bytes = file
        .metadata()
        .with_context(|| format!("failed to read metadata for {path}"))?
        .len();

    let sha256 = {
        let mut reader = std::io::BufReader::new(&file);
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = reader
                .read(&mut buf)
                .with_context(|| format!("failed to read {path} while hashing"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        hex::encode(hasher.finalize())
    };

    let meta = UploadMeta {
        name: args.name,
        size_bytes,
        sha256: sha256.clone(),
        dtype: artifact.dtype,
        storage_dtype: artifact.storage_dtype,
        num_tokens: artifact.num_tokens,
        num_layers: artifact.num_layers,
        quantized: artifact.quantized,
        visibility: args.visibility.clone(),
    };

    let mut client = ApiClient::new(cfg)?;
    let init = client
        .init_upload(&org, &project, &meta)
        .context("failed to initiate upload")?;
    println!(
        "uploading to {} (org: {}, project: {})",
        init.upload_url, org, project
    );

    file.rewind()
        .with_context(|| format!("failed to rewind {path}"))?;

    HttpTransfer::new()
        .upload(&init.upload_url, Box::new(file), size_bytes)
        .context("failed to upload artifact")?;

    client
        .complete_upload(&org, &project, &init.artifact_id)
        .context("upload completed but server-side verification failed")?;

    println!("Uploaded {} as artifact {}", args.path, init.artifact_id);
    Ok(())
}
