use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client as HttpClient;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::time::Duration;

pub trait Transfer: Send + Sync {
    fn upload(&self, url: &str, source: Box<dyn Read + Send>, size_bytes: u64) -> Result<()>;

    fn download(&self, url: &str, destination: &str, total_size: u64) -> Result<()>;
}

fn transfer_client() -> Result<HttpClient> {
    HttpClient::builder()
        .timeout(Duration::from_secs(300))
        .build()
        .context("failed to build transfer HTTP client")
}

pub struct HttpTransfer;

impl HttpTransfer {
    pub fn new() -> Self {
        Self
    }
}

impl Transfer for HttpTransfer {
    fn upload(&self, url: &str, source: Box<dyn Read + Send>, size_bytes: u64) -> Result<()> {
        let pb = ProgressBar::new(size_bytes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );
        let reader = pb.wrap_read(source);

        let client = transfer_client()?;
        let resp = client
            .put(url)
            .body(reqwest::blocking::Body::sized(reader, size_bytes))
            .send()
            .context("failed to upload artifact")?;

        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            pb.abandon_with_message("upload failed");
            anyhow::bail!("upload failed: {}", text);
        }

        pb.finish_with_message("upload complete");
        Ok(())
    }

    fn download(&self, url: &str, destination: &str, total_size: u64) -> Result<()> {
        let client = transfer_client()?;
        let resp = client.get(url).send().context("download request failed")?;
        if !resp.status().is_success() {
            let text = resp.text().unwrap_or_default();
            anyhow::bail!("download failed: {}", text);
        }

        let size = if total_size > 0 {
            total_size
        } else {
            resp.headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
                .unwrap_or(0)
        };

        let pb = ProgressBar::new(size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );

        let file =
            File::create(destination).with_context(|| format!("failed to create {destination}"))?;
        let mut writer = BufWriter::new(file);

        let mut reader = pb.wrap_read(resp);
        std::io::copy(&mut reader, &mut writer).context("failed to stream download to disk")?;
        writer.flush().context("failed to flush download")?;
        pb.finish_with_message("download complete");
        Ok(())
    }
}
