use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

fn temp_path_for(path: &Path) -> PathBuf {
    path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|e| e.to_str()).unwrap_or("")
    ))
}

/// Build the default user-data output directory for a command.
///
/// Uses `dirs::data_dir()` (or the current working directory if unavailable),
/// then appends `kvcdn/<command>/` and creates all missing directories.
pub fn default_output_dir(command: &str) -> Result<PathBuf> {
    let base = dirs::data_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let dir = base.join("kvcdn").join(command);
    fs::create_dir_all(&dir)
        .with_context(|| format!("creating output directory {}", dir.display()))?;
    Ok(dir)
}

fn sanitize_model(model_name: &str) -> String {
    model_name.replace('/', "_")
}

fn timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{}", now)
}

/// Ensure a KV artifact path ends with the `.kv` extension.
///
/// If `path` already ends in `.kv`, it is returned unchanged. Otherwise `.kv`
/// is appended and a warning is printed so the user knows the convention was
/// enforced.
pub fn ensure_kv_extension(path: &Path) -> PathBuf {
    let ext = path.extension().and_then(|e| e.to_str());
    if ext == Some("kv") {
        return path.to_path_buf();
    }
    eprintln!(
        "warning: KV artifact path '{}' does not end with '.kv'; appending '.kv'",
        path.display()
    );
    let mut name = path.as_os_str().to_owned();
    name.push(".kv");
    PathBuf::from(name)
}

/// Unique timestamped `.kv` path under the default data directory.
pub fn unique_kv_path(command: &str, model_name: &str) -> Result<PathBuf> {
    let dir = default_output_dir(command)?;
    let name = format!(
        "{}-{}-{}.kv",
        command,
        timestamp(),
        sanitize_model(model_name)
    );
    Ok(dir.join(name))
}

/// Unique timestamped `.csv` path under the default data directory.
pub fn unique_csv_path(command: &str, model_name: &str) -> Result<PathBuf> {
    let dir = default_output_dir(command)?;
    let name = format!("{}-{}-bench.csv", timestamp(), sanitize_model(model_name));
    Ok(dir.join(name))
}

pub fn unique_png_path(command: &str, model_name: &str) -> Result<PathBuf> {
    let dir = default_output_dir(command)?;
    let name = format!("{}-{}.png", timestamp(), sanitize_model(model_name));
    Ok(dir.join(name))
}

/// Atomically write a file by creating a temp file next to the destination,
/// invoking `writer` to fill it, then renaming it over the destination.
///
/// On failure the temp file is removed so no partial output is left behind.
pub fn atomic_write(
    path: &Path,
    mut writer: impl FnMut(&mut fs::File) -> Result<()>,
) -> Result<()> {
    let tmp = temp_path_for(path);
    let result = atomic_write_inner(&tmp, &mut writer);
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result?;

    fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn atomic_write_inner(
    tmp: &Path,
    writer: &mut impl FnMut(&mut fs::File) -> Result<()>,
) -> Result<()> {
    let mut file =
        fs::File::create(tmp).with_context(|| format!("creating temp file {}", tmp.display()))?;
    writer(&mut file).with_context(|| format!("writing temp file {}", tmp.display()))?;
    std::io::Write::flush(&mut file)
        .with_context(|| format!("flushing temp file {}", tmp.display()))?;
    file.sync_all()
        .with_context(|| format!("syncing temp file {}", tmp.display()))?;
    Ok(())
}

/// Resolve an explicit output path or generate a unique one under the data dir.
///
/// `extension` is used only when generating a default path: `"kv"` produces a
/// timestamped `.kv` file; `"csv"` produces a timestamped `-bench.csv` file.
/// Explicit KV paths are passed through `ensure_kv_extension` so every
/// generated artifact follows the `.kv` convention.
pub fn resolve_output_path(
    command: &str,
    model_name: &str,
    explicit: &Option<String>,
    extension: &str,
) -> Result<PathBuf> {
    if let Some(p) = explicit {
        let path = PathBuf::from(p);
        if extension == "kv" {
            return Ok(ensure_kv_extension(&path));
        }
        return Ok(path);
    }
    match extension {
        "csv" => unique_csv_path(command, model_name),
        "png" => unique_png_path(command, model_name),
        _ => unique_kv_path(command, model_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn default_output_dir_creates_kvcdn_command_dir() {
        let dir = default_output_dir("test-command").unwrap();
        assert!(dir.exists());
        assert!(dir.to_string_lossy().contains("kvcdn"));
        assert!(dir.to_string_lossy().contains("test-command"));
    }

    #[test]
    fn unique_kv_path_includes_command_timestamp_and_sanitized_model() {
        let path = unique_kv_path("verify", "Qwen/Qwen3-0.6B").unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("verify-"));
        assert!(name.ends_with("-Qwen_Qwen3-0.6B.kv"));
        assert!(name.len() > "verify-Qwen_Qwen3-0.6B.kv".len());
        assert_eq!(path.extension().and_then(|e| e.to_str()), Some("kv"));
    }

    #[test]
    fn unique_csv_path_uses_bench_csv_suffix() {
        let path = unique_csv_path("benchmark", "foo/bar").unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.ends_with("-foo_bar-bench.csv"));
    }

    #[test]
    fn unique_kv_path_produces_different_timestamps() {
        let a = unique_kv_path("verify", "m").unwrap();
        thread::sleep(Duration::from_millis(2));
        let b = unique_kv_path("verify", "m").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn resolve_output_path_uses_explicit_when_given() {
        let explicit = Some("my/custom/file.kv".to_string());
        let path = resolve_output_path("verify", "m", &explicit, "kv").unwrap();
        assert_eq!(path, PathBuf::from("my/custom/file.kv"));
    }

    #[test]
    fn resolve_output_path_defaults_to_unique_kv() {
        let path = resolve_output_path("diag", "m", &None, "kv").unwrap();
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("diag-"));
        assert!(name.ends_with("-m.kv"));
    }

    #[test]
    fn resolve_output_path_enforces_kv_extension() {
        let explicit = Some("my/custom/file".to_string());
        let path = resolve_output_path("verify", "m", &explicit, "kv").unwrap();
        assert_eq!(path, PathBuf::from("my/custom/file.kv"));
    }

    #[test]
    fn ensure_kv_extension_is_idempotent() {
        assert_eq!(
            ensure_kv_extension(Path::new("foo.kv")),
            PathBuf::from("foo.kv")
        );
        assert_eq!(
            ensure_kv_extension(Path::new("foo")),
            PathBuf::from("foo.kv")
        );
    }

    #[test]
    fn resolve_output_path_defaults_to_unique_csv() {
        let path = resolve_output_path("benchmark", "m", &None, "csv").unwrap();
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("-m-bench.csv")
        );
    }

    #[test]
    fn atomic_write_creates_final_file_and_no_temp() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("out.txt");
        atomic_write(&final_path, |f| {
            writeln!(f, "hello")?;
            Ok(())
        })
        .unwrap();

        assert!(final_path.exists());
        assert_eq!(fs::read_to_string(&final_path).unwrap().trim(), "hello");
        assert!(!dir.path().join("out.txt.tmp").exists());
    }

    #[test]
    fn atomic_write_cleans_up_temp_on_failure() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("out.txt");

        let result: Result<()> = atomic_write(&final_path, |_f| {
            anyhow::bail!("simulated failure");
        });
        assert!(result.is_err());
        assert!(!final_path.exists());
    }
}
