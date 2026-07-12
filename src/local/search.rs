use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::cli::SearchArgs;
use crate::core::common;
use crate::local::kv_io::{KVArtifact, read_kv_metadata};
use crate::local::tokenize::encode;
use crate::models::engine::{load_model_on, resolve_revision};

#[derive(Debug, Clone)]
pub struct ScoredMatch {
    pub path: PathBuf,
    pub artifact: KVArtifact,
    pub prefix_len: usize,
}

pub fn longest_common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn collect_kv_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                out.extend(collect_kv_files(&path));
            } else if path.extension().and_then(|e| e.to_str()) == Some("kv") {
                out.push(path);
            }
        }
    }
    out
}

fn read_artifact(path: &Path) -> Result<KVArtifact> {
    read_kv_metadata(path).with_context(|| format!("reading metadata for {}", path.display()))
}

fn rank_artifacts(files: &[PathBuf], model_name: &str, query_tokens: &[u32]) -> Vec<ScoredMatch> {
    let mut matches = Vec::new();
    for path in files {
        let artifact = match read_artifact(path) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("warning: {}", e);
                continue;
            }
        };
        if artifact.model_name != model_name {
            continue;
        }
        let Some(tokens) = artifact.tokens.as_ref() else {
            eprintln!(
                "warning: skipping {} (no tokens in sidecar)",
                path.display()
            );
            continue;
        };
        let prefix_len = longest_common_prefix(query_tokens, tokens);
        matches.push(ScoredMatch {
            path: path.clone(),
            artifact,
            prefix_len,
        });
    }
    matches.sort_by(|a, b| {
        b.prefix_len
            .cmp(&a.prefix_len)
            .then_with(|| b.artifact.num_tokens.cmp(&a.artifact.num_tokens))
    });
    matches
}

fn format_table(matches: &[ScoredMatch]) -> String {
    let mut out = String::new();
    out.push_str("path\tprefix_tokens\ttotal_tokens\tmodel\n");
    for m in matches {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            m.path.display(),
            m.prefix_len,
            m.artifact.num_tokens,
            m.artifact.model_name
        ));
    }
    out
}

fn format_json(matches: &[ScoredMatch]) -> Result<String> {
    let items: Vec<HashMap<String, serde_json::Value>> = matches
        .iter()
        .map(|m| {
            let mut map = HashMap::new();
            map.insert("path".to_string(), m.path.display().to_string().into());
            map.insert("prefix_tokens".to_string(), m.prefix_len.into());
            map.insert("total_tokens".to_string(), m.artifact.num_tokens.into());
            map.insert("model".to_string(), m.artifact.model_name.clone().into());
            map
        })
        .collect();
    Ok(serde_json::to_string_pretty(&items)?)
}

pub fn run(args: SearchArgs) -> Result<()> {
    let dir = Path::new(&args.dir);
    if !dir.is_dir() {
        anyhow::bail!("--dir '{}' is not a directory", args.dir);
    }

    let files = collect_kv_files(dir);
    if files.is_empty() {
        anyhow::bail!("no .kv artifacts found in {}", args.dir);
    }

    let device = args
        .device
        .clone()
        .unwrap_or_else(|| common::pick_device().unwrap_or(candle_core::Device::Cpu));
    // NOTE: default search only needs the tokenizer; loading the full model is a
    // temporary simplification. If search latency matters, split out a
    // tokenizer-only loader later.
    let mut bundle = load_model_on(
        &args.model,
        &resolve_revision(args.revision.as_deref()),
        candle_core::DType::F16,
        device.clone(),
    )?;
    let query_tokens =
        encode(&bundle.tokenizer, &args.query, false).with_context(|| "encoding query")?;

    let matches = rank_artifacts(&files, &args.model, &query_tokens);

    if matches.is_empty() {
        println!("no matching KV artifact found for model {}", args.model);
        if args.tree.is_some() {
            // With --tree we still want to generate from scratch.
            return crate::local::search_tree::run(&args, &mut bundle, &query_tokens, None);
        }
        return Ok(());
    }

    if args.tree.is_some() {
        return crate::local::search_tree::run(&args, &mut bundle, &query_tokens, matches.first());
    }

    match args.format.as_str() {
        "json" => println!("{}", format_json(&matches)?),
        _ => print!("{}", format_table(&matches)),
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_artifact(model_name: &str, tokens: Vec<u32>, num_tokens: usize) -> KVArtifact {
        KVArtifact {
            model_name: model_name.to_string(),
            num_layers: 1,
            num_tokens,
            dtype: "F16".to_string(),
            storage_dtype: Some("F16".to_string()),
            nbytes: 100,
            quantized: false,
            prompt: None,
            tokens: Some(tokens),
        }
    }

    #[test]
    fn longest_common_prefix_counts_matching_prefix() {
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(longest_common_prefix(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(longest_common_prefix(&[], &[1]), 0);
        assert_eq!(longest_common_prefix(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn rank_artifacts_picks_longest_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.kv");
        let b = dir.path().join("b.kv");
        fs::write(&a, b"dummy").unwrap();
        fs::write(&b, b"dummy").unwrap();
        let meta_a = make_artifact("m", vec![1, 2, 3, 4], 4);
        let meta_b = make_artifact("m", vec![1, 2, 5, 6], 4);
        fs::write(
            a.with_extension("kv.json"),
            serde_json::to_string_pretty(&meta_a).unwrap(),
        )
        .unwrap();
        fs::write(
            b.with_extension("kv.json"),
            serde_json::to_string_pretty(&meta_b).unwrap(),
        )
        .unwrap();

        let query = [1u32, 2, 3, 99];
        let files = vec![a.clone(), b.clone()];
        let ranked = rank_artifacts(&files, "m", &query);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].path, a);
        assert_eq!(ranked[0].prefix_len, 3);
        assert_eq!(ranked[1].path, b);
        assert_eq!(ranked[1].prefix_len, 2);
    }

    #[test]
    fn rank_artifacts_skips_missing_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.kv");
        fs::write(&a, b"dummy").unwrap();
        let mut meta = make_artifact("m", vec![1], 1);
        meta.tokens = None;
        fs::write(
            a.with_extension("kv.json"),
            serde_json::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();

        let files = vec![a];
        let ranked = rank_artifacts(&files, "m", &[1, 2, 3]);
        assert!(ranked.is_empty());
    }
}
