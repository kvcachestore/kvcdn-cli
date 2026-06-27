use std::fs;

use candle_core::DType;
use kvcdn::core::validation::validate_adapter;
use kvcdn::models::engine::{load_model, resolve_revision};

fn models_to_validate() -> Vec<String> {
    std::env::var("KVCDN_VALIDATE_MODELS")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn fixture_text() -> String {
    fs::read_to_string("tests/fixtures/context.txt").expect("reading fixture")
}

#[test]
fn validate_configured_adapters() {
    let models = models_to_validate();
    if models.is_empty() {
        eprintln!("KVCDN_VALIDATE_MODELS is unset; skipping adapter validation test");
        return;
    }

    let fixture = fixture_text();
    let parts: Vec<&str> = fixture.splitn(2, "Answer:").collect();
    let (context, question) = if parts.len() == 2 {
        (parts[0].trim(), "Answer:")
    } else {
        (fixture.as_str(), "What is the main point?")
    };

    let max_tokens: usize = std::env::var("KVCDN_VALIDATE_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let mut failures = Vec::new();
    for model_name in models {
        eprintln!("\n=== validating {model_name} ===");
        let revision = resolve_revision(None);
        let mut bundle = match load_model(&model_name, &revision, DType::F16) {
            Ok(b) => b,
            Err(e) => {
                failures.push(format!("{model_name}: load failed: {e}"));
                continue;
            }
        };

        let report = match validate_adapter(
            &mut *bundle.model,
            &bundle.device,
            &bundle.tokenizer,
            &model_name,
            context,
            question,
            max_tokens,
        ) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{model_name}: validation error: {e}"));
                continue;
            }
        };

        eprintln!(
            "{model_name}: reference={} tokens in {:?}, cache={} in {:?}, quant={} in {:?}",
            report.phases[0].tokens.len(),
            report.phases[0].duration,
            report.phases[1].tokens.len(),
            report.phases[1].duration,
            report.phases[2].tokens.len(),
            report.phases[2].duration,
        );

        if !report.passed {
            failures.push(format!("{model_name}: mismatch at {:?}", report.mismatches));
        }
    }

    assert!(
        failures.is_empty(),
        "adapter validation failed:\n{}",
        failures.join("\n")
    );
}
