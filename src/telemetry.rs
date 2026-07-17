use serde::Serialize;
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct TelemetryEvent {
    pub command: String,
    pub version: String,
    pub duration_ms: u64,
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    Auth,
    Network,
    Config,
    Usage,
    Runtime,
}

impl ErrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorKind::Auth => "auth_error",
            ErrorKind::Network => "network_error",
            ErrorKind::Config => "config_error",
            ErrorKind::Usage => "usage_error",
            ErrorKind::Runtime => "runtime_error",
        }
    }
}

pub fn categorize_error(err: &anyhow::Error) -> ErrorKind {
    let err_str = err.to_string().to_lowercase();
    let chain = err
        .chain()
        .map(|e| e.to_string().to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");

    if err_str.contains("session expired")
        || err_str.contains("unauthorized")
        || err_str.contains("invalid api key")
        || err_str.contains("credential")
        || err_str.contains("refresh token")
        || err_str.contains("login")
    {
        return ErrorKind::Auth;
    }

    if err_str.contains("timeout")
        || err_str.contains("connection refused")
        || err_str.contains("dns")
        || err_str.contains("request failed")
        || chain.contains("reqwest")
        || chain.contains("http")
    {
        return ErrorKind::Network;
    }

    if err_str.contains("api_url")
        || err_str.contains("issuer_url")
        || err_str.contains("configured")
        || err_str.contains("config.toml")
        || err_str.contains("malformed")
    {
        return ErrorKind::Config;
    }

    if err_str.contains("usage")
        || err_str.contains("required")
        || err_str.contains("invalid")
        || err_str.contains("missing")
        || err_str.contains("argument")
    {
        return ErrorKind::Usage;
    }

    ErrorKind::Runtime
}

pub fn duration_ms(start: Instant) -> u64 {
    start.elapsed().as_millis().clamp(0, u64::MAX as u128) as u64
}

/// Submit telemetry in a background thread. Returns a receiver that completes when the
/// request finishes or gives up. The caller can wait a short, bounded time for delivery.
pub fn submit(event: TelemetryEvent, api_url: String) -> Option<mpsc::Receiver<()>> {
    if api_url.is_empty() {
        return None;
    }

    let body = match serde_json::to_vec(&event) {
        Ok(b) => b,
        Err(_) => return None,
    };

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _send_completion = Completion(tx);

        let client = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(_) => return,
        };

        let url = format!("{api_url}/api/v1/telemetry");
        let _ = client
            .post(&url)
            .header("content-type", "application/json")
            .body(body)
            .send();
    });

    Some(rx)
}

struct Completion(mpsc::Sender<()>);

impl Drop for Completion {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_kind_strings_are_stable() {
        assert_eq!(ErrorKind::Auth.as_str(), "auth_error");
        assert_eq!(ErrorKind::Network.as_str(), "network_error");
        assert_eq!(ErrorKind::Config.as_str(), "config_error");
        assert_eq!(ErrorKind::Usage.as_str(), "usage_error");
        assert_eq!(ErrorKind::Runtime.as_str(), "runtime_error");
    }

    #[test]
    fn categorizes_auth_error() {
        let err = anyhow::anyhow!("session expired; run `kvcdn login`");
        assert_eq!(categorize_error(&err), ErrorKind::Auth);
    }

    #[test]
    fn categorizes_network_error() {
        let err = anyhow::anyhow!("request failed (https://api.example.com): timeout");
        assert_eq!(categorize_error(&err), ErrorKind::Network);
    }

    #[test]
    fn categorizes_config_error() {
        let err = anyhow::anyhow!("api_url is not configured");
        assert_eq!(categorize_error(&err), ErrorKind::Config);
    }

    #[test]
    fn categorizes_usage_error() {
        let err = anyhow::anyhow!("missing required argument");
        assert_eq!(categorize_error(&err), ErrorKind::Usage);
    }

    #[test]
    fn falls_back_to_runtime_error() {
        let err = anyhow::anyhow!("tensor shape mismatch");
        assert_eq!(categorize_error(&err), ErrorKind::Runtime);
    }

    #[test]
    fn event_serializes_without_error_kind_when_success() {
        let event = TelemetryEvent {
            command: "verify".to_string(),
            version: "0.2.0".to_string(),
            duration_ms: 123,
            success: true,
            error_kind: None,
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"success\":true"));
        assert!(!json.contains("error_kind"));
    }

    #[test]
    fn event_serializes_error_kind_when_failure() {
        let event = TelemetryEvent {
            command: "upload".to_string(),
            version: "0.2.0".to_string(),
            duration_ms: 456,
            success: false,
            error_kind: Some("auth_error".to_string()),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"success\":false"));
        assert!(json.contains("\"error_kind\":\"auth_error\""));
    }

    #[test]
    fn submit_returns_none_when_api_url_empty() {
        let event = TelemetryEvent {
            command: "verify".to_string(),
            version: "0.2.0".to_string(),
            duration_ms: 10,
            success: true,
            error_kind: None,
        };
        assert!(submit(event, String::new()).is_none());
    }

    #[test]
    fn submit_posts_event_and_signals_completion() {
        let mut server = mockito::Server::new();
        let expected_body = serde_json::json!({
            "command": "verify",
            "version": "0.2.0",
            "duration_ms": 42,
            "success": true,
        });
        let mock = server
            .mock("POST", "/api/v1/telemetry")
            .match_header("content-type", "application/json")
            .match_body(mockito::Matcher::Json(expected_body))
            .with_status(204)
            .create();

        let event = TelemetryEvent {
            command: "verify".to_string(),
            version: "0.2.0".to_string(),
            duration_ms: 42,
            success: true,
            error_kind: None,
        };
        let rx = submit(event, server.url()).expect("submit should return a receiver");
        rx.recv_timeout(Duration::from_secs(1))
            .expect("telemetry thread should signal completion");
        mock.assert();
    }

    #[test]
    fn submit_signals_completion_when_upstream_fails() {
        let mut server = mockito::Server::new();
        let mock = server
            .mock("POST", "/api/v1/telemetry")
            .with_status(503)
            .create();

        let event = TelemetryEvent {
            command: "list".to_string(),
            version: "0.2.0".to_string(),
            duration_ms: 7,
            success: true,
            error_kind: None,
        };
        let rx = submit(event, server.url()).expect("submit should return a receiver");
        rx.recv_timeout(Duration::from_secs(1))
            .expect("telemetry thread should signal completion even on 5xx");
        mock.assert();
    }
}
