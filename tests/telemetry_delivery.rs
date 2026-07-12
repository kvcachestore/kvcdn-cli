use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn cli_sends_telemetry_event_to_backend() {
    let received = Arc::new(Mutex::new(Vec::<String>::new()));
    let received_clone = received.clone();

    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind server");
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        _ => panic!("expected IP address"),
    };

    let telemetry_seen = Arc::new(AtomicBool::new(false));
    let telemetry_seen_clone = telemetry_seen.clone();

    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !telemetry_seen_clone.load(Ordering::Relaxed) {
            if let Ok(Some(mut request)) = server.try_recv() {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if body.contains("\"command\":\"api-key\"") {
                    telemetry_seen_clone.store(true, Ordering::Relaxed);
                }
                received_clone.lock().unwrap().push(body);
                let response = tiny_http::Response::from_string("").with_status_code(204);
                let _ = request.respond(response);
            }
            thread::sleep(Duration::from_millis(10));
        }
    });

    let bin =
        std::env::var("CARGO_BIN_EXE_kvcdn").unwrap_or_else(|_| "target/debug/kvcdn".to_string());

    let output = Command::new(&bin)
        .args(["api-key", "clear"])
        .env("KVCDN_API_URL", format!("http://127.0.0.1:{}", port))
        .env("KVCDN_TELEMETRY", "1")
        .output()
        .expect("failed to run kvcdn");

    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "kvcdn exited with non-zero status");

    handle.join().expect("server thread panicked");

    let bodies = received.lock().unwrap();
    let telemetry = bodies
        .iter()
        .find(|b| b.contains("\"command\":\"api-key\""))
        .expect("telemetry event not received by backend");

    let event: serde_json::Value = serde_json::from_str(telemetry).expect("valid json");
    assert_eq!(event["command"], "api-key");
    assert_eq!(event["success"], true);
    assert!(
        event["duration_ms"].as_u64().is_some(),
        "duration_ms should be present"
    );
}

#[test]
fn cli_does_not_send_telemetry_when_disabled() {
    let received = Arc::new(Mutex::new(Vec::<String>::new()));
    let received_clone = received.clone();

    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind server");
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        _ => panic!("expected IP address"),
    };

    let telemetry_seen = Arc::new(AtomicBool::new(false));
    let telemetry_seen_clone = telemetry_seen.clone();

    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline && !telemetry_seen_clone.load(Ordering::Relaxed) {
            if let Ok(Some(mut request)) = server.try_recv() {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if body.contains("\"command\"") {
                    telemetry_seen_clone.store(true, Ordering::Relaxed);
                }
                received_clone.lock().unwrap().push(body);
                let response = tiny_http::Response::from_string("").with_status_code(204);
                let _ = request.respond(response);
            }
            thread::sleep(Duration::from_millis(10));
        }
    });

    let bin =
        std::env::var("CARGO_BIN_EXE_kvcdn").unwrap_or_else(|_| "target/debug/kvcdn".to_string());

    let output = Command::new(&bin)
        .args(["api-key", "clear"])
        .env("KVCDN_API_URL", format!("http://127.0.0.1:{}", port))
        .env("KVCDN_TELEMETRY", "0")
        .output()
        .expect("failed to run kvcdn");

    assert!(output.status.success(), "kvcdn exited with non-zero status");

    handle.join().expect("server thread panicked");

    let bodies = received.lock().unwrap();
    assert!(
        !bodies.iter().any(|b| b.contains("\"command\"")),
        "telemetry event should not be sent when KVCDN_TELEMETRY=0"
    );
}

#[test]
fn cli_sends_telemetry_event_on_command_failure() {
    let received = Arc::new(Mutex::new(Vec::<String>::new()));
    let received_clone = received.clone();

    let server = tiny_http::Server::http("127.0.0.1:0").expect("bind server");
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        _ => panic!("expected IP address"),
    };

    let telemetry_seen = Arc::new(AtomicBool::new(false));
    let telemetry_seen_clone = telemetry_seen.clone();

    let handle = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !telemetry_seen_clone.load(Ordering::Relaxed) {
            if let Ok(Some(mut request)) = server.try_recv() {
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                if body.contains("\"command\":\"plot\"") {
                    telemetry_seen_clone.store(true, Ordering::Relaxed);
                }
                received_clone.lock().unwrap().push(body);
                let response = tiny_http::Response::from_string("").with_status_code(204);
                let _ = request.respond(response);
            }
            thread::sleep(Duration::from_millis(10));
        }
    });

    let bin =
        std::env::var("CARGO_BIN_EXE_kvcdn").unwrap_or_else(|_| "target/debug/kvcdn".to_string());

    let output = Command::new(&bin)
        .args(["plot", "--csv-path", "/tmp/does-not-exist-plot.csv"])
        .env("KVCDN_API_URL", format!("http://127.0.0.1:{}", port))
        .env("KVCDN_TELEMETRY", "1")
        .output()
        .expect("failed to run kvcdn");

    assert!(!output.status.success(), "kvcdn should have failed");

    handle.join().expect("server thread panicked");

    let bodies = received.lock().unwrap();
    let telemetry = bodies
        .iter()
        .find(|b| b.contains("\"command\":\"plot\""))
        .expect("telemetry event not received by backend");

    let event: serde_json::Value = serde_json::from_str(telemetry).expect("valid json");
    assert_eq!(event["command"], "plot");
    assert_eq!(event["success"], false);
    assert!(
        event["error_kind"].as_str().is_some(),
        "error_kind should be present on failure"
    );
}
