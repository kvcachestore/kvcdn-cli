use anyhow::{Context, Result, anyhow};
use std::io::Write;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use tiny_http::{Response, Server};
use url::Url;

pub struct CallbackServer {
    pub redirect_uri: String,
    rx: Option<Receiver<Result<String>>>,
    handle: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    server_addr: std::net::SocketAddr,
}

impl CallbackServer {
    /// Start a server on 127.0.0.1:0 and return the redirect URI + receiver for the auth code.
    pub fn start(state: &str) -> Result<Self> {
        let server = Server::http("127.0.0.1:0")
            .map_err(|e| anyhow!("failed to start callback server: {e}"))?;
        let server_addr = server
            .server_addr()
            .to_ip()
            .context("callback server not bound to an IP address")?;
        let redirect_uri = format!("http://127.0.0.1:{}/callback", server_addr.port());
        let expected_state = state.to_string();
        let (tx, rx): (Sender<Result<String>>, Receiver<Result<String>>) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let handle = thread::spawn(move || {
            let mut tx = Some(tx);
            for request in server.incoming_requests() {
                // If the server is being shut down without a real callback, stop accepting.
                if shutdown_clone.load(Ordering::SeqCst) {
                    let _ = request.respond(
                        Response::from_string("server shutting down").with_status_code(503),
                    );
                    break;
                }

                let url = format!("http://localhost{}", request.url());
                let parsed = match Url::parse(&url) {
                    Ok(u) => u,
                    Err(e) => {
                        let _ = request.respond(Response::from_string(format!("bad request: {e}")));
                        continue;
                    }
                };
                let pairs: std::collections::HashMap<_, _> = parsed
                    .query_pairs()
                    .map(|(k, v)| (k.into_owned(), v.into_owned()))
                    .collect();

                if pairs.get("state") != Some(&expected_state) {
                    if let Some(sender) = tx.take() {
                        let _ = sender.send(Err(anyhow!("state mismatch")));
                    }
                    let _ = request
                        .respond(Response::from_string("state mismatch").with_status_code(400));
                    break;
                }

                let response = if let Some(code) = pairs.get("code") {
                    if let Some(sender) = tx.take() {
                        let _ = sender.send(Ok(code.clone()));
                    }
                    Response::from_string("Login successful. You may close this tab.")
                } else if let Some(error) = pairs.get("error") {
                    if let Some(sender) = tx.take() {
                        let _ = sender.send(Err(anyhow!("OIDC error: {error}")));
                    }
                    Response::from_string(format!("error: {error}")).with_status_code(400)
                } else {
                    if let Some(sender) = tx.take() {
                        let _ = sender.send(Err(anyhow!("missing code or error")));
                    }
                    Response::from_string("missing code or error").with_status_code(400)
                };
                let _ = request.respond(response);
                break;
            }
        });

        Ok(Self {
            redirect_uri,
            rx: Some(rx),
            handle: Some(handle),
            shutdown,
            server_addr,
        })
    }

    /// Block until a callback is received.
    pub fn wait(mut self) -> Result<String> {
        let rx = self
            .rx
            .take()
            .context("callback server receiver already consumed")?;
        let result = rx
            .recv()
            .context("callback server channel closed")?
            .context("callback error");
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| anyhow!("callback server thread panicked"))?;
        }
        result
    }
}

impl Drop for CallbackServer {
    fn drop(&mut self) {
        if self.handle.is_some() {
            self.shutdown.store(true, Ordering::SeqCst);
            // Unblock the server thread by making a dummy request.
            if let Ok(mut stream) = TcpStream::connect(self.server_addr) {
                let _ = stream.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_returns_port_and_url() {
        let server = CallbackServer::start("abc123").unwrap();
        assert!(server.redirect_uri.starts_with("http://127.0.0.1:"));
        assert!(server.redirect_uri.contains("/callback"));
        drop(server);
    }

    #[test]
    fn callback_returns_code() {
        let server = CallbackServer::start("state123").unwrap();
        let uri = server.redirect_uri.clone();
        let request_handle = std::thread::spawn(move || {
            let url = format!("{uri}?code=abc&state=state123");
            let resp = reqwest::blocking::get(&url).unwrap();
            assert!(resp.status().is_success());
        });
        let code = server.wait().unwrap();
        assert_eq!(code, "abc");
        request_handle.join().unwrap();
    }

    #[test]
    fn callback_rejects_state_mismatch() {
        let server = CallbackServer::start("state123").unwrap();
        let uri = server.redirect_uri.clone();
        let request_handle = std::thread::spawn(move || {
            let url = format!("{uri}?code=abc&state=wrong");
            let resp = reqwest::blocking::get(&url).unwrap();
            assert_eq!(resp.status().as_u16(), 400);
        });
        let err = server.wait().unwrap_err();
        assert!(format!("{err:#}").contains("state mismatch"));
        request_handle.join().unwrap();
    }

    #[test]
    fn callback_returns_oidc_error() {
        let server = CallbackServer::start("state123").unwrap();
        let uri = server.redirect_uri.clone();
        let request_handle = std::thread::spawn(move || {
            let url = format!("{uri}?error=access_denied&state=state123");
            let resp = reqwest::blocking::get(&url).unwrap();
            assert_eq!(resp.status().as_u16(), 400);
        });
        let err = server.wait().unwrap_err();
        assert!(format!("{err:#}").contains("access_denied"));
        request_handle.join().unwrap();
    }
}
