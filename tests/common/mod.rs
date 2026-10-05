//! A stub decision endpoint, shared by the integration tests.
//!
//! A plain `TcpListener` speaking just enough HTTP/1.1, so the real request is
//! built, serialised and sent, and the real response parsed — with no network
//! call, API key, or HTTP-server dependency. Living in a subdirectory keeps
//! Cargo from compiling it as a test binary of its own.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::TcpListener;
use tokio::sync::oneshot;

/// What the stub received.
///
/// Each test binary inspects a different subset, so per-binary dead-code
/// analysis would flag the fields the other one uses.
#[allow(dead_code)]
pub struct Received {
    /// The request target, e.g. `/api/alpha/decisions`.
    pub path: String,
    /// The `Authorization` header, if one was sent.
    pub authorization: Option<String>,
    /// The request body, or `Value::Null` if it was not JSON.
    pub body: Value,
}

/// A stub endpoint that serves exactly one request.
pub struct Stub {
    /// The base URL to point a client at.
    pub url: String,
    /// Resolves once the single request has been served.
    pub received: oneshot::Receiver<Received>,
    /// Held so the port stays bound for the test's lifetime.
    _listener: Arc<TcpListener>,
}

/// Serve one request on a loopback port, replying with `status` and `body`.
pub async fn stub(status: u16, body: &str) -> Stub {
    let listener = Arc::new(
        TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a loopback port"),
    );
    let port = listener.local_addr().expect("local addr").port();
    let (tx, received) = oneshot::channel();
    let body = body.to_owned();
    let accept = Arc::clone(&listener);

    tokio::spawn(async move {
        let (mut socket, _) = accept.accept().await.expect("accept");

        // Read until the headers end, then exactly Content-Length more, so the
        // read neither blocks forever nor truncates the payload.
        let mut raw = Vec::new();
        let mut buffer = [0_u8; 4096];
        let (head_end, content_length) = loop {
            let read = socket.read(&mut buffer).await.expect("read");
            if read == 0 {
                break (raw.len(), 0);
            }
            raw.extend_from_slice(&buffer[..read]);
            if let Some(position) = find(&raw, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&raw[..position]).into_owned();
                let length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                break (position + 4, length);
            }
        };

        while raw.len() < head_end + content_length {
            let read = socket.read(&mut buffer).await.expect("read body");
            if read == 0 {
                break;
            }
            raw.extend_from_slice(&buffer[..read]);
        }

        let head = String::from_utf8_lossy(&raw[..head_end.saturating_sub(4)]).into_owned();
        let mut lines = head.lines();
        let request_line = lines.next().unwrap_or_default().to_owned();
        let headers: HashMap<String, String> = lines
            .filter_map(|line| {
                let (name, value) = line.split_once(':')?;
                Some((name.to_ascii_lowercase(), value.trim().to_owned()))
            })
            .collect();

        let payload = &raw[head_end.min(raw.len())..];
        let seen = Received {
            path: request_line
                .split_whitespace()
                .nth(1)
                .unwrap_or_default()
                .to_owned(),
            authorization: headers.get("authorization").cloned(),
            body: serde_json::from_slice(payload).unwrap_or(Value::Null),
        };

        let reason = if (200..300).contains(&status) {
            "OK"
        } else {
            "Error"
        };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        socket.write_all(response.as_bytes()).await.expect("write");
        socket.flush().await.expect("flush");
        let _ = tx.send(seen);
    });

    Stub {
        url: format!("http://127.0.0.1:{port}"),
        received,
        _listener: listener,
    }
}

/// A client on `provider` pointed at a stub that answers once with `body`.
///
/// Both test binaries drive the same stub through a client configured this
/// way, so the model and base-URL wiring lives here rather than once per
/// binary. `key` stays a caller's choice because tests assert on it.
#[allow(dead_code)]
pub async fn stub_client(
    provider: arbiter::Provider,
    key: &str,
    status: u16,
    body: &str,
) -> (arbiter::Arbiter, oneshot::Receiver<Received>) {
    let stub = stub(status, body).await;
    let client = arbiter::Arbiter::for_provider(provider, key)
        .expect("build a client")
        .base_url(stub.url)
        .model(provider.pinned_model());
    (client, stub.received)
}

/// The first index of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// The response from the API reference, verbatim.
pub const DOCUMENTED: &str = r#"{
  "answers": {
    "is_bug": { "noul": 0.96, "type": "noul" },
    "team": {
      "choice": "payments", "confidence": 0.75,
      "probabilities": { "account": 0, "frontend": 0.16, "payments": 0.84 },
      "type": "choice"
    },
    "urgency": {
      "confidence": 0.99,
      "legend": { "0": "Can wait", "1": "This week", "2": "Blocking now" },
      "probabilities": { "0": 0, "1": 0.01, "2": 0.99 },
      "score": 1.99, "type": "score"
    }
  },
  "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
  "model": "typesafe/jev-1.13-20260917",
  "provider": "TypeSafe",
  "usage": { "cost": 0.000019992, "input_tokens": 476, "output_tokens": 70 }
}"#;
