use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl RecordedRequest {
    pub fn json_body(&self) -> Value {
        serde_json::from_slice(&self.body).expect("request body must be valid JSON")
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(String::as_str)
    }
}

pub struct MockResponse {
    status: u16,
    content_type: String,
    body: Vec<u8>,
    include_content_length: bool,
}

impl MockResponse {
    pub fn json_ok(result: Value) -> Self {
        Self::json(200, serde_json::json!({ "ok": true, "result": result }))
    }

    pub fn json_error(error_code: i64, description: &str) -> Self {
        Self::json(
            200,
            serde_json::json!({
                "ok": false,
                "error_code": error_code,
                "description": description,
            }),
        )
    }

    pub fn bytes(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: content_type.to_string(),
            body: body.into(),
            include_content_length: true,
        }
    }

    pub fn bytes_without_content_length(
        status: u16,
        content_type: &str,
        body: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            status,
            content_type: content_type.to_string(),
            body: body.into(),
            include_content_length: false,
        }
    }

    fn json(status: u16, value: Value) -> Self {
        Self {
            status,
            content_type: "application/json".to_string(),
            body: serde_json::to_vec(&value).expect("mock JSON response must serialize"),
            include_content_length: true,
        }
    }
}

pub struct MockTelegram {
    api_base_url: String,
    file_base_url: String,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

impl MockTelegram {
    pub fn start(responses: Vec<MockResponse>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("mock server must bind");
        let addr = listener.local_addr().expect("mock server must have addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_requests = Arc::clone(&requests);

        thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("mock server accept failed");
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .expect("mock server read timeout failed");
                let request = read_request(&mut stream);
                thread_requests
                    .lock()
                    .expect("mock request lock poisoned")
                    .push(request);
                write_response(&mut stream, response);
            }
        });

        let root = format!("http://{addr}");
        Self {
            api_base_url: root.clone(),
            file_base_url: format!("{root}/file"),
            requests,
        }
    }

    pub fn start_stalled() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("mock server must bind");
        let addr = listener.local_addr().expect("mock server must have addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let thread_requests = Arc::clone(&requests);

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("mock server accept failed");
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .expect("mock server read timeout failed");
            let request = read_request(&mut stream);
            thread_requests
                .lock()
                .expect("mock request lock poisoned")
                .push(request);
            thread::sleep(Duration::from_secs(5));
        });

        let root = format!("http://{addr}");
        Self {
            api_base_url: root.clone(),
            file_base_url: format!("{root}/file"),
            requests,
        }
    }

    pub fn api_base_url(&self) -> &str {
        &self.api_base_url
    }

    pub fn file_base_url(&self) -> &str {
        &self.file_base_url
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.requests
            .lock()
            .expect("mock request lock poisoned")
            .clone()
    }
}

fn read_request(stream: &mut TcpStream) -> RecordedRequest {
    let mut buffer = Vec::new();
    let mut chunk = [0; 1024];
    let mut header_end = None;

    while header_end.is_none() {
        let read = stream.read(&mut chunk).expect("mock server read failed");
        assert!(read > 0, "client closed connection before headers");
        buffer.extend_from_slice(&chunk[..read]);
        header_end = find_header_end(&buffer);
    }

    let header_end = header_end.expect("headers must be complete");
    let header_text = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().expect("request line missing");
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts
        .next()
        .expect("request method missing")
        .to_string();
    let path = request_parts
        .next()
        .expect("request path missing")
        .to_string();

    let mut headers = BTreeMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
    }

    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buffer.len() < body_start + content_length {
        let read = stream
            .read(&mut chunk)
            .expect("mock server body read failed");
        assert!(read > 0, "client closed connection before body");
        buffer.extend_from_slice(&chunk[..read]);
    }

    RecordedRequest {
        method,
        path,
        headers,
        body: buffer[body_start..body_start + content_length].to_vec(),
    }
}

fn write_response(stream: &mut TcpStream, response: MockResponse) {
    let reason = if response.status == 200 {
        "OK"
    } else {
        "ERROR"
    };
    let content_length = if response.include_content_length {
        format!("Content-Length: {}\r\n", response.body.len())
    } else {
        String::new()
    };
    let headers = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\n{content_length}Connection: close\r\n\r\n",
        response.status, reason, response.content_type
    );
    stream
        .write_all(headers.as_bytes())
        .expect("mock response headers write failed");
    stream
        .write_all(&response.body)
        .expect("mock response body write failed");
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

pub fn body_contains(body: &[u8], needle: &[u8]) -> bool {
    body.windows(needle.len()).any(|window| window == needle)
}
