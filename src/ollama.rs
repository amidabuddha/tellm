//! Lifecycle management for the narrowly recognized local Ollama endpoint.
//!
//! This module owns all process-global Ollama state, readiness checks, model
//! unload requests, and tellm-started child cleanup. The runtime only decides
//! when those operations belong in a chat or shutdown flow.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::Mutex as AsyncMutex;
use tokio::task::spawn_blocking;
use tokio::time::{Instant as TokioInstant, sleep};

const CONNECT_TIMEOUT: Duration = Duration::from_millis(300);
const START_WAIT: Duration = Duration::from_secs(15);
const READY_POLL: Duration = Duration::from_millis(250);
const MODEL_INFO_TIMEOUT: Duration = Duration::from_secs(5);
const UNLOAD_TIMEOUT: Duration = Duration::from_secs(5);
const TERMINATE_WAIT: Duration = Duration::from_secs(2);
const TERMINATE_POLL: Duration = Duration::from_millis(100);

static START_LOCK: OnceLock<AsyncMutex<()>> = OnceLock::new();
static CHILD: OnceLock<Mutex<Option<ManagedChild>>> = OnceLock::new();
static LOADED_MODELS: OnceLock<Mutex<BTreeSet<LoadedModel>>> = OnceLock::new();
static MODEL_INFO_HTTP: OnceLock<reqwest::Client> = OnceLock::new();

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LoadedModel {
    base_url: String,
    model: String,
}

struct ManagedChild {
    child: Option<Child>,
}

impl ManagedChild {
    fn new(child: Child) -> Self {
        Self { child: Some(child) }
    }

    fn id(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    fn stop(mut self) -> Result<String, String> {
        let child = self
            .child
            .take()
            .ok_or_else(|| "Ollama child was already stopped".to_string())?;
        stop_child(child)
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        let Some(child) = self.child.take() else {
            return;
        };
        match stop_child(child) {
            Ok(message) => log::info!(target: "tellm::ollama", "{message}"),
            Err(error) => log::error!(
                target: "tellm::ollama",
                "spawned process stop failed during drop error={error:?}"
            ),
        }
    }
}

/// Panic-unwind backstop for a tellm-started Ollama child. Normal shutdown
/// calls [`stop_started`] first, leaving this guard with nothing to do.
pub(crate) struct CleanupGuard;

impl Drop for CleanupGuard {
    fn drop(&mut self) {
        stop_started_blocking();
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct UnloadSummary {
    attempted: usize,
    unloaded: Vec<String>,
    not_loaded: Vec<String>,
    failed: Vec<(String, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnloadOutcome {
    Unloaded,
    NotLoaded,
}

pub(crate) async fn ensure_ready(base_url: &str) -> Result<(), String> {
    let Some(addr) = local_addr(base_url) else {
        return Ok(());
    };

    if tcp_connects(addr.clone()).await? {
        return Ok(());
    }

    let _start_guard = start_lock().lock().await;
    if tcp_connects(addr.clone()).await? {
        return Ok(());
    }

    log::info!(
        target: "tellm::ollama",
        "local endpoint unreachable; starting `ollama serve` base_url={base_url:?}"
    );
    start_serve().await?;

    let deadline = TokioInstant::now() + START_WAIT;
    loop {
        sleep(READY_POLL).await;
        if tcp_connects(addr.clone()).await? {
            log::info!(target: "tellm::ollama", "local endpoint ready base_url={base_url:?}");
            return Ok(());
        }
        if TokioInstant::now() >= deadline {
            return Err(format!(
                "local Ollama endpoint {base_url} did not become reachable after {}s",
                START_WAIT.as_secs()
            ));
        }
    }
}

/// Reject image-bearing requests before inference when the narrowly
/// recognized local Ollama model does not advertise vision support. Other
/// chat-completions-compatible endpoints do not expose Ollama's model metadata
/// API and are left to their provider request path.
pub(crate) async fn require_vision_capability(base_url: &str, model: &str) -> Result<(), String> {
    let Some(addr) = local_addr(base_url) else {
        return Ok(());
    };

    require_vision_at(&format!("http://{addr}/api/show"), model).await
}

async fn require_vision_at(show_url: &str, model: &str) -> Result<(), String> {
    // Checked 2026-08-02 against docs.ollama.com/api-reference/show-model-details:
    // POST /api/show accepts `model` and returns a `capabilities` string array;
    // vision-capable models include the literal `vision`.
    let response = model_info_http()
        .post(show_url)
        .json(&serde_json::json!({ "model": model }))
        .send()
        .await
        .map_err(|error| format!("could not inspect local Ollama model {model:?}: {error}"))?;
    let status = response.status();
    let body: serde_json::Value = response.json().await.map_err(|error| {
        format!("local Ollama returned invalid model metadata for {model:?}: {error}")
    })?;

    if !status.is_success() {
        let detail = body
            .get("error")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("unknown Ollama model metadata error");
        return Err(format!(
            "could not inspect local Ollama model {model:?}: API error {}: {detail}",
            status.as_u16()
        ));
    }

    let capabilities = body
        .get("capabilities")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            format!(
                "could not verify image support for local Ollama model {model:?}: \
                 /api/show did not return capabilities"
            )
        })?;
    let capabilities = capabilities
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect::<Vec<_>>();

    if capabilities.contains(&"vision") {
        return Ok(());
    }

    let reported = if capabilities.is_empty() {
        "none".to_string()
    } else {
        capabilities.join(", ")
    };
    Err(format!(
        "local Ollama model {model:?} does not support image input \
         (reported capabilities: {reported}); choose a model whose `ollama show` output includes \
         `vision`"
    ))
}

fn model_info_http() -> &'static reqwest::Client {
    MODEL_INFO_HTTP.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(MODEL_INFO_TIMEOUT)
            .build()
            .expect("valid local Ollama model metadata HTTP client")
    })
}

pub(crate) fn remember_model(base_url: &str, model: &str) {
    if local_addr(base_url).is_none() {
        return;
    }

    loaded_models()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(LoadedModel {
            base_url: base_url.to_string(),
            model: model.to_string(),
        });
}

pub(crate) async fn unload_models() -> UnloadSummary {
    spawn_blocking(unload_models_blocking)
        .await
        .unwrap_or_else(|error| UnloadSummary {
            attempted: 1,
            failed: vec![(
                "tracked Ollama models".to_string(),
                format!("Ollama unload task failed: {error}"),
            )],
            ..UnloadSummary::default()
        })
}

pub(crate) fn unload_reply(summary: &UnloadSummary) -> String {
    if summary.attempted == 0 {
        return "No local Ollama models have been used by this tellm session.".to_string();
    }

    let mut parts = Vec::new();
    if !summary.unloaded.is_empty() {
        parts.push(format!(
            "Unloaded local Ollama model{}: {}.",
            plural(summary.unloaded.len()),
            summary.unloaded.join(", ")
        ));
    }
    if !summary.not_loaded.is_empty() {
        parts.push(format!(
            "Already not loaded local Ollama model{}: {}.",
            plural(summary.not_loaded.len()),
            summary.not_loaded.join(", ")
        ));
    }
    if !summary.failed.is_empty() {
        let failures = summary
            .failed
            .iter()
            .map(|(model, error)| format!("{model} ({error})"))
            .collect::<Vec<_>>()
            .join(", ");
        parts.push(format!(
            "Failed to unload local Ollama model{}: {failures}.",
            plural(summary.failed.len())
        ));
    }

    parts.join(" ")
}

pub(crate) async fn stop_started() {
    let Some(child) = take_child() else {
        return;
    };
    unload_started_models().await;
    match spawn_blocking(move || child.stop()).await {
        Ok(Ok(message)) => log::info!(target: "tellm::ollama", "{message}"),
        Ok(Err(error)) => {
            log::error!(target: "tellm::ollama", "spawned process stop failed error={error:?}")
        }
        Err(error) => {
            log::error!(target: "tellm::ollama", "shutdown task join failed error={error:?}")
        }
    }
}

fn start_lock() -> &'static AsyncMutex<()> {
    START_LOCK.get_or_init(|| AsyncMutex::new(()))
}

fn child() -> &'static Mutex<Option<ManagedChild>> {
    CHILD.get_or_init(|| Mutex::new(None))
}

fn loaded_models() -> &'static Mutex<BTreeSet<LoadedModel>> {
    LOADED_MODELS.get_or_init(|| Mutex::new(BTreeSet::new()))
}

async fn tcp_connects(addr: String) -> Result<bool, String> {
    spawn_blocking(move || {
        let addrs = addr
            .to_socket_addrs()
            .map_err(|error| format!("invalid Ollama listen address {addr}: {error}"))?;
        for socket_addr in addrs {
            if TcpStream::connect_timeout(&socket_addr, CONNECT_TIMEOUT).is_ok() {
                return Ok(true);
            }
        }
        Ok(false)
    })
    .await
    .map_err(|error| format!("Ollama readiness check task failed: {error}"))?
}

async fn start_serve() -> Result<(), String> {
    let spawned_child = spawn_blocking(|| {
        ProcessCommand::new("ollama")
            .arg("serve")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
    })
    .await
    .map_err(|error| format!("failed to start `ollama serve`: {error}"))?
    .map_err(|error| {
        format!("local Ollama is not running and `ollama serve` could not be started: {error}")
    })?;
    let managed_child = ManagedChild::new(spawned_child);
    let pid = managed_child.id().expect("newly spawned child has a pid");
    *child()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(managed_child);
    log::info!(target: "tellm::ollama", "started `ollama serve` pid={pid}");
    Ok(())
}

fn stop_started_blocking() {
    let Some(child) = take_child() else {
        return;
    };
    unload_started_models_blocking();
    match child.stop() {
        Ok(message) => log::info!(target: "tellm::ollama", "{message}"),
        Err(error) => {
            log::error!(target: "tellm::ollama", "spawned process stop failed error={error:?}")
        }
    }
}

fn take_child() -> Option<ManagedChild> {
    child()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
}

async fn unload_started_models() {
    let summary = unload_models().await;
    log_unload_summary(summary);
}

fn unload_started_models_blocking() {
    let summary = unload_models_blocking();
    log_unload_summary(summary);
}

fn log_unload_summary(summary: UnloadSummary) {
    for model in summary.unloaded {
        log::info!(target: "tellm::ollama", "model unloaded model={model:?}");
    }
    for model in summary.not_loaded {
        log::info!(target: "tellm::ollama", "model already not loaded model={model:?}");
    }
    for (model, error) in summary.failed {
        log::warn!(
            target: "tellm::ollama",
            "model unload failed model={model:?} error={error:?}"
        );
    }
}

fn unload_models_blocking() -> UnloadSummary {
    let models = {
        let models = loaded_models()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        models.iter().cloned().collect::<Vec<_>>()
    };
    let mut summary = UnloadSummary {
        attempted: models.len(),
        ..UnloadSummary::default()
    };

    for model in models {
        match unload_model(&model) {
            Ok(UnloadOutcome::Unloaded) => {
                loaded_models()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&model);
                summary.unloaded.push(model.model);
            }
            Ok(UnloadOutcome::NotLoaded) => {
                loaded_models()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&model);
                summary.not_loaded.push(model.model);
            }
            Err(error) => summary.failed.push((model.model, error)),
        }
    }

    summary
}

fn unload_model(model: &LoadedModel) -> Result<UnloadOutcome, String> {
    let addr = local_addr(&model.base_url)
        .ok_or_else(|| format!("not a local Ollama endpoint: {}", model.base_url))?;
    unload_model_blocking(&addr, &model.model)
}

fn unload_model_blocking(addr: &str, model: &str) -> Result<UnloadOutcome, String> {
    let mut stream = connect_tcp(addr)?;
    stream
        .set_read_timeout(Some(UNLOAD_TIMEOUT))
        .map_err(|error| format!("could not set Ollama unload read timeout: {error}"))?;
    stream
        .set_write_timeout(Some(UNLOAD_TIMEOUT))
        .map_err(|error| format!("could not set Ollama unload write timeout: {error}"))?;

    let request = unload_request(addr, model);
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("could not send Ollama unload request: {error}"))?;

    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|error| format!("could not read Ollama unload response: {error}"))?;
    unload_response_outcome(&response)
}

fn connect_tcp(addr: &str) -> Result<TcpStream, String> {
    let addrs = addr
        .to_socket_addrs()
        .map_err(|error| format!("invalid Ollama listen address {addr}: {error}"))?;
    for socket_addr in addrs {
        if let Ok(stream) = TcpStream::connect_timeout(&socket_addr, CONNECT_TIMEOUT) {
            return Ok(stream);
        }
    }
    Err(format!("could not connect to local Ollama endpoint {addr}"))
}

fn unload_request(addr: &str, model: &str) -> String {
    // Checked 2026-07-05 against docs.ollama.com/api/generate:
    // keep_alive accepts 0 to unload a model immediately.
    let body = serde_json::json!({
        "model": model,
        "prompt": "",
        "stream": false,
        "keep_alive": 0,
    })
    .to_string();
    format!(
        "POST /api/generate HTTP/1.1\r\n\
         Host: {addr}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    )
}

fn response_is_success(response: &str) -> bool {
    matches!(response_status_code(response), Some(code) if (200..300).contains(&code))
}

fn response_status_code(response: &str) -> Option<u16> {
    response
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
}

fn unload_response_outcome(response: &str) -> Result<UnloadOutcome, String> {
    if response_is_success(response) {
        return Ok(UnloadOutcome::Unloaded);
    }

    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or("");
    if response_status_code(response) == Some(404)
        || body.to_ascii_lowercase().contains("not found")
    {
        return Ok(UnloadOutcome::NotLoaded);
    }

    Err(format!(
        "Ollama unload returned {}",
        response.lines().next().unwrap_or("an empty HTTP response")
    ))
}

fn plural(count: usize) -> &'static str {
    if count == 1 { "" } else { "s" }
}

fn stop_child(mut child: Child) -> Result<String, String> {
    let pid = child.id();
    if let Some(status) = child
        .try_wait()
        .map_err(|error| format!("could not inspect pid {pid}: {error}"))?
    {
        return Ok(format!(
            "`ollama serve` pid {pid} already exited with {status}"
        ));
    }

    let fallback_reason = match terminate_child(&mut child) {
        Ok(method) => {
            if let Some(status) = wait_for_exit(&mut child)? {
                return Ok(format!(
                    "stopped tellm-started `ollama serve` pid {pid} after {method} with {status}"
                ));
            }
            format!("{method} timeout")
        }
        Err(error) => format!("failed graceful stop: {error}"),
    };

    if let Some(status) = child
        .try_wait()
        .map_err(|error| format!("could not inspect pid {pid}: {error}"))?
    {
        return Ok(format!(
            "stopped tellm-started `ollama serve` pid {pid} after {fallback_reason} with {status}"
        ));
    }

    child
        .kill()
        .map_err(|error| format!("could not kill pid {pid} after {fallback_reason}: {error}"))?;
    let status = child
        .wait()
        .map_err(|error| format!("could not wait for pid {pid}: {error}"))?;
    Ok(format!(
        "stopped tellm-started `ollama serve` pid {pid} with SIGKILL after {fallback_reason}: {status}"
    ))
}

#[cfg(unix)]
fn terminate_child(child: &mut Child) -> Result<&'static str, String> {
    let pid = child.id();
    let raw_pid = i32::try_from(pid).map_err(|_| format!("pid {pid} does not fit in pid_t"))?;
    let result = unsafe { libc::kill(raw_pid, libc::SIGTERM) };
    if result == 0 {
        Ok("SIGTERM")
    } else {
        Err(format!(
            "could not send SIGTERM to pid {pid}: {}",
            std::io::Error::last_os_error()
        ))
    }
}

#[cfg(not(unix))]
fn terminate_child(_child: &mut Child) -> Result<&'static str, String> {
    Err("graceful process termination is unsupported on this platform".to_string())
}

fn wait_for_exit(child: &mut Child) -> Result<Option<std::process::ExitStatus>, String> {
    let pid = child.id();
    let deadline = Instant::now() + TERMINATE_WAIT;
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| format!("could not inspect pid {pid}: {error}"))?
        {
            return Ok(Some(status));
        }
        if Instant::now() >= deadline {
            return Ok(None);
        }
        std::thread::sleep(TERMINATE_POLL);
    }
}

fn local_addr(base_url: &str) -> Option<String> {
    let rest = base_url.strip_prefix("http://")?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let (host, port) = split_host_port(authority)?;
    if port != "11434" {
        return None;
    }
    let normalized = host.trim_start_matches('[').trim_end_matches(']');
    match normalized {
        "localhost" | "127.0.0.1" | "::1" => Some(format!("{host}:{port}")),
        _ => None,
    }
}

fn split_host_port(authority: &str) -> Option<(&str, &str)> {
    if let Some(rest) = authority.strip_prefix('[') {
        let closing = rest.find(']')?;
        let host = &authority[..closing + 2];
        let port = rest[closing + 1..].strip_prefix(':')?;
        return Some((host, port));
    }
    authority.rsplit_once(':')
}

#[cfg(test)]
mod tests {
    use super::*;
    use tellm_test_support::{MockHttpServer, MockResponse};

    #[test]
    fn local_addr_only_matches_local_default_ollama_port() {
        assert_eq!(
            local_addr("http://localhost:11434/v1").as_deref(),
            Some("localhost:11434")
        );
        assert_eq!(
            local_addr("http://127.0.0.1:11434/v1/").as_deref(),
            Some("127.0.0.1:11434")
        );
        assert_eq!(
            local_addr("http://[::1]:11434/v1").as_deref(),
            Some("[::1]:11434")
        );
        assert_eq!(local_addr("https://api.mistral.ai/v1"), None);
        assert_eq!(local_addr("http://localhost:8080/v1"), None);
        assert_eq!(local_addr("http://192.168.1.10:11434/v1"), None);
    }

    #[tokio::test]
    async fn model_metadata_requires_reported_vision_capability() {
        let mock = MockHttpServer::start(vec![
            MockResponse::json(
                200,
                serde_json::json!({
                    "capabilities": ["completion", "vision"]
                }),
            ),
            MockResponse::json(
                200,
                serde_json::json!({
                    "capabilities": ["completion", "tools", "thinking"]
                }),
            ),
            MockResponse::json(200, serde_json::json!({ "details": {} })),
            MockResponse::json(404, serde_json::json!({ "error": "model not found" })),
        ]);
        let show_url = format!("{}/api/show", mock.base_url());

        require_vision_at(&show_url, "vision-model")
            .await
            .expect("vision capability must pass");

        let unsupported = require_vision_at(&show_url, "text-model")
            .await
            .expect_err("missing vision capability must fail");
        assert!(unsupported.contains("does not support image input"));
        assert!(unsupported.contains("completion, tools, thinking"));

        let missing = require_vision_at(&show_url, "unknown-capabilities")
            .await
            .expect_err("missing capabilities field must fail closed");
        assert!(missing.contains("did not return capabilities"));

        let not_found = require_vision_at(&show_url, "missing-model")
            .await
            .expect_err("metadata API errors must fail closed");
        assert!(not_found.contains("API error 404: model not found"));

        let requests = mock.requests();
        assert_eq!(requests.len(), 4);
        for (request, model) in requests.iter().zip([
            "vision-model",
            "text-model",
            "unknown-capabilities",
            "missing-model",
        ]) {
            assert_eq!(request.method, "POST");
            assert_eq!(request.path, "/api/show");
            assert_eq!(request.json_body(), serde_json::json!({ "model": model }));
        }
    }

    #[test]
    fn unload_request_uses_keep_alive_zero() {
        let request = unload_request("localhost:11434", "gemma4:31b-mlx");
        let (headers, body) = request.split_once("\r\n\r\n").unwrap();
        assert!(headers.starts_with("POST /api/generate HTTP/1.1"));
        assert!(headers.contains("Host: localhost:11434"));
        assert!(headers.contains(&format!("Content-Length: {}", body.len())));

        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["model"], "gemma4:31b-mlx");
        assert_eq!(body["prompt"], "");
        assert_eq!(body["stream"], false);
        assert_eq!(body["keep_alive"], 0);
    }

    #[test]
    fn unload_reply_reports_empty_success_and_failures() {
        assert_eq!(
            unload_reply(&UnloadSummary::default()),
            "No local Ollama models have been used by this tellm session."
        );
        assert_eq!(
            unload_reply(&UnloadSummary {
                attempted: 2,
                unloaded: vec!["llama3.3:70b".to_string(), "qwen3:32b".to_string()],
                not_loaded: Vec::new(),
                failed: Vec::new(),
            }),
            "Unloaded local Ollama models: llama3.3:70b, qwen3:32b."
        );
        let partial = unload_reply(&UnloadSummary {
            attempted: 3,
            unloaded: vec!["llama3.3:70b".to_string()],
            not_loaded: vec!["gemma4:31b-mlx".to_string()],
            failed: vec![("qwen3:32b".to_string(), "connection refused".to_string())],
        });
        assert!(
            partial.contains("Unloaded local Ollama model: llama3.3:70b."),
            "{partial}"
        );
        assert!(
            partial.contains("Already not loaded local Ollama model: gemma4:31b-mlx."),
            "{partial}"
        );
        assert!(
            partial.contains("Failed to unload local Ollama model: qwen3:32b"),
            "{partial}"
        );
    }

    #[test]
    fn response_success_only_accepts_2xx() {
        assert!(response_is_success("HTTP/1.1 200 OK\r\n\r\n{}"));
        assert!(response_is_success("HTTP/1.1 204 No Content\r\n\r\n"));
        assert!(!response_is_success("HTTP/1.1 404 Not Found\r\n\r\n"));
        assert!(!response_is_success(""));
    }

    #[test]
    fn unload_response_treats_missing_model_as_terminal() {
        assert_eq!(
            unload_response_outcome("HTTP/1.1 200 OK\r\n\r\n{}"),
            Ok(UnloadOutcome::Unloaded)
        );
        assert_eq!(
            unload_response_outcome("HTTP/1.1 404 Not Found\r\n\r\n{}"),
            Ok(UnloadOutcome::NotLoaded)
        );
        assert_eq!(
            unload_response_outcome(
                "HTTP/1.1 400 Bad Request\r\n\r\n{\"error\":\"model \\\"bad\\\" not found\"}"
            ),
            Ok(UnloadOutcome::NotLoaded)
        );

        let error = unload_response_outcome("HTTP/1.1 500 Server Error\r\n\r\n{}")
            .expect_err("500 remains a real unload failure");
        assert!(error.contains("HTTP/1.1 500 Server Error"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn stop_child_uses_sigterm_before_sigkill() {
        let spawned_child = ProcessCommand::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep test process");

        let message = stop_child(spawned_child).expect("stop child");
        assert!(message.contains("SIGTERM"), "{message}");
        assert!(!message.contains("SIGKILL"), "{message}");
    }

    #[cfg(unix)]
    #[test]
    fn cleanup_guard_drop_stops_tracked_child() {
        drop(take_child());
        let spawned_child = ProcessCommand::new("sleep")
            .arg("30")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn sleep test process");
        let pid = spawned_child.id();
        *child()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(ManagedChild::new(spawned_child));

        {
            let _guard = CleanupGuard;
        }

        assert!(take_child().is_none());
        assert!(!process_exists(pid), "pid {pid} should have exited");
    }

    #[cfg(unix)]
    fn process_exists(pid: u32) -> bool {
        let raw_pid = i32::try_from(pid).expect("test pid fits in pid_t");
        let result = unsafe { libc::kill(raw_pid, 0) };
        result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
    }
}
