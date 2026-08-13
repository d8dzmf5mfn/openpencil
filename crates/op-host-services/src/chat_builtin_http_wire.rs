//! SSE transport + wire helpers for the built-in HTTP chat providers.
//!
//! The response pump, endpoint/base-URL normalization, and the
//! Anthropic / OpenAI SSE payload → `ChatDelta` parsers, carved off
//! `chat_builtin_http.rs` to keep both files under the 800-line cap.
//! `chat_builtin_http` re-exports this module's surface so existing
//! paths (`chat_builtin_http::map_openai_stop_reason`, …) are unchanged.

use futures::StreamExt;
use op_ai::chat_provider::{ChatDelta, StopReason};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::chat_builtin_http::BuiltinHttpError;

/// Apply the provider-specific low-reasoning control for structured design
/// turns. The two supported wire shapes are deliberately centralized here so
/// classic streaming and the tool-executing agent loop cannot drift or send
/// mutually-exclusive fields together.
pub fn apply_reasoning_wire_control(body: &mut Value, model: &str, reduce_reasoning: bool) {
    if !reduce_reasoning {
        return;
    }
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    match op_orchestrator::reasoning_wire_control(model) {
        Some(op_orchestrator::ReasoningWireControl::ThinkingDisabled) => {
            obj.remove("reasoning_effort");
            obj.insert("thinking".into(), serde_json::json!({ "type": "disabled" }));
        }
        Some(op_orchestrator::ReasoningWireControl::ReasoningEffortLow) => {
            obj.remove("thinking");
            obj.insert("reasoning_effort".into(), Value::String("low".into()));
        }
        Some(op_orchestrator::ReasoningWireControl::ReasoningEffortNone) => {
            obj.remove("thinking");
            obj.insert("reasoning_effort".into(), Value::String("none".into()));
        }
        None => {}
    }
}

pub(crate) async fn pump_sse_response(
    resp: reqwest::Response,
    tx: &mpsc::Sender<ChatDelta>,
    parse: fn(&str) -> Option<ChatDelta>,
) -> Result<bool, BuiltinHttpError> {
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::<u8>::new();
    let mut event_data = String::new();
    let mut emitted_done = false;

    while let Some(chunk) = stream.next().await {
        if tx.is_closed() {
            return Ok(true);
        }
        let bytes = chunk.map_err(|e| BuiltinHttpError::SseStream {
            message: e.to_string(),
        })?;
        buf.extend_from_slice(&bytes);
        while let Some(nl_pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl_pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');
            if line.is_empty() {
                if emit_sse_event(&mut event_data, tx, parse).await {
                    emitted_done = true;
                    break;
                }
                continue;
            }
            if let Some(data) = line.strip_prefix("data:") {
                if !event_data.is_empty() {
                    event_data.push('\n');
                }
                event_data.push_str(data.trim_start());
            }
        }
        if emitted_done {
            break;
        }
    }

    if !emitted_done && !buf.is_empty() {
        let line = String::from_utf8_lossy(&buf);
        let line = line.trim_end_matches('\n').trim_end_matches('\r');
        if let Some(data) = line.strip_prefix("data:") {
            if !event_data.is_empty() {
                event_data.push('\n');
            }
            event_data.push_str(data.trim_start());
        }
    }
    if !emitted_done && emit_sse_event(&mut event_data, tx, parse).await {
        emitted_done = true;
    }

    Ok(emitted_done)
}

async fn emit_sse_event(
    event_data: &mut String,
    tx: &mpsc::Sender<ChatDelta>,
    parse: fn(&str) -> Option<ChatDelta>,
) -> bool {
    let data = event_data.trim();
    if data.is_empty() {
        event_data.clear();
        return false;
    }
    let Some(delta) = parse(data) else {
        event_data.clear();
        return false;
    };
    let emitted_done = matches!(delta, ChatDelta::Done { .. });
    let emitted_error = matches!(delta, ChatDelta::Error(_));
    if tx.send(delta).await.is_err() {
        event_data.clear();
        return true;
    }
    if emitted_error {
        let _ = tx
            .send(ChatDelta::Done {
                stop_reason: StopReason::Aborted,
            })
            .await;
        event_data.clear();
        return true;
    }
    event_data.clear();
    emitted_done
}

pub(crate) fn provider_endpoint(base_url: &str, path: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    if base.ends_with(path) {
        return base.to_string();
    }
    if path == "/v1/messages" && base.ends_with("/v1") {
        return format!("{base}/messages");
    }
    format!("{base}{path}")
}

pub(crate) fn normalize_provider_base_url(base_url: &str) -> Result<String, BuiltinHttpError> {
    let url =
        reqwest::Url::parse(base_url).map_err(|error| BuiltinHttpError::InvalidEndpointUrl {
            message: error.to_string(),
        })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(BuiltinHttpError::EndpointUnsupportedScheme {
            scheme: url.scheme().to_string(),
        });
    }
    if url.host_str().is_none() {
        return Err(BuiltinHttpError::EndpointMissingHost);
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(BuiltinHttpError::EndpointHasQueryOrFragment);
    }
    Ok(url.as_str().trim_end_matches('/').to_string())
}

pub(crate) fn parse_openai_sse_data(data: &str) -> Option<ChatDelta> {
    let data = data.trim();
    if data == "[DONE]" {
        return Some(ChatDelta::Done {
            stop_reason: StopReason::EndTurn,
        });
    }
    let value: Value = serde_json::from_str(data).ok()?;
    if value.get("error").is_some() {
        // A provider-controlled HTTP-200 SSE event can reflect request headers
        // or credentials in its message. Preserve only the error boundary.
        return Some(ChatDelta::Error(
            "OpenAI-compatible provider reported a stream error".into(),
        ));
    }
    let choice = value.get("choices")?.as_array()?.first()?;
    if let Some(delta) = choice.get("delta") {
        // Some OpenAI-compatible providers put reasoning and the first
        // content token in the SAME delta. This classic parser returns one
        // event, so content must win; preferring reasoning here used to drop
        // the only script-bearing token and could leave orchestration empty.
        if let Some(content) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return Some(ChatDelta::TextDelta(content.to_string()));
        }
        if let Some(reasoning) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return Some(ChatDelta::Thinking(reasoning.to_string()));
        }
    }
    choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(|reason| ChatDelta::Done {
            stop_reason: map_openai_stop_reason(reason),
        })
}

pub(crate) fn parse_anthropic_sse_data(data: &str) -> Option<ChatDelta> {
    let value: Value = serde_json::from_str(data.trim()).ok()?;
    match value.get("type").and_then(Value::as_str).unwrap_or("") {
        "content_block_delta" => {
            let delta = value.get("delta")?;
            match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                "text_delta" => delta
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(|s| ChatDelta::TextDelta(s.to_string())),
                "thinking_delta" => delta
                    .get("thinking")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(|s| ChatDelta::Thinking(s.to_string())),
                _ => None,
            }
        }
        "message_delta" => value
            .pointer("/delta/stop_reason")
            .and_then(Value::as_str)
            .map(|reason| ChatDelta::Done {
                stop_reason: map_anthropic_stop_reason(reason),
            }),
        "message_stop" => Some(ChatDelta::Done {
            stop_reason: StopReason::EndTurn,
        }),
        "error" => Some(ChatDelta::Error(
            "Anthropic provider reported a stream error".into(),
        )),
        _ => None,
    }
}

pub fn map_anthropic_stop_reason(reason: &str) -> StopReason {
    match reason {
        "max_tokens" => StopReason::MaxTokens,
        "tool_use" => StopReason::ToolUse,
        "aborted" | "user_abort" => StopReason::Aborted,
        _ => StopReason::EndTurn,
    }
}

pub fn map_openai_stop_reason(reason: &str) -> StopReason {
    match reason {
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::Aborted,
        _ => StopReason::EndTurn,
    }
}
