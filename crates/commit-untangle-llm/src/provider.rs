use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::debug;

use crate::{
    ChatMessage, ChatModel, ChatRequest, ChatRole, LlmError, SemanticModel, StructuredRequest,
    ToolCall, ToolChatMessage, ToolChatRequest, ToolChatResponse, ToolDefinition,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    Gemini,
    OpenAi,
    OpenRouter,
    Ollama,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub provider: ProviderKind,
    pub model: String,
}

impl ModelSpec {
    /// Parses a provider-qualified model string (`provider::model`) or infers the provider
    /// from well-known model name prefixes (`claude-*`, `gemini-*`, `gpt-*`, `o1*`, `o3*`, `o4*`).
    pub fn parse(raw: &str) -> Result<Self, LlmError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(LlmError::UnconfiguredProvider("<empty>".to_owned()));
        }

        if let Some((prov, model)) = trimmed.split_once("::") {
            let model = model.trim();
            if model.is_empty() {
                return Err(LlmError::UnconfiguredProvider(trimmed.to_owned()));
            }
            let provider = match prov.trim().to_ascii_lowercase().as_str() {
                "anthropic" => ProviderKind::Anthropic,
                "gemini" | "google" => ProviderKind::Gemini,
                "openai" => ProviderKind::OpenAi,
                "openrouter" => ProviderKind::OpenRouter,
                "ollama" => ProviderKind::Ollama,
                _ => return Err(LlmError::UnconfiguredProvider(trimmed.to_owned())),
            };
            return Ok(Self {
                provider,
                model: model.to_owned(),
            });
        }

        let lower = trimmed.to_ascii_lowercase();
        let provider = if lower.starts_with("claude") {
            ProviderKind::Anthropic
        } else if lower.starts_with("gemini") {
            ProviderKind::Gemini
        } else if lower.starts_with("gpt")
            || lower.starts_with("o1")
            || lower.starts_with("o3")
            || lower.starts_with("o4")
        {
            ProviderKind::OpenAi
        } else {
            return Err(LlmError::UnconfiguredProvider(trimmed.to_owned()));
        };

        Ok(Self {
            provider,
            model: trimmed.to_owned(),
        })
    }
}

/// Native multi-provider HTTP adapter supporting Anthropic, Gemini, OpenAI, OpenRouter, and Ollama
/// with automatic JSON fence extraction and 1-shot schema-repair retry.
pub struct ProviderModel {
    client: Client,
    spec: ModelSpec,
    api_key: Option<String>,
    base_url: String,
    max_repair_retries: usize,
    omit_temperature: AtomicBool,
}

fn is_temperature_unsupported_error(status: u16, body: &str) -> bool {
    if status != 400 {
        return false;
    }
    let lower = body.to_ascii_lowercase();
    lower.contains("temperature")
        && (lower.contains("deprecated")
            || lower.contains("unsupported")
            || lower.contains("not supported")
            || lower.contains("does not support")
            || lower.contains("not allowed")
            || lower.contains("invalid"))
}

impl ProviderModel {
    pub fn from_env(spec_str: &str) -> Result<Self, LlmError> {
        let spec = ModelSpec::parse(spec_str)?;
        let (api_key, base_url) = match spec.provider {
            ProviderKind::Anthropic => {
                let key = std::env::var("ANTHROPIC_API_KEY")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| LlmError::MissingApiKey("ANTHROPIC_API_KEY".to_owned()))?;
                let base = std::env::var("ANTHROPIC_BASE_URL")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| "https://api.anthropic.com".to_owned());
                (Some(key), base)
            }
            ProviderKind::Gemini => {
                let key = std::env::var("GEMINI_API_KEY")
                    .or_else(|_| std::env::var("GOOGLE_API_KEY"))
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| LlmError::MissingApiKey("GEMINI_API_KEY".to_owned()))?;
                let base = std::env::var("GEMINI_BASE_URL")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_owned());
                (Some(key), base)
            }
            ProviderKind::OpenAi => {
                let key = std::env::var("OPENAI_API_KEY")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| LlmError::MissingApiKey("OPENAI_API_KEY".to_owned()))?;
                let base = std::env::var("OPENAI_BASE_URL")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| "https://api.openai.com".to_owned());
                (Some(key), base)
            }
            ProviderKind::OpenRouter => {
                let key = std::env::var("OPENROUTER_API_KEY")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| LlmError::MissingApiKey("OPENROUTER_API_KEY".to_owned()))?;
                let base = std::env::var("OPENROUTER_BASE_URL")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| "https://openrouter.ai/api".to_owned());
                (Some(key), base)
            }
            ProviderKind::Ollama => {
                let base = std::env::var("OLLAMA_HOST")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
                    .unwrap_or_else(|| "http://127.0.0.1:11434".to_owned());
                (None, base)
            }
        };

        Self::new(spec, api_key, base_url)
    }

    pub fn new(
        spec: ModelSpec,
        api_key: Option<String>,
        base_url: impl Into<String>,
    ) -> Result<Self, LlmError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(180))
            .build()
            .map_err(|e| LlmError::Http(e.to_string()))?;
        Ok(Self {
            client,
            spec,
            api_key,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            max_repair_retries: 1,
            omit_temperature: AtomicBool::new(false),
        })
    }

    async fn call_raw_with_backoff(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<(String, bool), LlmError> {
        let mut attempt = 0usize;
        let max_http_retries = 2usize;
        loop {
            let res = self.call_raw_once(system, user, temperature).await;
            if let Err(LlmError::ProviderResponse { status, ref body }) = res {
                if !self.omit_temperature.load(Ordering::Relaxed)
                    && is_temperature_unsupported_error(status, body)
                {
                    debug!("model rejected `temperature`; retrying without temperature");
                    self.omit_temperature.store(true, Ordering::Relaxed);
                    continue;
                }
            }
            let should_retry = match &res {
                Err(LlmError::ProviderResponse { status, .. }) => {
                    matches!(status, 429 | 500 | 502 | 503 | 504 | 529)
                }
                _ => false,
            };
            if should_retry && attempt < max_http_retries {
                let delay_ms = 400u64 * (1u64 << attempt);
                debug!(
                    "transient provider error on attempt {}, backing off {}ms",
                    attempt + 1,
                    delay_ms
                );
                tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                attempt += 1;
                continue;
            }
            return res;
        }
    }

    async fn call_raw_once(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<(String, bool), LlmError> {
        match self.spec.provider {
            ProviderKind::Anthropic => self.call_anthropic(system, user, temperature).await,
            ProviderKind::Gemini => self
                .call_gemini(system, user, temperature)
                .await
                .map(|s| (s, false)),
            ProviderKind::OpenAi | ProviderKind::OpenRouter => self
                .call_openai(system, user, temperature)
                .await
                .map(|s| (s, false)),
            ProviderKind::Ollama => self
                .call_ollama(system, user, temperature)
                .await
                .map(|s| (s, false)),
        }
    }

    async fn call_anthropic(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<(String, bool), LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let mut body = json!({
            "model": self.spec.model,
            "max_tokens": 8192,
            "system": system,
            "messages": [
                { "role": "user", "content": user }
            ]
        });
        if !self.omit_temperature.load(Ordering::Relaxed) {
            body["temperature"] = json!(temperature);
        }

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        let truncated = val.get("stop_reason").and_then(Value::as_str) == Some("max_tokens");
        let mut text = String::new();
        if let Some(arr) = val.get("content").and_then(Value::as_array) {
            for block in arr {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
        }
        if text.is_empty() {
            return Err(LlmError::InvalidOutput(
                "Anthropic response contained no text blocks".to_owned(),
            ));
        }
        Ok((text, truncated))
    }

    async fn call_gemini(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.spec.model
        );
        let key = self.api_key.as_deref().unwrap_or("");
        let body = json!({
            "systemInstruction": {
                "parts": [{ "text": system }]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{ "text": user }]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json"
            }
        });

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        val.pointer("/candidates/0/content/parts/0/text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                LlmError::InvalidOutput("Gemini response missing candidates[0] text".to_owned())
            })
    }

    async fn call_openai(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let mut body = json!({
            "model": self.spec.model,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user }
            ]
        });
        if !self.omit_temperature.load(Ordering::Relaxed) {
            body["temperature"] = json!(temperature);
        }

        let resp = self
            .client
            .post(&url)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        val.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                LlmError::InvalidOutput(
                    "OpenAI response missing choices[0].message.content".to_owned(),
                )
            })
    }

    async fn call_ollama(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = json!({
            "model": self.spec.model,
            "stream": false,
            "format": "json",
            "options": { "temperature": temperature },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user }
            ]
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        val.pointer("/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                LlmError::InvalidOutput("Ollama response missing message.content".to_owned())
            })
    }
}

/// Extracts a JSON `Value` from raw model output, stripping optional Markdown code fences
/// or surrounding commentary around the outermost `{...}` object.
pub fn extract_json_value(raw: &str) -> Result<Value, String> {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Ok(v);
    }

    // Strip ```json ... ``` or ``` ... ``` fences
    if let Some(start) = trimmed.find("```") {
        let after_ticks = &trimmed[start + 3..];
        let body_start = after_ticks.find('\n').map_or(0, |i| i + 1);
        let rest = &after_ticks[body_start..];
        if let Some(end) = rest.rfind("```") {
            let candidate = rest[..end].trim();
            if let Ok(v) = serde_json::from_str::<Value>(candidate) {
                return Ok(v);
            }
        }
    }

    // Fallback: slice from first `{` to last `}`
    if let (Some(first), Some(last)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if last >= first {
            let candidate = &trimmed[first..=last];
            if let Ok(v) = serde_json::from_str::<Value>(candidate) {
                return Ok(v);
            }
        }
    }

    Err(format!(
        "could not parse JSON object from model response (prefix: {:?})",
        trimmed.chars().take(120).collect::<String>()
    ))
}

#[async_trait]
impl SemanticModel for ProviderModel {
    async fn generate_json(&self, request: StructuredRequest) -> Result<Value, LlmError> {
        let mut system = request.system_prompt.clone();
        system.push_str("\n\nExpected JSON Schema:\n");
        system.push_str(&request.json_schema.to_string());

        let mut prompt = request.user_prompt.clone();
        let mut last_err = String::new();

        for attempt in 0..=self.max_repair_retries {
            let (raw, truncated) = self
                .call_raw_with_backoff(&system, &prompt, request.temperature)
                .await?;
            match extract_json_value(&raw) {
                Ok(val) => return Ok(val),
                Err(err) => {
                    debug!(
                        "provider JSON extraction failed on attempt {}: {}",
                        attempt + 1,
                        err
                    );
                    last_err = if truncated {
                        format!("{err} (response truncated at max_tokens)")
                    } else {
                        err
                    };
                    if attempt < self.max_repair_retries {
                        let conciseness_note = if truncated {
                            " Keep string values concise so the complete JSON object fits within max_tokens."
                        } else {
                            ""
                        };
                        prompt = format!(
                            "{}\n\nYour previous reply was not valid JSON ({}). Return ONLY a valid JSON object matching the schema with no prose.{}",
                            request.user_prompt, last_err, conciseness_note
                        );
                    }
                }
            }
        }

        Err(LlmError::InvalidOutput(last_err))
    }
}

impl ProviderModel {
    async fn chat_anthropic(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let system: String = messages
            .iter()
            .filter(|m| m.role == ChatRole::System)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let turns: Vec<Value> = messages
            .iter()
            .filter(|m| m.role != ChatRole::System)
            .map(|m| {
                json!({
                    "role": match m.role {
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                        ChatRole::System => "user",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": self.spec.model,
            "max_tokens": 8192,
            "system": system,
            "messages": turns,
        });
        if !self.omit_temperature.load(Ordering::Relaxed) {
            body["temperature"] = json!(temperature);
        }

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        let mut text = String::new();
        if let Some(arr) = val.get("content").and_then(Value::as_array) {
            for block in arr {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
        }
        if text.is_empty() {
            return Err(LlmError::InvalidOutput(
                "Anthropic chat response contained no text blocks".to_owned(),
            ));
        }
        Ok(text)
    }

    async fn chat_gemini(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.spec.model
        );
        let key = self.api_key.as_deref().unwrap_or("");
        let system: String = messages
            .iter()
            .filter(|m| m.role == ChatRole::System)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let contents: Vec<Value> = messages
            .iter()
            .filter(|m| m.role != ChatRole::System)
            .map(|m| {
                json!({
                    "role": match m.role {
                        ChatRole::Assistant => "model",
                        _ => "user",
                    },
                    "parts": [{ "text": m.content }],
                })
            })
            .collect();

        let body = json!({
            "systemInstruction": {
                "parts": [{ "text": system }]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        });

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        val.pointer("/candidates/0/content/parts/0/text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                LlmError::InvalidOutput(
                    "Gemini chat response missing candidates[0] text".to_owned(),
                )
            })
    }

    async fn chat_openai(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let turns: Vec<Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let mut body = json!({
            "model": self.spec.model,
            "messages": turns,
        });
        if !self.omit_temperature.load(Ordering::Relaxed) {
            body["temperature"] = json!(temperature);
        }

        let resp = self
            .client
            .post(&url)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        val.pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                LlmError::InvalidOutput(
                    "OpenAI chat response missing choices[0].message.content".to_owned(),
                )
            })
    }

    async fn chat_ollama(
        &self,
        messages: &[ChatMessage],
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let turns: Vec<Value> = messages
            .iter()
            .map(|m| {
                json!({
                    "role": match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    },
                    "content": m.content,
                })
            })
            .collect();

        let body = json!({
            "model": self.spec.model,
            "stream": false,
            "options": { "temperature": temperature },
            "messages": turns,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        val.pointer("/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| {
                LlmError::InvalidOutput("Ollama chat response missing message.content".to_owned())
            })
    }

    async fn chat_with_tools_anthropic(
        &self,
        messages: &[ToolChatMessage],
        tools: &[ToolDefinition],
        temperature: f32,
    ) -> Result<ToolChatResponse, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let mut system_parts = Vec::new();
        let mut turns: Vec<Value> = Vec::new();

        let mut i = 0;
        while i < messages.len() {
            match &messages[i] {
                ToolChatMessage::System { content } => {
                    system_parts.push(content.as_str());
                    i += 1;
                }
                ToolChatMessage::User { content } => {
                    turns.push(json!({ "role": "user", "content": content }));
                    i += 1;
                }
                ToolChatMessage::Assistant {
                    content,
                    tool_calls,
                } => {
                    let mut blocks = Vec::new();
                    if !content.is_empty() {
                        blocks.push(json!({ "type": "text", "text": content }));
                    }
                    for call in tool_calls {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": call.arguments,
                        }));
                    }
                    if blocks.is_empty() {
                        blocks.push(json!({ "type": "text", "text": "" }));
                    }
                    turns.push(json!({ "role": "assistant", "content": blocks }));
                    i += 1;
                }
                ToolChatMessage::ToolResult { .. } => {
                    let mut blocks = Vec::new();
                    while let Some(ToolChatMessage::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                        ..
                    }) = messages.get(i)
                    {
                        let mut block = json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        });
                        if *is_error {
                            block["is_error"] = json!(true);
                        }
                        blocks.push(block);
                        i += 1;
                    }
                    turns.push(json!({ "role": "user", "content": blocks }));
                }
            }
        }

        let mut body = json!({
            "model": self.spec.model,
            "max_tokens": 8192,
            "system": system_parts.join("\n\n"),
            "messages": turns,
        });
        if !tools.is_empty() {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!(tool_defs);
        }
        if !self.omit_temperature.load(Ordering::Relaxed) {
            body["temperature"] = json!(temperature);
        }

        let resp = self
            .client
            .post(&url)
            .header("x-api-key", key)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        if let Some(arr) = val.get("content").and_then(Value::as_array) {
            for block in arr {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(Value::as_str) {
                            content.push_str(t);
                        }
                    }
                    Some("tool_use") => {
                        let id = block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("tool_call")
                            .to_owned();
                        let name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned();
                        let arguments = block.get("input").cloned().unwrap_or_else(|| json!({}));
                        if !name.is_empty() {
                            tool_calls.push(ToolCall {
                                id,
                                name,
                                arguments,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        if content.is_empty() && tool_calls.is_empty() {
            return Err(LlmError::InvalidOutput(
                "Anthropic tool-chat response contained neither text nor tool_use blocks"
                    .to_owned(),
            ));
        }
        Ok(ToolChatResponse {
            content,
            tool_calls,
        })
    }

    async fn chat_with_tools_openai(
        &self,
        messages: &[ToolChatMessage],
        tools: &[ToolDefinition],
        temperature: f32,
    ) -> Result<ToolChatResponse, LlmError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let turns: Vec<Value> = messages
            .iter()
            .map(|m| match m {
                ToolChatMessage::System { content } => {
                    json!({ "role": "system", "content": content })
                }
                ToolChatMessage::User { content } => {
                    json!({ "role": "user", "content": content })
                }
                ToolChatMessage::Assistant {
                    content,
                    tool_calls,
                } => {
                    let mut msg = json!({
                        "role": "assistant",
                        "content": if content.is_empty() && !tool_calls.is_empty() {
                            Value::Null
                        } else {
                            Value::String(content.clone())
                        },
                    });
                    if !tool_calls.is_empty() {
                        let tc: Vec<Value> = tool_calls
                            .iter()
                            .map(|c| {
                                json!({
                                    "id": c.id,
                                    "type": "function",
                                    "function": {
                                        "name": c.name,
                                        "arguments": c.arguments.to_string(),
                                    }
                                })
                            })
                            .collect();
                        msg["tool_calls"] = json!(tc);
                    }
                    msg
                }
                ToolChatMessage::ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }),
            })
            .collect();

        let mut body = json!({
            "model": self.spec.model,
            "messages": turns,
        });
        if !tools.is_empty() {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tool_defs);
        }
        if !self.omit_temperature.load(Ordering::Relaxed) {
            body["temperature"] = json!(temperature);
        }

        let resp = self
            .client
            .post(&url)
            .bearer_auth(key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        let msg = val.pointer("/choices/0/message").ok_or_else(|| {
            LlmError::InvalidOutput("OpenAI tool-chat response missing choices[0].message".to_owned())
        })?;
        let content = msg
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let mut tool_calls = Vec::new();
        if let Some(arr) = msg.get("tool_calls").and_then(Value::as_array) {
            for (idx, tc) in arr.iter().enumerate() {
                let id = tc
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("openai_call_{idx}"));
                let name = tc
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let args_val = tc.pointer("/function/arguments");
                let arguments = match args_val {
                    Some(Value::String(s)) if !s.trim().is_empty() => {
                        serde_json::from_str(s).unwrap_or_else(|_| json!({ "raw": s }))
                    }
                    Some(Value::Object(m)) => Value::Object(m.clone()),
                    _ => json!({}),
                };
                if !name.is_empty() {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
            }
        }
        Ok(ToolChatResponse {
            content,
            tool_calls,
        })
    }

    async fn chat_with_tools_gemini(
        &self,
        messages: &[ToolChatMessage],
        tools: &[ToolDefinition],
        temperature: f32,
    ) -> Result<ToolChatResponse, LlmError> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.spec.model
        );
        let key = self.api_key.as_deref().unwrap_or("");

        let mut system_parts = Vec::new();
        let mut contents: Vec<Value> = Vec::new();
        let mut i = 0;
        while i < messages.len() {
            match &messages[i] {
                ToolChatMessage::System { content } => {
                    system_parts.push(content.as_str());
                    i += 1;
                }
                ToolChatMessage::User { content } => {
                    contents.push(json!({
                        "role": "user",
                        "parts": [{ "text": content }]
                    }));
                    i += 1;
                }
                ToolChatMessage::Assistant {
                    content,
                    tool_calls,
                } => {
                    let mut parts = Vec::new();
                    if !content.is_empty() {
                        parts.push(json!({ "text": content }));
                    }
                    for call in tool_calls {
                        parts.push(json!({
                            "functionCall": {
                                "name": call.name,
                                "args": call.arguments,
                            }
                        }));
                    }
                    if parts.is_empty() {
                        parts.push(json!({ "text": "" }));
                    }
                    contents.push(json!({ "role": "model", "parts": parts }));
                    i += 1;
                }
                ToolChatMessage::ToolResult { .. } => {
                    let mut parts = Vec::new();
                    while let Some(ToolChatMessage::ToolResult {
                        tool_name,
                        content,
                        is_error,
                        ..
                    }) = messages.get(i)
                    {
                        parts.push(json!({
                            "functionResponse": {
                                "name": tool_name,
                                "response": {
                                    "result": content,
                                    "isError": is_error,
                                }
                            }
                        }));
                        i += 1;
                    }
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            }
        }

        let mut body = json!({
            "systemInstruction": {
                "parts": [{ "text": system_parts.join("\n\n") }]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": temperature
            }
        });
        if !tools.is_empty() {
            let decls: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!([{ "functionDeclarations": decls }]);
        }

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", key)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        let parts = val
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                LlmError::InvalidOutput(
                    "Gemini tool-chat response missing candidates[0].content.parts".to_owned(),
                )
            })?;

        let mut content = String::new();
        let mut tool_calls = Vec::new();
        for (idx, part) in parts.iter().enumerate() {
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                content.push_str(t);
            }
            if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let arguments = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                if !name.is_empty() {
                    tool_calls.push(ToolCall {
                        id: format!("gemini_call_{idx}"),
                        name,
                        arguments,
                    });
                }
            }
        }
        Ok(ToolChatResponse {
            content,
            tool_calls,
        })
    }

    async fn chat_with_tools_ollama(
        &self,
        messages: &[ToolChatMessage],
        tools: &[ToolDefinition],
        temperature: f32,
    ) -> Result<ToolChatResponse, LlmError> {
        let url = format!("{}/api/chat", self.base_url);
        let turns: Vec<Value> = messages
            .iter()
            .map(|m| match m {
                ToolChatMessage::System { content } => {
                    json!({ "role": "system", "content": content })
                }
                ToolChatMessage::User { content } => {
                    json!({ "role": "user", "content": content })
                }
                ToolChatMessage::Assistant {
                    content,
                    tool_calls,
                } => {
                    let mut msg = json!({ "role": "assistant", "content": content });
                    if !tool_calls.is_empty() {
                        let tc: Vec<Value> = tool_calls
                            .iter()
                            .map(|c| {
                                json!({
                                    "function": {
                                        "name": c.name,
                                        "arguments": c.arguments,
                                    }
                                })
                            })
                            .collect();
                        msg["tool_calls"] = json!(tc);
                    }
                    msg
                }
                ToolChatMessage::ToolResult {
                    tool_name,
                    content,
                    ..
                } => json!({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": content,
                }),
            })
            .collect();

        let mut body = json!({
            "model": self.spec.model,
            "stream": false,
            "options": { "temperature": temperature },
            "messages": turns,
        });
        if !tools.is_empty() {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = json!(tool_defs);
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = resp.status();
        let val: Value = resp
            .json()
            .await
            .map_err(|e| LlmError::Http(format!("HTTP {status}: {e}")))?;
        if !status.is_success() {
            return Err(LlmError::ProviderResponse {
                status: status.as_u16(),
                body: val.to_string(),
            });
        }

        let msg = val.get("message").ok_or_else(|| {
            LlmError::InvalidOutput("Ollama tool-chat response missing message".to_owned())
        })?;
        let content = msg
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let mut tool_calls = Vec::new();
        if let Some(arr) = msg.get("tool_calls").and_then(Value::as_array) {
            for (idx, tc) in arr.iter().enumerate() {
                let name = tc
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let args_val = tc.pointer("/function/arguments");
                let arguments = match args_val {
                    Some(Value::String(s)) if !s.trim().is_empty() => {
                        serde_json::from_str(s).unwrap_or_else(|_| json!({ "raw": s }))
                    }
                    Some(Value::Object(m)) => Value::Object(m.clone()),
                    _ => json!({}),
                };
                if !name.is_empty() {
                    tool_calls.push(ToolCall {
                        id: format!("ollama_call_{idx}"),
                        name,
                        arguments,
                    });
                }
            }
        }
        Ok(ToolChatResponse {
            content,
            tool_calls,
        })
    }
}

#[async_trait]
impl ChatModel for ProviderModel {
    async fn chat(&self, request: ChatRequest) -> Result<String, LlmError> {
        loop {
            let res = match self.spec.provider {
                ProviderKind::Anthropic => {
                    self.chat_anthropic(&request.messages, request.temperature)
                        .await
                }
                ProviderKind::Gemini => {
                    self.chat_gemini(&request.messages, request.temperature)
                        .await
                }
                ProviderKind::OpenAi | ProviderKind::OpenRouter => {
                    self.chat_openai(&request.messages, request.temperature)
                        .await
                }
                ProviderKind::Ollama => {
                    self.chat_ollama(&request.messages, request.temperature)
                        .await
                }
            };
            if let Err(LlmError::ProviderResponse { status, ref body }) = res {
                if !self.omit_temperature.load(Ordering::Relaxed)
                    && is_temperature_unsupported_error(status, body)
                {
                    debug!("chat model rejected `temperature`; retrying without temperature");
                    self.omit_temperature.store(true, Ordering::Relaxed);
                    continue;
                }
            }
            return res;
        }
    }

    async fn chat_with_tools(
        &self,
        request: ToolChatRequest,
    ) -> Result<ToolChatResponse, LlmError> {
        loop {
            let res = match self.spec.provider {
                ProviderKind::Anthropic => {
                    self.chat_with_tools_anthropic(
                        &request.messages,
                        &request.tools,
                        request.temperature,
                    )
                    .await
                }
                ProviderKind::Gemini => {
                    self.chat_with_tools_gemini(
                        &request.messages,
                        &request.tools,
                        request.temperature,
                    )
                    .await
                }
                ProviderKind::OpenAi | ProviderKind::OpenRouter => {
                    self.chat_with_tools_openai(
                        &request.messages,
                        &request.tools,
                        request.temperature,
                    )
                    .await
                }
                ProviderKind::Ollama => {
                    self.chat_with_tools_ollama(
                        &request.messages,
                        &request.tools,
                        request.temperature,
                    )
                    .await
                }
            };
            if let Err(LlmError::ProviderResponse { status, ref body }) = res {
                if !self.omit_temperature.load(Ordering::Relaxed)
                    && is_temperature_unsupported_error(status, body)
                {
                    debug!("tool-chat model rejected `temperature`; retrying without temperature");
                    self.omit_temperature.store(true, Ordering::Relaxed);
                    continue;
                }
            }
            return res;
        }
    }
}
