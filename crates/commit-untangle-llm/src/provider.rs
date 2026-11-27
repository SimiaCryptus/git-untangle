use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::debug;

use crate::{
    ChatMessage, ChatModel, ChatRequest, ChatRole, LlmError, SemanticModel, StructuredRequest,
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
        })
    }

    async fn call_raw(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<String, LlmError> {
        match self.spec.provider {
            ProviderKind::Anthropic => self.call_anthropic(system, user, temperature).await,
            ProviderKind::Gemini => self.call_gemini(system, user, temperature).await,
            ProviderKind::OpenAi | ProviderKind::OpenRouter => {
                self.call_openai(system, user, temperature).await
            }
            ProviderKind::Ollama => self.call_ollama(system, user, temperature).await,
        }
    }

    async fn call_anthropic(
        &self,
        system: &str,
        user: &str,
        temperature: f32,
    ) -> Result<String, LlmError> {
        let url = format!("{}/v1/messages", self.base_url);
        let key = self.api_key.as_deref().unwrap_or("");
        let body = json!({
            "model": self.spec.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": system,
            "messages": [
                { "role": "user", "content": user }
            ]
        });

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
                "Anthropic response contained no text blocks".to_owned(),
            ));
        }
        Ok(text)
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
        let body = json!({
            "model": self.spec.model,
            "temperature": temperature,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user }
            ]
        });

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
            let raw = self.call_raw(&system, &prompt, request.temperature).await?;
            match extract_json_value(&raw) {
                Ok(val) => return Ok(val),
                Err(err) => {
                    debug!(
                        "provider JSON extraction failed on attempt {}: {}",
                        attempt + 1,
                        err
                    );
                    last_err = err;
                    if attempt < self.max_repair_retries {
                        prompt = format!(
                            "{}\n\nYour previous reply was not valid JSON ({}). Return ONLY a valid JSON object matching the schema with no prose.",
                            request.user_prompt, last_err
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

        let body = json!({
            "model": self.spec.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": system,
            "messages": turns,
        });

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

        let body = json!({
            "model": self.spec.model,
            "temperature": temperature,
            "messages": turns,
        });

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
}

#[async_trait]
impl ChatModel for ProviderModel {
    async fn chat(&self, request: ChatRequest) -> Result<String, LlmError> {
        match self.spec.provider {
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
        }
    }
}
