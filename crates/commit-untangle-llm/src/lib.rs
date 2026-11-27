pub mod prompts;
pub mod provider;
pub mod schemas;

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub use crate::provider::{extract_json_value, ModelSpec, ProviderKind, ProviderModel};

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("unsupported or invalid model specifier '{0}' (expected e.g. anthropic::claude-sonnet-4-5, gemini::gemini-2.5-pro, openai::gpt-4.1, openrouter::..., or ollama::qwen2.5-coder)")]
    UnconfiguredProvider(String),

    #[error("missing required API key environment variable `{0}`")]
    MissingApiKey(String),

    #[error("HTTP transport error: {0}")]
    Http(String),

    #[error("provider returned HTTP {status}: {body}")]
    ProviderResponse { status: u16, body: String },

    #[error("model returned invalid structured output: {0}")]
    InvalidOutput(String),

    #[error("cassette replay miss for key {key} ({task:?})")]
    CassetteMiss { task: SemanticTask, key: String },

    #[error("JSON serialization/validation error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl LlmError {
    /// Returns `true` when the error indicates a permanent configuration, credential, or model-name
    /// error that will fail identically on subsequent calls and should not be silently ignored.
    pub fn is_fatal_config_or_auth(&self) -> bool {
        match self {
            Self::UnconfiguredProvider(_) | Self::MissingApiKey(_) => true,
            Self::ProviderResponse { status, body } => {
                if matches!(status, 401 | 403 | 404) {
                    return true;
                }
                if *status == 400 {
                    let lower = body.to_ascii_lowercase();
                    return lower.contains("not_found")
                        || lower.contains("model_not_found")
                        || lower.contains("does not exist")
                        || lower.contains("invalid_api_key")
                        || lower.contains("authentication_error")
                        || lower.contains("permission_error")
                        || (lower.contains("model") && lower.contains("invalid"));
                }
                false
            }
            _ => false,
        }
    }

    /// Optional actionable hint appended to CLI diagnostics for common provider errors.
    pub fn diagnostic_hint(&self) -> Option<&'static str> {
        match self {
            Self::ProviderResponse { status: 404, .. } => Some(
                "verify the model ID in `--model <provider::id>` (e.g. `anthropic::claude-haiku-4-5`, `anthropic::claude-sonnet-4-5`, `gemini::gemini-2.5-pro`, `openai::gpt-4.1`)",
            ),
            Self::ProviderResponse {
                status: 401 | 403, ..
            } => Some("check your provider API key environment variable and account permissions"),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticTask {
    RefineClusters,
    InferDependencies,
    DiagnoseBuildFailure,
    WriteCommitMessages,
    AutoCommitMessage,
    InteractiveChat,
    ToolChat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
}

impl ChatRequest {
    pub fn cache_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"interactive_chat");
        for m in &self.messages {
            hasher.update([0u8]);
            hasher.update(format!("{:?}", m.role).as_bytes());
            hasher.update([0u8]);
            hasher.update(m.content.as_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ToolChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(default)]
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChatRequest {
    pub messages: Vec<ToolChatMessage>,
    pub tools: Vec<ToolDefinition>,
    pub temperature: f32,
}

impl ToolChatRequest {
    pub fn cache_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(b"tool_chat");
        if let Ok(tools_json) = serde_json::to_string(&self.tools) {
            hasher.update([0u8]);
            hasher.update(tools_json.as_bytes());
        }
        for m in &self.messages {
            hasher.update([0u8]);
            if let Ok(msg_json) = serde_json::to_string(m) {
                hasher.update(msg_json.as_bytes());
            }
        }
        hex::encode(hasher.finalize())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolChatResponse {
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

#[async_trait]
pub trait ChatModel: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<String, LlmError>;

    async fn chat_with_tools(&self, request: ToolChatRequest)
        -> Result<ToolChatResponse, LlmError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredRequest {
    pub task: SemanticTask,
    pub prompt_version: String,
    pub system_prompt: String,
    pub user_prompt: String,
    pub json_schema: serde_json::Value,
    pub temperature: f32,
}

impl StructuredRequest {
    pub fn cache_key(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{:?}", self.task).as_bytes());
        hasher.update([0u8]);
        hasher.update(self.prompt_version.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.system_prompt.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.user_prompt.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.json_schema.to_string().as_bytes());
        hex::encode(hasher.finalize())
    }
}

#[async_trait]
pub trait SemanticModel: Send + Sync {
    async fn generate_json(
        &self,
        request: StructuredRequest,
    ) -> Result<serde_json::Value, LlmError>;
}

/// Typed convenience wrapper decoding structured JSON at the algorithm boundary,
/// with a 1-shot schema-validation repair retry when deserialization into `T` fails.
pub async fn generate_typed<T: DeserializeOwned>(
    model: &dyn SemanticModel,
    request: StructuredRequest,
) -> Result<T, LlmError> {
    let val = model.generate_json(request.clone()).await?;
    match serde_json::from_value::<T>(val) {
        Ok(parsed) => Ok(parsed),
        Err(first_err) => {
            let mut repair_req = request;
            repair_req.user_prompt.push_str(&format!(
                "\n\nYour previous JSON response failed schema validation ({first_err}). Return ONLY a valid JSON object strictly conforming to the schema."
            ));
            match model.generate_json(repair_req).await {
                Ok(repaired_val) => Ok(serde_json::from_value(repaired_val)?),
                Err(LlmError::CassetteMiss { .. }) => Err(LlmError::Json(first_err)),
                Err(other_err) => Err(other_err),
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CassetteEntry {
    pub key: String,
    pub task: SemanticTask,
    pub prompt_version: String,
    pub response: serde_json::Value,
}

/// Offline cassette replay model for deterministic testing and CI reproduction.
pub struct ReplayModel {
    entries: HashMap<String, serde_json::Value>,
}

impl ReplayModel {
    pub fn from_jsonl_file(path: &Path) -> Result<Self, LlmError> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut entries = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: CassetteEntry = serde_json::from_str(&line)?;
            entries.insert(entry.key, entry.response);
        }
        Ok(Self { entries })
    }
}

#[async_trait]
impl SemanticModel for ReplayModel {
    async fn generate_json(
        &self,
        request: StructuredRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let key = request.cache_key();
        self.entries
            .get(&key)
            .cloned()
            .ok_or(LlmError::CassetteMiss {
                task: request.task,
                key,
            })
    }
}

#[async_trait]
impl ChatModel for ReplayModel {
    async fn chat(&self, request: ChatRequest) -> Result<String, LlmError> {
        let key = request.cache_key();
        let val = self.entries.get(&key).ok_or(LlmError::CassetteMiss {
            task: SemanticTask::InteractiveChat,
            key,
        })?;
        val.as_str()
            .map(str::to_owned)
            .or_else(|| val.get("reply").and_then(|v| v.as_str()).map(str::to_owned))
            .ok_or_else(|| {
                LlmError::InvalidOutput(
                    "chat cassette entry must be a string or {reply}".to_owned(),
                )
            })
    }

    async fn chat_with_tools(
        &self,
        request: ToolChatRequest,
    ) -> Result<ToolChatResponse, LlmError> {
        let key = request.cache_key();
        let val = self.entries.get(&key).ok_or(LlmError::CassetteMiss {
            task: SemanticTask::ToolChat,
            key,
        })?;
        if let Ok(resp) = serde_json::from_value::<ToolChatResponse>(val.clone()) {
            return Ok(resp);
        }
        if let Some(reply) = val
            .as_str()
            .or_else(|| val.get("reply").and_then(|v| v.as_str()))
        {
            return Ok(ToolChatResponse {
                content: reply.to_owned(),
                tool_calls: Vec::new(),
            });
        }
        Err(LlmError::InvalidOutput(
            "tool chat cassette entry must be ToolChatResponse, string, or {reply}".to_owned(),
        ))
    }
}

/// Decorator recording requests and responses to a JSONL cassette file.
pub struct RecordingModel<M> {
    inner: M,
    path: PathBuf,
    lock: Mutex<()>,
}

impl<M> RecordingModel<M> {
    pub fn new(inner: M, path: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            path: path.into(),
            lock: Mutex::new(()),
        }
    }
}

#[async_trait]
impl<M: SemanticModel> SemanticModel for RecordingModel<M> {
    async fn generate_json(
        &self,
        request: StructuredRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let key = request.cache_key();
        let task = request.task;
        let prompt_version = request.prompt_version.clone();
        let response = self.inner.generate_json(request).await?;
        let entry = CassetteEntry {
            key,
            task,
            prompt_version,
            response: response.clone(),
        };
        self.append_entry(&entry)?;
        Ok(response)
    }
}

impl<M> RecordingModel<M> {
    fn append_entry(&self, entry: &CassetteEntry) -> Result<(), LlmError> {
        let serialized = serde_json::to_string(entry)?;
        let _guard = self.lock.lock().expect("cassette mutex");
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{serialized}")?;
        Ok(())
    }
}

#[async_trait]
impl<M: ChatModel> ChatModel for RecordingModel<M> {
    async fn chat(&self, request: ChatRequest) -> Result<String, LlmError> {
        let key = request.cache_key();
        let reply = self.inner.chat(request).await?;
        let entry = CassetteEntry {
            key,
            task: SemanticTask::InteractiveChat,
            prompt_version: prompts::PROMPT_VERSION.to_owned(),
            response: serde_json::json!({ "reply": reply }),
        };
        self.append_entry(&entry)?;
        Ok(reply)
    }

    async fn chat_with_tools(
        &self,
        request: ToolChatRequest,
    ) -> Result<ToolChatResponse, LlmError> {
        let key = request.cache_key();
        let response = self.inner.chat_with_tools(request).await?;
        let entry = CassetteEntry {
            key,
            task: SemanticTask::ToolChat,
            prompt_version: prompts::PROMPT_VERSION.to_owned(),
            response: serde_json::to_value(&response)?,
        };
        self.append_entry(&entry)?;
        Ok(response)
    }
}

/// Deterministic in-memory mock model for unit and integration tests.
#[derive(Default)]
pub struct MockModel {
    responses: Mutex<Vec<serde_json::Value>>,
}

impl MockModel {
    pub fn with_responses(responses: Vec<serde_json::Value>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }
}

#[async_trait]
impl SemanticModel for MockModel {
    async fn generate_json(
        &self,
        request: StructuredRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let mut guard = self.responses.lock().expect("mock mutex");
        if guard.is_empty() {
            return Err(LlmError::CassetteMiss {
                task: request.task,
                key: request.cache_key(),
            });
        }
        Ok(guard.remove(0))
    }
}
