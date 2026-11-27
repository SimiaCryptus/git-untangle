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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticTask {
    RefineClusters,
    InferDependencies,
    DiagnoseBuildFailure,
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

/// Typed convenience wrapper decoding structured JSON at the algorithm boundary.
pub async fn generate_typed<T: DeserializeOwned>(
    model: &dyn SemanticModel,
    request: StructuredRequest,
) -> Result<T, LlmError> {
    let val = model.generate_json(request).await?;
    Ok(serde_json::from_value(val)?)
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
        let serialized = serde_json::to_string(&entry)?;
        let _guard = self.lock.lock().expect("cassette mutex");
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(f, "{serialized}")?;
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
