pub mod cli_backend;
pub mod tempdir;
pub mod worktree;

use std::path::PathBuf;

use async_trait::async_trait;
use commit_untangle_core::bytes::RawBytes;
use commit_untangle_core::graph::ChangeGraph;
use commit_untangle_core::model::{ChangeItem, FileChange, Operation, RawFileDelta};
use commit_untangle_core::proposal::CommitMessageDraft;
use commit_untangle_core::schema::{CommitSummary, RangeArtifact, SeriesMetadata};
use indexmap::IndexMap;
use thiserror::Error;

pub use crate::cli_backend::GitCliBackend;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("repository precondition error: {0}")]
    Precondition(String),

    #[error("invalid revision or range: {0}")]
    InvalidTarget(String),

    #[error("git command `{cmd}` failed (exit {code}): {stderr}")]
    CommandFailed {
        cmd: String,
        code: i32,
        stderr: String,
    },

    #[error("git I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("core replay error: {0}")]
    Core(#[from] commit_untangle_core::error::CoreError),
}

#[derive(Clone, Debug)]
pub struct RepoInfo {
    pub toplevel: PathBuf,
    pub prefix: String,
    pub git_dir: PathBuf,
}

#[derive(Clone, Debug, Default)]
pub struct TargetSpec {
    pub spec: String,
    pub base_override: Option<String>,
    pub pathspecs: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedTarget {
    pub base: String,
    pub tip: String,
    pub short_id: String,
    pub is_range: bool,
    pub symmetric: bool,
    pub base_is_ancestor: bool,
    pub is_merge: bool,
    pub subject: String,
    pub message: String,
    pub author_name: String,
    pub author_email: String,
    pub author_date: String,
    pub commits: Vec<CommitSummary>,
    pub pathspecs: Vec<String>,
}

impl ResolvedTarget {
    pub fn to_range_artifact(&self) -> Option<RangeArtifact> {
        if !self.is_range {
            return None;
        }
        Some(RangeArtifact {
            base: self.base.clone(),
            tip: self.tip.clone(),
            symmetric: self.symmetric,
            base_is_ancestor: self.base_is_ancestor,
            commits: self.commits.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct EmitOptions {
    pub dry_run: bool,
    pub emit_branch: bool,
    pub branch_name: Option<String>,
    pub branch_prefix: String,
    pub emit_patches: bool,
    pub emit_item_diffs: bool,
    /// Optional LLM-authored commit messages keyed by `Operation.id`.
    pub commit_messages: IndexMap<String, CommitMessageDraft>,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            dry_run: false,
            emit_branch: true,
            branch_name: None,
            branch_prefix: "untangle/".to_owned(),
            emit_patches: false,
            emit_item_diffs: false,
            commit_messages: IndexMap::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DagEmitResult {
    pub metadata: SeriesMetadata,
    pub operation_patches: Vec<(String, RawBytes)>,
    pub item_patches: IndexMap<String, RawBytes>,
}

#[async_trait]
pub trait GitBackend: Send + Sync {
    async fn repo_info(&self) -> Result<RepoInfo, GitError>;

    async fn resolve_target(&self, spec: &TargetSpec) -> Result<ResolvedTarget, GitError>;

    async fn extract_deltas(&self, target: &ResolvedTarget) -> Result<Vec<RawFileDelta>, GitError>;

    async fn materialize_dag(
        &self,
        target: &ResolvedTarget,
        graph: &ChangeGraph,
        order: &[Operation],
        files: &[FileChange],
        items_by_id: &IndexMap<String, ChangeItem>,
        options: &EmitOptions,
    ) -> Result<DagEmitResult, GitError>;
}
