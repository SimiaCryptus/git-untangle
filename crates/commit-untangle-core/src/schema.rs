use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::graph::ChangeGraph;
use crate::model::{ChangeItem, Operation};
use crate::verify::VerificationReport;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitSummary {
    pub sha: String,
    pub subject: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RangeArtifact {
    pub base: String,
    pub tip: String,
    pub symmetric: bool,
    pub base_is_ancestor: bool,
    pub commits: Vec<CommitSummary>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ItemArtifact {
    pub id: String,
    pub kind: String,
    pub file: String,
    pub old_path: Option<String>,
    pub line_start: Option<usize>,
    pub old_line_count: Option<usize>,
    pub new_line_count: Option<usize>,
    pub fingerprint: String,
    pub commit: Option<String>,
    /// Resolved commit subject line when materialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_subject: Option<String>,
    /// Resolved commit body description (excluding trailers) when materialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit_body: Option<String>,
    pub diff: Option<String>,
    /// Optional inline preview diff (UTF-8 lossy) so standalone viewers work even without patch files.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diff_preview: Option<String>,
    pub defines: Vec<String>,
    pub substitution: Option<[String; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationArtifact {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub commit: Option<String>,
    pub base_commit: Option<String>,
    pub merge_parents: Vec<String>,
    pub mechanical: bool,
    pub rule: Option<String>,
    pub confidence: f64,
    pub verified: Option<bool>,
    pub suspicious: bool,
    pub suspicion_reason: Option<String>,
    pub files: Vec<String>,
    pub items: Vec<ItemArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeArtifact {
    pub before: String,
    pub after: String,
    pub kind: String,
    pub reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SeriesMetadata {
    pub head_sha: Option<String>,
    pub tree_matches: Option<bool>,
    pub tree_diff: String,
    pub branch_created: Option<String>,
    pub branch_error: Option<String>,
    pub item_commits: IndexMap<String, String>,
    pub item_subjects: IndexMap<String, String>,
    pub item_bodies: IndexMap<String, String>,
    pub item_diff_files: IndexMap<String, String>,
    pub item_diff_previews: IndexMap<String, String>,
    pub operation_commits: IndexMap<String, String>,
    pub operation_bases: IndexMap<String, String>,
    pub operation_parents: IndexMap<String, Vec<String>>,
    pub merge_commits: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeriesArtifact {
    pub layout: String,
    pub head: Option<String>,
    pub tree_matches: Option<bool>,
    pub branch: Option<String>,
    pub branch_error: Option<String>,
    pub item_commits: usize,
    pub merge_commits: Vec<String>,
}

/// Top-level JSON output schema (`*_graph.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisArtifact {
    pub commit: String,
    pub parent: String,
    pub subject: String,
    pub range: Option<RangeArtifact>,
    pub exact_reconstruction: bool,
    pub operations: Vec<OperationArtifact>,
    pub edges: Vec<EdgeArtifact>,
    pub order: Vec<String>,
    pub verification: VerificationReport,
    pub series: SeriesArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn build_analysis_artifact(
    commit: &str,
    parent: &str,
    subject: &str,
    range: Option<RangeArtifact>,
    exact_reconstruction: bool,
    graph: &ChangeGraph,
    order: &[Operation],
    items_by_id: &IndexMap<String, ChangeItem>,
    verification: &VerificationReport,
    series: &SeriesMetadata,
) -> AnalysisArtifact {
    let operations = order
        .iter()
        .map(|op| {
            let items = op
                .item_ids
                .iter()
                .filter_map(|id| items_by_id.get(id))
                .map(|item| {
                    let defines: Vec<String> = item
                        .lexical
                        .defined_added
                        .union(&item.lexical.defined_removed)
                        .cloned()
                        .collect();
                    ItemArtifact {
                        id: item.id.clone(),
                        kind: item.kind.as_str().to_owned(),
                        file: item.display_path.clone(),
                        old_path: item.old_path.clone(),
                        line_start: item.hunk.as_ref().map(|h| h.start + 1),
                        old_line_count: item.hunk.as_ref().map(|h| h.old_lines.len()),
                        new_line_count: item.hunk.as_ref().map(|h| h.new_lines.len()),
                        fingerprint: item.fingerprint.clone(),
                        commit: series.item_commits.get(&item.id).cloned(),
                        commit_subject: series.item_subjects.get(&item.id).cloned(),
                        commit_body: series.item_bodies.get(&item.id).cloned(),
                        diff: series.item_diff_files.get(&item.id).cloned(),
                        diff_preview: series.item_diff_previews.get(&item.id).cloned(),
                        defines,
                        substitution: item
                            .substitution
                            .as_ref()
                            .map(|(a, b)| [a.clone(), b.clone()]),
                    }
                })
                .collect();

            OperationArtifact {
                id: op.id.clone(),
                name: op.name.clone(),
                description: op.description.clone(),
                category: op.category.clone(),
                commit: series.operation_commits.get(&op.id).cloned(),
                base_commit: series.operation_bases.get(&op.id).cloned(),
                merge_parents: series
                    .operation_parents
                    .get(&op.id)
                    .cloned()
                    .unwrap_or_default(),
                mechanical: op.is_mechanical(items_by_id),
                rule: op.rule.clone(),
                confidence: op.confidence,
                verified: op.verified,
                suspicious: op.suspicious,
                suspicion_reason: op.suspicion_reason.clone(),
                files: op.files(items_by_id),
                items,
            }
        })
        .collect();

    let edges = graph
        .edges
        .iter()
        .map(|e| EdgeArtifact {
            before: e.before.clone(),
            after: e.after.clone(),
            kind: e.kind.as_str().to_owned(),
            reason: e.reason.clone(),
        })
        .collect();

    AnalysisArtifact {
        commit: commit.to_owned(),
        parent: parent.to_owned(),
        subject: subject.to_owned(),
        range,
        exact_reconstruction,
        operations,
        edges,
        order: order.iter().map(|op| op.id.clone()).collect(),
        verification: verification.clone(),
        series: SeriesArtifact {
            layout: "dag".to_owned(),
            head: series.head_sha.clone(),
            tree_matches: series.tree_matches,
            branch: series.branch_created.clone(),
            branch_error: series.branch_error.clone(),
            item_commits: series.item_commits.len(),
            merge_commits: series.merge_commits.clone(),
        },
    }
}
