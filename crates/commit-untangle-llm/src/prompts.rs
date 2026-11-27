use commit_untangle_core::graph::ChangeGraph;
use commit_untangle_core::model::{ChangeItem, ItemKind, Operation};
use indexmap::IndexMap;

use crate::schemas::{
    build_diagnosis_schema, cluster_refinement_schema, commit_messages_schema,
    dependency_proposal_schema,
};
use crate::{SemanticTask, StructuredRequest};

pub const PROMPT_VERSION: &str = "v1";
pub const DEFAULT_BATCH_CHAR_BUDGET: usize = 20_000;
pub const MAX_SUMMARIZED_ITEMS_PER_CLUSTER: usize = 40;

fn clip_line(s: &str, max_chars: usize) -> String {
    s.trim_end().chars().take(max_chars).collect()
}

pub fn item_summary(item: &ChangeItem, max_lines: usize) -> String {
    let mut out = String::new();
    out.push_str(&format!("- [{}] {}", item.id, item.display_path));
    if let Some(ref h) = item.hunk {
        out.push_str(&format!(":{}", h.start + 1));
    }
    out.push_str(&format!(" ({}", item.kind.as_str()));
    if let Some(ref h) = item.hunk {
        out.push_str(&format!(", -{}/+{}", h.old_lines.len(), h.new_lines.len()));
    }
    out.push(')');

    match item.kind {
        ItemKind::AddFile => {
            let n = item.added_bytes.to_display().lines().count();
            out.push_str(&format!(" new file, {n} lines"));
        }
        ItemKind::DeleteFile => {
            let n = item.removed_bytes.to_display().lines().count();
            out.push_str(&format!(" deleted file, {n} lines"));
        }
        ItemKind::ReplaceFile => {
            out.push_str(" whole-file replacement");
        }
        _ => {}
    }
    out.push('\n');

    if let Some(ref h) = item.hunk {
        for ol in h.old_lines.iter().take(max_lines) {
            out.push_str(&format!("    - {}\n", clip_line(&ol.to_display(), 140)));
        }
        if h.old_lines.len() > max_lines {
            out.push_str("    - ...\n");
        }
        for nl in h.new_lines.iter().take(max_lines) {
            out.push_str(&format!("    + {}\n", clip_line(&nl.to_display(), 140)));
        }
        if h.new_lines.len() > max_lines {
            out.push_str("    + ...\n");
        }
    } else {
        let defs: Vec<&str> = item
            .lexical
            .defined_added
            .union(&item.lexical.defined_removed)
            .take(8)
            .map(String::as_str)
            .collect();
        if !defs.is_empty() {
            out.push_str(&format!("    defines: {}\n", defs.join(", ")));
        }
    }
    out
}

/// Packs consecutive deterministic clusters into prompt-sized batches up to `char_budget`.
pub fn batch_clusters(
    clusters: &[Vec<String>],
    items_by_id: &IndexMap<String, ChangeItem>,
    char_budget: usize,
) -> Vec<Vec<Vec<String>>> {
    let mut batches = Vec::new();
    let mut current: Vec<Vec<String>> = Vec::new();
    let mut current_chars = 0usize;

    for cluster in clusters {
        let cluster_chars: usize = cluster
            .iter()
            .take(MAX_SUMMARIZED_ITEMS_PER_CLUSTER)
            .filter_map(|id| items_by_id.get(id))
            .map(|it| item_summary(it, 3).len())
            .sum();

        if !current.is_empty() && current_chars + cluster_chars > char_budget {
            batches.push(current);
            current = Vec::new();
            current_chars = 0;
        }
        current.push(cluster.clone());
        current_chars += cluster_chars;
    }

    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

pub fn build_cluster_refinement_request(
    batch: &[Vec<String>],
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    let system_prompt = "\
You are analyzing a git commit (or commit range) being decomposed into small, atomic, purpose-specific operations.
Each operation becomes its own commit branch and should express a single coherent intent.
Return ONLY valid JSON conforming to the provided schema with `actions` (merge_clusters, split_cluster, move_items) and `labels` keyed by resulting cluster ID (e.g. `c1`, `c1+c2`, `c3.1`).
Prefer operations that compile independently; keep a definition change together with tightly coupled call-site updates when inseparable.
Mark `suspicious: true` if an operation looks unrelated to the commit or hides a behavioral change inside a refactor/formatting change.".to_owned();

    let mut user = String::new();
    if !commit_context.trim().is_empty() {
        user.push_str("## Commit Context\n");
        user.push_str(&commit_context.chars().take(4000).collect::<String>());
        user.push_str("\n\n");
    }
    user.push_str("## Candidate Clusters\n");
    for (idx, cluster) in batch.iter().enumerate() {
        let cid = format!("c{}", idx + 1);
        user.push_str(&format!("### {cid}\n"));
        for item_id in cluster.iter().take(MAX_SUMMARIZED_ITEMS_PER_CLUSTER) {
            if let Some(item) = items_by_id.get(item_id) {
                user.push_str(&item_summary(item, 3));
            }
        }
        if cluster.len() > MAX_SUMMARIZED_ITEMS_PER_CLUSTER {
            let rest: Vec<&str> = cluster
                .iter()
                .skip(MAX_SUMMARIZED_ITEMS_PER_CLUSTER)
                .map(String::as_str)
                .collect();
            user.push_str(&format!(
                "- ...and {} more items in {cid}: {}\n",
                rest.len(),
                rest.join(", ")
            ));
        }
        user.push('\n');
    }
    user.push_str("Respond with JSON containing `actions` and `labels`.");

    StructuredRequest {
        task: SemanticTask::RefineClusters,
        prompt_version: PROMPT_VERSION.to_owned(),
        system_prompt,
        user_prompt: user,
        json_schema: cluster_refinement_schema(),
        temperature: 0.1,
    }
}

pub fn build_dependency_request(
    graph: &ChangeGraph,
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    let system_prompt = "\
You are ordering operations of a decomposed git commit into a DAG.
Infer semantic dependencies not visible in lexical def/use syntax (config keys required by code, schema migrations, behavioral changes required by tests) and propose a readable topological order among independent operations (refactor -> feature -> fix -> test -> docs/cleanup).
Return ONLY valid JSON conforming to the schema with `edges` and `suggested_order`.".to_owned();

    let mut user = String::new();
    if !commit_context.trim().is_empty() {
        user.push_str("## Commit Context\n");
        user.push_str(&commit_context.chars().take(3000).collect::<String>());
        user.push_str("\n\n");
    }

    user.push_str("## Operations\n");
    for op in &graph.ops {
        let desc_clip: String = op.description.chars().take(300).collect();
        user.push_str(&format!(
            "- {} [{}] {}: {}\n",
            op.id, op.category, op.name, desc_clip
        ));
        let files = op.files(items_by_id);
        let listed: Vec<&str> = files.iter().take(6).map(String::as_str).collect();
        if !listed.is_empty() {
            user.push_str(&format!("    files: {}\n", listed.join(", ")));
        }
        let defs: Vec<String> = op.defined_added(items_by_id).into_iter().take(10).collect();
        if !defs.is_empty() {
            user.push_str(&format!("    defines: {}\n", defs.join(", ")));
        }
        let refs: Vec<String> = op.refs_added(items_by_id).into_iter().take(10).collect();
        if !refs.is_empty() {
            user.push_str(&format!("    newly references: {}\n", refs.join(", ")));
        }
    }

    user.push_str("\n## Known (Syntactic) Dependencies\n");
    if graph.edges.is_empty() {
        user.push_str("(none)\n");
    } else {
        for edge in graph.edges.iter().take(200) {
            user.push_str(&format!(
                "- {} before {}: {}\n",
                edge.before, edge.after, edge.reason
            ));
        }
    }
    user.push_str("\nRespond with JSON containing `edges` and `suggested_order`.");

    StructuredRequest {
        task: SemanticTask::InferDependencies,
        prompt_version: PROMPT_VERSION.to_owned(),
        system_prompt,
        user_prompt: user,
        json_schema: dependency_proposal_schema(),
        temperature: 0.1,
    }
}

pub fn build_diagnosis_request(
    failing: &Operation,
    later: &[Operation],
    build_output: &str,
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    let system_prompt = "\
A git commit was decomposed into operations applied sequentially. After applying the failing operation, the build failed.
Identify which LATER operations must be applied before the failing operation so the build succeeds. Do not propose code edits.
Return ONLY valid JSON conforming to the schema with `required_operation_ids` and `explanation`.".to_owned();

    let mut user = String::new();
    if !commit_context.trim().is_empty() {
        user.push_str("## Commit Context\n");
        user.push_str(&commit_context.chars().take(1500).collect::<String>());
        user.push_str("\n\n");
    }
    let desc_clip: String = failing.description.chars().take(300).collect();
    user.push_str(&format!(
        "## Failing Operation\n{}: {} — {}\n",
        failing.id, failing.name, desc_clip
    ));
    let failing_files: Vec<String> = failing.files(items_by_id).into_iter().take(8).collect();
    user.push_str(&format!("files: {}\n\n", failing_files.join(", ")));

    let tail: String = build_output
        .chars()
        .rev()
        .take(4000)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    user.push_str("## Build Output (tail)\n```\n");
    user.push_str(&tail);
    user.push_str("\n```\n\n## Later Operations\n");
    for op in later.iter().take(100) {
        let files: Vec<String> = op.files(items_by_id).into_iter().take(4).collect();
        let defs: Vec<String> = op.defined_added(items_by_id).into_iter().take(8).collect();
        user.push_str(&format!(
            "- {}: {}; files: {}; defines: {}\n",
            op.id,
            op.name,
            files.join(", "),
            defs.join(", ")
        ));
    }
    user.push_str("\nRespond with JSON containing `required_operation_ids` and `explanation`.");

    StructuredRequest {
        task: SemanticTask::DiagnoseBuildFailure,
        prompt_version: PROMPT_VERSION.to_owned(),
        system_prompt,
        user_prompt: user,
        json_schema: build_diagnosis_schema(),
        temperature: 0.1,
    }
}

/// Packs operations into prompt-sized batches up to `char_budget` for commit-message authoring.
pub fn batch_operations(
    ops: &[Operation],
    items_by_id: &IndexMap<String, ChangeItem>,
    char_budget: usize,
) -> Vec<Vec<Operation>> {
    let mut batches = Vec::new();
    let mut current: Vec<Operation> = Vec::new();
    let mut current_chars = 0usize;

    for op in ops {
        let op_chars: usize = op.name.len()
            + op.description.len()
            + op.item_ids
                .iter()
                .take(MAX_SUMMARIZED_ITEMS_PER_CLUSTER)
                .filter_map(|id| items_by_id.get(id))
                .map(|it| item_summary(it, 3).len())
                .sum::<usize>();

        if !current.is_empty() && current_chars + op_chars > char_budget {
            batches.push(current);
            current = Vec::new();
            current_chars = 0;
        }
        current.push(op.clone());
        current_chars += op_chars;
    }

    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Builds a structured request asking the LLM to author crisp, idiomatic Git commit messages
/// (imperative subject <= 72 chars, concise explanatory body, and optional per-item subjects)
/// for a batch of operations.
pub fn build_commit_messages_request(
    ops: &[Operation],
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    let system_prompt = "\
You write clear, idiomatic Git commit messages for decomposed atomic operations.
For each operation id (e.g. `op1`), produce:
- `subject`: concise imperative commit subject (<= 72 characters, no trailing period)
- `body`: 1-3 concise sentences explaining why and what changed (wrap cleanly; omit git trailers as those are added automatically)
- `item_subjects` (optional): map from item id (e.g. `i3`) to a short imperative subject (<= 72 chars) when an operation spans multiple items.
Return ONLY valid JSON conforming to the schema with a top-level `messages` map keyed by operation id."
        .to_owned();

    let mut user = String::new();
    if !commit_context.trim().is_empty() {
        user.push_str("## Original Commit Context\n");
        user.push_str(&commit_context.chars().take(3000).collect::<String>());
        user.push_str("\n\n");
    }

    user.push_str("## Operations\n");
    for op in ops {
        user.push_str(&format!("### {} [{}] {}\n", op.id, op.category, op.name));
        if !op.description.trim().is_empty() {
            user.push_str(&format!("Current description: {}\n", op.description.trim()));
        }
        if let Some(ref rule) = op.rule {
            user.push_str(&format!("Mechanical rule: {rule}\n"));
        }
        for item_id in op.item_ids.iter().take(MAX_SUMMARIZED_ITEMS_PER_CLUSTER) {
            if let Some(item) = items_by_id.get(item_id) {
                user.push_str(&item_summary(item, 3));
            }
        }
        if op.item_ids.len() > MAX_SUMMARIZED_ITEMS_PER_CLUSTER {
            user.push_str(&format!(
                "- ...and {} more item(s)\n",
                op.item_ids.len() - MAX_SUMMARIZED_ITEMS_PER_CLUSTER
            ));
        }
        user.push('\n');
    }
    user.push_str("Respond with JSON containing `messages`.");

    StructuredRequest {
        task: SemanticTask::WriteCommitMessages,
        prompt_version: PROMPT_VERSION.to_owned(),
        system_prompt,
        user_prompt: user,
        json_schema: commit_messages_schema(),
        temperature: 0.2,
    }
}
