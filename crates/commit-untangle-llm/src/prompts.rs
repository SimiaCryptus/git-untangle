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
pub const MAX_ITEMS_PER_COMMIT_MSG_BATCH: usize = 12;

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
        match item.kind {
            ItemKind::AddFile | ItemKind::ReplaceFile => {
                let disp = item.added_bytes.to_display();
                let non_empty: Vec<&str> = disp
                    .lines()
                    .map(str::trim_end)
                    .filter(|l| !l.trim().is_empty())
                    .collect();
                for line in non_empty.iter().take(max_lines) {
                    out.push_str(&format!("    + {}\n", clip_line(line, 140)));
                }
                if non_empty.len() > max_lines {
                    out.push_str("    + ...\n");
                }
            }
            ItemKind::DeleteFile => {
                let disp = item.removed_bytes.to_display();
                let non_empty: Vec<&str> = disp
                    .lines()
                    .map(str::trim_end)
                    .filter(|l| !l.trim().is_empty())
                    .collect();
                for line in non_empty.iter().take(max_lines) {
                    out.push_str(&format!("    - {}\n", clip_line(line, 140)));
                }
                if non_empty.len() > max_lines {
                    out.push_str("    - ...\n");
                }
            }
            _ => {}
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

/// Unit of work for LLM commit-message authoring.
///
/// Small/medium operations (`<= MAX_ITEMS_PER_COMMIT_MSG_BATCH` items) are packed together via
/// [`CommitMessageBatch::Operations`], while large operations (`> MAX_ITEMS_PER_COMMIT_MSG_BATCH`
/// items) are sliced into bounded item chunks via [`CommitMessageBatch::OperationSlice`] so every
/// single item receives an authored commit subject and description without hitting output limits.
#[derive(Debug, Clone)]
pub enum CommitMessageBatch {
    Operations(Vec<Operation>),
    OperationSlice {
        op: Operation,
        slice_item_ids: Vec<String>,
        slice_index: usize,
        total_slices: usize,
    },
}

impl CommitMessageBatch {
    pub fn expected_op_ids(&self) -> Vec<String> {
        match self {
            Self::Operations(ops) => ops.iter().map(|o| o.id.clone()).collect(),
            Self::OperationSlice { op, .. } => vec![op.id.clone()],
        }
    }
}

/// Packs operations into prompt-sized batches up to `char_budget` and `MAX_ITEMS_PER_COMMIT_MSG_BATCH`
/// for commit-message authoring.
pub fn batch_operations(
    ops: &[Operation],
    items_by_id: &IndexMap<String, ChangeItem>,
    char_budget: usize,
) -> Vec<Vec<Operation>> {
    let mut batches = Vec::new();
    let mut current: Vec<Operation> = Vec::new();
    let mut current_chars = 0usize;
    let mut current_items = 0usize;

    for op in ops {
        let op_items = op.item_ids.len().max(1);
        let op_chars: usize = op.name.len()
            + op.description.len()
            + op.item_ids
                .iter()
                .take(MAX_SUMMARIZED_ITEMS_PER_CLUSTER)
                .filter_map(|id| items_by_id.get(id))
                .map(|it| item_summary(it, 4).len())
                .sum::<usize>();

        if !current.is_empty()
            && (current_chars + op_chars > char_budget
                || current_items + op_items > MAX_ITEMS_PER_COMMIT_MSG_BATCH)
        {
            batches.push(current);
            current = Vec::new();
            current_chars = 0;
            current_items = 0;
        }
        current.push(op.clone());
        current_chars += op_chars;
        current_items += op_items;
    }

    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Plans commit-message batches across operations, automatically slicing oversized operations
/// (`> MAX_ITEMS_PER_COMMIT_MSG_BATCH` items) into bounded item chunks so every item is covered.
pub fn plan_commit_message_batches(
    ops: &[Operation],
    items_by_id: &IndexMap<String, ChangeItem>,
    char_budget: usize,
) -> Vec<CommitMessageBatch> {
    let mut planned = Vec::new();
    let mut pending_small: Vec<Operation> = Vec::new();

    let flush_small = |small: &mut Vec<Operation>, out: &mut Vec<CommitMessageBatch>| {
        if !small.is_empty() {
            for group in batch_operations(small, items_by_id, char_budget) {
                out.push(CommitMessageBatch::Operations(group));
            }
            small.clear();
        }
    };

    for op in ops {
        if op.item_ids.len() > MAX_ITEMS_PER_COMMIT_MSG_BATCH {
            flush_small(&mut pending_small, &mut planned);
            let chunks: Vec<&[String]> =
                op.item_ids.chunks(MAX_ITEMS_PER_COMMIT_MSG_BATCH).collect();
            let total_slices = chunks.len();
            for (slice_index, chunk) in chunks.into_iter().enumerate() {
                planned.push(CommitMessageBatch::OperationSlice {
                    op: op.clone(),
                    slice_item_ids: chunk.to_vec(),
                    slice_index,
                    total_slices,
                });
            }
        } else {
            pending_small.push(op.clone());
        }
    }
    flush_small(&mut pending_small, &mut planned);
    planned
}

pub fn build_commit_message_batch_request(
    batch: &CommitMessageBatch,
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    match batch {
        CommitMessageBatch::Operations(ops) => {
            build_commit_messages_request(ops, commit_context, items_by_id)
        }
        CommitMessageBatch::OperationSlice {
            op,
            slice_item_ids,
            slice_index,
            total_slices,
        } => build_operation_slice_request(
            op,
            slice_item_ids,
            *slice_index,
            *total_slices,
            commit_context,
            items_by_id,
        ),
    }
}

/// Builds a structured request asking the LLM to author crisp, idiomatic Git commit messages
/// (imperative subject <= 72 chars, concise explanatory body, and per-item subjects + bodies)
/// for a batch of operations.
pub fn build_commit_messages_request(
    ops: &[Operation],
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    let system_prompt = "\
You write clear, idiomatic Git commit messages for decomposed atomic operations.
For EVERY operation id listed below (using exact keys such as `op1`), produce an entry in `messages`:
- `subject`: concise imperative commit subject (<= 72 characters, no trailing period; match repo style such as `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:` when appropriate). Never use generic counts like \"Add 32 files\".
- `body`: 1-3 substantive sentences explaining what changed and why (omit git trailers; never write boilerplate like \"Changes N change item(s)\").
- `item_subjects`: when an operation has more than 1 item, include an imperative commit subject (<= 72 chars) for EVERY item id (e.g. `i3`) in that operation.
- `item_bodies`: when an operation has more than 1 item, include a concise 1-2 sentence commit description for EVERY item id explaining that specific file or hunk change.
Return ONLY valid JSON conforming to the schema with a top-level `messages` map keyed by exact operation id."
        .to_owned();

    let mut user = String::new();
    if !commit_context.trim().is_empty() {
        user.push_str("## Original Commit Context\n");
        user.push_str(&commit_context.chars().take(3000).collect::<String>());
        user.push_str("\n\n");
    }

    user.push_str("## Operations\n");
    for op in ops {
        user.push_str(&format!(
            "### {} [{}] ({} item(s)): {}\n",
            op.id,
            op.category,
            op.item_ids.len(),
            op.name
        ));
        if !op.description.trim().is_empty() {
            user.push_str(&format!("Current summary: {}\n", op.description.trim()));
        }
        if let Some(ref rule) = op.rule {
            user.push_str(&format!("Mechanical rule: {rule}\n"));
        }
        for item_id in op.item_ids.iter().take(MAX_SUMMARIZED_ITEMS_PER_CLUSTER) {
            if let Some(item) = items_by_id.get(item_id) {
                user.push_str(&item_summary(item, 4));
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
    let op_list: Vec<&str> = ops.iter().map(|o| o.id.as_str()).collect();
    user.push_str(&format!(
        "Respond with JSON containing `messages` keyed by: {}.",
        op_list.join(", ")
    ));

    StructuredRequest {
        task: SemanticTask::WriteCommitMessages,
        prompt_version: PROMPT_VERSION.to_owned(),
        system_prompt,
        user_prompt: user,
        json_schema: commit_messages_schema(),
        temperature: 0.2,
    }
}

/// Builds a structured request for a single slice of items within a large multi-item operation.
pub fn build_operation_slice_request(
    op: &Operation,
    slice_item_ids: &[String],
    slice_index: usize,
    total_slices: usize,
    commit_context: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> StructuredRequest {
    let system_prompt = "\
You write clear, idiomatic Git commit messages for decomposed atomic operations and their individual item commits.
Produce a JSON object with `messages` containing exact key `".to_owned()
        + &op.id
        + "`:
- `subject`: concise imperative commit subject (<= 72 characters, no trailing period) summarizing the overall operation (not generic file counts).
- `body`: 1-3 substantive sentences explaining the overall operation's purpose and architecture.
- `item_subjects`: REQUIRED map with an imperative commit subject (<= 72 characters) for EVERY item id listed in this batch (`i...`).
- `item_bodies`: REQUIRED map with a concise 1-2 sentence commit description for EVERY item id listed in this batch explaining what that specific file/hunk introduces or updates.
Return ONLY valid JSON conforming to the schema.";

    let mut user = String::new();
    if !commit_context.trim().is_empty() {
        user.push_str("## Original Commit Context\n");
        user.push_str(&commit_context.chars().take(2500).collect::<String>());
        user.push_str("\n\n");
    }

    let all_files = op.files(items_by_id);
    let file_preview: Vec<&str> = all_files.iter().take(20).map(String::as_str).collect();
    let more_files = if all_files.len() > 20 {
        format!(" (+{} more)", all_files.len() - 20)
    } else {
        String::new()
    };

    user.push_str(&format!(
        "## Operation {} [{}] (slice {}/{} — {} of {} total item(s))\n",
        op.id,
        op.category,
        slice_index + 1,
        total_slices,
        slice_item_ids.len(),
        op.item_ids.len()
    ));
    user.push_str(&format!(
        "Overall operation files: {}{more_files}\n",
        file_preview.join(", ")
    ));
    if !op.description.trim().is_empty() {
        user.push_str(&format!("Current summary: {}\n", op.description.trim()));
    }
    if let Some(ref rule) = op.rule {
        user.push_str(&format!("Mechanical rule: {rule}\n"));
    }
    user.push_str(
        "\n### Items in This Slice (provide `item_subjects` and `item_bodies` for ALL of these)\n",
    );
    for item_id in slice_item_ids {
        if let Some(item) = items_by_id.get(item_id) {
            user.push_str(&item_summary(item, 4));
        }
    }
    user.push_str(&format!(
        "\nRespond with JSON containing `messages.{}` including `subject`, `body`, `item_subjects`, and `item_bodies` for items: {}.",
        op.id,
        slice_item_ids.join(", ")
    ));

    StructuredRequest {
        task: SemanticTask::WriteCommitMessages,
        prompt_version: PROMPT_VERSION.to_owned(),
        system_prompt,
        user_prompt: user,
        json_schema: commit_messages_schema(),
        temperature: 0.2,
    }
}
