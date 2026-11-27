use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cluster::{
    default_description, default_name, deterministic_operation, normalize_category, truncate_chars,
};
use crate::error::CoreError;
use crate::model::{ChangeItem, Operation};

fn null_to_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Normalizes LLM-emitted IDs such as `"[op1]"`, `` "`i12`" ``, `"op_1"`, or `" op1: "` into canonical IDs.
pub fn normalize_entity_id(raw: &str) -> String {
    let trimmed = raw
        .trim()
        .trim_matches(|c: char| matches!(c, '[' | ']' | '`' | '"' | '\'' | ':'))
        .trim();
    let lower = trimmed.to_ascii_lowercase();
    if let Some(rest) = lower.strip_prefix("operation ") {
        return normalize_entity_id(rest);
    }
    if let Some(rest) = lower.strip_prefix("item ") {
        return normalize_entity_id(rest);
    }
    if let Some(num) = lower.strip_prefix("op_") {
        if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
            return format!("op{num}");
        }
    }
    if let Some(num) = lower.strip_prefix("item_") {
        if !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()) {
            return format!("i{num}");
        }
    }
    trimmed.to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClusterAction {
    MergeClusters {
        #[serde(default, deserialize_with = "null_to_default")]
        clusters: Vec<String>,
        #[serde(default, deserialize_with = "null_to_default")]
        reason: String,
    },
    SplitCluster {
        #[serde(default, deserialize_with = "null_to_default")]
        cluster: String,
        #[serde(default, deserialize_with = "null_to_default")]
        groups: Vec<Vec<String>>,
        #[serde(default)]
        reason: Option<String>,
    },
    MoveItems {
        #[serde(default, deserialize_with = "null_to_default")]
        items: Vec<String>,
        #[serde(default, deserialize_with = "null_to_default")]
        from_cluster: String,
        #[serde(default, deserialize_with = "null_to_default")]
        to_cluster: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationLabel {
    #[serde(default, deserialize_with = "null_to_default")]
    pub name: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub description: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub suspicious: bool,
    #[serde(default)]
    pub suspicion_reason: Option<String>,
}

/// Constrained graph-patch proposal over deterministic clusters (`c1`, `c2`, ...).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterRefinementProposal {
    #[serde(default, deserialize_with = "null_to_default")]
    pub actions: Vec<ClusterAction>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub labels: IndexMap<String, OperationLabel>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannedEdge {
    #[serde(default, deserialize_with = "null_to_default")]
    pub before: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub after: String,
    #[serde(default, deserialize_with = "null_to_default")]
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyProposal {
    #[serde(default, deserialize_with = "null_to_default")]
    pub edges: Vec<PlannedEdge>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub suggested_order: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildDiagnosisProposal {
    #[serde(default, deserialize_with = "null_to_default")]
    pub required_operation_ids: Vec<String>,
    #[serde(default, deserialize_with = "null_to_default")]
    pub explanation: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct CommitMessageDraft {
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub body: String,
    /// Optional per-item subject overrides keyed by `ChangeItem.id` (e.g. `i12`).
    #[serde(default)]
    pub item_subjects: IndexMap<String, String>,
    /// Optional per-item body/description overrides keyed by `ChangeItem.id` (e.g. `i12`).
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub item_bodies: IndexMap<String, String>,
}

impl CommitMessageDraft {
    pub fn merge_from(&mut self, other: CommitMessageDraft) {
        if self.subject.trim().is_empty() && !other.subject.trim().is_empty() {
            self.subject = other.subject.trim().to_owned();
        }
        if self.body.trim().is_empty() && !other.body.trim().is_empty() {
            self.body = other.body.trim().to_owned();
        }
        for (k, v) in other.item_subjects {
            let clean_k = normalize_entity_id(&k);
            let clean_v = v.trim();
            if !clean_k.is_empty() && !clean_v.is_empty() {
                self.item_subjects.insert(clean_k, clean_v.to_owned());
            }
        }
        for (k, v) in other.item_bodies {
            let clean_k = normalize_entity_id(&k);
            let clean_v = v.trim();
            if !clean_k.is_empty() && !clean_v.is_empty() {
                self.item_bodies.insert(clean_k, clean_v.to_owned());
            }
        }
    }

    fn from_json_value(val: &Value) -> Option<Self> {
        if let Some(s) = val.as_str() {
            let mut lines = s.trim().lines();
            let subject = lines.next().unwrap_or("").trim().to_owned();
            let body = lines.collect::<Vec<_>>().join("\n").trim().to_owned();
            return Some(Self {
                subject,
                body,
                item_subjects: IndexMap::new(),
                item_bodies: IndexMap::new(),
            });
        }
        let obj = val.as_object()?;
        let pick_str = |keys: &[&str]| -> String {
            for &k in keys {
                if let Some(s) = obj.get(k).and_then(Value::as_str) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        return trimmed.to_owned();
                    }
                }
            }
            String::new()
        };

        let subject = pick_str(&["subject", "title", "name", "summary", "header"]);
        let body = pick_str(&["body", "description", "message", "details", "explanation"]);

        let mut item_subjects = IndexMap::new();
        let mut item_bodies = IndexMap::new();

        let ingest_item_entry =
            |raw_id: &str,
             entry: &Value,
             subjects: &mut IndexMap<String, String>,
             bodies: &mut IndexMap<String, String>| {
                let id = normalize_entity_id(raw_id);
                if id.is_empty() {
                    return;
                }
                if let Some(s) = entry.as_str() {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        subjects.insert(id, trimmed.to_owned());
                    }
                } else if let Some(item_obj) = entry.as_object() {
                    for sk in ["subject", "title", "name", "summary"] {
                        if let Some(s) = item_obj.get(sk).and_then(Value::as_str) {
                            let trimmed = s.trim();
                            if !trimmed.is_empty() {
                                subjects.insert(id.clone(), trimmed.to_owned());
                                break;
                            }
                        }
                    }
                    for bk in ["body", "description", "message", "details"] {
                        if let Some(b) = item_obj.get(bk).and_then(Value::as_str) {
                            let trimmed = b.trim();
                            if !trimmed.is_empty() {
                                bodies.insert(id.clone(), trimmed.to_owned());
                                break;
                            }
                        }
                    }
                }
            };

        if let Some(sub_map) = obj.get("item_subjects").and_then(Value::as_object) {
            for (k, v) in sub_map {
                ingest_item_entry(k, v, &mut item_subjects, &mut item_bodies);
            }
        }

        for body_key in ["item_bodies", "item_descriptions", "item_messages"] {
            if let Some(b_map) = obj.get(body_key).and_then(Value::as_object) {
                for (k, v) in b_map {
                    let id = normalize_entity_id(k);
                    if id.is_empty() {
                        continue;
                    }
                    if let Some(s) = v.as_str() {
                        let trimmed = s.trim();
                        if !trimmed.is_empty() {
                            item_bodies.insert(id, trimmed.to_owned());
                        }
                    } else {
                        ingest_item_entry(&id, v, &mut item_subjects, &mut item_bodies);
                    }
                }
            }
        }

        if let Some(items_val) = obj.get("items") {
            if let Some(items_map) = items_val.as_object() {
                for (k, v) in items_map {
                    ingest_item_entry(k, v, &mut item_subjects, &mut item_bodies);
                }
            } else if let Some(items_arr) = items_val.as_array() {
                for elem in items_arr {
                    if let Some(elem_obj) = elem.as_object() {
                        if let Some(raw_id) = elem_obj
                            .get("id")
                            .or_else(|| elem_obj.get("item_id"))
                            .and_then(Value::as_str)
                        {
                            ingest_item_entry(raw_id, elem, &mut item_subjects, &mut item_bodies);
                        }
                    }
                }
            }
        }

        Some(Self {
            subject,
            body,
            item_subjects,
            item_bodies,
        })
    }
}

impl<'de> Deserialize<'de> for CommitMessageDraft {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let val = Value::deserialize(deserializer)?;
        if val.is_null() {
            return Ok(Self::default());
        }
        Self::from_json_value(&val)
            .ok_or_else(|| serde::de::Error::custom("expected commit message object or string"))
    }
}

/// LLM-authored commit messages keyed by `Operation.id` (e.g. `op1`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct CommitMessageProposal {
    #[serde(default)]
    pub messages: IndexMap<String, CommitMessageDraft>,
}

impl<'de> Deserialize<'de> for CommitMessageProposal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let val = Value::deserialize(deserializer)?;
        if val.is_null() {
            return Ok(Self::default());
        }
        let mut messages = IndexMap::new();

        let ingest_collection = |coll: &Value, out: &mut IndexMap<String, CommitMessageDraft>| {
            if let Some(map) = coll.as_object() {
                for (k, v) in map {
                    let op_id = normalize_entity_id(k);
                    if !op_id.is_empty() {
                        if let Some(draft) = CommitMessageDraft::from_json_value(v) {
                            out.entry(op_id)
                                .and_modify(|existing: &mut CommitMessageDraft| {
                                    existing.merge_from(draft.clone())
                                })
                                .or_insert(draft);
                        }
                    }
                }
            } else if let Some(arr) = coll.as_array() {
                for elem in arr {
                    if let Some(obj) = elem.as_object() {
                        if let Some(raw_id) = obj
                            .get("id")
                            .or_else(|| obj.get("op_id"))
                            .or_else(|| obj.get("operation_id"))
                            .or_else(|| obj.get("operation"))
                            .and_then(Value::as_str)
                        {
                            let op_id = normalize_entity_id(raw_id);
                            if !op_id.is_empty() {
                                if let Some(draft) = CommitMessageDraft::from_json_value(elem) {
                                    out.entry(op_id)
                                        .and_modify(|existing: &mut CommitMessageDraft| {
                                            existing.merge_from(draft.clone())
                                        })
                                        .or_insert(draft);
                                }
                            }
                        }
                    }
                }
            }
        };

        if let Some(root) = val.as_object() {
            let mut found_wrapper = false;
            for wrapper_key in ["messages", "operations", "commits", "commit_messages"] {
                if let Some(inner) = root.get(wrapper_key) {
                    found_wrapper = true;
                    ingest_collection(inner, &mut messages);
                }
            }
            if !found_wrapper {
                // Fallback: flat map `{"op1": {"subject": ..., "body": ...}}`
                ingest_collection(&val, &mut messages);
            }
        } else if val.is_array() {
            ingest_collection(&val, &mut messages);
        } else {
            return Err(serde::de::Error::custom(
                "expected JSON object with `messages` map",
            ));
        }

        Ok(Self { messages })
    }
}

/// Applies a constrained `ClusterRefinementProposal` to initial clusters `c1..ck`,
/// strictly preserving Invariant I1 (every input item ID appears in exactly one output `Operation`).
pub fn apply_cluster_proposal(
    initial_clusters: &[Vec<String>],
    proposal: &ClusterRefinementProposal,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> Result<Vec<Operation>, CoreError> {
    let expected_items: BTreeSet<String> = initial_clusters
        .iter()
        .flat_map(|c| c.iter().cloned())
        .collect();

    let mut clusters: IndexMap<String, Vec<String>> = initial_clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_empty())
        .map(|(i, c)| (format!("c{}", i + 1), c.clone()))
        .collect();

    for action in &proposal.actions {
        match action {
            ClusterAction::MergeClusters {
                clusters: to_merge, ..
            } => {
                let mut unique_ids = Vec::new();
                let mut seen = BTreeSet::new();
                for cid in to_merge {
                    let trimmed = cid.trim();
                    if seen.insert(trimmed.to_owned()) {
                        unique_ids.push(trimmed.to_owned());
                    }
                }
                if unique_ids.len() < 2 || !unique_ids.iter().all(|id| clusters.contains_key(id)) {
                    continue;
                }
                let merged_id = unique_ids.join("+");
                let first_pos = clusters.get_index_of(&unique_ids[0]).unwrap_or(0);
                let mut combined = Vec::new();
                for id in &unique_ids {
                    if let Some(items) = clusters.shift_remove(id) {
                        combined.extend(items);
                    }
                }
                let insert_at = first_pos.min(clusters.len());
                clusters.shift_insert(insert_at, merged_id, combined);
            }
            ClusterAction::SplitCluster {
                cluster, groups, ..
            } => {
                let target_id = cluster.trim();
                let Some(pos) = clusters.get_index_of(target_id) else {
                    continue;
                };
                let orig_items = clusters[target_id].clone();
                let orig_set: BTreeSet<&str> = orig_items.iter().map(String::as_str).collect();
                let mut assigned: BTreeSet<String> = BTreeSet::new();
                let mut sub_groups: Vec<Vec<String>> = Vec::new();

                for g in groups {
                    let mut valid = Vec::new();
                    for raw_id in g {
                        let cleaned = raw_id.trim().trim_matches(['[', ']']);
                        if orig_set.contains(cleaned) && assigned.insert(cleaned.to_owned()) {
                            valid.push(cleaned.to_owned());
                        }
                    }
                    if !valid.is_empty() {
                        sub_groups.push(valid);
                    }
                }

                if sub_groups.is_empty() {
                    continue;
                }

                let leftovers: Vec<String> = orig_items
                    .into_iter()
                    .filter(|id| !assigned.contains(id))
                    .collect();
                if !leftovers.is_empty() {
                    if let Some(largest) = sub_groups.iter_mut().max_by_key(|g| g.len()) {
                        largest.extend(leftovers);
                    }
                }

                clusters.shift_remove(target_id);
                for (idx, sub) in sub_groups.into_iter().enumerate() {
                    let sub_id = format!("{target_id}.{}", idx + 1);
                    clusters.shift_insert((pos + idx).min(clusters.len()), sub_id, sub);
                }
            }
            ClusterAction::MoveItems {
                items,
                from_cluster,
                to_cluster,
            } => {
                let from_id = from_cluster.trim();
                let to_id = to_cluster.trim();
                if from_id == to_id
                    || !clusters.contains_key(from_id)
                    || !clusters.contains_key(to_id)
                {
                    continue;
                }
                let wanted: BTreeSet<&str> = items
                    .iter()
                    .map(|s| s.trim().trim_matches(['[', ']']))
                    .collect();
                let mut moving = Vec::new();
                if let Some(src) = clusters.get_mut(from_id) {
                    let mut retained = Vec::with_capacity(src.len());
                    for id in src.drain(..) {
                        if wanted.contains(id.as_str()) {
                            moving.push(id);
                        } else {
                            retained.push(id);
                        }
                    }
                    *src = retained;
                }
                if moving.is_empty() {
                    continue;
                }
                if let Some(dst) = clusters.get_mut(to_id) {
                    dst.extend(moving);
                }
                if clusters.get(from_id).is_some_and(Vec::is_empty) {
                    clusters.shift_remove(from_id);
                }
            }
        }
    }

    let normalized_labels: IndexMap<String, &OperationLabel> = proposal
        .labels
        .iter()
        .map(|(k, v)| (normalize_entity_id(k), v))
        .collect();

    let mut ops = Vec::with_capacity(clusters.len());
    for (cid, item_ids) in clusters {
        if item_ids.is_empty() {
            continue;
        }
        if let Some(label) = proposal
            .labels
            .get(&cid)
            .or_else(|| normalized_labels.get(&cid).copied())
        {
            let resolved: Vec<&ChangeItem> = item_ids
                .iter()
                .filter_map(|id| items_by_id.get(id))
                .collect();
            let name = if label.name.trim().is_empty() {
                default_name(&resolved)
            } else {
                truncate_chars(label.name.trim(), 120)
            };
            let description = if label.description.trim().is_empty() {
                default_description(&resolved)
            } else {
                label.description.trim().to_owned()
            };
            let category = normalize_category(label.category.as_deref());
            let mut op = Operation::new("", name, description, category, item_ids);
            op.suspicious = label.suspicious;
            op.suspicion_reason = label
                .suspicion_reason
                .as_ref()
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty());
            op.confidence = 0.75;
            ops.push(op);
        } else {
            ops.push(deterministic_operation(item_ids, items_by_id));
        }
    }

    verify_partition(&expected_items, &ops)?;
    Ok(ops)
}

/// Verifies Invariant I1: every expected item ID appears in `ops` exactly once and no operation is empty.
pub fn verify_partition(
    expected_items: &BTreeSet<String>,
    ops: &[Operation],
) -> Result<(), CoreError> {
    let mut seen = BTreeSet::new();
    for op in ops {
        if op.item_ids.is_empty() {
            return Err(CoreError::PartitionViolation(format!(
                "operation '{}' has zero items",
                op.name
            )));
        }
        for id in &op.item_ids {
            if !expected_items.contains(id) {
                return Err(CoreError::PartitionViolation(format!(
                    "unexpected item id '{id}'"
                )));
            }
            if !seen.insert(id.clone()) {
                return Err(CoreError::PartitionViolation(format!(
                    "duplicate item id '{id}'"
                )));
            }
        }
    }
    if seen.len() != expected_items.len() {
        return Err(CoreError::PartitionViolation(format!(
            "missing {} item(s) from partition",
            expected_items.len() - seen.len()
        )));
    }
    Ok(())
}
