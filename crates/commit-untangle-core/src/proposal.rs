use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::cluster::{
    default_description, default_name, deterministic_operation, normalize_category, truncate_chars,
};
use crate::error::CoreError;
use crate::model::{ChangeItem, Operation};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClusterAction {
    MergeClusters {
        clusters: Vec<String>,
        #[serde(default)]
        reason: String,
    },
    SplitCluster {
        cluster: String,
        groups: Vec<Vec<String>>,
        #[serde(default)]
        reason: Option<String>,
    },
    MoveItems {
        items: Vec<String>,
        from_cluster: String,
        to_cluster: String,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationLabel {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub suspicious: bool,
    #[serde(default)]
    pub suspicion_reason: Option<String>,
}

/// Constrained graph-patch proposal over deterministic clusters (`c1`, `c2`, ...).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterRefinementProposal {
    #[serde(default)]
    pub actions: Vec<ClusterAction>,
    #[serde(default)]
    pub labels: IndexMap<String, OperationLabel>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannedEdge {
    pub before: String,
    pub after: String,
    #[serde(default)]
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyProposal {
    #[serde(default)]
    pub edges: Vec<PlannedEdge>,
    #[serde(default)]
    pub suggested_order: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BuildDiagnosisProposal {
    #[serde(default)]
    pub required_operation_ids: Vec<String>,
    #[serde(default)]
    pub explanation: String,
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

    let mut ops = Vec::with_capacity(clusters.len());
    for (cid, item_ids) in clusters {
        if item_ids.is_empty() {
            continue;
        }
        if let Some(label) = proposal.labels.get(&cid) {
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
