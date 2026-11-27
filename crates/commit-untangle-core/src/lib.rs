pub mod bytes;
pub mod classify;
pub mod cluster;
pub mod error;
pub mod graph;
pub mod hunk;
pub mod model;
pub mod proposal;
pub mod replay;
pub mod schema;
pub mod symbols;
pub mod verify;

use indexmap::IndexMap;

use crate::cluster::{
    cluster_semantic_items, deterministic_operation, generated_operations, mechanical_operations,
};
use crate::error::CoreError;
use crate::graph::{ChangeGraph, MechanicalPlacement};
use crate::model::{ChangeItem, FileChange, Operation};

#[derive(Clone, Debug)]
pub struct DeterministicPlanConfig {
    pub similarity_threshold: f64,
    pub max_cluster_size: usize,
    pub mechanical_placement: MechanicalPlacement,
}

impl Default for DeterministicPlanConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            max_cluster_size: 40,
            mechanical_placement: MechanicalPlacement::First,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeterministicPlan {
    pub graph: ChangeGraph,
    pub order: Vec<Operation>,
    pub cycle_merges: Vec<Vec<String>>,
    pub semantic_clusters: Vec<Vec<String>>,
}

/// Runs Stages 2 & 3 deterministically (mechanical + generated + similarity clustering,
/// syntactic def/use dependencies, Tarjan SCC cycle contraction, and priority Kahn topological sort).
pub fn build_deterministic_plan(
    files: &[FileChange],
    items: &IndexMap<String, ChangeItem>,
    cfg: &DeterministicPlanConfig,
) -> Result<DeterministicPlan, CoreError> {
    let mechanical_ops = mechanical_operations(items, files);
    let generated_ops = generated_operations(items, files);

    let semantic_refs: Vec<&ChangeItem> = items
        .values()
        .filter(|it| !it.kind.is_mechanical() && !files[it.file_index].generated)
        .collect();

    let raw_clusters =
        cluster_semantic_items(&semantic_refs, cfg.similarity_threshold, cfg.max_cluster_size);
    let semantic_clusters: Vec<Vec<String>> = raw_clusters
        .into_iter()
        .map(|c| c.into_iter().map(|it| it.id.clone()).collect())
        .collect();

    let mut semantic_ops: Vec<Operation> = semantic_clusters
        .iter()
        .map(|ids| deterministic_operation(ids.clone(), items))
        .collect();
    semantic_ops.sort_by_key(|op| op.position(items));

    let mut all_ops = Vec::with_capacity(
        mechanical_ops.len() + semantic_ops.len() + generated_ops.len(),
    );
    all_ops.extend(mechanical_ops);
    all_ops.extend(semantic_ops);
    all_ops.extend(generated_ops);

    for (idx, op) in all_ops.iter_mut().enumerate() {
        op.id = format!("op{}", idx + 1);
    }

    let mut graph = ChangeGraph::new(all_ops);
    graph.add_deterministic_edges(items);
    let cycle_merges = graph.merge_cycles();
    let order = graph.topological_order(cfg.mechanical_placement, &IndexMap::new(), items)?;

    Ok(DeterministicPlan {
        graph,
        order,
        cycle_merges,
        semantic_clusters,
    })
}
