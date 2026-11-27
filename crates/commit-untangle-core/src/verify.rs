use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::graph::{ChangeGraph, EdgeKind};
use crate::model::{ChangeItem, Operation};
use crate::proposal::BuildDiagnosisProposal;
use crate::symbols::tokens;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationReport {
    pub status: String,
    pub steps: Vec<String>,
    pub fixes: Vec<String>,
}

impl Default for VerificationReport {
    fn default() -> Self {
        Self {
            status: "skipped".to_owned(),
            steps: Vec::new(),
            fixes: Vec::new(),
        }
    }
}

/// Orders `first_id` before `second_id`, merging them immediately if `second_id` already reaches `first_id`.
pub fn order_before(
    graph: &mut ChangeGraph,
    first_id: &str,
    second_id: &str,
    reason: &str,
    report: &mut VerificationReport,
) -> bool {
    if graph.op(first_id).is_none() || graph.op(second_id).is_none() || first_id == second_id {
        return false;
    }
    if graph.reachable(second_id, first_id) {
        let note =
            format!("(Merged during build verification: mutually dependent with {first_id}.)");
        let ids = vec![second_id.to_owned(), first_id.to_owned()];
        if graph.merge(&ids, &note).is_ok() {
            report.fixes.push(format!(
                "Merged {first_id} into {second_id} (mutual dependency)"
            ));
            return true;
        }
        false
    } else if graph.add_edge(
        first_id,
        second_id,
        EdgeKind::BuildVerification,
        reason,
        true,
    ) {
        report
            .fixes
            .push(format!("Added edge {first_id} -> {second_id}: {reason}"));
        true
    } else {
        false
    }
}

/// Executes deterministic repair Tiers 1 & 2 from compiler output tokens.
/// Returns `true` if at least one repair edge or merge was applied.
pub fn repair_deterministic_tiers(
    graph: &mut ChangeGraph,
    failing: &Operation,
    later: &[Operation],
    build_output: &str,
    items_by_id: &IndexMap<String, ChangeItem>,
    report: &mut VerificationReport,
) -> bool {
    let out_tokens = tokens(build_output);
    let mut fixed = false;

    // Tier 1: Missing symbol introduced by a later operation
    for op in later {
        if op
            .defined_added(items_by_id)
            .iter()
            .any(|sym| out_tokens.contains(sym))
        {
            let reason = format!(
                "Build failed when {} was applied before {}, which introduces a referenced symbol",
                failing.id, op.id
            );
            fixed = order_before(graph, &op.id, &failing.id, &reason, report) || fixed;
        }
    }

    // Tier 2: Failing operation deletes a definition that later operations stop referencing
    if !fixed {
        let removed_defs = failing
            .defined_removed(items_by_id)
            .into_iter()
            .filter(|sym| out_tokens.contains(sym))
            .collect::<std::collections::BTreeSet<_>>();
        if !removed_defs.is_empty() {
            for op in later {
                if op
                    .refs_removed(items_by_id)
                    .iter()
                    .any(|sym| removed_defs.contains(sym))
                {
                    let reason = format!(
                        "{} removes remaining uses of a symbol deleted by {}",
                        op.id, failing.id
                    );
                    fixed = order_before(graph, &op.id, &failing.id, &reason, report) || fixed;
                }
            }
        }
    }

    if fixed {
        for cycle in graph.merge_cycles() {
            report
                .fixes
                .push(format!("Merged dependency cycle: {}", cycle.join(", ")));
        }
    }
    fixed
}

/// Applies Tier 3 LLM diagnosis or Tier 4 guaranteed-convergence fallback merge.
pub fn repair_with_diagnosis_or_fallback(
    graph: &mut ChangeGraph,
    failing_id: &str,
    later: &[Operation],
    diagnosis: Option<&BuildDiagnosisProposal>,
    report: &mut VerificationReport,
) -> bool {
    let mut fixed = false;
    if let Some(diag) = diagnosis {
        for req in &diag.required_operation_ids {
            let trimmed = req.trim();
            if later.iter().any(|op| op.id == trimmed) {
                let reason = format!(
                    "LLM diagnosis: {}",
                    diag.explanation.chars().take(200).collect::<String>()
                );
                fixed = order_before(graph, trimmed, failing_id, &reason, report) || fixed;
            }
        }
    }

    if !fixed {
        if let Some(partner) = later.iter().find(|op| graph.op(&op.id).is_some()) {
            if graph.op(failing_id).is_some() {
                let ids = vec![failing_id.to_owned(), partner.id.clone()];
                if graph
                    .merge(
                        &ids,
                        "(Merged during build verification: no separable dependency was found.)",
                    )
                    .is_ok()
                {
                    report.fixes.push(format!(
                        "Merged {} into {failing_id} (no dependency could be identified)",
                        partner.id
                    ));
                    fixed = true;
                }
            }
        }
    }

    if fixed {
        for cycle in graph.merge_cycles() {
            report
                .fixes
                .push(format!("Merged dependency cycle: {}", cycle.join(", ")));
        }
    }
    fixed
}
