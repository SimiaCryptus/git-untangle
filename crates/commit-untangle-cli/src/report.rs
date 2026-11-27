use std::collections::BTreeMap;

use commit_untangle_core::graph::ChangeGraph;
use commit_untangle_core::model::{ChangeItem, FileChange, Operation};
use commit_untangle_core::schema::SeriesMetadata;
use commit_untangle_core::verify::VerificationReport;
use commit_untangle_git::ResolvedTarget;
use indexmap::IndexMap;

#[allow(clippy::too_many_arguments)]
pub fn build_markdown_report(
    target: &ResolvedTarget,
    files: &[FileChange],
    items: &IndexMap<String, ChangeItem>,
    graph: &ChangeGraph,
    order: &[Operation],
    verification: &VerificationReport,
    series: &SeriesMetadata,
    notes: &[String],
    mismatches: &[String],
) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("# Commit Untangle Report\n\n");

    let tip_label = if target.is_range { "Tip" } else { "Commit" };
    let base_label = if target.is_range { "Base" } else { "Parent" };
    out.push_str(&format!(
        "- **{tip_label}:** `{}` — {}\n",
        target.tip, target.subject
    ));
    out.push_str(&format!("- **{base_label}:** `{}`\n", target.base));

    if target.is_range {
        let mut qualifiers = vec![format!("{} commit(s)", target.commits.len())];
        if target.symmetric {
            qualifiers.push("base = merge-base".to_owned());
        }
        if !target.base_is_ancestor {
            qualifiers.push("warning: base is not an ancestor of tip".to_owned());
        }
        out.push_str(&format!(
            "- **Range:** `{}..{}` ({})\n",
            &target.base[..10.min(target.base.len())],
            &target.tip[..10.min(target.tip.len())],
            qualifiers.join(", ")
        ));
    }
    if !target.pathspecs.is_empty() {
        out.push_str(&format!(
            "- **Pathspecs:** `{}`\n",
            target.pathspecs.join("`, `")
        ));
    }

    let mut kind_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for item in items.values() {
        *kind_counts.entry(item.kind.as_str()).or_insert(0) += 1;
    }
    let kind_summary = kind_counts
        .into_iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect::<Vec<_>>()
        .join(", ");

    let mech_count = order.iter().filter(|op| op.is_mechanical(items)).count();

    out.push_str(&format!("- **Files changed:** {}\n", files.len()));
    out.push_str(&format!(
        "- **Change items:** {} ({kind_summary})\n",
        items.len()
    ));
    out.push_str(&format!(
        "- **Operations:** {} ({mech_count} mechanical)\n",
        order.len()
    ));
    out.push_str(&format!("- **Dependency edges:** {}\n", graph.edges.len()));
    if mismatches.is_empty() {
        out.push_str("- **Exact reconstruction:** verified\n");
    } else {
        out.push_str(&format!(
            "- **Exact reconstruction:** FAILED ({})\n",
            mismatches.join(", ")
        ));
    }
    out.push_str(&format!(
        "- **Build verification:** {}\n",
        verification.status
    ));
    if let Some(tree_ok) = series.tree_matches {
        out.push_str(&format!(
            "- **Commit series tree identical to original:** {}\n",
            if tree_ok { "yes" } else { "NO" }
        ));
    }
    if let Some(ref branch) = series.branch_created {
        out.push_str(&format!("- **Branch:** `{branch}`\n"));
    }
    if let Some(ref head) = series.head_sha {
        out.push_str(&format!(
            "- **Item commits:** {} (series head `{head}`)\n",
            series.item_commits.len()
        ));
    }
    if !series.merge_commits.is_empty() {
        out.push_str(&format!(
            "- **Merge commits:** {} (commit DAG mirrors the dependency graph)\n",
            series.merge_commits.len()
        ));
    }
    out.push('\n');

    if target.is_range && !target.commits.is_empty() {
        out.push_str("## Commits in Range (oldest first)\n\n");
        for c in target.commits.iter().take(200) {
            out.push_str(&format!(
                "- `{}` {}\n",
                &c.sha[..10.min(c.sha.len())],
                c.subject
            ));
        }
        if target.commits.len() > 200 {
            out.push_str(&format!("- _...and {} more_\n", target.commits.len() - 200));
        }
        out.push('\n');
    }

    let suspicious: Vec<&Operation> = order.iter().filter(|op| op.suspicious).collect();
    if !suspicious.is_empty() {
        out.push_str("## Flagged Operations\n\n");
        for op in suspicious {
            let reason = op.suspicion_reason.as_deref().unwrap_or("flagged");
            out.push_str(&format!("- **{} {}**: {reason}\n", op.id, op.name));
        }
        out.push('\n');
    }

    let mechanical: Vec<&Operation> = order.iter().filter(|op| op.is_mechanical(items)).collect();
    if !mechanical.is_empty() {
        out.push_str("## Mechanical Changes (can be approved in bulk)\n\n");
        for op in mechanical {
            let rule_suffix = op
                .rule
                .as_ref()
                .map(|r| {
                    let clean: String = r.chars().take(120).collect();
                    format!(", rule: `{}`", clean.replace('`', "'"))
                })
                .unwrap_or_default();
            out.push_str(&format!(
                "- **{} {}** — {} item(s){rule_suffix}\n",
                op.id,
                op.name,
                op.item_ids.len()
            ));
        }
        out.push('\n');
    }

    out.push_str("## Operations in Order\n\n");
    for (idx, op) in order.iter().enumerate() {
        out.push_str(&operation_markdown(idx + 1, op, items, series));
        out.push('\n');
    }

    out.push_str("## Dependencies\n\n");
    if graph.edges.is_empty() {
        out.push_str("_No dependencies detected; operations are independent._\n\n");
    } else {
        for e in &graph.edges {
            out.push_str(&format!(
                "- {} -> {} _({})_: {}\n",
                e.before,
                e.after,
                e.kind.as_str(),
                e.reason
            ));
        }
        out.push('\n');
    }

    if !verification.steps.is_empty() || !verification.fixes.is_empty() {
        out.push_str("## Verification Log\n\n");
        for step in &verification.steps {
            out.push_str(&format!("- {step}\n"));
        }
        if !verification.fixes.is_empty() {
            out.push_str("\n### Graph Repairs\n\n");
            for fix in &verification.fixes {
                out.push_str(&format!("- {fix}\n"));
            }
        }
        out.push('\n');
    }

    if series.tree_matches == Some(false) && !series.tree_diff.is_empty() {
        out.push_str("## Tree Differences\n\n```\n");
        out.push_str(&series.tree_diff);
        out.push_str("\n```\n\n");
    }

    if !notes.is_empty() {
        out.push_str("## Notes\n\n");
        for note in notes {
            out.push_str(&format!("- {note}\n"));
        }
        out.push('\n');
    }

    out
}

fn operation_markdown(
    num: usize,
    op: &Operation,
    items: &IndexMap<String, ChangeItem>,
    series: &SeriesMetadata,
) -> String {
    let mut s = String::new();
    let flag = if op.suspicious { " [SUSPICIOUS]" } else { "" };
    let ver = match op.verified {
        Some(true) => " [VERIFIED]",
        Some(false) => " [FAILED]",
        None => "",
    };
    s.push_str(&format!("### {num}. {}{flag}{ver}\n\n", op.name));
    let mech = if op.is_mechanical(items) {
        " · mechanical"
    } else {
        ""
    };
    s.push_str(&format!(
        "`{}` · **{}**{mech} · confidence {:.2} · {} item(s)\n\n",
        op.id,
        op.category,
        op.confidence,
        op.item_ids.len()
    ));
    s.push_str(&op.description);
    s.push('\n');

    if let Some(ref r) = op.rule {
        let clean: String = r.chars().take(300).collect();
        s.push_str(&format!("\nRule: `{}`\n", clean.replace('`', "'")));
    }
    if op.suspicious {
        let reason = op
            .suspicion_reason
            .as_deref()
            .unwrap_or("flagged by analysis");
        s.push_str(&format!("\n> Warning: {reason}\n"));
    }
    s.push('\n');

    for id in op.item_ids.iter().take(25) {
        if let Some(item) = items.get(id) {
            let loc = item
                .hunk
                .as_ref()
                .map(|h| format!(":{}", h.start + 1))
                .unwrap_or_default();
            let subj_note = if op.item_ids.len() > 1 {
                series
                    .item_subjects
                    .get(id)
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| format!(" — {s}"))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            s.push_str(&format!(
                "- {} {} `{}{loc}`{subj_note}\n",
                item.id,
                item.kind.as_str(),
                item.display_path
            ));
        }
    }
    if op.item_ids.len() > 25 {
        s.push_str(&format!("- ...and {} more\n", op.item_ids.len() - 25));
    }
    s
}

pub fn slugify(input: &str) -> String {
    let mut slug = String::with_capacity(input.len());
    let mut last_dash = false;
    for ch in input.chars() {
        let lower = ch.to_ascii_lowercase();
        if lower.is_ascii_alphanumeric() {
            slug.push(lower);
            last_dash = false;
        } else if !last_dash && !slug.is_empty() {
            slug.push('-');
            last_dash = true;
        }
        if slug.len() >= 50 {
            break;
        }
    }
    let trimmed = slug.trim_matches('-');
    if trimmed.is_empty() {
        "op".to_owned()
    } else {
        trimmed.to_owned()
    }
}
