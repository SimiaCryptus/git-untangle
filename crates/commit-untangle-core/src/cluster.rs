use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;

use crate::model::{ChangeItem, FileChange, ItemKind, Operation};
use crate::symbols::{is_significant, jaccard};

pub const CATEGORIES: &[&str] = &[
    "refactor",
    "feature",
    "fix",
    "test",
    "docs",
    "config",
    "build",
    "cleanup",
    "formatting",
    "generated",
    "change",
    "mixed",
];

const CONFIG_EXTENSIONS: &[&str] = &[
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".properties",
    ".xml",
    ".ini",
    ".cfg",
    ".conf",
    ".gradle",
    ".gradle.kts",
    ".editorconfig",
];

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = x;
        while self.parent[c] != r {
            let next = self.parent[c];
            self.parent[c] = r;
            c = next;
        }
        r
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            let hi = ra.max(rb);
            let lo = ra.min(rb);
            self.parent[hi] = lo;
        }
    }
}

/// Builds Stratum 1 mechanical operations (`MOVE`, `MODE_CHANGE`, `WHITESPACE`, `REORDER`, `BULK_REPLACE`).
pub fn mechanical_operations(
    items: &IndexMap<String, ChangeItem>,
    files: &[FileChange],
) -> Vec<Operation> {
    let mut ops = Vec::new();

    let moves: Vec<&ChangeItem> = items
        .values()
        .filter(|it| it.kind == ItemKind::Move)
        .collect();
    if !moves.is_empty() {
        let name = if moves.len() == 1 {
            let f = &files[moves[0].file_index];
            format!(
                "Move {} to {}",
                f.old_path.as_deref().unwrap_or("?"),
                f.new_path.as_deref().unwrap_or("?")
            )
        } else {
            format!("Move/rename {} files", moves.len())
        };
        let rule = moves
            .iter()
            .map(|it| {
                let f = &files[it.file_index];
                format!(
                    "git mv {} {}",
                    f.old_path.as_deref().unwrap_or("?"),
                    f.new_path.as_deref().unwrap_or("?")
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let mut op = Operation::new(
            "",
            name,
            "Pure file moves detected by git rename detection; content edits inside moved files are separate operations.",
            "refactor",
            moves.iter().map(|it| it.id.clone()).collect(),
        );
        op.rule = Some(rule);
        op.confidence = 0.95;
        ops.push(op);
    }

    let modes: Vec<&ChangeItem> = items
        .values()
        .filter(|it| it.kind == ItemKind::ModeChange)
        .collect();
    if !modes.is_empty() {
        let file_count = modes
            .iter()
            .map(|it| it.display_path.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let rule = modes
            .iter()
            .map(|it| {
                let f = &files[it.file_index];
                format!(
                    "chmod {}->{} {}",
                    f.old_mode.map_or("?", |m| m.as_octal()),
                    f.new_mode.map_or("?", |m| m.as_octal()),
                    f.display_path()
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let mut op = Operation::new(
            "",
            format!("Update file modes in {file_count} file(s)"),
            format!("Executable or tree mode transitions across {file_count} file(s)."),
            "config",
            modes.iter().map(|it| it.id.clone()).collect(),
        );
        op.rule = Some(rule);
        op.confidence = 0.95;
        ops.push(op);
    }

    let ws: Vec<&ChangeItem> = items
        .values()
        .filter(|it| it.kind == ItemKind::Whitespace)
        .collect();
    if !ws.is_empty() {
        let file_count = ws
            .iter()
            .map(|it| it.display_path.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let mut op = Operation::new(
            "",
            "Whitespace-only changes",
            format!(
                "Indentation, trailing whitespace, line ending and blank line changes in {file_count} file(s). No non-whitespace content changes."
            ),
            "formatting",
            ws.iter().map(|it| it.id.clone()).collect(),
        );
        op.rule = Some("whitespace-insensitive equality".to_owned());
        op.confidence = 0.95;
        ops.push(op);
    }

    let reorders: Vec<&ChangeItem> = items
        .values()
        .filter(|it| it.kind == ItemKind::Reorder)
        .collect();
    if !reorders.is_empty() {
        let file_count = reorders
            .iter()
            .map(|it| it.display_path.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let mut op = Operation::new(
            "",
            "Reorder lines (imports/includes)",
            format!("Hunks that only permute existing lines in {file_count} file(s)."),
            "formatting",
            reorders.iter().map(|it| it.id.clone()).collect(),
        );
        op.rule = Some("line permutation".to_owned());
        op.confidence = 0.95;
        ops.push(op);
    }

    let mut bulk_groups: IndexMap<(String, String), Vec<&ChangeItem>> = IndexMap::new();
    for it in items.values().filter(|it| it.kind == ItemKind::BulkReplace) {
        if let Some(ref sub) = it.substitution {
            bulk_groups.entry(sub.clone()).or_default().push(it);
        }
    }
    for ((from, to), group) in bulk_groups {
        let file_count = group
            .iter()
            .map(|it| it.display_path.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let from_short: String = from.chars().take(40).collect();
        let to_short: String = to.chars().take(40).collect();
        let mut op = Operation::new(
            "",
            format!("Replace `{from_short}` with `{to_short}`"),
            format!(
                "Bulk textual substitution applied at {} location(s) in {file_count} file(s).",
                group.len()
            ),
            "refactor",
            group.iter().map(|it| it.id.clone()).collect(),
        );
        op.rule = Some(format!("s/{from}/{to}/"));
        op.confidence = 0.95;
        ops.push(op);
    }

    ops
}

/// Builds Stratum 2 generated/lock/vendored operation if any generated non-mechanical items exist.
pub fn generated_operations(
    items: &IndexMap<String, ChangeItem>,
    files: &[FileChange],
) -> Vec<Operation> {
    let gen_items: Vec<&ChangeItem> = items
        .values()
        .filter(|it| !it.kind.is_mechanical() && files[it.file_index].generated)
        .collect();
    if gen_items.is_empty() {
        return Vec::new();
    }
    let file_count = gen_items
        .iter()
        .map(|it| it.display_path.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let mut op = Operation::new(
        "",
        "Update generated, lock and vendored files",
        format!(
            "Opaque changes to generated/lock/vendored files ({file_count} file(s)), kept as a single unit."
        ),
        "generated",
        gen_items.iter().map(|it| it.id.clone()).collect(),
    );
    op.confidence = 0.90;
    vec![op]
}

/// Clusters Stratum 3 semantic items using TF-bounded signature Jaccard similarity,
/// symbol definition-reference affinity, and intra-file hunk locality.
pub fn cluster_semantic_items<'a>(
    items: &[&'a ChangeItem],
    threshold: f64,
    max_cluster_size: usize,
) -> Vec<Vec<&'a ChangeItem>> {
    let n = items.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![vec![items[0]]];
    }

    let max_size = max_cluster_size.max(2);
    let mut weights: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut add_weight = |a: usize, b: usize, w: f64| {
        if a == b || w == 0.0 {
            return;
        }
        let key = (a.min(b), a.max(b));
        *weights.entry(key).or_insert(0.0) += w;
    };

    // 1. Inverted index over item signatures
    let mut inv: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (idx, item) in items.iter().enumerate() {
        for tok in &item.lexical.signature {
            inv.entry(tok.as_str()).or_default().push(idx);
        }
    }

    let max_df = 8usize.max(n / 20);
    let mut seen_pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
    for posting in inv.values() {
        if posting.len() >= 2 && posting.len() <= max_df {
            for x in 0..posting.len() {
                for y in (x + 1)..posting.len() {
                    let a = posting[x];
                    let b = posting[y];
                    let key = (a.min(b), a.max(b));
                    if seen_pairs.insert(key) {
                        let sim = jaccard(&items[a].lexical.signature, &items[b].lexical.signature);
                        add_weight(a, b, sim);
                    }
                }
            }
        }
    }

    // 2. Symbol-level relationships: definition change + reference updates
    for (idx, item) in items.iter().enumerate() {
        for sym in item
            .lexical
            .defined_added
            .union(&item.lexical.defined_removed)
        {
            if is_significant(sym) {
                if let Some(posting) = inv.get(sym.as_str()) {
                    if posting.len() <= 50 {
                        for &other in posting {
                            if other != idx {
                                add_weight(idx, other, 0.5);
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Structural intra-file locality (adjacent hunks within 3 lines)
    let mut by_file: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, item) in items.iter().enumerate() {
        by_file.entry(item.file_index).or_default().push(idx);
    }
    for mut file_indices in by_file.into_values() {
        file_indices.sort_by_key(|&i| items[i].line());
        for win in file_indices.windows(2) {
            let a = win[0];
            let b = win[1];
            if let (Some(ha), Some(hb)) = (&items[a].hunk, &items[b].hunk) {
                let end_a = ha.start + ha.old_count;
                if hb.start.saturating_sub(end_a) <= 3 {
                    add_weight(a, b, 0.3);
                }
            }
        }
    }

    let mut uf = UnionFind::new(n);
    for ((a, b), w) in weights {
        if w >= threshold {
            uf.union(a, b);
        }
    }

    let mut groups_map: IndexMap<usize, Vec<&'a ChangeItem>> = IndexMap::new();
    for (idx, &item) in items.iter().enumerate() {
        let root = uf.find(idx);
        groups_map.entry(root).or_default().push(item);
    }

    let mut result = Vec::new();
    for group in groups_map.into_values() {
        if group.len() <= max_size {
            result.push(group);
        } else if threshold < 1.5 {
            result.extend(cluster_semantic_items(&group, threshold + 0.2, max_size));
        } else {
            for chunk in group.chunks(max_size) {
                result.push(chunk.to_vec());
            }
        }
    }

    result.sort_by_key(|c| c.iter().map(|it| it.position()).min().unwrap_or(u64::MAX));
    result
}

pub fn deterministic_operation(
    item_ids: Vec<String>,
    items_by_id: &IndexMap<String, ChangeItem>,
) -> Operation {
    let resolved: Vec<&ChangeItem> = item_ids
        .iter()
        .filter_map(|id| items_by_id.get(id))
        .collect();
    let name = default_name(&resolved);
    let description = default_description(&resolved);
    let category = guess_category(&resolved);
    Operation::new("", name, description, category, item_ids)
}

pub fn default_name(items: &[&ChangeItem]) -> String {
    let mut basenames = Vec::new();
    let mut seen = BTreeSet::new();
    for it in items {
        let base = it
            .display_path
            .rsplit('/')
            .next()
            .unwrap_or(&it.display_path);
        if seen.insert(base) {
            basenames.push(base);
        }
    }
    let where_str = if basenames.len() == 1 {
        basenames[0].to_owned()
    } else {
        format!("{} files", basenames.len())
    };

    if !items.is_empty() && items.iter().all(|it| it.kind == ItemKind::AddFile) {
        return truncate_chars(&format!("Add {where_str}"), 120);
    }
    if !items.is_empty() && items.iter().all(|it| it.kind == ItemKind::DeleteFile) {
        return truncate_chars(&format!("Delete {where_str}"), 120);
    }

    let mut def_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for it in items {
        for sym in it
            .lexical
            .defined_added
            .union(&it.lexical.defined_removed)
        {
            *def_counts.entry(sym.as_str()).or_insert(0) += 1;
        }
    }
    let top_sym = def_counts
        .into_iter()
        .max_by_key(|&(sym, count)| (count, std::cmp::Reverse(sym)))
        .map(|(sym, _)| sym)
        .or_else(|| {
            let mut sig_counts: BTreeMap<&str, usize> = BTreeMap::new();
            for it in items {
                for sym in &it.lexical.signature {
                    *sig_counts.entry(sym.as_str()).or_insert(0) += 1;
                }
            }
            sig_counts
                .into_iter()
                .max_by_key(|&(sym, count)| (count, std::cmp::Reverse(sym)))
                .map(|(sym, _)| sym)
        });

    let raw = match top_sym {
        Some(sym) => format!("Update `{sym}` in {where_str}"),
        None => format!("Edit {where_str}"),
    };
    truncate_chars(&raw, 120)
}

pub fn default_description(items: &[&ChangeItem]) -> String {
    let mut files = Vec::new();
    let mut seen = BTreeSet::new();
    for it in items {
        if seen.insert(it.display_path.as_str()) {
            files.push(it.display_path.as_str());
        }
    }
    let listed: Vec<&str> = files.iter().copied().take(5).collect();
    let suffix = if files.len() > 5 {
        format!(" and {} more", files.len() - 5)
    } else {
        String::new()
    };
    format!(
        "Changes {} change item(s) in {}{suffix}.",
        items.len(),
        listed.join(", ")
    )
}

pub fn guess_category(items: &[&ChangeItem]) -> String {
    if items.is_empty() {
        return "change".to_owned();
    }
    let paths: Vec<String> = items
        .iter()
        .map(|it| it.display_path.to_lowercase())
        .collect();
    if paths.iter().all(|p| p.contains("test")) {
        "test".to_owned()
    } else if paths.iter().all(|p| {
        p.ends_with(".md")
            || p.ends_with(".rst")
            || p.ends_with(".txt")
            || p.starts_with("docs/")
    }) {
        "docs".to_owned()
    } else if paths
        .iter()
        .all(|p| CONFIG_EXTENSIONS.iter().any(|ext| p.ends_with(ext)))
    {
        "config".to_owned()
    } else {
        "change".to_owned()
    }
}

pub fn normalize_category(category: Option<&str>) -> String {
    let Some(raw) = category else {
        return "change".to_owned();
    };
    let lower = raw.trim().to_lowercase();
    if CATEGORIES.contains(&lower.as_str()) {
        lower
    } else {
        "change".to_owned()
    }
}

pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}
