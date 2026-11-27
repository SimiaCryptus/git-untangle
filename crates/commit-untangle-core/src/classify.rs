use indexmap::IndexMap;

use crate::hunk::ZeroContextHunk;
use crate::model::{ChangeItem, ItemKind};

const GENERATED_NAMES: &[&str] = &[
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "go.sum",
    "poetry.lock",
    "Gemfile.lock",
    "composer.lock",
    "Pipfile.lock",
    "gradle.lockfile",
    "flake.lock",
];

pub fn is_generated_path(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    GENERATED_NAMES.contains(&name)
        || name.ends_with(".lock")
        || name.contains(".generated.")
        || name.ends_with(".min.js")
        || name.ends_with(".pb.go")
        || path
            .split('/')
            .any(|seg| matches!(seg, "vendor" | "node_modules" | "generated"))
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Checks whether a hunk is a pure permutation of existing lines (ignoring trailing whitespace).
pub fn is_reorder(h: &ZeroContextHunk) -> bool {
    if h.old_lines.len() != h.new_lines.len() || h.old_lines.len() < 2 {
        return false;
    }
    let old_trimmed: Vec<Vec<u8>> = h
        .old_lines
        .iter()
        .map(|l| l.trim_end_whitespace().to_vec())
        .collect();
    let new_trimmed: Vec<Vec<u8>> = h
        .new_lines
        .iter()
        .map(|l| l.trim_end_whitespace().to_vec())
        .collect();
    if old_trimmed == new_trimmed {
        return false;
    }
    let mut old_sorted = old_trimmed;
    let mut new_sorted = new_trimmed;
    old_sorted.sort();
    new_sorted.sort();
    old_sorted == new_sorted
}

/// Detects a single identifier-bounded literal substitution `a -> b` that transforms every line
/// of `old_lines` into the corresponding line of `new_lines`.
///
/// Operates on `char` slices over valid UTF-8 lines so multibyte characters never split mid-codepoint.
pub fn substitution_of(h: &ZeroContextHunk) -> Option<(String, String)> {
    if h.old_lines.len() != h.new_lines.len() || h.old_lines.is_empty() {
        return None;
    }

    let mut old_strs = Vec::with_capacity(h.old_lines.len());
    let mut new_strs = Vec::with_capacity(h.new_lines.len());
    for (ol, nl) in h.old_lines.iter().zip(h.new_lines.iter()) {
        old_strs.push(std::str::from_utf8(ol).ok()?);
        new_strs.push(std::str::from_utf8(nl).ok()?);
    }

    let o_chars: Vec<char> = old_strs[0].chars().collect();
    let n_chars: Vec<char> = new_strs[0].chars().collect();

    let mut p = 0usize;
    while p < o_chars.len() && p < n_chars.len() && o_chars[p] == n_chars[p] {
        p += 1;
    }
    let mut s = 0usize;
    while s < o_chars.len() - p
        && s < n_chars.len() - p
        && o_chars[o_chars.len() - 1 - s] == n_chars[n_chars.len() - 1 - s]
    {
        s += 1;
    }

    while p > 0 && is_ident_char(o_chars[p - 1]) {
        p -= 1;
    }
    while s > 0 && is_ident_char(o_chars[o_chars.len() - s]) {
        s -= 1;
    }

    if p > o_chars.len().saturating_sub(s) || p > n_chars.len().saturating_sub(s) {
        return None;
    }

    let a: String = o_chars[p..o_chars.len() - s].iter().collect();
    let b: String = n_chars[p..n_chars.len() - s].iter().collect();

    if a.trim().is_empty()
        || a.len() > 200
        || b.len() > 200
        || a.contains('\n')
        || b.contains('\n')
        || a == b
    {
        return None;
    }

    let applies_all = old_strs
        .iter()
        .zip(new_strs.iter())
        .all(|(old_line, new_line)| old_line.replace(&a, &b) == *new_line);

    if applies_all {
        Some((a, b))
    } else {
        None
    }
}

/// Stage 1B: Classifies `EDIT` items into `WHITESPACE`, `REORDER`, or `BULK_REPLACE` where applicable.
pub fn classify_items(items: &mut [ChangeItem], bulk_min_occurrences: usize) {
    let min_bulk = bulk_min_occurrences.max(2);

    for item in items.iter_mut() {
        if item.kind != ItemKind::Edit {
            continue;
        }
        let Some(ref hunk) = item.hunk else {
            continue;
        };
        if item.removed_bytes.strip_whitespace() == item.added_bytes.strip_whitespace() {
            item.kind = ItemKind::Whitespace;
        } else if is_reorder(hunk) {
            item.kind = ItemKind::Reorder;
        } else {
            item.substitution = substitution_of(hunk);
        }
    }

    let mut sub_counts: IndexMap<(String, String), Vec<usize>> = IndexMap::new();
    for (idx, item) in items.iter().enumerate() {
        if item.kind == ItemKind::Edit {
            if let Some(ref sub) = item.substitution {
                sub_counts.entry(sub.clone()).or_default().push(idx);
            }
        }
    }

    for (_, indices) in sub_counts {
        if indices.len() >= min_bulk {
            for idx in indices {
                items[idx].kind = ItemKind::BulkReplace;
            }
        }
    }
}
