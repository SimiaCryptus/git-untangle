use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::bytes::RawBytes;
use crate::error::CoreError;
use crate::hunk::{apply_hunks, ZeroContextHunk};
use crate::model::{ChangeItem, FileChange, FileMode, FileStatus};

/// Concrete state of a single repository path for a given subset of applied `ChangeItem`s.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterializedEntry {
    pub path: String,
    pub mode: FileMode,
    pub content: RawBytes,
}

/// Evaluates `fileState(f, applied)` purely in memory.
/// Returns `Ok(None)` when the file is absent in the given state.
pub fn file_state(
    file: &FileChange,
    items_by_id: &IndexMap<String, ChangeItem>,
    applied: &BTreeSet<String>,
) -> Result<Option<MaterializedEntry>, CoreError> {
    match file.status {
        FileStatus::Added => {
            let is_applied = file
                .whole_file_item_id
                .as_ref()
                .is_some_and(|id| applied.contains(id));
            if is_applied {
                Ok(Some(MaterializedEntry {
                    path: file.new_path.clone().expect("Added file has new_path"),
                    mode: file.new_mode.unwrap_or(FileMode::Regular),
                    content: file.new_content.clone().unwrap_or_default(),
                }))
            } else {
                Ok(None)
            }
        }
        FileStatus::Deleted => {
            let is_applied = file
                .whole_file_item_id
                .as_ref()
                .is_some_and(|id| applied.contains(id));
            if is_applied {
                Ok(None)
            } else {
                Ok(Some(MaterializedEntry {
                    path: file.old_path.clone().expect("Deleted file has old_path"),
                    mode: file.old_mode.unwrap_or(FileMode::Regular),
                    content: file.old_content.clone().unwrap_or_default(),
                }))
            }
        }
        FileStatus::Modified | FileStatus::Renamed => {
            let moved = file
                .move_item_id
                .as_ref()
                .is_some_and(|id| applied.contains(id));
            let path = if moved {
                file.new_path.clone().expect("Renamed/Modified file has new_path")
            } else {
                file.old_path.clone().expect("Renamed/Modified file has old_path")
            };

            let mode_changed = file
                .mode_item_id
                .as_ref()
                .is_some_and(|id| applied.contains(id));
            let mode = if mode_changed || file.mode_item_id.is_none() && moved && file.old_mode == file.new_mode {
                file.new_mode.unwrap_or(FileMode::Regular)
            } else if file.mode_item_id.is_none() {
                file.new_mode
                    .or(file.old_mode)
                    .unwrap_or(FileMode::Regular)
            } else {
                file.old_mode.unwrap_or(FileMode::Regular)
            };

            let content = if let Some(ref whole_id) = file.whole_file_item_id {
                if applied.contains(whole_id) {
                    file.new_content.clone().unwrap_or_default()
                } else {
                    file.old_content.clone().unwrap_or_default()
                }
            } else {
                let selected_hunks: Vec<&ZeroContextHunk> = file
                    .hunk_item_ids
                    .iter()
                    .filter(|id| applied.contains(*id))
                    .filter_map(|id| items_by_id.get(id)?.hunk.as_ref())
                    .collect();
                apply_hunks(
                    file.old_content.as_ref().expect("Modified file has old_content"),
                    &selected_hunks,
                )?
            };

            Ok(Some(MaterializedEntry {
                path,
                mode,
                content,
            }))
        }
    }
}

/// Computes the desired state (`Some(MaterializedEntry)` or `None` if absent/deleted) for every
/// path touched by `touched_files` under `applied`.
///
/// Handles swap/overwrite cases where one `FileChange` vacates `p` and another `FileChange`
/// populates `p` in the same state.
pub fn desired_path_states(
    all_files: &[FileChange],
    touched_files: &[&FileChange],
    items_by_id: &IndexMap<String, ChangeItem>,
    applied: &BTreeSet<String>,
) -> Result<IndexMap<String, Option<MaterializedEntry>>, CoreError> {
    let mut candidate_paths = BTreeSet::new();
    for f in touched_files {
        if let Some(ref p) = f.old_path {
            candidate_paths.insert(p.clone());
        }
        if let Some(ref p) = f.new_path {
            candidate_paths.insert(p.clone());
        }
    }

    let mut result = IndexMap::with_capacity(candidate_paths.len());
    for path in candidate_paths {
        let mut found: Option<MaterializedEntry> = None;
        for f in all_files {
            if f.old_path.as_deref() == Some(path.as_str())
                || f.new_path.as_deref() == Some(path.as_str())
            {
                if let Some(entry) = file_state(f, items_by_id, applied)? {
                    if entry.path == path {
                        found = Some(entry);
                        break;
                    }
                }
            }
        }
        result.insert(path, found);
    }
    Ok(result)
}

/// Verifies Invariant I3: applying all items reproduces every `FileChange`'s expected tip state.
/// Returns the display paths of any mismatched files (empty when `exact_reconstruction == true`).
pub fn verify_exact_reconstruction(
    files: &[FileChange],
    items_by_id: &IndexMap<String, ChangeItem>,
) -> Vec<String> {
    let all_applied: BTreeSet<String> = items_by_id.keys().cloned().collect();
    let mut mismatches = Vec::new();

    for f in files {
        let expected = match f.status {
            FileStatus::Deleted => None,
            _ => Some(MaterializedEntry {
                path: f.new_path.clone().unwrap_or_default(),
                mode: f.new_mode.unwrap_or(FileMode::Regular),
                content: f.new_content.clone().unwrap_or_default(),
            }),
        };
        match file_state(f, items_by_id, &all_applied) {
            Ok(actual) if actual == expected => {}
            _ => mismatches.push(f.display_path().to_owned()),
        }
    }
    mismatches
}
