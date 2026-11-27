use std::collections::BTreeSet;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sha1::{Digest, Sha1};

use crate::bytes::RawBytes;
use crate::classify::{classify_items, is_generated_path};
use crate::hunk::{apply_hunk_slice, parse_zero_context_diff, ZeroContextHunk};
use crate::symbols::LexicalInfo;

/// Git tree entry mode.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileMode {
    #[default]
    Regular, // 100644
    Executable, // 100755
    Symlink,    // 120000
    Gitlink,    // 160000
}

impl FileMode {
    pub fn from_octal(mode: &str) -> Option<Self> {
        match mode.trim().trim_start_matches('0') {
            "100644" | "100664" => Some(Self::Regular),
            "100755" => Some(Self::Executable),
            "120000" => Some(Self::Symlink),
            "160000" => Some(Self::Gitlink),
            "" => None,
            _ => Some(Self::Regular),
        }
    }

    pub fn as_octal(&self) -> &'static str {
        match self {
            Self::Regular => "100644",
            Self::Executable => "100755",
            Self::Symlink => "120000",
            Self::Gitlink => "160000",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FileStatus {
    Added,
    Deleted,
    Modified,
    Renamed,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ItemKind {
    Edit,
    Whitespace,
    Reorder,
    BulkReplace,
    Move,
    ModeChange,
    AddFile,
    DeleteFile,
    ReplaceFile,
}

impl ItemKind {
    pub fn is_mechanical(&self) -> bool {
        matches!(
            self,
            Self::Whitespace | Self::Reorder | Self::BulkReplace | Self::Move | Self::ModeChange
        )
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Edit => "EDIT",
            Self::Whitespace => "WHITESPACE",
            Self::Reorder => "REORDER",
            Self::BulkReplace => "BULK_REPLACE",
            Self::Move => "MOVE",
            Self::ModeChange => "MODE_CHANGE",
            Self::AddFile => "ADD_FILE",
            Self::DeleteFile => "DELETE_FILE",
            Self::ReplaceFile => "REPLACE_FILE",
        }
    }
}

/// Pure input delta extracted from Git for a single file entry.
#[derive(Clone, Debug)]
pub struct RawFileDelta {
    pub status: char,
    pub old_path: Option<String>,
    pub new_path: Option<String>,
    pub old_mode: Option<FileMode>,
    pub new_mode: Option<FileMode>,
    pub old_content: Option<RawBytes>,
    pub new_content: Option<RawBytes>,
    /// Output of `git diff -U0` when applicable (for Modified/Renamed non-binary files).
    pub zero_context_diff: Option<RawBytes>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileChange {
    pub index: usize,
    pub status: FileStatus,
    pub old_path: Option<String>,
    pub new_path: Option<String>,
    pub old_mode: Option<FileMode>,
    pub new_mode: Option<FileMode>,
    pub old_content: Option<RawBytes>,
    pub new_content: Option<RawBytes>,
    pub hunks: Vec<ZeroContextHunk>,
    pub hunk_item_ids: Vec<String>,
    pub move_item_id: Option<String>,
    pub mode_item_id: Option<String>,
    pub whole_file_item_id: Option<String>,
    pub generated: bool,
}

impl FileChange {
    pub fn display_path(&self) -> &str {
        self.new_path
            .as_deref()
            .or(self.old_path.as_deref())
            .unwrap_or("?")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangeItem {
    pub id: String,
    pub kind: ItemKind,
    pub file_index: usize,
    pub display_path: String,
    pub old_path: Option<String>,
    pub hunk: Option<ZeroContextHunk>,
    pub removed_bytes: RawBytes,
    pub added_bytes: RawBytes,
    pub substitution: Option<(String, String)>,
    pub fingerprint: String,
    pub lexical: LexicalInfo,
}

impl ChangeItem {
    pub fn new(
        id: String,
        kind: ItemKind,
        file: &FileChange,
        hunk: Option<ZeroContextHunk>,
        removed_bytes: RawBytes,
        added_bytes: RawBytes,
    ) -> Self {
        let display_path = file.display_path().to_owned();
        let fingerprint = compute_fingerprint(&display_path, &removed_bytes, &added_bytes);
        let removed_disp = removed_bytes.to_display();
        let added_disp = added_bytes.to_display();
        let lexical = LexicalInfo::analyze(&removed_disp, &added_disp);

        Self {
            id,
            kind,
            file_index: file.index,
            display_path,
            old_path: file.old_path.clone(),
            hunk,
            removed_bytes,
            added_bytes,
            substitution: None,
            fingerprint,
            lexical,
        }
    }

    pub fn line(&self) -> usize {
        self.hunk.as_ref().map_or(0, |h| h.start)
    }

    pub fn position(&self) -> u64 {
        (self.file_index as u64) * 10_000_000 + (self.line() as u64)
    }
}

pub fn compute_fingerprint(path: &str, removed: &[u8], added: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(path.as_bytes());
    hasher.update([0u8]);
    hasher.update(removed);
    hasher.update([0u8]);
    hasher.update(added);
    hex::encode(hasher.finalize())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Operation {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub item_ids: Vec<String>,
    pub rule: Option<String>,
    pub suspicious: bool,
    pub suspicion_reason: Option<String>,
    pub confidence: f64,
    pub verified: Option<bool>,
}

impl Operation {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        description: impl Into<String>,
        category: impl Into<String>,
        item_ids: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            category: category.into(),
            item_ids,
            rule: None,
            suspicious: false,
            suspicion_reason: None,
            confidence: 0.6,
            verified: None,
        }
    }

    pub fn is_mechanical(&self, items_by_id: &IndexMap<String, ChangeItem>) -> bool {
        !self.item_ids.is_empty()
            && self.item_ids.iter().all(|id| {
                items_by_id
                    .get(id)
                    .is_some_and(|it| it.kind.is_mechanical())
            })
    }

    pub fn is_pure_move_or_mode(&self, items_by_id: &IndexMap<String, ChangeItem>) -> bool {
        !self.item_ids.is_empty()
            && self.item_ids.iter().all(|id| {
                items_by_id
                    .get(id)
                    .is_some_and(|it| matches!(it.kind, ItemKind::Move | ItemKind::ModeChange))
            })
    }

    pub fn defined_added(&self, items_by_id: &IndexMap<String, ChangeItem>) -> BTreeSet<String> {
        self.item_ids
            .iter()
            .filter_map(|id| items_by_id.get(id))
            .flat_map(|it| it.lexical.defined_added.iter().cloned())
            .collect()
    }

    pub fn defined_removed(&self, items_by_id: &IndexMap<String, ChangeItem>) -> BTreeSet<String> {
        self.item_ids
            .iter()
            .filter_map(|id| items_by_id.get(id))
            .flat_map(|it| it.lexical.defined_removed.iter().cloned())
            .collect()
    }

    pub fn refs_added(&self, items_by_id: &IndexMap<String, ChangeItem>) -> BTreeSet<String> {
        self.item_ids
            .iter()
            .filter_map(|id| items_by_id.get(id))
            .flat_map(|it| it.lexical.new_refs.iter().cloned())
            .collect()
    }

    pub fn refs_removed(&self, items_by_id: &IndexMap<String, ChangeItem>) -> BTreeSet<String> {
        self.item_ids
            .iter()
            .filter_map(|id| items_by_id.get(id))
            .flat_map(|it| it.lexical.dropped_refs.iter().cloned())
            .collect()
    }

    pub fn files(&self, items_by_id: &IndexMap<String, ChangeItem>) -> Vec<String> {
        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for id in &self.item_ids {
            if let Some(it) = items_by_id.get(id) {
                if seen.insert(it.display_path.clone()) {
                    out.push(it.display_path.clone());
                }
            }
        }
        out
    }

    pub fn position(&self, items_by_id: &IndexMap<String, ChangeItem>) -> u64 {
        self.item_ids
            .iter()
            .filter_map(|id| items_by_id.get(id))
            .map(|it| it.position())
            .min()
            .unwrap_or(u64::MAX)
    }
}

#[derive(Clone, Debug)]
pub struct ParsedCommit {
    pub files: Vec<FileChange>,
    pub items: IndexMap<String, ChangeItem>,
    pub notes: Vec<String>,
}

/// Stage 1: Parses raw file deltas into `FileChange`s and `ChangeItem`s, verifies byte-exact
/// zero-context hunk reconstruction (with whole-file fallback), decouples `MOVE` and `MODE_CHANGE`,
/// and runs deterministic item classification.
pub fn parse_and_classify(deltas: Vec<RawFileDelta>, bulk_min_occurrences: usize) -> ParsedCommit {
    let mut files = Vec::with_capacity(deltas.len());
    let mut items_vec = Vec::new();
    let mut notes = Vec::new();
    let mut next_id = 1usize;

    let mut alloc_id = || {
        let id = format!("i{next_id}");
        next_id += 1;
        id
    };

    for (idx, delta) in deltas.into_iter().enumerate() {
        match delta.status {
            'A' | 'C' => {
                let new_path = delta
                    .new_path
                    .expect("Added/Copied delta requires new_path");
                let content = delta.new_content.unwrap_or_default();
                let new_mode = Some(delta.new_mode.unwrap_or(FileMode::Regular));
                let generated = is_generated_path(&new_path);
                let mut f = FileChange {
                    index: idx,
                    status: FileStatus::Added,
                    old_path: None,
                    new_path: Some(new_path),
                    old_mode: None,
                    new_mode,
                    old_content: None,
                    new_content: Some(content.clone()),
                    hunks: Vec::new(),
                    hunk_item_ids: Vec::new(),
                    move_item_id: None,
                    mode_item_id: None,
                    whole_file_item_id: None,
                    generated,
                };
                let added_lexical = if content.is_binary() || new_mode == Some(FileMode::Gitlink) {
                    RawBytes::default()
                } else {
                    content
                };
                let item_id = alloc_id();
                let item = ChangeItem::new(
                    item_id.clone(),
                    ItemKind::AddFile,
                    &f,
                    None,
                    RawBytes::default(),
                    added_lexical,
                );
                f.whole_file_item_id = Some(item_id);
                items_vec.push(item);
                files.push(f);
            }
            'D' => {
                let old_path = delta.old_path.expect("Deleted delta requires old_path");
                let content = delta.old_content.unwrap_or_default();
                let old_mode = Some(delta.old_mode.unwrap_or(FileMode::Regular));
                let generated = is_generated_path(&old_path);
                let mut f = FileChange {
                    index: idx,
                    status: FileStatus::Deleted,
                    old_path: Some(old_path),
                    new_path: None,
                    old_mode,
                    new_mode: None,
                    old_content: Some(content.clone()),
                    new_content: None,
                    hunks: Vec::new(),
                    hunk_item_ids: Vec::new(),
                    move_item_id: None,
                    mode_item_id: None,
                    whole_file_item_id: None,
                    generated,
                };
                let removed_lexical = if content.is_binary() || old_mode == Some(FileMode::Gitlink)
                {
                    RawBytes::default()
                } else {
                    content
                };
                let item_id = alloc_id();
                let item = ChangeItem::new(
                    item_id.clone(),
                    ItemKind::DeleteFile,
                    &f,
                    None,
                    removed_lexical,
                    RawBytes::default(),
                );
                f.whole_file_item_id = Some(item_id);
                items_vec.push(item);
                files.push(f);
            }
            other_status => {
                let old_path = delta
                    .old_path
                    .expect("Modified/Renamed delta requires old_path");
                let new_path = delta.new_path.unwrap_or_else(|| old_path.clone());
                let old_content = delta.old_content.unwrap_or_default();
                let new_content = delta.new_content.unwrap_or_default();
                let old_mode = Some(delta.old_mode.unwrap_or(FileMode::Regular));
                let new_mode = Some(delta.new_mode.unwrap_or(FileMode::Regular));
                let is_rename = old_path != new_path;
                let generated = is_generated_path(&new_path);

                let mut f = FileChange {
                    index: idx,
                    status: if is_rename {
                        FileStatus::Renamed
                    } else {
                        FileStatus::Modified
                    },
                    old_path: Some(old_path),
                    new_path: Some(new_path.clone()),
                    old_mode,
                    new_mode,
                    old_content: Some(old_content.clone()),
                    new_content: Some(new_content.clone()),
                    hunks: Vec::new(),
                    hunk_item_ids: Vec::new(),
                    move_item_id: None,
                    mode_item_id: None,
                    whole_file_item_id: None,
                    generated,
                };

                if is_rename {
                    let move_id = alloc_id();
                    let move_item = ChangeItem::new(
                        move_id.clone(),
                        ItemKind::Move,
                        &f,
                        None,
                        RawBytes::default(),
                        RawBytes::default(),
                    );
                    f.move_item_id = Some(move_id);
                    items_vec.push(move_item);
                }

                if old_mode != new_mode && other_status != 'T' {
                    let mode_id = alloc_id();
                    let mode_item = ChangeItem::new(
                        mode_id.clone(),
                        ItemKind::ModeChange,
                        &f,
                        None,
                        RawBytes::default(),
                        RawBytes::default(),
                    );
                    f.mode_item_id = Some(mode_id);
                    items_vec.push(mode_item);
                }

                if old_content == new_content && other_status != 'T' {
                    files.push(f);
                    continue;
                }

                let is_non_text = other_status == 'T'
                    || old_mode == Some(FileMode::Gitlink)
                    || new_mode == Some(FileMode::Gitlink)
                    || old_mode == Some(FileMode::Symlink)
                    || new_mode == Some(FileMode::Symlink)
                    || old_content.is_binary()
                    || new_content.is_binary();

                if is_non_text {
                    if other_status == 'T' && old_mode != new_mode && f.mode_item_id.is_none() {
                        let mode_id = alloc_id();
                        let mode_item = ChangeItem::new(
                            mode_id.clone(),
                            ItemKind::ModeChange,
                            &f,
                            None,
                            RawBytes::default(),
                            RawBytes::default(),
                        );
                        f.mode_item_id = Some(mode_id);
                        items_vec.push(mode_item);
                    }
                    let item_id = alloc_id();
                    let item = ChangeItem::new(
                        item_id.clone(),
                        ItemKind::ReplaceFile,
                        &f,
                        None,
                        RawBytes::default(),
                        RawBytes::default(),
                    );
                    f.whole_file_item_id = Some(item_id);
                    items_vec.push(item);
                    files.push(f);
                    continue;
                }

                let parsed_hunks = delta
                    .zero_context_diff
                    .as_ref()
                    .and_then(|raw| parse_zero_context_diff(raw).ok());

                let exact = parsed_hunks.as_ref().is_some_and(|hs| {
                    !hs.is_empty()
                        && apply_hunk_slice(&old_content, hs)
                            .is_ok_and(|reconstructed| reconstructed == new_content)
                });

                if !exact {
                    notes.push(format!(
                        "`{new_path}`: could not be reproduced exactly from zero-context hunks; treated as a whole-file replacement"
                    ));
                    let item_id = alloc_id();
                    let item = ChangeItem::new(
                        item_id.clone(),
                        ItemKind::ReplaceFile,
                        &f,
                        None,
                        old_content,
                        new_content,
                    );
                    f.whole_file_item_id = Some(item_id);
                    items_vec.push(item);
                    files.push(f);
                    continue;
                }

                let hunks = parsed_hunks.expect("verified Some");
                for h in hunks {
                    let mut removed = Vec::new();
                    for l in &h.old_lines {
                        removed.extend_from_slice(l);
                    }
                    let mut added = Vec::new();
                    for l in &h.new_lines {
                        added.extend_from_slice(l);
                    }
                    let item_id = alloc_id();
                    let item = ChangeItem::new(
                        item_id.clone(),
                        ItemKind::Edit,
                        &f,
                        Some(h.clone()),
                        RawBytes(removed),
                        RawBytes(added),
                    );
                    f.hunks.push(h);
                    f.hunk_item_ids.push(item_id);
                    items_vec.push(item);
                }
                files.push(f);
            }
        }
    }

    classify_items(&mut items_vec, bulk_min_occurrences);

    let items: IndexMap<String, ChangeItem> = items_vec
        .into_iter()
        .map(|it| (it.id.clone(), it))
        .collect();

    ParsedCommit {
        files,
        items,
        notes,
    }
}
