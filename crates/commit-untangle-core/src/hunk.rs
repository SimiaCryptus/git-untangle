use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::bytes::RawBytes;
use crate::error::CoreError;

static HUNK_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@").expect("valid hunk regex")
});

/// A zero-context diff hunk anchored to 0-based line indices in `old_content`.
///
/// For replacements/deletions (`old_count > 0`), `start` is `old_start - 1`.
/// For pure insertions (`old_count == 0`), `start` is `old_start` (0 = top of file,
/// `lines.len()` = end of file).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZeroContextHunk {
    pub start: usize,
    pub old_count: usize,
    pub new_start: usize,
    pub old_lines: Vec<RawBytes>,
    pub new_lines: Vec<RawBytes>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum LastSide {
    Old,
    New,
}

/// Parses raw `git diff -U0` output bytes into non-overlapping `ZeroContextHunk`s,
/// preserving exact line terminators (`\n`, `\r\n`, and missing EOF newlines).
pub fn parse_zero_context_diff(diff: &[u8]) -> Result<Vec<ZeroContextHunk>, CoreError> {
    if diff.is_empty() {
        return Ok(Vec::new());
    }

    let raw_lines = RawBytes(diff.to_vec()).split_keeping_terminators();
    let mut hunks = Vec::new();

    let mut cur_header: Option<(usize, usize, usize, usize)> = None;
    let mut old_lines: Vec<RawBytes> = Vec::new();
    let mut new_lines: Vec<RawBytes> = Vec::new();
    let mut last_side: Option<LastSide> = None;

    let finish = |header: Option<(usize, usize, usize, usize)>,
                  old: Vec<RawBytes>,
                  new: Vec<RawBytes>,
                  out: &mut Vec<ZeroContextHunk>|
     -> Result<(), CoreError> {
        let Some((old_start, old_count, new_start, new_count)) = header else {
            return Ok(());
        };
        if old.len() != old_count {
            return Err(CoreError::HunkOldCountMismatch {
                expected: old_count,
                actual: old.len(),
            });
        }
        if new.len() != new_count {
            return Err(CoreError::HunkNewCountMismatch {
                expected: new_count,
                actual: new.len(),
            });
        }
        let start = if old_count == 0 {
            old_start
        } else {
            old_start.checked_sub(1).ok_or_else(|| {
                CoreError::MalformedHunkHeader(format!(
                    "old_start=0 with nonzero old_count={old_count}"
                ))
            })?
        };
        out.push(ZeroContextHunk {
            start,
            old_count,
            new_start,
            old_lines: old,
            new_lines: new,
        });
        Ok(())
    };

    for raw_line in raw_lines {
        let bytes = raw_line.as_slice();
        if bytes.starts_with(b"@@") {
            let old = std::mem::take(&mut old_lines);
            let new = std::mem::take(&mut new_lines);
            finish(cur_header.take(), old, new, &mut hunks)?;

            let header_str = String::from_utf8_lossy(bytes);
            let caps = HUNK_HEADER_RE
                .captures(&header_str)
                .ok_or_else(|| CoreError::MalformedHunkHeader(header_str.trim().to_owned()))?;
            let old_start: usize = caps[1]
                .parse()
                .map_err(|_| CoreError::MalformedHunkHeader(header_str.trim().to_owned()))?;
            let old_count: usize = caps
                .get(2)
                .map_or("1", |m| m.as_str())
                .parse()
                .map_err(|_| CoreError::MalformedHunkHeader(header_str.trim().to_owned()))?;
            let new_start: usize = caps[3]
                .parse()
                .map_err(|_| CoreError::MalformedHunkHeader(header_str.trim().to_owned()))?;
            let new_count: usize = caps
                .get(4)
                .map_or("1", |m| m.as_str())
                .parse()
                .map_err(|_| CoreError::MalformedHunkHeader(header_str.trim().to_owned()))?;

            cur_header = Some((old_start, old_count, new_start, new_count));
            last_side = None;
            continue;
        }

        if cur_header.is_none() || bytes.is_empty() {
            continue;
        }

        match bytes[0] {
            b'-' => {
                let mut content = bytes[1..].to_vec();
                if !content.ends_with(b"\n") {
                    content.push(b'\n');
                }
                old_lines.push(RawBytes(content));
                last_side = Some(LastSide::Old);
            }
            b'+' => {
                let mut content = bytes[1..].to_vec();
                if !content.ends_with(b"\n") {
                    content.push(b'\n');
                }
                new_lines.push(RawBytes(content));
                last_side = Some(LastSide::New);
            }
            b'\\' => {
                let target = match last_side {
                    Some(LastSide::Old) => old_lines.last_mut(),
                    Some(LastSide::New) => new_lines.last_mut(),
                    None => None,
                };
                if let Some(last) = target {
                    if last.0.ends_with(b"\n") {
                        last.0.pop();
                    }
                }
            }
            _ => {}
        }
    }

    finish(cur_header, old_lines, new_lines, &mut hunks)?;
    Ok(hunks)
}

/// Applies any subset of non-overlapping zero-context hunks to `base`.
/// Because zero-context hunks never overlap on `base`, any subset commutes cleanly.
pub fn apply_hunks(base: &RawBytes, hunks: &[&ZeroContextHunk]) -> Result<RawBytes, CoreError> {
    if hunks.is_empty() {
        return Ok(base.clone());
    }
    let lines = base.split_keeping_terminators();
    let mut sorted: Vec<&ZeroContextHunk> = hunks.to_vec();
    sorted.sort_by_key(|h| (h.start, h.old_count));

    let mut out = Vec::with_capacity(base.len() + 256);
    let mut pos = 0usize;

    for h in sorted {
        if h.start < pos || h.start + h.old_count > lines.len() {
            return Err(CoreError::HunkOutOfRange {
                start: h.start,
                old_count: h.old_count,
                file_lines: lines.len(),
                cursor: pos,
            });
        }
        for line in &lines[pos..h.start] {
            out.extend_from_slice(line);
        }
        for line in &h.new_lines {
            out.extend_from_slice(line);
        }
        pos = h.start + h.old_count;
    }

    for line in &lines[pos..] {
        out.extend_from_slice(line);
    }

    Ok(RawBytes(out))
}

/// Convenience wrapper applying a slice of owned `ZeroContextHunk`s.
pub fn apply_hunk_slice(base: &RawBytes, hunks: &[ZeroContextHunk]) -> Result<RawBytes, CoreError> {
    let refs: Vec<&ZeroContextHunk> = hunks.iter().collect();
    apply_hunks(base, &refs)
}
