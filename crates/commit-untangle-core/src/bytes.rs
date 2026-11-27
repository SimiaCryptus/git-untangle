use std::fmt;
use std::ops::Deref;

use memchr::memchr_iter;
use serde::{Deserialize, Serialize};

/// Canonical byte-exact representation of repository content, lines, and patches.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RawBytes(pub Vec<u8>);

impl RawBytes {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Converts raw bytes to a UTF-8 display string using U+FFFD replacement at the boundary.
    pub fn to_display(&self) -> DisplayText {
        DisplayText(String::from_utf8_lossy(&self.0).into_owned())
    }

    /// Matches Git's binary heuristic: NUL byte within the first 8,000 bytes.
    pub fn is_binary(&self) -> bool {
        let limit = self.0.len().min(8000);
        self.0[..limit].contains(&0u8)
    }

    /// Splits byte content into lines while preserving each line's exact `\n` (or `\r\n`) terminator.
    /// An empty slice produces an empty `Vec`.
    pub fn split_keeping_terminators(&self) -> Vec<RawBytes> {
        if self.0.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut start = 0usize;
        for idx in memchr_iter(b'\n', &self.0) {
            out.push(RawBytes(self.0[start..=idx].to_vec()));
            start = idx + 1;
        }
        if start < self.0.len() {
            out.push(RawBytes(self.0[start..].to_vec()));
        }
        out
    }

    /// Strips all whitespace characters (Unicode whitespace if valid UTF-8, ASCII whitespace otherwise).
    pub fn strip_whitespace(&self) -> Vec<u8> {
        if let Ok(s) = std::str::from_utf8(&self.0) {
            s.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
                .into_bytes()
        } else {
            self.0
                .iter()
                .copied()
                .filter(|b| !b.is_ascii_whitespace())
                .collect()
        }
    }

    /// Returns a slice with trailing whitespace (including `\r` and `\n`) trimmed.
    pub fn trim_end_whitespace(&self) -> &[u8] {
        if let Ok(s) = std::str::from_utf8(&self.0) {
            s.trim_end().as_bytes()
        } else {
            let mut end = self.0.len();
            while end > 0 && self.0[end - 1].is_ascii_whitespace() {
                end -= 1;
            }
            &self.0[..end]
        }
    }
}

impl Deref for RawBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[u8]> for RawBytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Vec<u8>> for RawBytes {
    fn from(v: Vec<u8>) -> Self {
        Self(v)
    }
}

impl From<&[u8]> for RawBytes {
    fn from(s: &[u8]) -> Self {
        Self(s.to_vec())
    }
}

impl From<&str> for RawBytes {
    fn from(s: &str) -> Self {
        Self(s.as_bytes().to_vec())
    }
}

impl From<String> for RawBytes {
    fn from(s: String) -> Self {
        Self(s.into_bytes())
    }
}

impl fmt::Debug for RawBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RawBytes({:?})", String::from_utf8_lossy(&self.0))
    }
}

/// Lossy UTF-8 projection used exclusively for lexical analysis, prompts, and display artifacts.
#[derive(Clone, Default, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DisplayText(pub String);

impl DisplayText {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Deref for DisplayText {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for DisplayText {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for DisplayText {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl fmt::Display for DisplayText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
