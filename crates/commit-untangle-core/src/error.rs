use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CoreError {
    #[error("malformed diff hunk header: {0}")]
    MalformedHunkHeader(String),

    #[error("hunk old line count mismatch: expected {expected}, got {actual}")]
    HunkOldCountMismatch { expected: usize, actual: usize },

    #[error("hunk new line count mismatch: expected {expected}, got {actual}")]
    HunkNewCountMismatch { expected: usize, actual: usize },

    #[error("hunk out of range at start={start}, old_count={old_count}, file_lines={file_lines}, cursor={cursor}")]
    HunkOutOfRange {
        start: usize,
        old_count: usize,
        file_lines: usize,
        cursor: usize,
    },

    #[error("unknown operation id: {0}")]
    UnknownOperation(String),

    #[error("change graph contains a dependency cycle ({visited}/{total} operations ordered)")]
    GraphCycle { visited: usize, total: usize },

    #[error("partition invariant violated: {0}")]
    PartitionViolation(String),
}
