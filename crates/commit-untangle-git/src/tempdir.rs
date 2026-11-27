use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Lightweight RAII temporary directory cleaned up automatically on `Drop`.
#[derive(Debug)]
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(prefix: &str) -> std::io::Result<Self> {
        let base = std::env::temp_dir();
        let pid = unsafe { libc::getpid() };
        for attempt in 0..100u64 {
            let seq = COUNTER.fetch_add(1, Ordering::Relaxed) + attempt;
            let candidate = base.join(format!("{prefix}-{pid}-{seq}"));
            match std::fs::create_dir(&candidate) {
                Ok(()) => return Ok(Self { path: candidate }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "failed to allocate unique temporary directory",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
