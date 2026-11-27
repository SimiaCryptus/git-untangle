use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use commit_untangle_core::model::{ChangeItem, FileChange, FileMode};
use commit_untangle_core::replay::{desired_path_states, MaterializedEntry};
use indexmap::IndexMap;

use crate::tempdir::TempDir;
use crate::GitError;

/// RAII scratch Git worktree checked out at `base_commit`, automatically removed and pruned on `Drop`.
#[derive(Debug)]
pub struct TempWorktree {
    repo_root: PathBuf,
    _temp_root: TempDir,
    worktree_dir: PathBuf,
}

impl TempWorktree {
    pub fn open(repo_root: &Path, base_commit: &str) -> Result<Self, GitError> {
        let temp_root = TempDir::new("git-untangle-wt")?;
        let worktree_dir = temp_root.path().join("wt");
        let wt_str = worktree_dir.to_string_lossy().into_owned();

        let out = Command::new("git")
            .current_dir(repo_root)
            .args(["worktree", "add", "--detach", &wt_str, base_commit])
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()?;

        if !out.status.success() {
            return Err(GitError::CommandFailed {
                cmd: format!("git worktree add --detach {wt_str} {base_commit}"),
                code: out.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&out.stderr).trim().to_owned(),
            });
        }

        Ok(Self {
            repo_root: repo_root.to_path_buf(),
            _temp_root: temp_root,
            worktree_dir,
        })
    }

    pub fn path(&self) -> &Path {
        &self.worktree_dir
    }
}

impl Drop for TempWorktree {
    fn drop(&mut self) {
        let wt_str = self.worktree_dir.to_string_lossy().into_owned();
        let _ = Command::new("git")
            .current_dir(&self.repo_root)
            .args(["worktree", "remove", "--force", &wt_str])
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = Command::new("git")
            .current_dir(&self.repo_root)
            .args(["worktree", "prune"])
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}

/// Incremental filesystem replayer syncing subsets of `ChangeItem`s into a scratch worktree.
pub struct WorktreeReplayer {
    worktree_dir: PathBuf,
    files: Vec<FileChange>,
    items_by_id: IndexMap<String, ChangeItem>,
    last: IndexMap<String, Option<MaterializedEntry>>,
}

impl WorktreeReplayer {
    pub fn new(
        worktree_dir: PathBuf,
        files: Vec<FileChange>,
        items_by_id: IndexMap<String, ChangeItem>,
    ) -> Result<Self, GitError> {
        let all_refs: Vec<&FileChange> = files.iter().collect();
        let last = desired_path_states(&files, &all_refs, &items_by_id, &BTreeSet::new())?;
        Ok(Self {
            worktree_dir,
            files,
            items_by_id,
            last,
        })
    }

    /// Synchronizes the worktree to match `applied`. If `touched_indices` is `None`, checks all files.
    /// Returns the list of paths that were actually modified on disk.
    pub fn sync(
        &mut self,
        applied: &BTreeSet<String>,
        touched_indices: Option<&[usize]>,
    ) -> Result<Vec<String>, GitError> {
        let touched: Vec<&FileChange> = match touched_indices {
            Some(idxs) => idxs
                .iter()
                .filter_map(|&i| self.files.get(i))
                .collect(),
            None => self.files.iter().collect(),
        };

        let desired = desired_path_states(&self.files, &touched, &self.items_by_id, applied)?;
        let mut changed = Vec::new();

        // Deletions first to avoid file <-> directory collisions
        for (path, entry_opt) in &desired {
            if entry_opt.is_none() && self.last.get(path).is_some_and(Option::is_some) {
                let target = self.worktree_dir.join(path);
                let _ = remove_path_any(&target);
                self.last.insert(path.clone(), None);
                changed.push(path.clone());
            }
        }

        // Creations / updates
        for (path, entry_opt) in desired {
            let Some(entry) = entry_opt else {
                continue;
            };
            if self.last.get(&path) == Some(&Some(entry.clone())) {
                continue;
            }
            if entry.mode == FileMode::Gitlink {
                // Submodule gitlink entries cannot be written as plain blobs in worktree
                self.last.insert(path.clone(), Some(entry));
                continue;
            }
            let target = self.worktree_dir.join(&path);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let _ = remove_path_any(&target);

            match entry.mode {
                FileMode::Symlink => {
                    write_symlink(&target, entry.content.as_slice())?;
                }
                FileMode::Regular | FileMode::Executable => {
                    fs::write(&target, entry.content.as_slice())?;
                    apply_mode(&target, entry.mode)?;
                }
                FileMode::Gitlink => {}
            }

            self.last.insert(path.clone(), Some(entry));
            changed.push(path);
        }

        Ok(changed)
    }
}

fn remove_path_any(path: &Path) -> std::io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(unix)]
fn write_symlink(target: &Path, link_bytes: &[u8]) -> std::io::Result<()> {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    std::os::unix::fs::symlink(OsStr::from_bytes(link_bytes), target)
}

#[cfg(not(unix))]
fn write_symlink(target: &Path, link_bytes: &[u8]) -> std::io::Result<()> {
    fs::write(target, link_bytes)
}

#[cfg(unix)]
fn apply_mode(path: &Path, mode: FileMode) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let perm = match mode {
        FileMode::Executable => 0o755,
        _ => 0o644,
    };
    fs::set_permissions(path, fs::Permissions::from_mode(perm))
}

#[cfg(not(unix))]
fn apply_mode(_path: &Path, _mode: FileMode) -> std::io::Result<()> {
    Ok(())
}

/// Runs a shell verification command in `dir` with a hard timeout, capturing the combined
/// stdout/stderr tail (up to 6,000 characters) without pipe deadlocks.
pub fn run_verify_command(
    command: &str,
    dir: &Path,
    timeout_secs: u64,
) -> Result<(bool, String), GitError> {
    let log_dir = TempDir::new("git-untangle-build")?;
    let log_path = log_dir.path().join("build.log");
    let stdout_file = fs::File::create(&log_path)?;
    let stderr_file = stdout_file.try_clone()?;

    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };

    let mut child = cmd
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()?;

    let (tx, rx) = mpsc::channel();
    let pid = child.id();
    let handle = std::thread::spawn(move || {
        let status = child.wait();
        let _ = tx.send(status);
    });

    let result = match rx.recv_timeout(Duration::from_secs(timeout_secs)) {
        Ok(Ok(status)) => {
            let _ = handle.join();
            let raw = fs::read(&log_path).unwrap_or_default();
            let tail = tail_chars(&String::from_utf8_lossy(&raw), 6000);
            (status.success(), tail)
        }
        Ok(Err(e)) => {
            let _ = handle.join();
            return Err(GitError::Io(e));
        }
        Err(_) => {
            kill_pid(pid);
            let _ = handle.join();
            let raw = fs::read(&log_path).unwrap_or_default();
            let tail = tail_chars(&String::from_utf8_lossy(&raw), 6000);
            (false, format!("TIMEOUT after {timeout_secs}s\n{tail}"))
        }
    };

    Ok(result)
}

#[cfg(unix)]
fn kill_pid(pid: u32) {
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_pid(_pid: u32) {}

fn tail_chars(s: &str, max_chars: usize) -> String {
    let count = s.chars().count();
    if count <= max_chars {
        s.to_owned()
    } else {
        s.chars().skip(count - max_chars).collect()
    }
}
