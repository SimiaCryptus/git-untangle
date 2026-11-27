use std::collections::{BTreeSet, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use commit_untangle_core::bytes::RawBytes;
use commit_untangle_core::graph::ChangeGraph;
use commit_untangle_core::model::{ChangeItem, FileChange, FileMode, Operation, RawFileDelta};
use commit_untangle_core::replay::desired_path_states;
use commit_untangle_core::schema::CommitSummary;
use indexmap::IndexMap;
use tracing::debug;

use crate::tempdir::TempDir;
use crate::{
    DagEmitResult, EmitOptions, GitBackend, GitError, RepoInfo, ResolvedTarget, TargetSpec,
};

#[derive(Clone, Debug)]
pub struct GitCliBackend {
    start_dir: PathBuf,
}

#[derive(Debug)]
struct RawDiffEntry {
    old_mode: Option<FileMode>,
    new_mode: Option<FileMode>,
    old_oid: String,
    new_oid: String,
    status: char,
    old_path: Option<String>,
    new_path: Option<String>,
}

impl GitCliBackend {
    pub fn discover(start_dir: impl Into<PathBuf>) -> Result<(Self, RepoInfo), GitError> {
        let dir = start_dir.into();
        if !dir.is_dir() {
            return Err(GitError::Precondition(format!(
                "repository path does not exist: {}",
                dir.display()
            )));
        }
        let probe = Self {
            start_dir: dir.clone(),
        };
        let toplevel_str = probe
            .git_text(&dir, &["rev-parse", "--show-toplevel"], &[])
            .map_err(|_| {
                GitError::Precondition(format!("not a git repository: {}", dir.display()))
            })?;
        let toplevel = std::fs::canonicalize(toplevel_str.trim())
            .unwrap_or_else(|_| PathBuf::from(toplevel_str.trim()));
        let prefix = probe
            .git_text(&dir, &["rev-parse", "--show-prefix"], &[])
            .unwrap_or_default();
        let git_dir_raw = probe.git_text(&toplevel, &["rev-parse", "--git-dir"], &[])?;
        let git_dir_path = PathBuf::from(git_dir_raw.trim());
        let git_dir = if git_dir_path.is_absolute() {
            git_dir_path
        } else {
            toplevel.join(git_dir_path)
        };

        let info = RepoInfo {
            toplevel: toplevel.clone(),
            prefix: prefix.trim().to_owned(),
            git_dir,
        };
        Ok((
            Self {
                start_dir: toplevel,
            },
            info,
        ))
    }

    pub fn root(&self) -> &Path {
        &self.start_dir
    }

    fn git_exec(
        &self,
        cwd: &Path,
        args: &[&str],
        env: &[(&str, &str)],
        stdin_bytes: Option<&[u8]>,
        allowed_codes: &[i32],
    ) -> Result<(i32, Vec<u8>, String), GitError> {
        let mut cmd = Command::new("git");
        cmd.current_dir(cwd)
            .arg("-c")
            .arg("core.quotePath=false")
            .args(args)
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        for &(k, v) in env {
            cmd.env(k, v);
        }

        if stdin_bytes.is_some() {
            cmd.stdin(Stdio::piped());
        } else {
            cmd.stdin(Stdio::null());
        }

        let mut child = cmd.spawn()?;
        if let Some(input) = stdin_bytes {
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(input)?;
            }
        }

        let out = child.wait_with_output()?;
        let code = out.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
        if !allowed_codes.contains(&code) {
            return Err(GitError::CommandFailed {
                cmd: format!("git {}", args.join(" ")),
                code,
                stderr,
            });
        }
        Ok((code, out.stdout, stderr))
    }

    fn git_bytes(
        &self,
        cwd: &Path,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<Vec<u8>, GitError> {
        Ok(self.git_exec(cwd, args, env, None, &[0])?.1)
    }

    fn git_text(
        &self,
        cwd: &Path,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<String, GitError> {
        let bytes = self.git_bytes(cwd, args, env)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
    }

    fn git_ok(&self, cwd: &Path, args: &[&str]) -> bool {
        self.git_exec(cwd, args, &[], None, &[0]).is_ok()
    }

    fn resolve_commit(&self, rev: &str) -> Result<String, GitError> {
        let name = if rev.trim().is_empty() {
            "HEAD"
        } else {
            rev.trim()
        };
        let spec = format!("{name}^{{commit}}");
        if !self.git_ok(&self.start_dir, &["rev-parse", "--verify", "--quiet", &spec]) {
            return Err(GitError::InvalidTarget(format!("unknown revision: {name}")));
        }
        self.git_text(&self.start_dir, &["rev-parse", "--verify", &spec], &[])
    }

    fn resolve_target_sync(&self, spec: &TargetSpec) -> Result<ResolvedTarget, GitError> {
        let raw_spec = if spec.spec.trim().is_empty() {
            "HEAD"
        } else {
            spec.spec.trim()
        };
        let base_override = spec
            .base_override
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());

        if base_override.is_some() && raw_spec.contains("..") {
            return Err(GitError::InvalidTarget(format!(
                "target must be a single revision when --base is set, got range: {raw_spec}"
            )));
        }
        if raw_spec.split("...").count() > 2
            || (!raw_spec.contains("...") && raw_spec.split("..").count() > 2)
        {
            return Err(GitError::InvalidTarget(format!(
                "not a valid revision or range: {raw_spec}"
            )));
        }

        let (base_spec, tip_spec, symmetric) = if let Some(b) = base_override {
            (Some(b), raw_spec, false)
        } else if let Some((left, right)) = raw_spec.split_once("...") {
            (Some(left), right, true)
        } else if let Some((left, right)) = raw_spec.split_once("..") {
            (Some(left), right, false)
        } else {
            (None, raw_spec, false)
        };

        let tip = self.resolve_commit(tip_spec)?;
        let author_raw = self.git_text(
            &self.start_dir,
            &["log", "-1", "--format=%an%x00%ae%x00%aI", &tip],
            &[],
        )?;
        let author_parts: Vec<&str> = author_raw.split('\0').collect();
        let author_name = author_parts.first().copied().unwrap_or("Unknown").to_owned();
        let author_email = author_parts
            .get(1)
            .copied()
            .unwrap_or("unknown@localhost")
            .to_owned();
        let author_date = author_parts.get(2).copied().unwrap_or("").to_owned();

        if let Some(b_spec) = base_spec {
            let base_sha = self.resolve_commit(b_spec)?;
            let base = if symmetric {
                let (code, out, _) = self.git_exec(
                    &self.start_dir,
                    &["merge-base", &base_sha, &tip],
                    &[],
                    None,
                    &[0, 1],
                )?;
                if code != 0 {
                    return Err(GitError::Precondition(format!(
                        "no merge-base between '{}' and '{}'",
                        if b_spec.is_empty() { "HEAD" } else { b_spec },
                        if tip_spec.is_empty() { "HEAD" } else { tip_spec }
                    )));
                }
                String::from_utf8_lossy(&out).trim().to_owned()
            } else {
                base_sha
            };

            if base == tip {
                return Err(GitError::Precondition(format!(
                    "commit range '{raw_spec}' is empty (base and tip are both {})",
                    &tip[..10.min(tip.len())]
                )));
            }

            let base_is_ancestor =
                self.git_ok(&self.start_dir, &["merge-base", "--is-ancestor", &base, &tip]);
            let range_expr = format!("{base}..{tip}");
            let log_raw = self
                .git_text(
                    &self.start_dir,
                    &["log", "--reverse", "--format=%H%x00%s", &range_expr],
                    &[],
                )
                .unwrap_or_default();
            let commits: Vec<CommitSummary> = log_raw
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|line| {
                    let (sha, subj) = line.split_once('\0').unwrap_or((line, ""));
                    CommitSummary {
                        sha: sha.to_owned(),
                        subject: subj.to_owned(),
                    }
                })
                .collect();

            let message = self
                .git_text(
                    &self.start_dir,
                    &[
                        "log",
                        "--reverse",
                        "--format=- %h %s%n%w(0,4,4)%b",
                        &range_expr,
                    ],
                    &[],
                )
                .unwrap_or_default();

            let short_id = format!(
                "{}-{}",
                &base[..8.min(base.len())],
                &tip[..8.min(tip.len())]
            );
            let subject = format!(
                "Combined changes of {} commit(s) {}..{}",
                commits.len(),
                &base[..10.min(base.len())],
                &tip[..10.min(tip.len())]
            );

            Ok(ResolvedTarget {
                base,
                tip,
                short_id,
                is_range: true,
                symmetric,
                base_is_ancestor,
                is_merge: false,
                subject,
                message,
                author_name,
                author_email,
                author_date,
                commits,
                pathspecs: spec.pathspecs.clone(),
            })
        } else {
            let parent_spec = format!("{tip}^1");
            if !self.git_ok(
                &self.start_dir,
                &["rev-parse", "--verify", "--quiet", &parent_spec],
            ) {
                return Err(GitError::Precondition(format!(
                    "commit {tip} has no parent (root commit); nothing to untangle against"
                )));
            }
            let base = self.git_text(&self.start_dir, &["rev-parse", &parent_spec], &[])?;
            let merge_spec = format!("{tip}^2");
            let is_merge = self.git_ok(
                &self.start_dir,
                &["rev-parse", "--verify", "--quiet", &merge_spec],
            );
            let message =
                self.git_text(&self.start_dir, &["log", "-1", "--format=%B", &tip], &[])?;
            let subject = message.lines().next().unwrap_or("").to_owned();
            let short_id = tip[..10.min(tip.len())].to_owned();

            Ok(ResolvedTarget {
                base,
                tip,
                short_id,
                is_range: false,
                symmetric: false,
                base_is_ancestor: true,
                is_merge,
                subject,
                message,
                author_name,
                author_email,
                author_date,
                commits: Vec::new(),
                pathspecs: spec.pathspecs.clone(),
            })
        }
    }

    fn batch_read_blobs(&self, oids: &BTreeSet<String>) -> Result<HashMap<String, RawBytes>, GitError> {
        let mut map = HashMap::new();
        if oids.is_empty() {
            return Ok(map);
        }

        let mut child = Command::new("git")
            .current_dir(&self.start_dir)
            .args(["cat-file", "--batch"])
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        {
            let mut stdin = child.stdin.take().expect("piped stdin");
            for oid in oids {
                writeln!(stdin, "{oid}")?;
            }
        }

        let stdout = child.stdout.take().expect("piped stdout");
        let mut reader = BufReader::new(stdout);
        for _ in 0..oids.len() {
            let mut header = String::new();
            let read_hdr = reader.read_line(&mut header)?;
            if read_hdr == 0 {
                break;
            }
            let parts: Vec<&str> = header.split_whitespace().collect();
            if parts.len() < 3 || parts[1] == "missing" {
                return Err(GitError::Precondition(format!(
                    "git cat-file --batch missing object: {}",
                    header.trim()
                )));
            }
            let oid = parts[0].to_owned();
            let size: usize = parts[2].parse().map_err(|_| {
                GitError::Precondition(format!("invalid cat-file size header: {}", header.trim()))
            })?;
            let mut buf = vec![0u8; size];
            reader.read_exact(&mut buf)?;
            let mut nl = [0u8; 1];
            reader.read_exact(&mut nl)?;
            map.insert(oid, RawBytes(buf));
        }

        let status = child.wait()?;
        if !status.success() {
            return Err(GitError::CommandFailed {
                cmd: "git cat-file --batch".to_owned(),
                code: status.code().unwrap_or(-1),
                stderr: String::new(),
            });
        }
        Ok(map)
    }

    fn extract_deltas_sync(&self, target: &ResolvedTarget) -> Result<Vec<RawFileDelta>, GitError> {
        let mut args: Vec<&str> = vec![
            "diff",
            "--raw",
            "-z",
            "-M",
            "--no-abbrev",
            "--no-color",
            &target.base,
            &target.tip,
        ];
        if !target.pathspecs.is_empty() {
            args.push("--");
            for p in &target.pathspecs {
                args.push(p.as_str());
            }
        }

        let raw = self.git_bytes(&self.start_dir, &args, &[])?;
        let tokens: Vec<&[u8]> = raw.split(|&b| b == 0).collect();

        let mut entries = Vec::new();
        let mut i = 0usize;
        while i < tokens.len() {
            let meta = tokens[i];
            if meta.is_empty() || meta[0] != b':' {
                i += 1;
                continue;
            }
            let meta_str = String::from_utf8_lossy(&meta[1..]);
            let parts: Vec<&str> = meta_str.split_whitespace().collect();
            if parts.len() < 5 {
                i += 1;
                continue;
            }
            let old_mode = FileMode::from_octal(parts[0]);
            let new_mode = FileMode::from_octal(parts[1]);
            let old_oid = parts[2].to_owned();
            let new_oid = parts[3].to_owned();
            let status = parts[4].chars().next().unwrap_or('M');

            if matches!(status, 'R' | 'C') && i + 2 < tokens.len() {
                let p1 = String::from_utf8_lossy(tokens[i + 1]).into_owned();
                let p2 = String::from_utf8_lossy(tokens[i + 2]).into_owned();
                entries.push(RawDiffEntry {
                    old_mode,
                    new_mode,
                    old_oid,
                    new_oid,
                    status,
                    old_path: Some(p1),
                    new_path: Some(p2),
                });
                i += 3;
            } else if i + 1 < tokens.len() {
                let p = String::from_utf8_lossy(tokens[i + 1]).into_owned();
                let (old_path, new_path) = match status {
                    'A' => (None, Some(p)),
                    'D' => (Some(p), None),
                    _ => (Some(p.clone()), Some(p)),
                };
                entries.push(RawDiffEntry {
                    old_mode,
                    new_mode,
                    old_oid,
                    new_oid,
                    status,
                    old_path,
                    new_path,
                });
                i += 2;
            } else {
                i += 1;
            }
        }

        let is_zero_oid = |oid: &str| oid.chars().all(|c| c == '0');
        let mut wanted_oids = BTreeSet::new();
        for e in &entries {
            if !is_zero_oid(&e.old_oid) && e.old_mode != Some(FileMode::Gitlink) {
                wanted_oids.insert(e.old_oid.clone());
            }
            if !is_zero_oid(&e.new_oid) && e.new_mode != Some(FileMode::Gitlink) {
                wanted_oids.insert(e.new_oid.clone());
            }
        }

        let blobs = self.batch_read_blobs(&wanted_oids)?;
        let resolve_content = |oid: &str, mode: Option<FileMode>| -> Option<RawBytes> {
            if is_zero_oid(oid) {
                None
            } else if mode == Some(FileMode::Gitlink) {
                Some(RawBytes::from(oid))
            } else {
                blobs.get(oid).cloned()
            }
        };

        let mut deltas = Vec::with_capacity(entries.len());
        for e in entries {
            let old_content = resolve_content(&e.old_oid, e.old_mode);
            let new_content = resolve_content(&e.new_oid, e.new_mode);

            let needs_u0_diff = matches!(e.status, 'M' | 'R')
                && e.old_oid != e.new_oid
                && e.old_mode != Some(FileMode::Gitlink)
                && e.new_mode != Some(FileMode::Gitlink)
                && e.old_mode != Some(FileMode::Symlink)
                && e.new_mode != Some(FileMode::Symlink)
                && old_content.as_ref().is_some_and(|b| !b.is_binary())
                && new_content.as_ref().is_some_and(|b| !b.is_binary());

            let zero_context_diff = if needs_u0_diff {
                self.git_exec(
                    &self.start_dir,
                    &[
                        "diff",
                        "-U0",
                        "--no-color",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--text",
                        &e.old_oid,
                        &e.new_oid,
                    ],
                    &[],
                    None,
                    &[0, 1],
                )
                .ok()
                .map(|(_, bytes, _)| RawBytes(bytes))
            } else {
                None
            };

            deltas.push(RawFileDelta {
                status: e.status,
                old_path: e.old_path,
                new_path: e.new_path,
                old_mode: e.old_mode,
                new_mode: e.new_mode,
                old_content,
                new_content,
                zero_context_diff,
            });
        }

        Ok(deltas)
    }

    #[allow(clippy::too_many_arguments)]
    fn materialize_dag_sync(
        &self,
        target: &ResolvedTarget,
        graph: &ChangeGraph,
        order: &[Operation],
        files: &[FileChange],
        items_by_id: &IndexMap<String, ChangeItem>,
        options: &EmitOptions,
    ) -> Result<DagEmitResult, GitError> {
        let tr = graph.transitive_reduction(order);
        let ancestors = tr.ancestors;
        let reduced_parents = tr.reduced_parents;
        let sinks = tr.sinks;
        let mut result = DagEmitResult::default();
        result.metadata.operation_parents = reduced_parents.clone();

        // Populate inline previews for every item so `--dry-run` still produces rich viewer diffs.
        for item in items_by_id.values() {
            result
                .metadata
                .item_diff_previews
                .insert(item.id.clone(), build_inline_item_preview(item, &files[item.file_index]));
        }

        if options.dry_run {
            return Ok(result);
        }

        let temp_dir = TempDir::new("git-untangle-idx")?;
        let index_path = temp_dir.path().join("index");
        let index_str = index_path.to_string_lossy().into_owned();

        let has_user_identity = self.git_ok(&self.start_dir, &["config", "user.name"])
            && self.git_ok(&self.start_dir, &["config", "user.email"]);

        let mut base_env: Vec<(&str, &str)> = vec![
            ("GIT_INDEX_FILE", &index_str),
            ("GIT_AUTHOR_NAME", &target.author_name),
            ("GIT_AUTHOR_EMAIL", &target.author_email),
        ];
        if !target.author_date.is_empty() {
            base_env.push(("GIT_AUTHOR_DATE", &target.author_date));
        }
        if !has_user_identity {
            base_env.push(("GIT_COMMITTER_NAME", &target.author_name));
            base_env.push(("GIT_COMMITTER_EMAIL", &target.author_email));
            if !target.author_date.is_empty() {
                base_env.push(("GIT_COMMITTER_DATE", &target.author_date));
            }
        }

        let op_by_id: HashMap<&str, &Operation> =
            order.iter().map(|o| (o.id.as_str(), o)).collect();
        let items_of_ops = |op_ids: &BTreeSet<String>| -> BTreeSet<String> {
            let mut set = BTreeSet::new();
            for oid in op_ids {
                if let Some(op) = op_by_id.get(oid.as_str()) {
                    set.extend(op.item_ids.iter().cloned());
                }
            }
            set
        };
        let mut blob_oid_cache: HashMap<Vec<u8>, String> = HashMap::new();
        let mut current_index_commit = String::new();
        let reset_index_to = |commit_sha: &str, current: &mut String| -> Result<(), GitError> {
            if current != commit_sha {
                self.git_bytes(&self.start_dir, &["read-tree", commit_sha], &base_env)?;
                *current = commit_sha.to_owned();
            }
            Ok(())
        };

        let all_file_refs: Vec<&FileChange> = files.iter().collect();

        let stage_desired = |touched: &[&FileChange],
                             applied: &BTreeSet<String>,
                             cache: &mut HashMap<Vec<u8>, String>|
         -> Result<(), GitError> {
            let desired = desired_path_states(files, touched, items_by_id, applied)?;
            // Process deletions first so file<->directory transitions (`a` vs `a/b`) never collide in index.
            for (path, entry_opt) in &desired {
                if entry_opt.is_none() {
                    self.git_bytes(
                        &self.start_dir,
                        &["update-index", "--force-remove", "--", path],
                        &base_env,
                    )?;
                }
            }
            for (path, entry_opt) in &desired {
                if let Some(entry) = entry_opt {
                    let oid = if entry.mode == FileMode::Gitlink {
                        String::from_utf8_lossy(&entry.content).trim().to_owned()
                    } else if let Some(cached) = cache.get(entry.content.as_slice()) {
                        cached.clone()
                    } else {
                        let out = self.git_exec(
                            &self.start_dir,
                            &["hash-object", "-w", "--no-filters", "--stdin"],
                            &base_env,
                            Some(&entry.content),
                            &[0],
                        )?.1;
                        let sha = String::from_utf8_lossy(&out).trim().to_owned();
                        cache.insert(entry.content.0.clone(), sha.clone());
                        sha
                    };
                    let cacheinfo = format!("{},{oid},{path}", entry.mode.as_octal());
                    self.git_bytes(
                        &self.start_dir,
                        &["update-index", "--add", "--cacheinfo", &cacheinfo],
                        &base_env,
                    )?;
                }
            }
            Ok(())
        };

        let commit_current_index =
            |parents: &[String], message: &str, current: &mut String| -> Result<String, GitError> {
                let tree = self.git_text(&self.start_dir, &["write-tree"], &base_env)?;
                let mut args: Vec<&str> = vec!["commit-tree", &tree];
                for p in parents {
                    args.push("-p");
                    args.push(p.as_str());
                }
                let out = self
                    .git_exec(
                        &self.start_dir,
                        &args,
                        &base_env,
                        Some(message.as_bytes()),
                        &[0],
                    )?
                    .1;
                let sha = String::from_utf8_lossy(&out).trim().to_owned();
                *current = sha.clone();
                Ok(sha)
            };

        reset_index_to(&target.base, &mut current_index_commit)?;

        let total_items: usize = order.iter().map(|o| o.item_ids.len()).sum();
        let mut step = 0usize;

        for op in order {
            let reduced = reduced_parents
                .get(&op.id)
                .cloned()
                .unwrap_or_default();
            let anc = ancestors.get(&op.id).cloned().unwrap_or_default();
            let base_state = items_of_ops(&anc);

            let op_base_sha = match reduced.len() {
                0 => target.base.clone(),
                1 => result.metadata.operation_commits[&reduced[0]].clone(),
                _ => {
                    let mut parent_shas = Vec::new();
                    for pid in &reduced {
                        if let Some(sha) = result.metadata.operation_commits.get(pid) {
                            if !parent_shas.contains(sha) {
                                parent_shas.push(sha.clone());
                            }
                        }
                    }
                    if parent_shas.len() == 1 {
                        parent_shas[0].clone()
                    } else {
                        reset_index_to(&parent_shas[0], &mut current_index_commit)?;
                        stage_desired(&all_file_refs, &base_state, &mut blob_oid_cache)?;
                        let msg =
                            format_merge_message(op, &reduced, &op_by_id, &target.short_id);
                        let merge_sha = commit_current_index(
                            &parent_shas,
                            &msg,
                            &mut current_index_commit,
                        )?;
                        result.metadata.merge_commits.push(merge_sha.clone());
                        debug!("created prerequisite merge {merge_sha} for {}", op.id);
                        merge_sha
                    }
                }
            };

            reset_index_to(&op_base_sha, &mut current_index_commit)?;
            result
                .metadata
                .operation_bases
                .insert(op.id.clone(), op_base_sha.clone());

            let mut applied = base_state;
            let mut sorted_items: Vec<&ChangeItem> = op
                .item_ids
                .iter()
                .filter_map(|id| items_by_id.get(id))
                .collect();
            sorted_items.sort_by(|a, b| {
                a.file_index
                    .cmp(&b.file_index)
                    .then_with(|| a.line().cmp(&b.line()))
                    .then_with(|| a.id.cmp(&b.id))
            });

            let mut prev_sha = op_base_sha.clone();
            for item in sorted_items {
                step += 1;
                applied.insert(item.id.clone());
                let touched = [&files[item.file_index]];
                stage_desired(&touched, &applied, &mut blob_oid_cache)?;
                let msg = format_item_message(op, item, step, total_items, &target.short_id);
                let item_sha = commit_current_index(
                    std::slice::from_ref(&prev_sha),
                    &msg,
                    &mut current_index_commit,
                )?;

                if options.emit_item_diffs || options.emit_patches {
                    let patch_bytes = self
                        .git_exec(
                            &self.start_dir,
                            &["diff", "--binary", "-M", "--no-color", &prev_sha, &item_sha],
                            &[],
                            None,
                            &[0, 1],
                        )?
                        .1;
                    let preview = String::from_utf8_lossy(&patch_bytes);
                    let clipped: String = preview.chars().take(12_000).collect();
                    if !clipped.is_empty() {
                        result
                            .metadata
                            .item_diff_previews
                            .insert(item.id.clone(), clipped);
                    }
                    result
                        .item_patches
                        .insert(item.id.clone(), RawBytes(patch_bytes));
                }

                result
                    .metadata
                    .item_commits
                    .insert(item.id.clone(), item_sha.clone());
                prev_sha = item_sha;
            }

            result
                .metadata
                .operation_commits
                .insert(op.id.clone(), prev_sha.clone());

            if options.emit_patches {
                let op_patch = self
                    .git_exec(
                        &self.start_dir,
                        &[
                            "diff",
                            "--binary",
                            "-M",
                            "--no-color",
                            &op_base_sha,
                            &prev_sha,
                        ],
                        &[],
                        None,
                        &[0, 1],
                    )?
                    .1;
                result
                    .operation_patches
                    .push((op.name.clone(), RawBytes(op_patch)));
            }
        }

        // Join all sink operations
        let head_sha = match sinks.len() {
            0 => target.base.clone(),
            1 => result.metadata.operation_commits[&sinks[0]].clone(),
            _ => {
                let mut parent_shas = Vec::new();
                for sid in &sinks {
                    if let Some(sha) = result.metadata.operation_commits.get(sid) {
                        if !parent_shas.contains(sha) {
                            parent_shas.push(sha.clone());
                        }
                    }
                }
                if parent_shas.len() == 1 {
                    parent_shas[0].clone()
                } else {
                    let all_items: BTreeSet<String> = items_by_id.keys().cloned().collect();
                    reset_index_to(&parent_shas[0], &mut current_index_commit)?;
                    stage_desired(&all_file_refs, &all_items, &mut blob_oid_cache)?;
                    let msg = format_final_merge_message(&sinks, &target.short_id, &target.subject);
                    let merge_sha =
                        commit_current_index(&parent_shas, &msg, &mut current_index_commit)?;
                    result.metadata.merge_commits.push(merge_sha.clone());
                    merge_sha
                }
            }
        };

        result.metadata.head_sha = Some(head_sha.clone());

        // Verify Git tree equivalence (Invariant I5)
        let tree_matches = if target.pathspecs.is_empty() {
            let head_tree =
                self.git_text(&self.start_dir, &["rev-parse", &format!("{head_sha}^{{tree}}")], &[])?;
            let tip_tree = self.git_text(
                &self.start_dir,
                &["rev-parse", &format!("{}^{{tree}}", target.tip)],
                &[],
            )?;
            head_tree == tip_tree
        } else {
            let mut diff_args: Vec<&str> = vec!["diff", "--quiet", &head_sha, &target.tip, "--"];
            for p in &target.pathspecs {
                diff_args.push(p.as_str());
            }
            self.git_ok(&self.start_dir, &diff_args)
        };
        result.metadata.tree_matches = Some(tree_matches);

        if !tree_matches {
            let mut stat_args: Vec<&str> = vec!["diff", "--stat", "--summary", &head_sha, &target.tip];
            if !target.pathspecs.is_empty() {
                stat_args.push("--");
                for p in &target.pathspecs {
                    stat_args.push(p.as_str());
                }
            }
            if let Ok((_, bytes, _)) =
                self.git_exec(&self.start_dir, &stat_args, &[], None, &[0, 1])
            {
                result.metadata.tree_diff = String::from_utf8_lossy(&bytes)
                    .chars()
                    .take(4000)
                    .collect();
            }
        }

        if options.emit_branch && tree_matches {
            let mut branch = options
                .branch_name
                .clone()
                .unwrap_or_else(|| format!("{}{}", options.branch_prefix, target.short_id));
            if options.branch_name.is_none()
                && self.git_ok(
                    &self.start_dir,
                    &["show-ref", "--verify", "--quiet", &format!("refs/heads/{branch}")],
                )
            {
                let ts = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                branch = format!("{branch}-{ts}");
            }

            match self.git_exec(
                &self.start_dir,
                &["branch", &branch, &head_sha],
                &[],
                None,
                &[0],
            ) {
                Ok(_) => result.metadata.branch_created = Some(branch),
                Err(e) => {
                    result.metadata.branch_error =
                        Some(format!("could not create branch '{branch}': {e}"));
                }
            }
        }

        Ok(result)
    }
}

#[async_trait]
impl GitBackend for GitCliBackend {
    async fn repo_info(&self) -> Result<RepoInfo, GitError> {
        let dir = self.start_dir.clone();
        tokio::task::spawn_blocking(move || Ok(GitCliBackend::discover(dir)?.1))
            .await
            .map_err(|e| GitError::Io(std::io::Error::other(e.to_string())))?
    }

    async fn resolve_target(&self, spec: &TargetSpec) -> Result<ResolvedTarget, GitError> {
        let this = self.clone();
        let spec = spec.clone();
        tokio::task::spawn_blocking(move || this.resolve_target_sync(&spec))
            .await
            .map_err(|e| GitError::Io(std::io::Error::other(e.to_string())))?
    }

    async fn extract_deltas(&self, target: &ResolvedTarget) -> Result<Vec<RawFileDelta>, GitError> {
        let this = self.clone();
        let target = target.clone();
        tokio::task::spawn_blocking(move || this.extract_deltas_sync(&target))
            .await
            .map_err(|e| GitError::Io(std::io::Error::other(e.to_string())))?
    }

    async fn materialize_dag(
        &self,
        target: &ResolvedTarget,
        graph: &ChangeGraph,
        order: &[Operation],
        files: &[FileChange],
        items_by_id: &IndexMap<String, ChangeItem>,
        options: &EmitOptions,
    ) -> Result<DagEmitResult, GitError> {
        let this = self.clone();
        let target = target.clone();
        let graph = graph.clone();
        let order = order.to_vec();
        let files = files.to_vec();
        let items_by_id = items_by_id.clone();
        let options = options.clone();
        tokio::task::spawn_blocking(move || {
            this.materialize_dag_sync(&target, &graph, &order, &files, &items_by_id, &options)
        })
        .await
        .map_err(|e| GitError::Io(std::io::Error::other(e.to_string())))?
    }
}

fn first_line_clipped(s: &str, limit: usize) -> String {
    s.lines().next().unwrap_or("").chars().take(limit).collect()
}

fn format_merge_message(
    op: &Operation,
    parents: &[String],
    op_by_id: &HashMap<&str, &Operation>,
    short_sha: &str,
) -> String {
    let mut out = String::new();
    let header = format!("Merge prerequisites of {}: {}", op.id, op.name);
    out.push_str(&first_line_clipped(&header, 100));
    out.push_str("\n\n");
    for p in parents {
        let name = op_by_id
            .get(p.as_str())
            .map_or(p.as_str(), |o| o.name.as_str());
        out.push_str(&format!("- {p}: {}\n", first_line_clipped(name, 100)));
    }
    out.push('\n');
    out.push_str(&format!("Commit-Analyzer-Source: {short_sha}\n"));
    out.push_str(&format!("Commit-Analyzer-Merge-For: {}\n", op.id));
    out
}

fn format_final_merge_message(sinks: &[String], short_sha: &str, subject: &str) -> String {
    let mut out = String::new();
    let header = format!("Merge exploded commit {short_sha}: {subject}");
    out.push_str(&first_line_clipped(&header, 100));
    out.push_str("\n\n");
    out.push_str(&format!(
        "Joins the independent operation branches {};\n",
        sinks.join(", ")
    ));
    out.push_str("the resulting tree is identical to the original commit.\n\n");
    out.push_str(&format!("Commit-Analyzer-Source: {short_sha}\n"));
    out
}

fn format_item_message(
    op: &Operation,
    item: &ChangeItem,
    step: usize,
    total: usize,
    short_sha: &str,
) -> String {
    let mut out = String::new();
    let header = format!(
        "{}: {} {}",
        op.name,
        item.kind.as_str().to_lowercase(),
        item.display_path
    );
    out.push_str(&first_line_clipped(&header, 100));
    out.push_str("\n\n");
    out.push_str(op.description.trim());
    out.push('\n');
    if let Some(ref rule) = op.rule {
        let r: String = rule.chars().take(500).collect();
        out.push_str(&format!("\nRule: {r}\n"));
    }
    out.push('\n');
    out.push_str(&format!("Commit-Analyzer-Source: {short_sha}\n"));
    out.push_str(&format!("Commit-Analyzer-Operation: {}\n", op.id));
    out.push_str(&format!("Commit-Analyzer-Item: {}\n", item.id));
    out.push_str(&format!("Commit-Analyzer-Step: {step}/{total}\n"));
    out
}

fn build_inline_item_preview(item: &ChangeItem, file: &FileChange) -> String {
    if let Some(ref h) = item.hunk {
        let mut s = format!(
            "--- a/{}\n+++ b/{}\n@@ -{},{} +{},{} @@\n",
            file.old_path.as_deref().unwrap_or(&item.display_path),
            item.display_path,
            if h.old_count == 0 { h.start } else { h.start + 1 },
            h.old_count,
            h.new_start,
            h.new_lines.len()
        );
        for ol in h.old_lines.iter().take(60) {
            s.push('-');
            s.push_str(&ol.to_display());
            if !ol.ends_with(b"\n") {
                s.push('\n');
            }
        }
        for nl in h.new_lines.iter().take(60) {
            s.push('+');
            s.push_str(&nl.to_display());
            if !nl.ends_with(b"\n") {
                s.push('\n');
            }
        }
        s
    } else {
        match item.kind {
            commit_untangle_core::model::ItemKind::Move => format!(
                "rename from {}\nrename to {}\n",
                file.old_path.as_deref().unwrap_or("?"),
                file.new_path.as_deref().unwrap_or("?")
            ),
            commit_untangle_core::model::ItemKind::ModeChange => format!(
                "old mode {}\nnew mode {}\n",
                file.old_mode.map_or("?", |m| m.as_octal()),
                file.new_mode.map_or("?", |m| m.as_octal())
            ),
            commit_untangle_core::model::ItemKind::AddFile => {
                let mut s = format!("new file {}\n", item.display_path);
                for line in item.added_bytes.to_display().lines().take(60) {
                    s.push('+');
                    s.push_str(line);
                    s.push('\n');
                }
                s
            }
            commit_untangle_core::model::ItemKind::DeleteFile => {
                let mut s = format!("deleted file {}\n", item.display_path);
                for line in item.removed_bytes.to_display().lines().take(60) {
                    s.push('-');
                    s.push_str(line);
                    s.push('\n');
                }
                s
            }
            _ => format!("{} {}\n", item.kind.as_str(), item.display_path),
        }
    }
}
