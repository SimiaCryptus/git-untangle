use std::path::{Path, PathBuf};
use std::process::Command;

use commit_untangle_llm::ToolDefinition;
use serde_json::{json, Value};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader, DuplexStream};

use crate::chat::truncate_utf8_bytes;

pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";
pub const MCP_SERVER_NAME: &str = "git-chat-log-mcp";
const DEFAULT_TOOL_MAX_BYTES: usize = 16_384;
const MAX_TOOL_MAX_BYTES: usize = 65_536;
const MIN_TOOL_MAX_BYTES: usize = 512;

#[derive(Debug, Clone)]
pub struct McpToolCallResult {
    pub text: String,
    pub is_error: bool,
}

/// Returns the MCP tool specifications exposed by the Git revision-history server.
pub fn mcp_tool_schemas() -> Vec<Value> {
    vec![
        json!({
            "name": "git_log",
            "description": "Navigate and paginate Git commit history with optional revision range, author, date, path, and message filters.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rev_range": { "type": "string", "description": "Revision or range (e.g. 'HEAD', 'main..feature', 'v1.0..HEAD')." },
                    "limit": { "type": "integer", "description": "Maximum number of commits to return (default 20, max 200)." },
                    "skip": { "type": "integer", "description": "Number of commits to skip for pagination (default 0)." },
                    "author": { "type": "string", "description": "Filter commits by author name or email pattern." },
                    "since": { "type": "string", "description": "Show commits more recent than a specific date (e.g. '2 weeks ago', '2025-01-01')." },
                    "until": { "type": "string", "description": "Show commits older than a specific date." },
                    "path": { "type": "string", "description": "Restrict history to a specific file or directory path." },
                    "grep": { "type": "string", "description": "Filter commits whose message matches this pattern." },
                    "first_parent": { "type": "boolean", "description": "Follow only the first parent commit upon seeing a merge commit." },
                    "no_merges": { "type": "boolean", "description": "Exclude merge commits." },
                    "merges_only": { "type": "boolean", "description": "Include only merge commits." },
                    "stat": { "type": "boolean", "description": "Include file diffstat for each commit." }
                }
            }
        }),
        json!({
            "name": "git_show_commit",
            "description": "Inspect a specific commit's full metadata, commit message, file stats, and optionally its unified diff.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rev": { "type": "string", "description": "Commit hash, branch, tag, or revision expression (e.g. 'HEAD', 'HEAD~2', 'abc1234')." },
                    "stat_only": { "type": "boolean", "description": "If true, show only commit metadata and --stat without patch hunks." },
                    "path": { "type": "string", "description": "Restrict diff output to a specific file or directory path." },
                    "max_bytes": { "type": "integer", "description": "Maximum output bytes (default 16384, max 65536)." }
                },
                "required": ["rev"]
            }
        }),
        json!({
            "name": "git_search_commits",
            "description": "Search Git revision history by commit message (--grep), added/removed literal code string (-S pickaxe), or added/removed code regex (-G pickaxe).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query string or regex." },
                    "mode": {
                        "type": "string",
                        "enum": ["message", "pickaxe_string", "pickaxe_regex"],
                        "description": "Search mode: 'message' (--grep), 'pickaxe_string' (-S), or 'pickaxe_regex' (-G). Default: 'message'."
                    },
                    "rev_range": { "type": "string", "description": "Optional revision or range to search within." },
                    "path": { "type": "string", "description": "Optional file or directory path filter." },
                    "limit": { "type": "integer", "description": "Maximum commits to return (default 20, max 100)." },
                    "ignore_case": { "type": "boolean", "description": "Perform case-insensitive matching (default true)." }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "git_diff_revisions",
            "description": "Compare two Git revisions (or a revision against HEAD) with full patch, --stat, or --name-status.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "from_rev": { "type": "string", "description": "Base revision (e.g. 'HEAD~3', 'main', commit hash)." },
                    "to_rev": { "type": "string", "description": "Target revision (default 'HEAD')." },
                    "path": { "type": "string", "description": "Optional file or directory path filter." },
                    "stat_only": { "type": "boolean", "description": "If true, return --stat --name-status summary only." },
                    "max_bytes": { "type": "integer", "description": "Maximum output bytes (default 16384, max 65536)." }
                },
                "required": ["from_rev"]
            }
        }),
        json!({
            "name": "git_file_history",
            "description": "Follow the revision history of a specific file across renames (`git log --follow`).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Repository-relative file path." },
                    "rev": { "type": "string", "description": "Starting revision (default 'HEAD')." },
                    "limit": { "type": "integer", "description": "Maximum commits to return (default 20, max 100)." },
                    "skip": { "type": "integer", "description": "Number of commits to skip for pagination (default 0)." },
                    "with_patch": { "type": "boolean", "description": "Include patch diffs for each commit (default false)." },
                    "max_bytes": { "type": "integer", "description": "Maximum output bytes (default 16384, max 65536)." }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "git_blame",
            "description": "Show line-by-line commit attribution (`git blame`) for a file at a given revision and optional line range.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Repository-relative file path." },
                    "rev": { "type": "string", "description": "Revision to blame at (default 'HEAD')." },
                    "start_line": { "type": "integer", "description": "1-based start line number (optional)." },
                    "end_line": { "type": "integer", "description": "1-based end line number (optional)." },
                    "max_bytes": { "type": "integer", "description": "Maximum output bytes (default 16384, max 65536)." }
                },
                "required": ["path"]
            }
        }),
        json!({
            "name": "git_list_refs",
            "description": "List branches, tags, and current HEAD to orient revision history navigation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["all", "branches", "tags"],
                        "description": "Which refs to list: 'all', 'branches', or 'tags' (default 'all')."
                    },
                    "pattern": { "type": "string", "description": "Optional glob pattern to filter ref names (e.g. 'v*')." },
                    "limit": { "type": "integer", "description": "Maximum refs to return (default 50, max 200)." }
                }
            }
        }),
        json!({
            "name": "git_show_file",
            "description": "Read the contents of a file at a specific revision (`git show <rev>:<path>`), optionally sliced by line range.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rev": { "type": "string", "description": "Revision (e.g. 'HEAD', 'HEAD~2', branch, or commit hash)." },
                    "path": { "type": "string", "description": "Repository-relative file path." },
                    "start_line": { "type": "integer", "description": "1-based start line number (optional)." },
                    "end_line": { "type": "integer", "description": "1-based end line number (optional)." },
                    "max_bytes": { "type": "integer", "description": "Maximum output bytes (default 16384, max 65536)." }
                },
                "required": ["rev", "path"]
            }
        }),
        json!({
            "name": "git_commit_ancestry",
            "description": "Inspect commit parents, branches/tags containing a commit, and optional merge-base / ahead-behind counts against another revision.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rev": { "type": "string", "description": "Target revision." },
                    "other_rev": { "type": "string", "description": "Optional second revision to compute merge-base and left/right commit counts." }
                },
                "required": ["rev"]
            }
        }),
    ]
}

fn validate_non_flag(label: &str, val: &str) -> Result<String, String> {
    let trimmed = val.trim();
    if trimmed.is_empty() {
        return Err(format!("`{label}` must not be empty"));
    }
    if trimmed.starts_with('-') {
        return Err(format!("`{label}` must not start with '-'"));
    }
    Ok(trimmed.to_owned())
}

fn clamp_max_bytes(args: &Value) -> usize {
    args.get("max_bytes")
        .and_then(Value::as_u64)
        .map(|n| (n as usize).clamp(MIN_TOOL_MAX_BYTES, MAX_TOOL_MAX_BYTES))
        .unwrap_or(DEFAULT_TOOL_MAX_BYTES)
}

fn run_git_readonly(repo_dir: &Path, args: &[String]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(repo_dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .output()
        .map_err(|e| format!("failed to execute git: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    if !output.status.success() {
        let detail = if !stderr.trim().is_empty() {
            stderr.trim().to_owned()
        } else if !stdout.trim().is_empty() {
            stdout.trim().to_owned()
        } else {
            format!("git exited with status {}", output.status)
        };
        return Err(detail);
    }
    Ok(stdout)
}

fn cap_output(raw: &str, max_bytes: usize) -> String {
    let (prefix, truncated) = truncate_utf8_bytes(raw, max_bytes);
    if truncated {
        format!(
            "{prefix}\n[... truncated at {max_bytes} bytes; total {} bytes ...]",
            raw.len()
        )
    } else {
        raw.to_owned()
    }
}

/// Executes a single MCP Git revision-history tool synchronously.
pub fn execute_git_tool(repo_dir: &Path, name: &str, args: &Value) -> McpToolCallResult {
    let res = match name {
        "git_log" => tool_git_log(repo_dir, args),
        "git_show_commit" => tool_git_show_commit(repo_dir, args),
        "git_search_commits" => tool_git_search_commits(repo_dir, args),
        "git_diff_revisions" => tool_git_diff_revisions(repo_dir, args),
        "git_file_history" => tool_git_file_history(repo_dir, args),
        "git_blame" => tool_git_blame(repo_dir, args),
        "git_list_refs" => tool_git_list_refs(repo_dir, args),
        "git_show_file" => tool_git_show_file(repo_dir, args),
        "git_commit_ancestry" => tool_git_commit_ancestry(repo_dir, args),
        other => Err(format!("unknown MCP tool `{other}`")),
    };

    match res {
        Ok(text) => McpToolCallResult {
            text,
            is_error: false,
        },
        Err(err) => McpToolCallResult {
            text: format!("error: {err}"),
            is_error: true,
        },
    }
}

fn tool_git_log(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(1, 200);
    let skip = args.get("skip").and_then(Value::as_u64).unwrap_or(0);
    let max_bytes = clamp_max_bytes(args);

    let mut cmd = vec![
        "log".to_owned(),
        format!("--max-count={limit}"),
        format!("--skip={skip}"),
        "--date=short".to_owned(),
        "--format=%H | %ad | %an | %D | %s".to_owned(),
    ];

    if let Some(author) = args.get("author").and_then(Value::as_str) {
        if !author.trim().is_empty() {
            cmd.push(format!("--author={}", author.trim()));
        }
    }
    if let Some(since) = args.get("since").and_then(Value::as_str) {
        if !since.trim().is_empty() {
            cmd.push(format!("--since={}", since.trim()));
        }
    }
    if let Some(until) = args.get("until").and_then(Value::as_str) {
        if !until.trim().is_empty() {
            cmd.push(format!("--until={}", until.trim()));
        }
    }
    if let Some(grep) = args.get("grep").and_then(Value::as_str) {
        if !grep.trim().is_empty() {
            cmd.push("--regexp-ignore-case".to_owned());
            cmd.push(format!("--grep={}", grep.trim()));
        }
    }
    if args.get("first_parent").and_then(Value::as_bool) == Some(true) {
        cmd.push("--first-parent".to_owned());
    }
    if args.get("no_merges").and_then(Value::as_bool) == Some(true) {
        cmd.push("--no-merges".to_owned());
    }
    if args.get("merges_only").and_then(Value::as_bool) == Some(true) {
        cmd.push("--merges".to_owned());
    }
    if args.get("stat").and_then(Value::as_bool) == Some(true) {
        cmd.push("--stat".to_owned());
    }
    if let Some(rev) = args.get("rev_range").and_then(Value::as_str) {
        if !rev.trim().is_empty() {
            cmd.push(validate_non_flag("rev_range", rev)?);
        }
    }
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        if !path.trim().is_empty() {
            cmd.push("--".to_owned());
            cmd.push(validate_non_flag("path", path)?);
        }
    }

    let out = run_git_readonly(repo_dir, &cmd)?;
    if out.trim().is_empty() {
        Ok("(no matching commits)".to_owned())
    } else {
        Ok(cap_output(&out, max_bytes))
    }
}

fn tool_git_show_commit(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let rev = args
        .get("rev")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `rev`".to_owned())?;
    let rev = validate_non_flag("rev", rev)?;
    let stat_only = args.get("stat_only").and_then(Value::as_bool).unwrap_or(false);
    let max_bytes = clamp_max_bytes(args);

    let mut cmd = vec!["show".to_owned(), "--stat".to_owned()];
    if stat_only {
        cmd.push("--no-patch".to_owned());
    } else {
        cmd.push("--patch".to_owned());
    }
    cmd.push(rev);

    if let Some(path) = args.get("path").and_then(Value::as_str) {
        if !path.trim().is_empty() {
            cmd.push("--".to_owned());
            cmd.push(validate_non_flag("path", path)?);
        }
    }

    let out = run_git_readonly(repo_dir, &cmd)?;
    Ok(cap_output(&out, max_bytes))
}

fn tool_git_search_commits(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let query = args
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "missing required parameter `query`".to_owned())?;
    let mode = args
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("message");
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(1, 100);
    let ignore_case = args
        .get("ignore_case")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let max_bytes = clamp_max_bytes(args);

    let mut cmd = vec![
        "log".to_owned(),
        format!("--max-count={limit}"),
        "--date=short".to_owned(),
        "--format=%H | %ad | %an | %s".to_owned(),
        "--stat".to_owned(),
    ];
    if ignore_case {
        cmd.push("--regexp-ignore-case".to_owned());
    }
    match mode {
        "message" => {
            cmd.push(format!("--grep={query}"));
        }
        "pickaxe_string" => {
            cmd.push(format!("-S{query}"));
        }
        "pickaxe_regex" => {
            cmd.push(format!("-G{query}"));
        }
        other => {
            return Err(format!(
                "invalid search mode `{other}` (expected 'message', 'pickaxe_string', or 'pickaxe_regex')"
            ));
        }
    }
    if let Some(rev) = args.get("rev_range").and_then(Value::as_str) {
        if !rev.trim().is_empty() {
            cmd.push(validate_non_flag("rev_range", rev)?);
        }
    }
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        if !path.trim().is_empty() {
            cmd.push("--".to_owned());
            cmd.push(validate_non_flag("path", path)?);
        }
    }

    let out = run_git_readonly(repo_dir, &cmd)?;
    if out.trim().is_empty() {
        Ok(format!("(no commits matched {mode} query `{query}`)"))
    } else {
        Ok(cap_output(&out, max_bytes))
    }
}

fn tool_git_diff_revisions(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let from_rev = args
        .get("from_rev")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `from_rev`".to_owned())?;
    let from_rev = validate_non_flag("from_rev", from_rev)?;
    let to_rev = args
        .get("to_rev")
        .and_then(Value::as_str)
        .unwrap_or("HEAD");
    let to_rev = validate_non_flag("to_rev", to_rev)?;
    let stat_only = args.get("stat_only").and_then(Value::as_bool).unwrap_or(false);
    let max_bytes = clamp_max_bytes(args);

    let mut cmd = vec!["diff".to_owned()];
    if stat_only {
        cmd.push("--stat".to_owned());
        cmd.push("--name-status".to_owned());
    }
    cmd.push(from_rev);
    cmd.push(to_rev);
    if let Some(path) = args.get("path").and_then(Value::as_str) {
        if !path.trim().is_empty() {
            cmd.push("--".to_owned());
            cmd.push(validate_non_flag("path", path)?);
        }
    }

    let out = run_git_readonly(repo_dir, &cmd)?;
    if out.trim().is_empty() {
        Ok("(no differences)".to_owned())
    } else {
        Ok(cap_output(&out, max_bytes))
    }
}

fn tool_git_file_history(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `path`".to_owned())?;
    let path = validate_non_flag("path", path)?;
    let rev = args.get("rev").and_then(Value::as_str).unwrap_or("HEAD");
    let rev = validate_non_flag("rev", rev)?;
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(20)
        .clamp(1, 100);
    let skip = args.get("skip").and_then(Value::as_u64).unwrap_or(0);
    let with_patch = args
        .get("with_patch")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let max_bytes = clamp_max_bytes(args);

    let mut cmd = vec![
        "log".to_owned(),
        "--follow".to_owned(),
        format!("--max-count={limit}"),
        format!("--skip={skip}"),
        "--date=short".to_owned(),
        "--format=%H | %ad | %an | %s".to_owned(),
    ];
    if with_patch {
        cmd.push("-p".to_owned());
    } else {
        cmd.push("--name-status".to_owned());
    }
    cmd.push(rev);
    cmd.push("--".to_owned());
    cmd.push(path);

    let out = run_git_readonly(repo_dir, &cmd)?;
    if out.trim().is_empty() {
        Ok("(no history found for path)".to_owned())
    } else {
        Ok(cap_output(&out, max_bytes))
    }
}

fn tool_git_blame(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `path`".to_owned())?;
    let path = validate_non_flag("path", path)?;
    let rev = args.get("rev").and_then(Value::as_str).unwrap_or("HEAD");
    let rev = validate_non_flag("rev", rev)?;
    let max_bytes = clamp_max_bytes(args);

    let mut cmd = vec![
        "blame".to_owned(),
        "--date=short".to_owned(),
        "-w".to_owned(),
    ];
    if let (Some(start), Some(end)) = (
        args.get("start_line").and_then(Value::as_u64),
        args.get("end_line").and_then(Value::as_u64),
    ) {
        if start == 0 || end < start {
            return Err("invalid line range (require 1 <= start_line <= end_line)".to_owned());
        }
        cmd.push(format!("-L{start},{end}"));
    } else if let Some(start) = args.get("start_line").and_then(Value::as_u64) {
        if start == 0 {
            return Err("start_line must be >= 1".to_owned());
        }
        cmd.push(format!("-L{start},+100"));
    }
    cmd.push(rev);
    cmd.push("--".to_owned());
    cmd.push(path);

    let out = run_git_readonly(repo_dir, &cmd)?;
    Ok(cap_output(&out, max_bytes))
}

fn tool_git_list_refs(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let kind = args.get("kind").and_then(Value::as_str).unwrap_or("all");
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(50)
        .clamp(1, 200);

    let mut out = String::new();
    if let Ok(head) = run_git_readonly(
        repo_dir,
        &[
            "rev-parse".to_owned(),
            "--abbrev-ref".to_owned(),
            "HEAD".to_owned(),
        ],
    ) {
        let sha = run_git_readonly(
            repo_dir,
            &["rev-parse".to_owned(), "--short".to_owned(), "HEAD".to_owned()],
        )
        .unwrap_or_default();
        out.push_str(&format!("HEAD: {} ({})\n", head.trim(), sha.trim()));
    }

    let mut cmd = vec![
        "for-each-ref".to_owned(),
        "--sort=-committerdate".to_owned(),
        format!("--count={limit}"),
        "--format=%(refname:short) | %(objectname:short) | %(committerdate:short) | %(subject)"
            .to_owned(),
    ];
    let prefix = match kind {
        "branches" => Some("refs/heads/"),
        "tags" => Some("refs/tags/"),
        "all" => None,
        other => {
            return Err(format!(
                "invalid kind `{other}` (expected 'all', 'branches', or 'tags')"
            ))
        }
    };
    if let Some(pat) = args.get("pattern").and_then(Value::as_str) {
        let pat = pat.trim();
        if !pat.is_empty() {
            if pat.starts_with('-') {
                return Err("`pattern` must not start with '-'".to_owned());
            }
            if let Some(pref) = prefix {
                cmd.push(format!("{pref}{pat}"));
            } else {
                cmd.push(format!("refs/heads/{pat}"));
                cmd.push(format!("refs/tags/{pat}"));
                cmd.push(format!("refs/remotes/{pat}"));
            }
        } else if let Some(pref) = prefix {
            cmd.push(pref.to_owned());
        }
    } else if let Some(pref) = prefix {
        cmd.push(pref.to_owned());
    }

    let refs = run_git_readonly(repo_dir, &cmd)?;
    if refs.trim().is_empty() {
        out.push_str("(no matching refs)\n");
    } else {
        out.push_str(&refs);
    }
    Ok(cap_output(&out, DEFAULT_TOOL_MAX_BYTES))
}

fn tool_git_show_file(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let rev = args
        .get("rev")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `rev`".to_owned())?;
    let rev = validate_non_flag("rev", rev)?;
    let path = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `path`".to_owned())?;
    let path = validate_non_flag("path", path)?;
    let max_bytes = clamp_max_bytes(args);

    let spec = format!("{rev}:{path}");
    let content = run_git_readonly(repo_dir, &["show".to_owned(), spec])?;

    let start_line = args
        .get("start_line")
        .and_then(Value::as_u64)
        .map(|n| n.max(1) as usize);
    let end_line = args
        .get("end_line")
        .and_then(Value::as_u64)
        .map(|n| n as usize);

    let sliced = if start_line.is_some() || end_line.is_some() {
        let s = start_line.unwrap_or(1);
        let e = end_line.unwrap_or(usize::MAX);
        if e < s {
            return Err("end_line must be >= start_line".to_owned());
        }
        let mut buf = String::new();
        for (idx, line) in content.lines().enumerate() {
            let line_no = idx + 1;
            if line_no >= s && line_no <= e {
                buf.push_str(&format!("{line_no}: {line}\n"));
            }
        }
        buf
    } else {
        content
    };

    Ok(cap_output(&sliced, max_bytes))
}

fn tool_git_commit_ancestry(repo_dir: &Path, args: &Value) -> Result<String, String> {
    let rev = args
        .get("rev")
        .and_then(Value::as_str)
        .ok_or_else(|| "missing required parameter `rev`".to_owned())?;
    let rev = validate_non_flag("rev", rev)?;

    let info = run_git_readonly(
        repo_dir,
        &[
            "show".to_owned(),
            "-s".to_owned(),
            "--format=commit: %H%nparents: %P%nauthor: %an <%ae>%ndate: %ad%nsubject: %s".to_owned(),
            "--date=iso-strict".to_owned(),
            rev.clone(),
        ],
    )?;

    let mut out = info;
    if let Ok(branches) = run_git_readonly(
        repo_dir,
        &["branch".to_owned(), "--contains".to_owned(), rev.clone()],
    ) {
        if !branches.trim().is_empty() {
            out.push_str("\nbranches containing commit:\n");
            out.push_str(&branches);
        }
    }
    if let Ok(tags) = run_git_readonly(
        repo_dir,
        &["tag".to_owned(), "--contains".to_owned(), rev.clone()],
    ) {
        if !tags.trim().is_empty() {
            out.push_str("\ntags containing commit:\n");
            out.push_str(&tags);
        }
    }

    if let Some(other) = args.get("other_rev").and_then(Value::as_str) {
        if !other.trim().is_empty() {
            let other = validate_non_flag("other_rev", other)?;
            let mb = run_git_readonly(
                repo_dir,
                &["merge-base".to_owned(), rev.clone(), other.clone()],
            )
            .unwrap_or_else(|e| format!("(none: {e})"));
            let counts = run_git_readonly(
                repo_dir,
                &[
                    "rev-list".to_owned(),
                    "--left-right".to_owned(),
                    "--count".to_owned(),
                    format!("{rev}...{other}"),
                ],
            )
            .unwrap_or_default();
            out.push_str(&format!("\ncomparison vs `{other}`:\n"));
            out.push_str(&format!("merge-base: {}\n", mb.trim()));
            if !counts.trim().is_empty() {
                let parts: Vec<&str> = counts.split_whitespace().collect();
                if parts.len() == 2 {
                    out.push_str(&format!(
                        "commits only in `{rev}`: {}, commits only in `{other}`: {}\n",
                        parts[0], parts[1]
                    ));
                }
            }
        }
    }

    Ok(cap_output(&out, DEFAULT_TOOL_MAX_BYTES))
}

/// Handles a single JSON-RPC 2.0 MCP message and returns an optional JSON-RPC response.
pub async fn handle_mcp_request(repo_dir: &Path, msg: &Value) -> Option<Value> {
    let id = msg.get("id").cloned();
    let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
    let params = msg.get("params").cloned().unwrap_or_else(|| json!({}));

    // Notifications (no `id`) do not receive responses.
    let id = id?;

    match method {
        "initialize" => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": MCP_SERVER_NAME,
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        })),
        "ping" => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {}
        })),
        "tools/list" => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "tools": mcp_tool_schemas()
            }
        })),
        "tools/call" => {
            let tool_name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let tool_args = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            if tool_name.is_empty() {
                return Some(json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {
                        "code": -32602,
                        "message": "missing tool name in tools/call"
                    }
                }));
            }
            let repo = repo_dir.to_path_buf();
            let call_res = tokio::task::spawn_blocking(move || {
                execute_git_tool(&repo, &tool_name, &tool_args)
            })
            .await
            .unwrap_or_else(|e| McpToolCallResult {
                text: format!("error: tool task panicked: {e}"),
                is_error: true,
            });

            Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "content": [{ "type": "text", "text": call_res.text }],
                    "isError": call_res.is_error
                }
            }))
        }
        other => Some(json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": -32601,
                "message": format!("Method not found: {other}")
            }
        })),
    }
}

/// Serves MCP JSON-RPC 2.0 newline-delimited messages over any async reader/writer pair.
pub async fn serve_mcp_stream<R, W>(repo_dir: PathBuf, reader: R, mut writer: W) -> Result<(), String>
where
    R: AsyncBufRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = reader.lines();
    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|e| format!("MCP stream read error: {e}"))?
    {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let err_resp = json!({
                    "jsonrpc": "2.0",
                    "id": Value::Null,
                    "error": {
                        "code": -32700,
                        "message": format!("Parse error: {e}")
                    }
                });
                let mut encoded = serde_json::to_string(&err_resp).unwrap_or_default();
                encoded.push('\n');
                writer
                    .write_all(encoded.as_bytes())
                    .await
                    .map_err(|e| format!("MCP stream write error: {e}"))?;
                writer
                    .flush()
                    .await
                    .map_err(|e| format!("MCP stream flush error: {e}"))?;
                continue;
            }
        };

        if let Some(resp) = handle_mcp_request(&repo_dir, &msg).await {
            let mut encoded = serde_json::to_string(&resp)
                .map_err(|e| format!("MCP response serialization error: {e}"))?;
            encoded.push('\n');
            writer
                .write_all(encoded.as_bytes())
                .await
                .map_err(|e| format!("MCP stream write error: {e}"))?;
            writer
                .flush()
                .await
                .map_err(|e| format!("MCP stream flush error: {e}"))?;
        }
    }
    Ok(())
}

/// Runs the standalone MCP Git revision-history server on stdin/stdout (`git-chat --mcp-log-server`).
pub async fn run_mcp_stdio_server(repo_dir: PathBuf) -> Result<(), String> {
    let stdin = BufReader::new(tokio::io::stdin());
    let stdout = tokio::io::stdout();
    serve_mcp_stream(repo_dir, stdin, stdout).await
}

/// Async MCP JSON-RPC 2.0 client connected to the Git revision-history MCP server.
pub struct McpLogClient {
    reader: BufReader<DuplexStream>,
    writer: DuplexStream,
    next_id: u64,
    server_name: String,
    server_version: String,
}

impl McpLogClient {
    /// Connects to an interactive in-process MCP server speaking full JSON-RPC 2.0 over duplex streams.
    pub async fn connect_in_process(repo_dir: PathBuf) -> Result<Self, String> {
        let (client_write, server_read) = tokio::io::duplex(256 * 1024);
        let (server_write, client_read) = tokio::io::duplex(256 * 1024);

        tokio::spawn(async move {
            let _ = serve_mcp_stream(repo_dir, BufReader::new(server_read), server_write).await;
        });

        let mut client = Self {
            reader: BufReader::new(client_read),
            writer: client_write,
            next_id: 1,
            server_name: MCP_SERVER_NAME.to_owned(),
            server_version: env!("CARGO_PKG_VERSION").to_owned(),
        };

        let init = client
            .send_request(
                "initialize",
                json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "git-chat",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
            )
            .await?;

        if let Some(name) = init.pointer("/serverInfo/name").and_then(Value::as_str) {
            client.server_name = name.to_owned();
        }
        if let Some(ver) = init.pointer("/serverInfo/version").and_then(Value::as_str) {
            client.server_version = ver.to_owned();
        }

        client
            .send_notification("notifications/initialized", json!({}))
            .await?;

        Ok(client)
    }

    pub fn server_info(&self) -> (&str, &str) {
        (&self.server_name, &self.server_version)
    }

    async fn send_notification(&mut self, method: &str, params: Value) -> Result<(), String> {
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        let mut encoded = serde_json::to_string(&msg)
            .map_err(|e| format!("MCP notification encode error: {e}"))?;
        encoded.push('\n');
        self.writer
            .write_all(encoded.as_bytes())
            .await
            .map_err(|e| format!("MCP write error: {e}"))?;
        self.writer
            .flush()
            .await
            .map_err(|e| format!("MCP flush error: {e}"))?;
        Ok(())
    }

    async fn send_request(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });
        let mut encoded =
            serde_json::to_string(&msg).map_err(|e| format!("MCP request encode error: {e}"))?;
        encoded.push('\n');
        self.writer
            .write_all(encoded.as_bytes())
            .await
            .map_err(|e| format!("MCP write error: {e}"))?;
        self.writer
            .flush()
            .await
            .map_err(|e| format!("MCP flush error: {e}"))?;

        let mut line = String::new();
        let n = self
            .reader
            .read_line(&mut line)
            .await
            .map_err(|e| format!("MCP read error: {e}"))?;
        if n == 0 {
            return Err("MCP server closed connection unexpectedly".to_owned());
        }
        let resp: Value = serde_json::from_str(line.trim())
            .map_err(|e| format!("MCP response decode error: {e}"))?;
        if let Some(err) = resp.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown MCP error");
            return Err(format!("MCP error: {msg}"));
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| "MCP response missing `result`".to_owned())
    }

    /// Lists all tools advertised by the MCP server.
    pub async fn list_tools(&mut self) -> Result<Vec<ToolDefinition>, String> {
        let result = self.send_request("tools/list", json!({})).await?;
        let arr = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| "MCP tools/list response missing `tools` array".to_owned())?;

        let mut tools = Vec::with_capacity(arr.len());
        for item in arr {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let input_schema = item
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            if !name.is_empty() {
                tools.push(ToolDefinition {
                    name,
                    description,
                    input_schema,
                });
            }
        }
        Ok(tools)
    }

    /// Invokes a named MCP tool with JSON arguments.
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> Result<McpToolCallResult, String> {
        let result = self
            .send_request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments
                }),
            )
            .await?;
        let is_error = result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let mut text = String::new();
        if let Some(blocks) = result.get("content").and_then(Value::as_array) {
            for block in blocks {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
        }
        Ok(McpToolCallResult { text, is_error })
    }
}
