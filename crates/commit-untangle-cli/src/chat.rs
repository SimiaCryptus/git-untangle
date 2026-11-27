use std::ffi::OsString;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use commit_untangle_git::tempdir::TempDir;
use commit_untangle_llm::prompts::{
    build_auto_commit_message_request, AutoCommitContext, AutoCommitMessageDraft,
};
use commit_untangle_llm::{
    generate_typed, ChatMessage, ChatModel, ChatRequest, ChatRole, ProviderModel, RecordingModel,
    ReplayModel, SemanticModel, ToolChatMessage, ToolChatRequest, ToolDefinition,
};

use crate::config::{resolve_chat_defaults, MAX_CHAT_MAX_BYTES, MIN_CHAT_MAX_BYTES};
use crate::mcp_log::McpLogClient;
use crate::{write_stdout, EXIT_OK, EXIT_PRECONDITION, EXIT_USAGE_OR_CONFIG};

const MAX_MCP_TOOL_ROUNDS: usize = 12;
const EMPTY_TREE_OID: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

#[derive(Debug, Clone)]
pub struct ChatInvocation {
    pub model: Option<String>,
    pub max_bytes: Option<usize>,
    pub prompt: Option<String>,
    pub one_shot: bool,
    pub no_chat: bool,
    pub replay_llm: Option<PathBuf>,
    pub record_llm: Option<PathBuf>,
    pub git_args: Vec<OsString>,
}

#[derive(Debug, Clone)]
pub struct CapturedGitOutput {
    pub command_display: String,
    pub exit_code: i32,
    pub context_text: String,
    pub total_bytes: usize,
    pub truncated: bool,
}

/// Parses leading `git-chat` options up to the first Git subcommand, leaving all subsequent
/// arguments untouched for `git`.
pub fn parse_chat_invocation(args: &[OsString]) -> Result<ChatInvocation, String> {
    let mut model = None;
    let mut max_bytes = None;
    let mut prompt = None;
    let mut one_shot = false;
    let mut no_chat = false;
    let mut replay_llm = None;
    let mut record_llm = None;

    let mut i = 1usize; // skip argv[0]
    while i < args.len() {
        let cur = args[i].to_string_lossy();
        match cur.as_ref() {
            "-m" | "--model" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing value for --model".to_owned())?;
                model = Some(val.to_string_lossy().into_owned());
            }
            "--max-bytes" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing value for --max-bytes".to_owned())?;
                let parsed: usize =
                    val.to_string_lossy().trim().parse().map_err(|_| {
                        format!("invalid --max-bytes value: {}", val.to_string_lossy())
                    })?;
                max_bytes = Some(parsed.clamp(MIN_CHAT_MAX_BYTES, MAX_CHAT_MAX_BYTES));
            }
            "-p" | "--prompt" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing value for --prompt".to_owned())?;
                prompt = Some(val.to_string_lossy().into_owned());
            }
            "--one-shot" => {
                one_shot = true;
            }
            "--no-chat" | "--no-llm" => {
                no_chat = true;
            }
            "--replay-llm" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing value for --replay-llm".to_owned())?;
                replay_llm = Some(PathBuf::from(val));
            }
            "--record-llm" => {
                i += 1;
                let val = args
                    .get(i)
                    .ok_or_else(|| "missing value for --record-llm".to_owned())?;
                record_llm = Some(PathBuf::from(val));
            }
            "--" => {
                i += 1;
                break;
            }
            other if other.starts_with("--max-bytes=") => {
                let raw = other.trim_start_matches("--max-bytes=");
                let parsed: usize = raw
                    .trim()
                    .parse()
                    .map_err(|_| format!("invalid --max-bytes value: {raw}"))?;
                max_bytes = Some(parsed.clamp(MIN_CHAT_MAX_BYTES, MAX_CHAT_MAX_BYTES));
            }
            other if other.starts_with("--model=") => {
                model = Some(other.trim_start_matches("--model=").to_owned());
            }
            other if other.starts_with("--prompt=") => {
                prompt = Some(other.trim_start_matches("--prompt=").to_owned());
            }
            other if other.starts_with("--replay-llm=") => {
                replay_llm = Some(PathBuf::from(other.trim_start_matches("--replay-llm=")));
            }
            other if other.starts_with("--record-llm=") => {
                record_llm = Some(PathBuf::from(other.trim_start_matches("--record-llm=")));
            }
            _ => break,
        }
        i += 1;
    }

    Ok(ChatInvocation {
        model,
        max_bytes,
        prompt,
        one_shot,
        no_chat,
        replay_llm,
        record_llm,
        git_args: args[i..].to_vec(),
    })
}

/// Truncates `text` to at most `max_bytes` bytes on a valid UTF-8 character boundary.
pub fn truncate_utf8_bytes(text: &str, max_bytes: usize) -> (&str, bool) {
    if text.len() <= max_bytes {
        return (text, false);
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    (&text[..end], true)
}

/// Runs `git <git_args...>`, echoes its direct stdout/stderr to the user, and returns a
/// byte-limited context snapshot for the chat session.
pub fn capture_git_command(
    git_args: &[OsString],
    max_bytes: usize,
) -> Result<CapturedGitOutput, String> {
    let output = Command::new("git")
        .args(git_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("failed to execute `git`: {e}"))?;

    let _ = io::stdout().write_all(&output.stdout);
    let _ = io::stdout().flush();
    let _ = io::stderr().write_all(&output.stderr);
    let _ = io::stderr().flush();

    let exit_code = output.status.code().unwrap_or(1);
    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let stderr_str = String::from_utf8_lossy(&output.stderr);

    let mut combined = String::new();
    if !stdout_str.is_empty() {
        combined.push_str(&stdout_str);
    }
    if !stderr_str.is_empty() {
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str("--- stderr ---\n");
        combined.push_str(&stderr_str);
    }

    let total_bytes = combined.len();
    let (prefix, truncated) = truncate_utf8_bytes(&combined, max_bytes);
    let mut context_text = prefix.to_owned();
    if truncated {
        context_text.push_str(&format!(
            "\n[... output truncated at {max_bytes} bytes; total {total_bytes} bytes ...]"
        ));
    }

    let cmd_parts: Vec<String> = git_args
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();

    Ok(CapturedGitOutput {
        command_display: format!("git {}", cmd_parts.join(" ")),
        exit_code,
        context_text,
        total_bytes,
        truncated,
    })
}

pub fn build_initial_chat_messages(
    captured: &CapturedGitOutput,
    max_bytes: usize,
) -> Vec<ChatMessage> {
    let system = ChatMessage {
        role: ChatRole::System,
        content: format!(
            "You are git-chat, a concise interactive Git assistant. \
             The user just ran `{}` (exit code {}). \
             Answer questions directly and accurately based on the captured output below (self-limited to {} bytes). \
             If needed information was truncated or not present, say so clearly and suggest the exact git command to inspect it.",
            captured.command_display, captured.exit_code, max_bytes
        ),
    };

    let context = ChatMessage {
        role: ChatRole::User,
        content: format!(
            "Command executed: `{}` (exit {})\nCaptured output ({} bytes{}):\n```\n{}\n```",
            captured.command_display,
            captured.exit_code,
            captured.total_bytes,
            if captured.truncated {
                format!(", truncated to {max_bytes} bytes")
            } else {
                String::new()
            },
            captured.context_text
        ),
    };

    vec![system, context]
}

/// Keeps conversational history bounded within `max_history_bytes` while always preserving
/// the system prompt (`messages[0]`) and captured Git command context (`messages[1]`).
pub fn trim_chat_history(messages: &mut Vec<ChatMessage>, max_history_bytes: usize) {
    while messages.len() > 4 {
        let conv_bytes: usize = messages[2..].iter().map(|m| m.content.len()).sum();
        if conv_bytes <= max_history_bytes {
            break;
        }
        messages.remove(2);
    }
}

fn build_chat_model(
    inv: &ChatInvocation,
    default_model: Option<&str>,
) -> Result<Option<Box<dyn ChatModel>>, String> {
    if inv.no_chat {
        return Ok(None);
    }
    if let Some(ref replay_path) = inv.replay_llm {
        let replay = ReplayModel::from_jsonl_file(replay_path)
            .map_err(|e| format!("failed to load cassette '{}': {e}", replay_path.display()))?;
        return if let Some(ref rec_path) = inv.record_llm {
            Ok(Some(Box::new(RecordingModel::new(replay, rec_path))))
        } else {
            Ok(Some(Box::new(replay)))
        };
    }

    let chosen = inv.model.as_deref().or(default_model);
    let Some(spec) = chosen else {
        return Ok(None);
    };
    let live = ProviderModel::from_env(spec).map_err(|e| e.to_string())?;
    if let Some(ref rec_path) = inv.record_llm {
        Ok(Some(Box::new(RecordingModel::new(live, rec_path))))
    } else {
        Ok(Some(Box::new(live)))
    }
}

fn build_semantic_model_for_chat(
    inv: &ChatInvocation,
    default_model: Option<&str>,
) -> Result<Option<Box<dyn SemanticModel>>, String> {
    if inv.no_chat {
        return Ok(None);
    }
    if let Some(ref replay_path) = inv.replay_llm {
        let replay = ReplayModel::from_jsonl_file(replay_path)
            .map_err(|e| format!("failed to load cassette '{}': {e}", replay_path.display()))?;
        return if let Some(ref rec_path) = inv.record_llm {
            Ok(Some(Box::new(RecordingModel::new(replay, rec_path))))
        } else {
            Ok(Some(Box::new(replay)))
        };
    }

    let chosen = inv.model.as_deref().or(default_model);
    let Some(spec) = chosen else {
        return Ok(None);
    };
    let live = ProviderModel::from_env(spec).map_err(|e| e.to_string())?;
    if let Some(ref rec_path) = inv.record_llm {
        Ok(Some(Box::new(RecordingModel::new(live, rec_path))))
    } else {
        Ok(Some(Box::new(live)))
    }
}

fn run_git_passthrough(git_args: &[OsString]) -> i32 {
    match Command::new("git")
        .args(git_args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
    {
        Ok(status) => status.code().unwrap_or(1),
        Err(e) => {
            eprintln!("error: failed to execute `git`: {e}");
            EXIT_PRECONDITION
        }
    }
}

pub fn run_chat_command(inv: ChatInvocation) -> i32 {
    if inv.git_args.is_empty() {
        eprintln!("error: missing git command (e.g. `git chat status`, `git chat log`, `git chat commit`, or `git chat untangle HEAD`)");
        return EXIT_USAGE_OR_CONFIG;
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let defaults = match resolve_chat_defaults(&cwd) {
        Ok(d) => d,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
        }
    };

    let max_bytes = inv.max_bytes.unwrap_or(defaults.max_bytes);
    let subcmd = inv
        .git_args
        .first()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    if subcmd == "commit" {
        return run_chat_commit(&inv, &cwd, defaults.model.as_deref(), max_bytes);
    }

    if subcmd == "log" {
        return run_chat_log(&inv, &cwd, defaults.model.as_deref(), max_bytes);
    }

    let model_opt = match build_chat_model(&inv, defaults.model.as_deref()) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
        }
    };

    let Some(model) = model_opt else {
        if inv.prompt.is_some() {
            eprintln!(
                "error: --prompt requires an LLM model (set --model <provider::id> or `git config untangle.model ...`)"
            );
            return EXIT_USAGE_OR_CONFIG;
        }
        // No model configured or --no-chat requested: run git directly as a clean passthrough.
        return match capture_git_command(&inv.git_args, max_bytes) {
            Ok(cap) => cap.exit_code,
            Err(msg) => {
                eprintln!("error: {msg}");
                EXIT_PRECONDITION
            }
        };
    };

    let captured = match capture_git_command(&inv.git_args, max_bytes) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_PRECONDITION;
        }
    };

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to initialize runtime: {e}");
            return EXIT_PRECONDITION;
        }
    };

    rt.block_on(run_chat_loop(model.as_ref(), captured, max_bytes, &inv))
}

#[derive(Debug, Default)]
struct CommitFlagSummary {
    has_explicit_message: bool,
    all: bool,
    amend: bool,
    dry_run: bool,
    allow_empty: bool,
    pathspecs: Vec<String>,
}

fn inspect_commit_args(args: &[OsString]) -> CommitFlagSummary {
    let mut summary = CommitFlagSummary::default();
    let mut after_double_dash = false;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].to_string_lossy();
        if after_double_dash {
            summary.pathspecs.push(arg.into_owned());
            i += 1;
            continue;
        }
        match arg.as_ref() {
            "--" => {
                after_double_dash = true;
            }
            "-m" | "--message" | "-F" | "--file" | "-C" | "--reuse-message" | "-c"
            | "--reedit-message" | "-t" | "--template" | "--fixup" | "--squash" => {
                summary.has_explicit_message = true;
                i += 1; // skip value
            }
            "-h" | "--help" => {
                summary.has_explicit_message = true;
            }
            "-a" | "--all" => {
                summary.all = true;
            }
            "--amend" => {
                summary.amend = true;
            }
            "--dry-run" => {
                summary.dry_run = true;
            }
            "--allow-empty" => {
                summary.allow_empty = true;
            }
            other
                if other.starts_with("--message=")
                    || other.starts_with("--file=")
                    || other.starts_with("--reuse-message=")
                    || other.starts_with("--reedit-message=")
                    || other.starts_with("--template=")
                    || other.starts_with("--fixup=")
                    || other.starts_with("--squash=") =>
            {
                summary.has_explicit_message = true;
            }
            other if other.starts_with('-') && !other.starts_with("--") && other.len() > 2 => {
                // Combined short flags like `-am` or `-as`
                for ch in other[1..].chars() {
                    match ch {
                        'm' | 'F' | 'C' | 'c' | 't' => {
                            summary.has_explicit_message = true;
                        }
                        'a' => {
                            summary.all = true;
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }
    summary
}

fn run_git_capture_text(repo_dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .current_dir(repo_dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn run_git_capture_owned(repo_dir: &Path, args: &[String]) -> Result<String, String> {
    let out = Command::new("git")
        .current_dir(repo_dir)
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_PAGER", "cat")
        .output()
        .map_err(|e| format!("failed to execute `git`: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("git {} failed: {}", args.join(" "), stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn run_chat_commit(
    inv: &ChatInvocation,
    cwd: &Path,
    default_model: Option<&str>,
    max_bytes: usize,
) -> i32 {
    let commit_args = &inv.git_args[1..];
    let flags = inspect_commit_args(commit_args);

    if inv.no_chat || flags.has_explicit_message {
        return run_git_passthrough(&inv.git_args);
    }

    let model_opt = match build_semantic_model_for_chat(inv, default_model) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
        }
    };
    let Some(model) = model_opt else {
        eprintln!(
            "error: `git chat commit` requires an LLM model to autogenerate a commit message \
             (set --model <provider::id>, `git config untangle.model ...`, or pass --no-chat / -m <msg>)"
        );
        return EXIT_USAGE_OR_CONFIG;
    };

    let has_head = run_git_capture_text(cwd, &["rev-parse", "--verify", "HEAD"]).is_some();
    let has_head_parent =
        run_git_capture_text(cwd, &["rev-parse", "--verify", "HEAD~1"]).is_some();

    let mut diff_base_args: Vec<String> = vec!["diff".to_owned()];
    if flags.amend {
        if !has_head {
            eprintln!("error: cannot use --amend on a repository with no commits");
            return EXIT_PRECONDITION;
        }
        if !flags.all {
            diff_base_args.push("--cached".to_owned());
        }
        diff_base_args.push(if has_head_parent {
            "HEAD~1".to_owned()
        } else {
            EMPTY_TREE_OID.to_owned()
        });
    } else if flags.all && has_head {
        diff_base_args.push("HEAD".to_owned());
    } else {
        diff_base_args.push("--cached".to_owned());
    }

    let mut stat_args = diff_base_args.clone();
    stat_args.push("--stat".to_owned());
    if !flags.pathspecs.is_empty() {
        stat_args.push("--".to_owned());
        stat_args.extend(flags.pathspecs.iter().cloned());
        diff_base_args.push("--".to_owned());
        diff_base_args.extend(flags.pathspecs.iter().cloned());
    }

    let diff_stat = match run_git_capture_owned(cwd, &stat_args) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_PRECONDITION;
        }
    };
    let full_diff = match run_git_capture_owned(cwd, &diff_base_args) {
        Ok(s) => s,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_PRECONDITION;
        }
    };

    if full_diff.trim().is_empty() && !flags.allow_empty && !flags.amend {
        eprintln!(
            "error: nothing staged to commit (stage files with `git add` or pass `git chat commit -a`)"
        );
        return EXIT_PRECONDITION;
    }

    let branch = run_git_capture_text(cwd, &["rev-parse", "--abbrev-ref", "HEAD"])
        .unwrap_or_default();
    let recent_subjects: Vec<String> = run_git_capture_text(cwd, &["log", "-n", "8", "--format=%s"])
        .map(|s| {
            s.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let amend_previous = if flags.amend {
        run_git_capture_text(cwd, &["log", "-1", "--format=%B", "HEAD"])
    } else {
        None
    };

    let total_diff_bytes = full_diff.len();
    let (diff_prefix, diff_truncated) = truncate_utf8_bytes(&full_diff, max_bytes);
    let ctx = AutoCommitContext {
        branch: branch.trim(),
        recent_subjects: &recent_subjects,
        diff_stat: diff_stat.trim(),
        diff_text: diff_prefix,
        diff_truncated,
        total_diff_bytes,
        amend_previous_message: amend_previous.as_deref(),
        user_hint: inv.prompt.as_deref(),
    };
    let req = build_auto_commit_message_request(&ctx);

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to initialize runtime: {e}");
            return EXIT_PRECONDITION;
        }
    };

    let draft = match rt.block_on(generate_typed::<AutoCommitMessageDraft>(
        model.as_ref(),
        req,
    )) {
        Ok(d) => d,
        Err(err) => {
            eprintln!("error: failed to generate commit message: {err}");
            if let Some(hint) = err.diagnostic_hint() {
                eprintln!("hint: {hint}");
            }
            return if err.is_fatal_config_or_auth() {
                EXIT_USAGE_OR_CONFIG
            } else {
                EXIT_PRECONDITION
            };
        }
    };

    let formatted = draft.format_commit_message();
    if formatted.trim().is_empty() {
        eprintln!("error: LLM generated an empty commit message");
        return EXIT_PRECONDITION;
    }

    if flags.dry_run {
        let _ = write_stdout(&formatted);
        return EXIT_OK;
    }

    eprintln!("[git-chat] Generated commit message:\n{formatted}");

    let tmp_dir = match TempDir::new("git-chat-commit") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: failed to create temporary directory: {e}");
            return EXIT_PRECONDITION;
        }
    };
    let msg_path = tmp_dir.path().join("COMMIT_MSG");
    if let Err(e) = std::fs::write(&msg_path, formatted.as_bytes()) {
        eprintln!("error: failed to write generated commit message: {e}");
        return EXIT_PRECONDITION;
    }

    let mut final_args: Vec<OsString> = Vec::with_capacity(commit_args.len() + 3);
    final_args.push(OsString::from("commit"));
    final_args.push(OsString::from("-F"));
    final_args.push(msg_path.into_os_string());
    final_args.extend(commit_args.iter().cloned());

    run_git_passthrough(&final_args)
}

fn run_chat_log(
    inv: &ChatInvocation,
    cwd: &Path,
    default_model: Option<&str>,
    max_bytes: usize,
) -> i32 {
    let model_opt = match build_chat_model(inv, default_model) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
        }
    };

    let Some(model) = model_opt else {
        if inv.prompt.is_some() {
            eprintln!(
                "error: --prompt requires an LLM model (set --model <provider::id> or `git config untangle.model ...`)"
            );
            return EXIT_USAGE_OR_CONFIG;
        }
        return run_git_passthrough(&inv.git_args);
    };

    // For bare `git chat log`, show a concise recent 20-commit snapshot instead of dumping the
    // entire repository history; the MCP server gives the chat session full navigation & search.
    let initial_git_args: Vec<OsString> = if inv.git_args.len() == 1 {
        vec![
            OsString::from("log"),
            OsString::from("-n"),
            OsString::from("20"),
            OsString::from("--oneline"),
            OsString::from("--decorate"),
        ]
    } else {
        inv.git_args.clone()
    };

    let captured = match capture_git_command(&initial_git_args, max_bytes) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_PRECONDITION;
        }
    };

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("error: failed to initialize runtime: {e}");
            return EXIT_PRECONDITION;
        }
    };

    rt.block_on(run_mcp_log_chat_loop(
        model.as_ref(),
        cwd.to_path_buf(),
        captured,
        max_bytes,
        inv,
    ))
}

pub fn build_initial_log_mcp_messages(
    captured: &CapturedGitOutput,
    max_bytes: usize,
    tools: &[ToolDefinition],
) -> Vec<ToolChatMessage> {
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    let system = ToolChatMessage::System {
        content: format!(
            "You are git-chat, an interactive Git revision-history assistant connected to an MCP tool server. \
             You have access to these read-only Git history MCP tools: {}. \
             Use these tools actively to navigate commit history, paginate logs, search messages or code changes (pickaxe -S/-G), \
             inspect commits and diffs, follow file histories across renames, view blame or files at past revisions, and check branch/tag ancestry. \
             Ground your answers in actual tool results and cite commit hashes and file paths accurately.",
            tool_names.join(", ")
        ),
    };

    let context = ToolChatMessage::User {
        content: format!(
            "Initial command executed: `{}` (exit {})\nCaptured output ({} bytes{}):\n```\n{}\n```",
            captured.command_display,
            captured.exit_code,
            captured.total_bytes,
            if captured.truncated {
                format!(", truncated to {max_bytes} bytes")
            } else {
                String::new()
            },
            captured.context_text
        ),
    };

    vec![system, context]
}

fn trim_tool_chat_history(messages: &mut Vec<ToolChatMessage>, max_history_bytes: usize) {
    while messages.len() > 6 {
        let conv_bytes: usize = messages[2..]
            .iter()
            .map(|m| match m {
                ToolChatMessage::System { content } | ToolChatMessage::User { content } => {
                    content.len()
                }
                ToolChatMessage::Assistant { content, .. } => content.len(),
                ToolChatMessage::ToolResult { content, .. } => content.len(),
            })
            .sum();
        if conv_bytes <= max_history_bytes {
            break;
        }
        messages.remove(2);
        // Never leave orphaned ToolResult messages immediately after the initial context.
        while matches!(messages.get(2), Some(ToolChatMessage::ToolResult { .. })) {
            messages.remove(2);
        }
    }
}

async fn run_tool_agent_turn(
    model: &dyn ChatModel,
    mcp: &mut McpLogClient,
    tools: &[ToolDefinition],
    messages: &mut Vec<ToolChatMessage>,
) -> Result<String, String> {
    for _round in 0..MAX_MCP_TOOL_ROUNDS {
        let resp = model
            .chat_with_tools(ToolChatRequest {
                messages: messages.clone(),
                tools: tools.to_vec(),
                temperature: 0.2,
            })
            .await
            .map_err(|e| format!("chat request failed: {e}"))?;

        if resp.tool_calls.is_empty() {
            messages.push(ToolChatMessage::Assistant {
                content: resp.content.clone(),
                tool_calls: Vec::new(),
            });
            return Ok(resp.content);
        }

        let calls = resp.tool_calls.clone();
        messages.push(ToolChatMessage::Assistant {
            content: resp.content,
            tool_calls: calls.clone(),
        });

        for call in calls {
            eprintln!("[mcp] calling {}({})", call.name, call.arguments);
            let res = match mcp.call_tool(&call.name, call.arguments).await {
                Ok(r) => r,
                Err(err) => crate::mcp_log::McpToolCallResult {
                    text: format!("MCP tool invocation error: {err}"),
                    is_error: true,
                },
            };
            messages.push(ToolChatMessage::ToolResult {
                tool_call_id: call.id,
                tool_name: call.name,
                content: res.text,
                is_error: res.is_error,
            });
        }
    }

    Err(format!(
        "exceeded maximum MCP tool rounds ({MAX_MCP_TOOL_ROUNDS}) without a final answer"
    ))
}

async fn run_mcp_log_chat_loop(
    model: &dyn ChatModel,
    repo_dir: PathBuf,
    captured: CapturedGitOutput,
    max_bytes: usize,
    inv: &ChatInvocation,
) -> i32 {
    let mut mcp = match McpLogClient::connect_in_process(repo_dir).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: failed to start MCP git-log server: {e}");
            return EXIT_PRECONDITION;
        }
    };
    let tools = match mcp.list_tools().await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: failed to list MCP tools: {e}");
            return EXIT_PRECONDITION;
        }
    };

    let mut messages = build_initial_log_mcp_messages(&captured, max_bytes, &tools);
    let interactive_tty = io::stdin().is_terminal() && io::stderr().is_terminal();

    if let Some(ref initial_q) = inv.prompt {
        messages.push(ToolChatMessage::User {
            content: initial_q.clone(),
        });
        match run_tool_agent_turn(model, &mut mcp, &tools, &mut messages).await {
            Ok(reply) => {
                let _ = write_stdout(&format!("{reply}\n"));
            }
            Err(e) => {
                eprintln!("error: {e}");
                return EXIT_PRECONDITION;
            }
        }
        if inv.one_shot || !interactive_tty {
            return EXIT_OK;
        }
    }

    let (srv_name, srv_ver) = mcp.server_info();
    eprintln!(
        "\n[git-chat log] MCP server `{srv_name}` v{srv_ver} connected ({} tools). \
         Ask about revision history, use /tools or /tool <name> [json], or /quit to exit.",
        tools.len()
    );

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        if interactive_tty {
            let _ = write!(io::stderr(), "git-chat log> ");
            let _ = io::stderr().flush();
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("error reading stdin: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "/quit" | "/exit" | "quit" | "exit" => break,
            "/tools" => {
                for t in &tools {
                    eprintln!("  - {}: {}", t.name, t.description);
                }
                continue;
            }
            "/context" => {
                let (sn, sv) = mcp.server_info();
                eprintln!(
                    "[git-chat log] `{}` — {} bytes captured (limit: {} bytes, truncated: {}); MCP server: {sn} v{sv} ({} tools)",
                    captured.command_display,
                    captured.total_bytes,
                    max_bytes,
                    captured.truncated,
                    tools.len()
                );
                continue;
            }
            _ => {}
        }

        if let Some(rest) = trimmed.strip_prefix("/tool ") {
            let rest = rest.trim();
            let (name, json_str) = match rest.split_once(char::is_whitespace) {
                Some((n, a)) => (n.trim(), a.trim()),
                None => (rest, "{}"),
            };
            let args_val: serde_json::Value = match serde_json::from_str(json_str) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("error: invalid JSON arguments for /tool {name}: {e}");
                    continue;
                }
            };
            match mcp.call_tool(name, args_val).await {
                Ok(res) => {
                    let _ = write_stdout(&format!("{}\n", res.text));
                }
                Err(e) => {
                    eprintln!("error: MCP tool call failed: {e}");
                }
            }
            continue;
        }

        messages.push(ToolChatMessage::User {
            content: trimmed.to_owned(),
        });
        trim_tool_chat_history(&mut messages, max_bytes.saturating_mul(2));

        match run_tool_agent_turn(model, &mut mcp, &tools, &mut messages).await {
            Ok(reply) => {
                let _ = write_stdout(&format!("{reply}\n"));
            }
            Err(e) => {
                eprintln!("error: {e}");
                return EXIT_PRECONDITION;
            }
        }
    }

    EXIT_OK
}

async fn run_chat_loop(
    model: &dyn ChatModel,
    captured: CapturedGitOutput,
    max_bytes: usize,
    inv: &ChatInvocation,
) -> i32 {
    let mut messages = build_initial_chat_messages(&captured, max_bytes);
    let interactive_tty = io::stdin().is_terminal() && io::stderr().is_terminal();

    if let Some(ref initial_q) = inv.prompt {
        messages.push(ChatMessage {
            role: ChatRole::User,
            content: initial_q.clone(),
        });
        match model
            .chat(ChatRequest {
                messages: messages.clone(),
                temperature: 0.2,
            })
            .await
        {
            Ok(reply) => {
                let _ = write_stdout(&format!("{reply}\n"));
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: reply,
                });
            }
            Err(e) => {
                eprintln!("error: chat request failed: {e}");
                return EXIT_PRECONDITION;
            }
        }
        if inv.one_shot || !interactive_tty {
            return EXIT_OK;
        }
    }

    eprintln!(
        "\n[git-chat] Context: {} bytes from `{}` (limit: {} bytes{}). Ask a question, or type /quit to exit.",
        captured.total_bytes,
        captured.command_display,
        max_bytes,
        if captured.truncated { ", truncated" } else { "" }
    );

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();

    loop {
        if interactive_tty {
            let _ = write!(io::stderr(), "git-chat> ");
            let _ = io::stderr().flush();
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("error reading stdin: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "/quit" | "/exit" | "quit" | "exit" => break,
            "/context" => {
                eprintln!(
                    "[git-chat] `{}` — {} bytes captured (limit: {} bytes, truncated: {})",
                    captured.command_display, captured.total_bytes, max_bytes, captured.truncated
                );
                continue;
            }
            _ => {}
        }

        messages.push(ChatMessage {
            role: ChatRole::User,
            content: trimmed.to_owned(),
        });
        trim_chat_history(&mut messages, max_bytes);

        match model
            .chat(ChatRequest {
                messages: messages.clone(),
                temperature: 0.2,
            })
            .await
        {
            Ok(reply) => {
                let _ = write_stdout(&format!("{reply}\n"));
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: reply,
                });
            }
            Err(e) => {
                eprintln!("error: chat request failed: {e}");
                return EXIT_PRECONDITION;
            }
        }
    }

    EXIT_OK
}
