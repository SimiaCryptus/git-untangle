use std::ffi::OsString;
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use commit_untangle_llm::{
    ChatMessage, ChatModel, ChatRequest, ChatRole, ProviderModel, RecordingModel, ReplayModel,
};

use crate::config::{resolve_chat_defaults, MAX_CHAT_MAX_BYTES, MIN_CHAT_MAX_BYTES};
use crate::{write_stdout, EXIT_OK, EXIT_PRECONDITION, EXIT_USAGE_OR_CONFIG};

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

pub fn run_chat_command(inv: ChatInvocation) -> i32 {
    if inv.git_args.is_empty() {
        eprintln!("error: missing git command (e.g. `git chat status`, `git chat log -n 5`, or `git chat untangle HEAD`)");
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
