use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use commit_untangle_core::graph::MechanicalPlacement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EmitTarget {
    Branch,
    Json,
    Report,
    Patches,
    Viewer,
}

#[derive(Debug, Parser, Clone)]
#[command(
    name = "git-untangle",
    version,
    about = "Decompose a commit or commit range into an ordered DAG of atomic operations"
)]
pub struct CliArgs {
    /// Run as if started in <REPO> instead of the current working directory
    #[arg(short = 'C', long = "repo")]
    pub repo: Option<PathBuf>,

    /// Commit (default: HEAD) or commit range (A..B, A...B) to untangle
    #[arg(value_name = "REVISION_OR_RANGE")]
    pub target: Option<String>,

    /// Explicit base revision (equivalent to <BASE>..<REVISION>)
    #[arg(long = "base", value_name = "REV")]
    pub base: Option<String>,

    /// Run strictly in deterministic mode without LLM calls
    #[arg(long = "no-llm")]
    pub no_llm: bool,

    /// Provider-qualified LLM identifier (e.g. anthropic::claude-sonnet-4-5)
    #[arg(short = 'm', long = "model", value_name = "PROVIDER::ID")]
    pub model: Option<String>,

    /// Shell command used to verify intermediate states build
    #[arg(long = "verify", value_name = "CMD")]
    pub verify: Option<String>,

    /// Per-build timeout in seconds
    #[arg(long = "verify-timeout", value_name = "SECS")]
    pub verify_timeout: Option<u64>,

    /// Maximum verification graph repair iterations
    #[arg(long = "max-retries", value_name = "N")]
    pub max_retries: Option<usize>,

    /// Comma-separated outputs: branch,json,report,patches,viewer (or 'none')
    #[arg(long = "emit", value_name = "TARGETS")]
    pub emit: Option<String>,

    /// Explicit output branch name [default: untangle/<short-id>]
    #[arg(short = 'b', long = "branch", value_name = "NAME")]
    pub branch: Option<String>,

    /// Directory for JSON, Markdown report, patches, and HTML viewer
    #[arg(short = 'o', long = "output-dir", value_name = "DIR")]
    pub output_dir: Option<PathBuf>,

    /// Analyze and emit artifacts without writing Git commits or branches
    #[arg(long = "dry-run")]
    pub dry_run: bool,

    /// Placement of mechanical operations: first | last
    #[arg(long = "mechanical-placement", value_name = "POS")]
    pub mechanical_placement: Option<String>,

    /// Similarity threshold for deterministic clustering in (0.0, 2.0]
    #[arg(long = "similarity", value_name = "FLOAT")]
    pub similarity: Option<f64>,

    /// Minimum occurrences for bulk-replace promotion
    #[arg(long = "bulk-min", value_name = "N")]
    pub bulk_min: Option<usize>,

    /// Maximum items per deterministic cluster before recursive splitting
    #[arg(long = "max-cluster-size", value_name = "N")]
    pub max_cluster_size: Option<usize>,

    /// Maximum operations for LLM dependency reasoning
    #[arg(long = "max-llm-ops", value_name = "N")]
    pub max_llm_ops: Option<usize>,

    /// Record LLM requests and responses to a JSONL cassette file
    #[arg(long = "record-llm", value_name = "PATH")]
    pub record_llm: Option<PathBuf>,

    /// Replay LLM responses offline from a JSONL cassette file
    #[arg(long = "replay-llm", value_name = "PATH")]
    pub replay_llm: Option<PathBuf>,

    /// Print machine-readable JSON summary to stdout
    #[arg(long = "json")]
    pub json_stdout: bool,

    /// Generate shell completions (bash, zsh, fish, powershell) and exit
    #[arg(long = "generate-completions", value_name = "SHELL")]
    pub generate_completions: Option<String>,

    /// Generate roff manpage (git-untangle.1) to stdout and exit
    #[arg(long = "generate-manpage")]
    pub generate_manpage: bool,

    /// Suppress non-essential status messages on stderr
    #[arg(short = 'q', long = "quiet")]
    pub quiet: bool,

    /// Increase diagnostic verbosity (-v, -vv)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Optional repository pathspecs restricting the analyzed diff
    #[arg(last = true, value_name = "PATHSPECS")]
    pub pathspecs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub repo_dir: PathBuf,
    pub target_spec: String,
    pub base_override: Option<String>,
    pub pathspecs: Vec<String>,
    pub use_llm: bool,
    pub model: Option<String>,
    pub verify_command: Option<String>,
    pub verify_timeout_secs: u64,
    pub max_retries: usize,
    pub emit_targets: BTreeSet<EmitTarget>,
    pub branch_name: Option<String>,
    pub branch_prefix: String,
    pub output_dir: Option<PathBuf>,
    pub dry_run: bool,
    pub mechanical_placement: MechanicalPlacement,
    pub similarity_threshold: f64,
    pub bulk_min: usize,
    pub max_cluster_size: usize,
    pub max_llm_ops: usize,
    pub record_llm: Option<PathBuf>,
    pub replay_llm: Option<PathBuf>,
    pub json_stdout: bool,
    pub quiet: bool,
    pub verbose: u8,
}

#[derive(Default, Debug)]
struct PartialConfig {
    use_llm: Option<bool>,
    model: Option<String>,
    verify_command: Option<String>,
    verify_timeout_secs: Option<u64>,
    max_retries: Option<usize>,
    emit: Option<String>,
    branch_prefix: Option<String>,
    mechanical_placement: Option<String>,
    similarity_threshold: Option<f64>,
    bulk_min: Option<usize>,
    max_cluster_size: Option<usize>,
    max_llm_ops: Option<usize>,
}

impl PartialConfig {
    fn merge_lower_priority(&mut self, lower: Self) {
        if self.use_llm.is_none() {
            self.use_llm = lower.use_llm;
        }
        if self.model.is_none() {
            self.model = lower.model;
        }
        if self.verify_command.is_none() {
            self.verify_command = lower.verify_command;
        }
        if self.verify_timeout_secs.is_none() {
            self.verify_timeout_secs = lower.verify_timeout_secs;
        }
        if self.max_retries.is_none() {
            self.max_retries = lower.max_retries;
        }
        if self.emit.is_none() {
            self.emit = lower.emit;
        }
        if self.branch_prefix.is_none() {
            self.branch_prefix = lower.branch_prefix;
        }
        if self.mechanical_placement.is_none() {
            self.mechanical_placement = lower.mechanical_placement;
        }
        if self.similarity_threshold.is_none() {
            self.similarity_threshold = lower.similarity_threshold;
        }
        if self.bulk_min.is_none() {
            self.bulk_min = lower.bulk_min;
        }
        if self.max_cluster_size.is_none() {
            self.max_cluster_size = lower.max_cluster_size;
        }
        if self.max_llm_ops.is_none() {
            self.max_llm_ops = lower.max_llm_ops;
        }
    }
}

fn is_forbidden_credential_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    lower.contains("apikey")
        || lower.contains("api_key")
        || lower.contains("api-key")
        || lower.contains("secret")
        || lower.ends_with("token")
}

fn load_git_config_layer(repo_dir: &Path, scope_flag: &str) -> Result<PartialConfig, String> {
    let out = Command::new("git")
        .current_dir(repo_dir)
        .args(["config", scope_flag, "--get-regexp", r"^untangle\."])
        .env("GIT_TERMINAL_PROMPT", "0")
        .output();

    let Ok(output) = out else {
        return Ok(PartialConfig::default());
    };
    if !output.status.success() {
        return Ok(PartialConfig::default());
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut cfg = PartialConfig::default();
    for line in text.lines() {
        let Some((raw_key, raw_val)) = line.split_once(char::is_whitespace) else {
            continue;
        };
        let key = raw_key
            .trim()
            .strip_prefix("untangle.")
            .unwrap_or(raw_key.trim());
        let val = raw_val.trim();

        if is_forbidden_credential_key(key) {
            return Err(format!(
                "refusing API credential in git config key `untangle.{key}`; store credentials in environment variables instead"
            ));
        }

        match key.to_ascii_lowercase().as_str() {
            "usellm" | "use-llm" | "use_llm" => {
                cfg.use_llm = Some(parse_bool(val)?);
            }
            "model" => cfg.model = Some(val.to_owned()),
            "verifycommand" | "verify-command" | "verify_command" => {
                cfg.verify_command = Some(val.to_owned());
            }
            "verifytimeout" | "verify-timeout" | "verify_timeout" => {
                cfg.verify_timeout_secs = Some(
                    val.parse()
                        .map_err(|_| format!("invalid untangle.{key}: {val}"))?,
                );
            }
            "maxretries" | "max-retries" | "max_retries" => {
                cfg.max_retries = Some(
                    val.parse()
                        .map_err(|_| format!("invalid untangle.{key}: {val}"))?,
                );
            }
            "emit" => cfg.emit = Some(val.to_owned()),
            "branchprefix" | "branch-prefix" | "branch_prefix" => {
                cfg.branch_prefix = Some(val.to_owned());
            }
            "mechanicalplacement" | "mechanical-placement" | "mechanical_placement" => {
                cfg.mechanical_placement = Some(val.to_owned());
            }
            "similaritythreshold" | "similarity-threshold" | "similarity_threshold"
            | "similarity" => {
                cfg.similarity_threshold = Some(
                    val.parse()
                        .map_err(|_| format!("invalid untangle.{key}: {val}"))?,
                );
            }
            "bulkmin" | "bulk-min" | "bulk_min" => {
                cfg.bulk_min = Some(
                    val.parse()
                        .map_err(|_| format!("invalid untangle.{key}: {val}"))?,
                );
            }
            "maxclustersize" | "max-cluster-size" | "max_cluster_size" => {
                cfg.max_cluster_size = Some(
                    val.parse()
                        .map_err(|_| format!("invalid untangle.{key}: {val}"))?,
                );
            }
            "maxllmops" | "max-llm-ops" | "max_llm_ops" => {
                cfg.max_llm_ops = Some(
                    val.parse()
                        .map_err(|_| format!("invalid untangle.{key}: {val}"))?,
                );
            }
            _ => {}
        }
    }
    Ok(cfg)
}

fn xdg_config_path() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.trim().is_empty() {
            return Some(PathBuf::from(xdg).join("git-untangle").join("config.toml"));
        }
    }
    xdg_home::home_dir().map(|h| h.join(".config").join("git-untangle").join("config.toml"))
}

fn load_xdg_toml_layer() -> Result<PartialConfig, String> {
    let Some(path) = xdg_config_path() else {
        return Ok(PartialConfig::default());
    };
    if !path.is_file() {
        return Ok(PartialConfig::default());
    }
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let doc = content
        .parse::<toml_edit::Document>()
        .map_err(|e| format!("invalid TOML in {}: {e}", path.display()))?;

    let mut cfg = PartialConfig::default();
    for (key, item) in doc.iter() {
        if is_forbidden_credential_key(key) {
            return Err(format!(
                "refusing API credential `{key}` in {}; store credentials in environment variables instead",
                path.display()
            ));
        }
        match key {
            "use_llm" => cfg.use_llm = item.as_bool(),
            "model" => cfg.model = item.as_str().map(str::to_owned),
            "verify_command" => cfg.verify_command = item.as_str().map(str::to_owned),
            "verify_timeout" => {
                cfg.verify_timeout_secs = item.as_integer().and_then(|v| u64::try_from(v).ok());
            }
            "max_retries" => {
                cfg.max_retries = item.as_integer().and_then(|v| usize::try_from(v).ok());
            }
            "emit" => cfg.emit = item.as_str().map(str::to_owned),
            "branch_prefix" => cfg.branch_prefix = item.as_str().map(str::to_owned),
            "mechanical_placement" => {
                cfg.mechanical_placement = item.as_str().map(str::to_owned);
            }
            "similarity_threshold" | "similarity" => {
                cfg.similarity_threshold = item
                    .as_float()
                    .or_else(|| item.as_integer().map(|i| i as f64));
            }
            "bulk_min" => {
                cfg.bulk_min = item.as_integer().and_then(|v| usize::try_from(v).ok());
            }
            "max_cluster_size" => {
                cfg.max_cluster_size = item.as_integer().and_then(|v| usize::try_from(v).ok());
            }
            "max_llm_ops" => {
                cfg.max_llm_ops = item.as_integer().and_then(|v| usize::try_from(v).ok());
            }
            _ => {}
        }
    }
    Ok(cfg)
}

fn parse_bool(val: &str) -> Result<bool, String> {
    match val.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => Err(format!("expected boolean value, got '{other}'")),
    }
}

fn parse_emit_targets(raw: &str) -> Result<BTreeSet<EmitTarget>, String> {
    let mut set = BTreeSet::new();
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("none") || trimmed.is_empty() {
        return Ok(set);
    }
    for part in trimmed.split(',') {
        match part.trim().to_ascii_lowercase().as_str() {
            "" => {}
            "none" => return Ok(BTreeSet::new()),
            "branch" => {
                set.insert(EmitTarget::Branch);
            }
            "json" => {
                set.insert(EmitTarget::Json);
            }
            "report" => {
                set.insert(EmitTarget::Report);
            }
            "patches" | "patch" => {
                set.insert(EmitTarget::Patches);
            }
            "viewer" | "html" => {
                set.insert(EmitTarget::Viewer);
            }
            "all" => {
                set.extend([
                    EmitTarget::Branch,
                    EmitTarget::Json,
                    EmitTarget::Report,
                    EmitTarget::Patches,
                    EmitTarget::Viewer,
                ]);
            }
            other => {
                return Err(format!(
                    "unknown --emit target '{other}' (expected branch,json,report,patches,viewer)"
                ));
            }
        }
    }
    Ok(set)
}

fn validate_branch_name(name: &str) -> Result<(), String> {
    let trimmed = name.trim();
    if trimmed.is_empty()
        || trimmed.starts_with('-')
        || trimmed.ends_with('/')
        || trimmed.ends_with(".lock")
        || trimmed.contains("..")
        || trimmed.contains("@{")
        || trimmed
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '~' | '^' | ':' | '?' | '*' | '[' | '\\'))
    {
        return Err(format!("invalid git branch name: '{name}'"));
    }
    Ok(())
}

/// Resolves the 5-layer configuration hierarchy:
/// CLI flags > local `.git/config` > global `~/.gitconfig` > XDG `config.toml` > defaults.
pub fn resolve_config(cli: CliArgs) -> Result<ResolvedConfig, String> {
    let repo_dir = cli.repo.unwrap_or_else(|| PathBuf::from("."));

    let mut layered = PartialConfig::default();
    if repo_dir.is_dir() {
        layered.merge_lower_priority(load_git_config_layer(&repo_dir, "--local")?);
        layered.merge_lower_priority(load_git_config_layer(&repo_dir, "--global")?);
    }
    layered.merge_lower_priority(load_xdg_toml_layer()?);

    let model = cli.model.or(layered.model).filter(|s| !s.trim().is_empty());
    let use_llm = if cli.no_llm {
        false
    } else if cli.replay_llm.is_some() || model.is_some() {
        layered.use_llm.unwrap_or(true)
    } else {
        layered.use_llm.unwrap_or(false)
    };

    let placement_raw = cli
        .mechanical_placement
        .or(layered.mechanical_placement)
        .unwrap_or_else(|| "first".to_owned());
    let mechanical_placement = match placement_raw.trim().to_ascii_lowercase().as_str() {
        "first" => MechanicalPlacement::First,
        "last" => MechanicalPlacement::Last,
        other => {
            return Err(format!(
                "--mechanical-placement must be 'first' or 'last', got '{other}'"
            ));
        }
    };

    let similarity_threshold = cli
        .similarity
        .or(layered.similarity_threshold)
        .unwrap_or(0.5);
    if !(similarity_threshold > 0.0 && similarity_threshold <= 2.0) {
        return Err(format!(
            "--similarity must be in (0.0, 2.0], got {similarity_threshold}"
        ));
    }

    let emit_raw = cli
        .emit
        .or(layered.emit)
        .unwrap_or_else(|| "branch,json,report,viewer".to_owned());
    let mut emit_targets = parse_emit_targets(&emit_raw)?;
    if cli.dry_run {
        emit_targets.remove(&EmitTarget::Branch);
    }

    let branch_name = cli
        .branch
        .map(|b| b.trim().to_owned())
        .filter(|b| !b.is_empty());
    if let Some(ref b) = branch_name {
        validate_branch_name(b)?;
    }

    let branch_prefix = layered
        .branch_prefix
        .unwrap_or_else(|| "untangle/".to_owned());

    let verify_command = cli
        .verify
        .or(layered.verify_command)
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());

    Ok(ResolvedConfig {
        repo_dir,
        target_spec: cli.target.unwrap_or_else(|| "HEAD".to_owned()),
        base_override: cli.base,
        pathspecs: cli.pathspecs,
        use_llm,
        model,
        verify_command,
        verify_timeout_secs: cli
            .verify_timeout
            .or(layered.verify_timeout_secs)
            .unwrap_or(600)
            .clamp(10, 86_400),
        max_retries: cli
            .max_retries
            .or(layered.max_retries)
            .unwrap_or(10)
            .min(100),
        emit_targets,
        branch_name,
        branch_prefix,
        output_dir: cli.output_dir,
        dry_run: cli.dry_run,
        mechanical_placement,
        similarity_threshold,
        bulk_min: cli.bulk_min.or(layered.bulk_min).unwrap_or(3).clamp(2, 1000),
        max_cluster_size: cli
            .max_cluster_size
            .or(layered.max_cluster_size)
            .unwrap_or(40)
            .clamp(2, 1000),
        max_llm_ops: cli
            .max_llm_ops
            .or(layered.max_llm_ops)
            .unwrap_or(200)
            .min(2000),
        record_llm: cli.record_llm,
        replay_llm: cli.replay_llm,
        json_stdout: cli.json_stdout,
        quiet: cli.quiet,
        verbose: cli.verbose,
    })
}
