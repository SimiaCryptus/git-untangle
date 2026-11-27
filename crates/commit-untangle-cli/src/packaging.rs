use clap::CommandFactory;

use crate::config::CliArgs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellKind {
    Bash,
    Zsh,
    Fish,
    PowerShell,
}

impl ShellKind {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bash" => Ok(Self::Bash),
            "zsh" => Ok(Self::Zsh),
            "fish" => Ok(Self::Fish),
            "powershell" | "pwsh" => Ok(Self::PowerShell),
            other => Err(format!(
                "unsupported shell '{other}' (expected bash, zsh, fish, or powershell)"
            )),
        }
    }
}

/// Generates shell completion scripts directly from the `clap::Command` metadata.
pub fn render_completions(shell: ShellKind) -> String {
    let cmd = CliArgs::command();
    let flags: Vec<(Option<char>, Option<String>, String)> = cmd
        .get_arguments()
        .filter(|a| !a.is_positional() && !a.is_hide_set())
        .map(|a| {
            let short = a.get_short();
            let long = a.get_long().map(|l| format!("--{l}"));
            let help = a
                .get_help()
                .map(|h| h.to_string().replace(['\'', '"', '\n'], " "))
                .unwrap_or_default();
            (short, long, help)
        })
        .collect();

    match shell {
        ShellKind::Bash => {
            let mut words = Vec::new();
            for (s, l, _) in &flags {
                if let Some(c) = s {
                    words.push(format!("-{c}"));
                }
                if let Some(long) = l {
                    words.push(long.clone());
                }
            }
            format!(
                "# Bash completion for git-untangle\n\
                 _git_untangle() {{\n\
                 \tlocal cur=\"${{COMP_WORDS[COMP_CWORD]}}\"\n\
                 \tlocal opts=\"{}\"\n\
                 \tif [[ \"$cur\" == -* ]]; then\n\
                 \t\tCOMPREPLY=( $(compgen -W \"$opts\" -- \"$cur\") )\n\
                 \telse\n\
                 \t\tlocal refs=$(git rev-parse --symbolic --branches --tags 2>/dev/null)\n\
                 \t\tCOMPREPLY=( $(compgen -W \"HEAD $refs\" -- \"$cur\") )\n\
                 \tfi\n\
                 }}\n\
                 complete -F _git_untangle git-untangle\n",
                words.join(" ")
            )
        }
        ShellKind::Zsh => {
            let mut specs = Vec::new();
            for (s, l, help) in &flags {
                match (s, l) {
                    (Some(c), Some(long)) => {
                        specs.push(format!("    '(-{c} {long})'{{-{c},{long}}}'[{help}]'"));
                    }
                    (None, Some(long)) => {
                        specs.push(format!("    '{long}[{help}]'"));
                    }
                    (Some(c), None) => {
                        specs.push(format!("    '-{c}[{help}]'"));
                    }
                    (None, None) => {}
                }
            }
            format!(
                "#compdef git-untangle\n\
                 _git_untangle() {{\n\
                 \t_arguments -s \\\n\
                 {} \\\n\
                 \t\t'1:revision:_git_revisions' \\\n\
                 \t\t'*:file:_files'\n\
                 }}\n\
                 _git_untangle \"$@\"\n",
                specs.join(" \\\n")
            )
        }
        ShellKind::Fish => {
            let mut lines = vec!["# Fish completion for git-untangle".to_owned()];
            for (s, l, help) in &flags {
                let mut line = String::from("complete -c git-untangle");
                if let Some(c) = s {
                    line.push_str(&format!(" -s {c}"));
                }
                if let Some(long) = l {
                    line.push_str(&format!(" -l {}", long.trim_start_matches("--")));
                }
                line.push_str(&format!(" -d '{help}'"));
                lines.push(line);
            }
            lines.join("\n") + "\n"
        }
        ShellKind::PowerShell => {
            let mut items = Vec::new();
            for (_, l, help) in &flags {
                if let Some(long) = l {
                    items.push(format!(
                        "        [System.Management.Automation.CompletionResult]::new('{long}', '{long}', 'ParameterName', '{help}')"
                    ));
                }
            }
            format!(
                "# PowerShell completion for git-untangle\n\
                 Register-ArgumentCompleter -Native -CommandName 'git-untangle' -ScriptBlock {{\n\
                 \tparam($wordToComplete, $commandAst, $cursorPosition)\n\
                 \t@(\n{}\n\t) | Where-Object {{ $_.CompletionText -like \"$wordToComplete*\" }}\n\
                 }}\n",
                items.join("\n")
            )
        }
    }
}

/// Generates a troff/roff manpage (`git-untangle.1`) from `CliArgs`.
pub fn render_manpage_untangle() -> String {
    let cmd = CliArgs::command();
    let version = env!("CARGO_PKG_VERSION");
    let mut out = format!(
        ".TH GIT-UNTANGLE 1 \"2026-11-27\" \"git-untangle {version}\" \"Git Manual\"\n\
         .SH NAME\n\
         git-untangle \\- decompose a commit or commit range into an ordered DAG of atomic operations\n\
         .SH SYNOPSIS\n\
         .B git untangle\n\
         [\\fIOPTIONS\\fR] [\\fIREVISION_OR_RANGE\\fR] [\\fB--\\fR \\fIPATHSPECS\\fR...]\n\
         .SH DESCRIPTION\n\
         \\fBgit-untangle\\fR analyzes the net tree delta of a commit (or range \\fIA..B\\fR / \\fIA...B\\fR), \
         separates mechanical edits (moves, file mode changes, whitespace, line reorders, bulk replacements) \
         and generated files from semantic hunks, clusters related changes, infers syntactic and optional LLM \
         semantic dependencies, optionally verifies intermediate builds with automatic graph repair, and \
         materializes a Git commit DAG whose final tree matches the original target byte-for-byte.\n\
         .SH OPTIONS\n"
    );

    for arg in cmd.get_arguments() {
        if arg.is_positional() || arg.is_hide_set() {
            continue;
        }
        let mut names = Vec::new();
        if let Some(s) = arg.get_short() {
            names.push(format!("\\fB\\-{s}\\fR"));
        }
        if let Some(l) = arg.get_long() {
            let escaped = l.replace('-', "\\-");
            names.push(format!("\\fB\\-\\-{escaped}\\fR"));
        }
        if names.is_empty() {
            continue;
        }
        let help = arg
            .get_help()
            .map(|h| h.to_string().replace('-', "\\-"))
            .unwrap_or_default();
        out.push_str(".TP\n");
        out.push_str(&names.join(", "));
        out.push('\n');
        out.push_str(&help);
        out.push('\n');
    }

    out.push_str(
        ".SH EXIT STATUS\n\
         .TP\n.B 0\nSuccess; byte-exact reconstruction verified.\n\
         .TP\n.B 2\nInvalid CLI usage, revision syntax, or configuration.\n\
         .TP\n.B 3\nRepository precondition failure.\n\
         .TP\n.B 4\nBuild verification retry budget exhausted (partial verification).\n\
         .TP\n.B 5\nInternal reconstruction invariant violation.\n\
         .SH CONFIGURATION\n\
         Settings resolve in order: CLI flags > repo .git/config (untangle.*) > global ~/.gitconfig > \
         $XDG_CONFIG_HOME/git-untangle/config.toml > defaults. API keys are read exclusively from environment variables.\n",
    );
    out
}

/// Generates a troff/roff manpage (`git-chat.1`).
pub fn render_manpage_chat() -> String {
    let version = env!("CARGO_PKG_VERSION");
    format!(
        ".TH GIT-CHAT 1 \"2026-11-27\" \"git-chat {version}\" \"Git Manual\"\n\
         .SH NAME\n\
         git-chat \\- interactive Git wrapper with semantic subcommands\n\
         .SH SYNOPSIS\n\
         .B git chat\n\
         [\\fICHAT_OPTIONS\\fR] \\fI<git-command | untangle>\\fR [\\fIARGS\\fR...]\n\
         .SH DESCRIPTION\n\
         \\fBgit-chat\\fR runs standard Git commands, echoes their direct output, and opens an interactive chat session \
         informed by that output (self-limited to 20480 bytes by default; configurable via \\fB\\-\\-max\\-bytes\\fR, \
         \\fBgit config untangle.chatMaxBytes\\fR, or \\fBGIT_CHAT_MAX_BYTES\\fR). \
         It also dispatches semantic subcommands such as \\fBgit chat untangle\\fR.\n\
         .SH OPTIONS\n\
         .TP\n\\fB\\-m\\fR, \\fB\\-\\-model\\fR \\fIPROVIDER::ID\\fR\nLLM provider and model.\n\
         .TP\n\\fB\\-\\-max\\-bytes\\fR \\fIN\\fR\nMaximum bytes of captured Git output passed into chat context (default: 20480).\n\
         .TP\n\\fB\\-p\\fR, \\fB\\-\\-prompt\\fR \\fIQUESTION\\fR\nAsk an initial question about the command output.\n\
         .TP\n\\fB\\-\\-one\\-shot\\fR\nExit after answering \\fB\\-\\-prompt\\fR.\n\
         .TP\n\\fB\\-\\-no\\-chat\\fR\nRun the Git command directly without starting a chat session.\n\
         .SH SEE ALSO\n\
         \\fBgit-untangle\\fR(1), \\fBgit\\fR(1)\n"
    )
}
