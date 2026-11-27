# `git-untangle`

> **Git is the execution substrate, Rust owns the invariants, and the LLM proposes semantic graph edits.**

`git-untangle` decomposes a large or tangled Git commit—or a multi-commit range (`A..B`, `A...B`)—into an ordered DAG of coherent, atomic operations, optionally refined by an LLM and verified via intermediate builds, while guaranteeing byte-for-byte final tree equivalence (`HEAD^{tree} == tip^{tree}`).

## Installation

### From Source (Cargo)

```bash
cargo install --path crates/commit-untangle-cli
```

This installs both `git-untangle` and `git-chat` onto `PATH`. Because Git automatically discovers `git-*` executables on `PATH`, you can invoke them immediately as Git subcommands:

```bash
git untangle HEAD --no-llm
git chat untangle main...feature --no-llm
git chat status   # ordinary Git commands forward transparently to `git`
```

### Shell Completions & Manpages

```bash
# Bash
git-untangle --generate-completions bash > ~/.local/share/bash-completion/completions/git-untangle

# Zsh
git-untangle --generate-completions zsh > ~/.zfunc/_git-untangle

# Fish
git-untangle --generate-completions fish > ~/.config/fish/completions/git-untangle.fish

# Manpages
git-untangle --generate-manpage > /usr/local/share/man/man1/git-untangle.1
git-chat --generate-manpage > /usr/local/share/man/man1/git-chat.1
```

## Quick Start

```bash
# Pure deterministic decomposition (no network / no API keys)
git untangle HEAD --no-llm

# Decompose a diverged branch range restricted to src/
git untangle main...HEAD --no-llm -- src/

# Verify every intermediate operation builds and auto-repair dependency order
git untangle HEAD --no-llm --verify "cargo check --quiet"

# Refine clusters, dependencies, and write commit messages with an LLM
export ANTHROPIC_API_KEY="sk-ant-..."
git untangle HEAD --model anthropic::claude-sonnet-4-5 --verify "cargo check --quiet"

# Interactive chat about revision history backed by an MCP tool server
git chat --model anthropic::claude-sonnet-4-5 log
git chat --model anthropic::claude-sonnet-4-5 --prompt "When was auth added?" log

# Autogenerate a commit message from staged changes and commit
git chat --model anthropic::claude-sonnet-4-5 commit
git chat --model anthropic::claude-sonnet-4-5 commit --dry-run

# Interactive chat about any Git command (direct output captured up to 20 KiB by default)
git chat --model anthropic::claude-sonnet-4-5 diff main..HEAD

# Configure default chat byte limit in Git config (or GIT_CHAT_MAX_BYTES)
git config untangle.chatMaxBytes 20480

# Record and replay LLM responses deterministically for CI / offline testing
git untangle HEAD --model gemini::gemini-2.5-pro --record-llm cassette.jsonl
git untangle HEAD --replay-llm cassette.jsonl
```

## Supported LLM Providers

Credentials are read strictly from environment variables (never `.git/config` or `config.toml`):

| Provider | Model Specifier | Environment Variable | Optional Base URL |
|---|---|---|---|
| Anthropic | `anthropic::claude-sonnet-4-5` | `ANTHROPIC_API_KEY` | `ANTHROPIC_BASE_URL` |
| Google Gemini | `gemini::gemini-2.5-pro` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `GEMINI_BASE_URL` |
| OpenAI | `openai::gpt-4.1` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| OpenRouter | `openrouter::anthropic/claude-sonnet-4` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| Ollama (local) | `ollama::qwen2.5-coder:32b` | _(none)_ | `OLLAMA_HOST` |

## Configuration Precedence

1. CLI flags (`--model`, `--verify`, `--similarity`, `--emit`, ...)
2. Repository `.git/config` (`git config untangle.verifyCommand "cargo check"`)
3. User global `~/.gitconfig` (`git config --global untangle.model gemini::gemini-2.5-pro`)
4. XDG tool config (`$XDG_CONFIG_HOME/git-untangle/config.toml`)
5. Built-in defaults

## Documentation

- [Algorithm Specification](algorithm.md) — Formal invariants, zero-context hunk algebra, clustering, SCC contraction, repair cascade, and DAG emission.
- [Software Design & Architecture](design.md) — 4-crate workspace layout, proposal algebra, and implementation milestones.
