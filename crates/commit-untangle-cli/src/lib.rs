pub mod chat;
pub mod config;
pub mod packaging;
pub mod report;
pub mod viewer;

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use commit_untangle_core::cluster::{
    cluster_semantic_items, generated_operations, mechanical_operations,
};
use commit_untangle_core::graph::ChangeGraph;
use commit_untangle_core::model::{parse_and_classify, ChangeItem, FileChange, Operation};
use commit_untangle_core::proposal::{
    apply_cluster_proposal, normalize_entity_id, BuildDiagnosisProposal, ClusterRefinementProposal,
    CommitMessageDraft, CommitMessageProposal, DependencyProposal,
};
use commit_untangle_core::replay::verify_exact_reconstruction;
use commit_untangle_core::schema::build_analysis_artifact;
use commit_untangle_core::verify::{
    repair_deterministic_tiers, repair_with_diagnosis_or_fallback, VerificationReport,
};
use commit_untangle_core::{build_deterministic_plan, DeterministicPlanConfig};
use commit_untangle_git::worktree::{run_verify_command, TempWorktree, WorktreeReplayer};
use commit_untangle_git::{EmitOptions, GitBackend, GitCliBackend, GitError, TargetSpec};
use commit_untangle_llm::prompts::{
    batch_clusters, build_cluster_refinement_request, build_commit_message_batch_request,
    build_dependency_request, build_diagnosis_request, plan_commit_message_batches,
    CommitMessageBatch, DEFAULT_BATCH_CHAR_BUDGET,
};
use commit_untangle_llm::{
    generate_typed, LlmError, ProviderModel, RecordingModel, ReplayModel, SemanticModel,
};
use indexmap::IndexMap;
use tokio::sync::Semaphore;

use crate::config::{resolve_config, CliArgs, EmitTarget, ResolvedConfig};
use crate::packaging::{
    render_completions, render_manpage_chat, render_manpage_untangle, ShellKind,
};
use crate::report::{build_markdown_report, slugify};
use crate::viewer::render_viewer_html;

/// Exit codes per `design.md` §3.4.4.
pub const EXIT_OK: i32 = 0;
pub const EXIT_USAGE_OR_CONFIG: i32 = 2;
pub const EXIT_PRECONDITION: i32 = 3;
pub const EXIT_VERIFY_EXHAUSTED: i32 = 4;
pub const EXIT_INVARIANT_VIOLATION: i32 = 5;

pub fn entrypoint_untangle<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let collected: Vec<OsString> = args.into_iter().map(Into::into).collect();
    if let Some(arg0) = collected.first() {
        let stem = std::path::Path::new(arg0)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if stem == "git-chat" {
            return entrypoint_chat(collected);
        }
    }

    let cli = match CliArgs::try_parse_from(collected) {
        Ok(c) => c,
        Err(e) => {
            let kind = e.kind();
            let _ = e.print();
            return match kind {
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
                    EXIT_OK
                }
                _ => EXIT_USAGE_OR_CONFIG,
            };
        }
    };

    if let Some(ref shell_name) = cli.generate_completions {
        match ShellKind::parse(shell_name) {
            Ok(shell) => {
                let _ = write_stdout(&render_completions(shell));
                return EXIT_OK;
            }
            Err(msg) => {
                eprintln!("error: {msg}");
                return EXIT_USAGE_OR_CONFIG;
            }
        }
    }
    if cli.generate_manpage {
        let _ = write_stdout(&render_manpage_untangle());
        return EXIT_OK;
    }

    let cfg = match resolve_config(cli) {
        Ok(cfg) => cfg,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
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

    rt.block_on(run_untangle(cfg))
}

/// Syntax-compatible `git chat <cmd>` wrapper:
/// - `git chat untangle [args...]` -> runs `git-untangle`
/// - `git chat --generate-manpage` -> emits `git-chat.1`
/// - `git chat --help` / `git chat --version` -> prints wrapper info
/// - any other command -> forwards transparently to `git`
pub fn entrypoint_chat<I, T>(args: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let collected: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let sub = collected.get(1).and_then(|s| s.to_str()).unwrap_or("");

    match sub {
        "" | "-h" | "--help" | "help-chat" => {
            let _ = write_stdout(
                "git-chat — Interactive Git wrapper with semantic subcommands\n\n\
                 Usage:\n  \
                 git chat [CHAT_OPTIONS] <git-command> [GIT_ARGS...]\n      \
                 Runs <git-command>, displays its direct output, and opens an interactive chat session\n      \
                 informed by that output (self-limited to 20480 bytes by default; configurable via --max-bytes\n      \
                 or `git config untangle.chatMaxBytes`).\n\n  \
                 git chat untangle [OPTIONS] [REV_OR_RANGE] [-- <PATHS>...]\n      \
                 Decompose a commit or range into an ordered DAG of atomic operations\n\n\
                 Chat Options (place before <git-command>):\n  \
                 -m, --model <PROVIDER::ID>   LLM provider and model\n      \
                 --max-bytes <N>          Max bytes of Git output captured into chat context [default: 20480]\n  \
                 -p, --prompt <QUESTION>      Ask an initial question about the command output\n      \
                 --one-shot               Exit after answering --prompt\n      \
                 --no-chat                Run <git-command> directly without starting chat\n      \
                 --record-llm <PATH>      Record chat turns to a JSONL cassette\n      \
                 --replay-llm <PATH>      Replay chat turns from a JSONL cassette\n",
            );
            EXIT_OK
        }
        "-V" | "--version" => {
            let _ = write_stdout(&format!("git-chat {}\n", env!("CARGO_PKG_VERSION")));
            EXIT_OK
        }
        "--generate-manpage" => {
            let _ = write_stdout(&render_manpage_chat());
            EXIT_OK
        }
        "untangle" => {
            let mut forwarded = Vec::with_capacity(collected.len());
            forwarded.push(OsString::from("git-untangle"));
            forwarded.extend(collected.into_iter().skip(2));
            entrypoint_untangle(forwarded)
        }
        _ => match chat::parse_chat_invocation(&collected) {
            Ok(inv) => chat::run_chat_command(inv),
            Err(msg) => {
                eprintln!("error: {msg}");
                EXIT_USAGE_OR_CONFIG
            }
        },
    }
}

const MAX_CONCURRENT_COMMIT_MSG_BATCHES: usize = 3;

enum PlanBuildError {
    Core(commit_untangle_core::error::CoreError),
    FatalLlm(LlmError),
}

fn print_fatal_llm_error(err: &LlmError) {
    eprintln!("error: LLM request failed: {err}");
    if let Some(hint) = err.diagnostic_hint() {
        eprintln!("hint: {hint}");
    }
}

fn build_semantic_model(cfg: &ResolvedConfig) -> Result<Option<Arc<dyn SemanticModel>>, String> {
    if !cfg.use_llm {
        return Ok(None);
    }

    if let Some(ref replay_path) = cfg.replay_llm {
        let model = ReplayModel::from_jsonl_file(replay_path)
            .map_err(|e| format!("failed to load cassette '{}': {e}", replay_path.display()))?;
        return if let Some(ref rec_path) = cfg.record_llm {
            Ok(Some(Arc::new(RecordingModel::new(model, rec_path))))
        } else {
            Ok(Some(Arc::new(model)))
        };
    }

    let model_spec = cfg
        .model
        .as_deref()
        .ok_or_else(|| "LLM mode enabled without --model or untangle.model (pass --no-llm for deterministic mode)".to_owned())?;
    let live = ProviderModel::from_env(model_spec).map_err(|e| e.to_string())?;
    if let Some(ref rec_path) = cfg.record_llm {
        Ok(Some(Arc::new(RecordingModel::new(live, rec_path))))
    } else {
        Ok(Some(Arc::new(live)))
    }
}

pub async fn run_untangle(cfg: ResolvedConfig) -> i32 {
    let status = |msg: &str| {
        if !cfg.quiet && !cfg.json_stdout {
            eprintln!("{msg}");
        }
    };

    let semantic_model = match build_semantic_model(&cfg) {
        Ok(m) => m,
        Err(msg) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
        }
    };

    // Stage 0: Discover repository & resolve target
    let (backend, repo_info) = match GitCliBackend::discover(&cfg.repo_dir) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_PRECONDITION;
        }
    };

    let target_spec = TargetSpec {
        spec: cfg.target_spec.clone(),
        base_override: cfg.base_override.clone(),
        pathspecs: cfg.pathspecs.clone(),
    };

    let target = match backend.resolve_target(&target_spec).await {
        Ok(t) => t,
        Err(GitError::InvalidTarget(msg)) => {
            eprintln!("error: {msg}");
            return EXIT_USAGE_OR_CONFIG;
        }
        Err(e) => {
            eprintln!("error: {e}");
            return EXIT_PRECONDITION;
        }
    };

    status(&format!(
        "[git-untangle] Analyzing {} ({}..{})",
        target.short_id,
        &target.base[..10.min(target.base.len())],
        &target.tip[..10.min(target.tip.len())]
    ));

    // Stage 1: Extract & classify change items
    let deltas = match backend.extract_deltas(&target).await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: failed to extract git diff: {e}");
            return EXIT_PRECONDITION;
        }
    };

    let parsed = parse_and_classify(deltas, cfg.bulk_min);
    if parsed.items.is_empty() {
        status(&format!(
            "[git-untangle] Target {} contains no changes to untangle.",
            target.short_id
        ));
        if cfg.json_stdout {
            let _ = write_stdout(&format!(
                "{{\"commit\":{:?},\"parent\":{:?},\"operations\":[],\"exact_reconstruction\":true}}\n",
                target.tip, target.base
            ));
        }
        return EXIT_OK;
    }

    status(&format!(
        "[git-untangle] Stage 1: parsed {} file(s) into {} change item(s)",
        parsed.files.len(),
        parsed.items.len()
    ));

    // Stage 2 & Stage 3: Cluster & infer dependencies (deterministic + optional LLM)
    let (mut graph, mut order, llm_ranks) = if let Some(ref model) = semantic_model {
        match build_model_refined_plan(
            &parsed.files,
            &parsed.items,
            &target.message,
            &cfg,
            model.as_ref(),
        )
        .await
        {
            Ok(res) => res,
            Err(PlanBuildError::FatalLlm(err)) => {
                print_fatal_llm_error(&err);
                return EXIT_USAGE_OR_CONFIG;
            }
            Err(PlanBuildError::Core(e)) => {
                eprintln!("error: plan construction failed: {e}");
                return EXIT_INVARIANT_VIOLATION;
            }
        }
    } else {
        let plan_cfg = DeterministicPlanConfig {
            similarity_threshold: cfg.similarity_threshold,
            max_cluster_size: cfg.max_cluster_size,
            mechanical_placement: cfg.mechanical_placement,
        };
        match build_deterministic_plan(&parsed.files, &parsed.items, &plan_cfg) {
            Ok(plan) => (plan.graph, plan.order, IndexMap::new()),
            Err(e) => {
                eprintln!("error: deterministic plan construction failed: {e}");
                return EXIT_INVARIANT_VIOLATION;
            }
        }
    };

    status(&format!(
        "[git-untangle] Stage 2-3: built DAG with {} operation(s) and {} dependency edge(s)",
        order.len(),
        graph.edges.len()
    ));

    // Stage 4: Closed-loop build verification & 4-tier repair cascade
    let mut verify_exhausted = false;
    let verification = if let Some(ref verify_cmd) = cfg.verify_command {
        status(&format!(
            "[git-untangle] Stage 4: verifying intermediate states with `{verify_cmd}`"
        ));
        match verify_graph_in_worktree(
            &repo_info.toplevel,
            &repo_info.prefix,
            &target.base,
            &target.message,
            verify_cmd,
            &cfg,
            &mut graph,
            &llm_ranks,
            &parsed.files,
            &parsed.items,
            semantic_model.as_deref(),
        )
        .await
        {
            Ok((rep, exhausted)) => {
                verify_exhausted = exhausted;
                match graph.topological_order(cfg.mechanical_placement, &llm_ranks, &parsed.items) {
                    Ok(new_order) => order = new_order,
                    Err(e) => {
                        eprintln!("error: post-verification topological sort failed: {e}");
                        return EXIT_INVARIANT_VIOLATION;
                    }
                }
                status(&format!(
                    "[git-untangle] Stage 4 verification: {}",
                    rep.status
                ));
                rep
            }
            Err(e) => {
                eprintln!("error: build verification setup failed: {e}");
                return EXIT_PRECONDITION;
            }
        }
    } else {
        VerificationReport::default()
    };

    // Stage 4B: LLM-authored commit messages when LLM is active
    let commit_messages = if let Some(ref model) = semantic_model {
        let planned_batches =
            plan_commit_message_batches(&order, &parsed.items, DEFAULT_BATCH_CHAR_BUDGET);
        if planned_batches.len() > 1 {
            status(&format!(
                "[git-untangle] Authoring commit messages for {} operation(s) ({} item(s) across {} LLM batch(es))",
                order.len(),
                parsed.items.len(),
                planned_batches.len()
            ));
        } else {
            status(&format!(
                "[git-untangle] Authoring commit messages for {} operation(s) with LLM",
                order.len()
            ));
        }
        let drafted = match author_commit_messages(
            &planned_batches,
            &target.message,
            &parsed.items,
            Arc::clone(model),
            cfg.quiet,
        )
        .await
        {
            Ok(d) => d,
            Err(fatal_err) => {
                print_fatal_llm_error(&fatal_err);
                return EXIT_USAGE_OR_CONFIG;
            }
        };

        let mut authored_ops = 0usize;
        let mut authored_items = 0usize;
        for op in &mut order {
            if let Some(d) = drafted.get(&op.id) {
                let has_subj = !d.subject.trim().is_empty();
                let has_body = !d.body.trim().is_empty();
                if has_subj {
                    op.name = d
                        .subject
                        .trim()
                        .trim_end_matches('.')
                        .chars()
                        .take(120)
                        .collect();
                }
                if has_body {
                    op.description = d.body.trim().to_owned();
                }
                if has_subj || has_body || !d.item_subjects.is_empty() {
                    authored_ops += 1;
                    op.confidence = op.confidence.max(0.75);
                }
                for item_id in &op.item_ids {
                    if d.item_subjects.contains_key(item_id)
                        || (op.item_ids.len() == 1 && has_subj)
                        || d.item_bodies.contains_key(item_id)
                        || has_body
                    {
                        authored_items += 1;
                    }
                }
            }
        }
        for op in &mut graph.ops {
            if let Some(d) = drafted.get(&op.id) {
                let has_subj = !d.subject.trim().is_empty();
                let has_body = !d.body.trim().is_empty();
                if has_subj {
                    op.name = d
                        .subject
                        .trim()
                        .trim_end_matches('.')
                        .chars()
                        .take(120)
                        .collect();
                }
                if has_body {
                    op.description = d.body.trim().to_owned();
                }
                if has_subj || has_body || !d.item_subjects.is_empty() {
                    op.confidence = op.confidence.max(0.75);
                }
            }
        }
        if authored_ops == 0 && !order.is_empty() && !cfg.quiet {
            eprintln!(
                "[git-untangle] warning: LLM produced 0 usable commit descriptions; falling back to heuristic messages"
            );
        } else {
            status(&format!(
                "[git-untangle] Authored LLM commit descriptions for {authored_ops}/{} operation(s) ({authored_items}/{} item commit(s))",
                order.len(),
                parsed.items.len()
            ));
        }
        drafted
    } else {
        IndexMap::new()
    };

    // Stage 5: Materialize Commit DAG & verify tree equivalence
    let emit_patches = cfg.emit_targets.contains(&EmitTarget::Patches);
    let emit_viewer = cfg.emit_targets.contains(&EmitTarget::Viewer);
    let emit_json = cfg.emit_targets.contains(&EmitTarget::Json);
    let emit_report = cfg.emit_targets.contains(&EmitTarget::Report);
    let emit_branch = cfg.emit_targets.contains(&EmitTarget::Branch) && !cfg.dry_run;

    let emit_opts = EmitOptions {
        dry_run: cfg.dry_run,
        emit_branch,
        branch_name: cfg.branch_name.clone(),
        branch_prefix: cfg.branch_prefix.clone(),
        emit_patches,
        emit_item_diffs: emit_patches || emit_viewer,
        commit_messages,
    };

    let mut dag_res = match backend
        .materialize_dag(
            &target,
            &graph,
            &order,
            &parsed.files,
            &parsed.items,
            &emit_opts,
        )
        .await
    {
        Ok(res) => res,
        Err(e) => {
            eprintln!("error: failed to materialize commit DAG: {e}");
            return EXIT_INVARIANT_VIOLATION;
        }
    };

    // In-memory exact reconstruction check (Invariant I3)
    let mismatches = verify_exact_reconstruction(&parsed.files, &parsed.items);
    let exact_reconstruction = mismatches.is_empty();

    let out_dir: PathBuf = cfg
        .output_dir
        .clone()
        .unwrap_or_else(|| repo_info.git_dir.join("untangle").join(&target.short_id));

    let base_prefix = format!("untangle_{}", target.short_id);
    let needs_out_dir = emit_json || emit_report || emit_patches || emit_viewer;
    if needs_out_dir {
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            eprintln!(
                "error: could not create output directory '{}': {e}",
                out_dir.display()
            );
            return EXIT_PRECONDITION;
        }
    }

    if emit_patches {
        for (item_id, patch_bytes) in &dag_res.item_patches {
            if let Some(item) = parsed.items.get(item_id) {
                let fname = format!(
                    "{base_prefix}_item_{item_id}_{}.patch",
                    slugify(&item.display_path)
                );
                let path = out_dir.join(&fname);
                if std::fs::write(&path, patch_bytes.as_slice()).is_ok() {
                    dag_res
                        .metadata
                        .item_diff_files
                        .insert(item_id.clone(), fname);
                }
            }
        }
        for (idx, (op_name, patch_bytes)) in dag_res.operation_patches.iter().enumerate() {
            let fname = format!("{base_prefix}_{:04}_{}.patch", idx + 1, slugify(op_name));
            let _ = std::fs::write(out_dir.join(fname), patch_bytes.as_slice());
        }
    }

    let artifact = build_analysis_artifact(
        &target.tip,
        &target.base,
        &target.subject,
        target.to_range_artifact(),
        exact_reconstruction,
        &graph,
        &order,
        &parsed.items,
        &verification,
        &dag_res.metadata,
    );

    let graph_json = serde_json::to_string_pretty(&artifact).unwrap_or_else(|_| "{}".to_owned());
    let json_filename = format!("{base_prefix}_graph.json");

    if emit_json {
        let _ = std::fs::write(out_dir.join(&json_filename), graph_json.as_bytes());
    }

    if emit_report {
        let md = build_markdown_report(
            &target,
            &parsed.files,
            &parsed.items,
            &graph,
            &order,
            &verification,
            &dag_res.metadata,
            &parsed.notes,
            &mismatches,
        );
        let _ = std::fs::write(
            out_dir.join(format!("{base_prefix}_report.md")),
            md.as_bytes(),
        );
    }

    if emit_viewer {
        let html = render_viewer_html(
            &json_filename,
            &target.short_id,
            &target.subject,
            &graph_json,
        );
        let _ = std::fs::write(
            out_dir.join(format!("{base_prefix}_viewer.html")),
            html.as_bytes(),
        );
    }

    if let Some(ref b) = dag_res.metadata.branch_created {
        status(&format!(
            "[git-untangle] Created branch `{b}` ({} item commit(s), {} merge commit(s))",
            dag_res.metadata.item_commits.len(),
            dag_res.metadata.merge_commits.len()
        ));
    }
    if needs_out_dir {
        status(&format!(
            "[git-untangle] Artifacts written to {}",
            out_dir.display()
        ));
    }

    if cfg.json_stdout {
        let _ = write_stdout(&format!("{graph_json}\n"));
    }

    if !exact_reconstruction || dag_res.metadata.tree_matches == Some(false) {
        eprintln!("error: reconstruction invariant check failed");
        return EXIT_INVARIANT_VIOLATION;
    }

    if verify_exhausted {
        return EXIT_VERIFY_EXHAUSTED;
    }

    EXIT_OK
}

async fn build_model_refined_plan(
    files: &[FileChange],
    items: &IndexMap<String, ChangeItem>,
    commit_context: &str,
    cfg: &ResolvedConfig,
    model: &dyn SemanticModel,
) -> Result<(ChangeGraph, Vec<Operation>, IndexMap<String, usize>), PlanBuildError> {
    let mechanical_ops = mechanical_operations(items, files);
    let generated_ops = generated_operations(items, files);

    let semantic_refs: Vec<&ChangeItem> = items
        .values()
        .filter(|it| !it.kind.is_mechanical() && !files[it.file_index].generated)
        .collect();
    let raw_clusters = cluster_semantic_items(
        &semantic_refs,
        cfg.similarity_threshold,
        cfg.max_cluster_size,
    );
    let cluster_ids: Vec<Vec<String>> = raw_clusters
        .into_iter()
        .map(|c| c.into_iter().map(|it| it.id.clone()).collect())
        .collect();

    let mut semantic_ops = Vec::new();
    let batches = batch_clusters(&cluster_ids, items, DEFAULT_BATCH_CHAR_BUDGET);
    for (batch_idx, batch) in batches.into_iter().enumerate() {
        let req = build_cluster_refinement_request(&batch, commit_context, items);
        let proposal = match generate_typed::<ClusterRefinementProposal>(model, req).await {
            Ok(p) => p,
            Err(err) if err.is_fatal_config_or_auth() => {
                return Err(PlanBuildError::FatalLlm(err));
            }
            Err(err) => {
                if !cfg.quiet {
                    eprintln!(
                        "[git-untangle] warning: LLM cluster refinement batch {} failed ({err}); using deterministic clusters",
                        batch_idx + 1
                    );
                }
                ClusterRefinementProposal::default()
            }
        };
        semantic_ops.extend(
            apply_cluster_proposal(&batch, &proposal, items).map_err(PlanBuildError::Core)?,
        );
    }

    semantic_ops.sort_by_key(|op| op.position(items));

    let mut all_ops =
        Vec::with_capacity(mechanical_ops.len() + semantic_ops.len() + generated_ops.len());
    all_ops.extend(mechanical_ops);
    all_ops.extend(semantic_ops);
    all_ops.extend(generated_ops);
    for (idx, op) in all_ops.iter_mut().enumerate() {
        op.id = format!("op{}", idx + 1);
    }

    let mut graph = ChangeGraph::new(all_ops);
    graph.add_deterministic_edges(items);
    graph.merge_cycles();

    let mut llm_ranks = IndexMap::new();
    if graph.ops.len() >= 2 && graph.ops.len() <= cfg.max_llm_ops {
        let req = build_dependency_request(&graph, commit_context, items);
        match generate_typed::<DependencyProposal>(model, req).await {
            Ok(prop) => {
                let (_, _, ranks) = graph.apply_dependency_proposal(&prop);
                llm_ranks = ranks;
            }
            Err(err) if err.is_fatal_config_or_auth() => {
                return Err(PlanBuildError::FatalLlm(err));
            }
            Err(err) => {
                if !cfg.quiet {
                    eprintln!(
                        "[git-untangle] warning: LLM dependency inference failed ({err}); using syntactic dependencies"
                    );
                }
            }
        }
    }

    let order = graph
        .topological_order(cfg.mechanical_placement, &llm_ranks, items)
        .map_err(PlanBuildError::Core)?;
    Ok((graph, order, llm_ranks))
}

#[allow(clippy::too_many_arguments)]
async fn verify_graph_in_worktree(
    repo_root: &std::path::Path,
    repo_prefix: &str,
    base_commit: &str,
    commit_context: &str,
    verify_cmd: &str,
    cfg: &ResolvedConfig,
    graph: &mut ChangeGraph,
    llm_ranks: &IndexMap<String, usize>,
    files: &[FileChange],
    items_by_id: &IndexMap<String, ChangeItem>,
    model: Option<&dyn SemanticModel>,
) -> Result<(VerificationReport, bool), GitError> {
    let wt = TempWorktree::open(repo_root, base_commit)?;
    let build_dir = if repo_prefix.is_empty() {
        wt.path().to_path_buf()
    } else {
        wt.path().join(repo_prefix)
    };
    let mut replayer =
        WorktreeReplayer::new(wt.path().to_path_buf(), files.to_vec(), items_by_id.clone())?;

    let mut report = VerificationReport::default();

    // Baseline check (Assumption A0)
    replayer.sync(&BTreeSet::new(), None)?;
    let (base_ok, _) = run_verify_command(verify_cmd, &build_dir, cfg.verify_timeout_secs)?;
    if !base_ok {
        report.status = "skipped: parent does not build (assumption A0 violated)".to_owned();
        report
            .steps
            .push("Baseline parent build failed; skipping intermediate verification".to_owned());
        return Ok((report, false));
    }

    let mut attempt = 0usize;
    loop {
        let order = graph.topological_order(cfg.mechanical_placement, llm_ranks, items_by_id)?;
        for op in &mut graph.ops {
            op.verified = None;
        }
        replayer.sync(&BTreeSet::new(), None)?;

        let mut applied = BTreeSet::new();
        let mut failure: Option<(usize, String)> = None;

        // Final operation reproduces tip (assumed valid per A1); verify prefixes 0..len-1
        let prefix_len = order.len().saturating_sub(1);
        for (k, op) in order.iter().enumerate().take(prefix_len) {
            for item_id in &op.item_ids {
                applied.insert(item_id.clone());
            }
            let touched_indices: Vec<usize> = op
                .item_ids
                .iter()
                .filter_map(|id| items_by_id.get(id).map(|it| it.file_index))
                .collect();
            let changed = replayer.sync(&applied, Some(&touched_indices))?;
            if changed.is_empty() {
                if let Some(g_op) = graph.op_mut(&op.id) {
                    g_op.verified = Some(true);
                }
                continue;
            }

            let (ok, output) = run_verify_command(verify_cmd, &build_dir, cfg.verify_timeout_secs)?;
            if ok {
                if let Some(g_op) = graph.op_mut(&op.id) {
                    g_op.verified = Some(true);
                    g_op.confidence = g_op.confidence.max(0.9);
                }
            } else {
                if let Some(g_op) = graph.op_mut(&op.id) {
                    g_op.verified = Some(false);
                }
                failure = Some((k, output));
                break;
            }
        }

        let Some((fail_idx, build_out)) = failure else {
            if let Some(last) = order.last() {
                if let Some(g_op) = graph.op_mut(&last.id) {
                    g_op.verified = Some(true);
                }
            }
            report.status = "passed".to_owned();
            report.steps.push(format!(
                "Attempt {}: all {} intermediate states build",
                attempt + 1,
                order.len()
            ));
            return Ok((report, false));
        };

        let failing = &order[fail_idx];
        report.steps.push(format!(
            "Attempt {}: build failed after applying {} ({})",
            attempt + 1,
            failing.id,
            failing.name
        ));

        attempt += 1;
        if attempt > cfg.max_retries {
            report.status = format!(
                "failed: retry budget exhausted (best partial result: first {fail_idx} operation(s) verified)"
            );
            return Ok((report, true));
        }

        let later = &order[fail_idx + 1..];
        let fixed_det =
            repair_deterministic_tiers(graph, failing, later, &build_out, items_by_id, &mut report);

        if !fixed_det {
            let diagnosis = if let (Some(m), false) = (model, later.is_empty()) {
                let req = build_diagnosis_request(
                    failing,
                    later,
                    &build_out,
                    commit_context,
                    items_by_id,
                );
                generate_typed::<BuildDiagnosisProposal>(m, req).await.ok()
            } else {
                None
            };

            let fixed_tail = repair_with_diagnosis_or_fallback(
                graph,
                &failing.id,
                later,
                diagnosis.as_ref(),
                &mut report,
            );
            if !fixed_tail {
                report.status =
                    "failed: no dependency could be identified for the failing state".to_owned();
                return Ok((report, true));
            }
        }
    }
}

fn extract_batch_messages(
    expected_op_ids: &[String],
    mut prop: CommitMessageProposal,
) -> IndexMap<String, CommitMessageDraft> {
    let expected_set: BTreeSet<&str> = expected_op_ids.iter().map(String::as_str).collect();
    let mut matched = IndexMap::new();

    for (raw_id, draft) in prop.messages.drain(..) {
        let norm = normalize_entity_id(&raw_id);
        if expected_set.contains(norm.as_str()) {
            matched
                .entry(norm)
                .and_modify(|existing: &mut CommitMessageDraft| existing.merge_from(draft.clone()))
                .or_insert(draft);
        } else {
            // Reinsert unmatched in case single-op fallback remap applies below
            matched.insert(norm, draft);
        }
    }

    let any_expected = expected_op_ids
        .iter()
        .any(|id| matched.contains_key(id.as_str()));
    if !any_expected && expected_op_ids.len() == 1 && matched.len() == 1 {
        let (_, single_draft) = matched.into_iter().next().expect("len == 1");
        let mut remapped = IndexMap::new();
        remapped.insert(expected_op_ids[0].clone(), single_draft);
        return remapped;
    }

    matched.retain(|k, _| expected_set.contains(k.as_str()));
    matched
}

async fn run_single_commit_message_batch(
    batch: CommitMessageBatch,
    commit_context: &str,
    items: &IndexMap<String, ChangeItem>,
    model: &dyn SemanticModel,
) -> Result<IndexMap<String, CommitMessageDraft>, LlmError> {
    let expected = batch.expected_op_ids();
    let req = build_commit_message_batch_request(&batch, commit_context, items);
    let prop = generate_typed::<CommitMessageProposal>(model, req.clone()).await?;
    let matched = extract_batch_messages(&expected, prop);
    if !matched.is_empty() || expected.is_empty() {
        return Ok(matched);
    }

    // 1-shot key-alignment repair if the model returned valid JSON without matching operation IDs
    let mut repair_req = req;
    repair_req.user_prompt.push_str(&format!(
        "\n\nYour previous reply did not contain entries keyed by the required operation ID(s): {}. Return ONLY JSON with `messages` keyed by those exact operation IDs.",
        expected.join(", ")
    ));
    match generate_typed::<CommitMessageProposal>(model, repair_req).await {
        Ok(repaired) => Ok(extract_batch_messages(&expected, repaired)),
        Err(LlmError::CassetteMiss { .. }) => Ok(IndexMap::new()),
        Err(e) => Err(e),
    }
}

async fn author_commit_messages(
    batches: &[CommitMessageBatch],
    commit_context: &str,
    items: &IndexMap<String, ChangeItem>,
    model: Arc<dyn SemanticModel>,
    quiet: bool,
) -> Result<IndexMap<String, CommitMessageDraft>, LlmError> {
    let mut drafted: IndexMap<String, CommitMessageDraft> = IndexMap::new();
    if batches.is_empty() {
        return Ok(drafted);
    }

    if batches.len() == 1 {
        match run_single_commit_message_batch(
            batches[0].clone(),
            commit_context,
            items,
            model.as_ref(),
        )
        .await
        {
            Ok(batch_map) => {
                for (op_id, draft) in batch_map {
                    drafted
                        .entry(op_id)
                        .and_modify(|existing| existing.merge_from(draft.clone()))
                        .or_insert(draft);
                }
            }
            Err(err) if err.is_fatal_config_or_auth() => return Err(err),
            Err(err) => {
                if !quiet {
                    eprintln!(
                        "[git-untangle] warning: LLM commit message generation failed ({err})"
                    );
                }
            }
        }
        return Ok(drafted);
    }

    let sem = Arc::new(Semaphore::new(MAX_CONCURRENT_COMMIT_MSG_BATCHES));
    let shared_items = Arc::new(items.clone());
    let shared_ctx = Arc::new(commit_context.to_owned());
    let mut handles = Vec::with_capacity(batches.len());

    for (idx, batch) in batches.iter().cloned().enumerate() {
        let permit = Arc::clone(&sem)
            .acquire_owned()
            .await
            .expect("semaphore open");
        let m = Arc::clone(&model);
        let it = Arc::clone(&shared_items);
        let ctx = Arc::clone(&shared_ctx);
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let res = run_single_commit_message_batch(batch, &ctx, &it, m.as_ref()).await;
            (idx, res)
        }));
    }

    for handle in handles {
        let Ok((idx, res)) = handle.await else {
            continue;
        };
        match res {
            Ok(batch_map) => {
                for (op_id, draft) in batch_map {
                    drafted
                        .entry(op_id)
                        .and_modify(|existing| existing.merge_from(draft.clone()))
                        .or_insert(draft);
                }
            }
            Err(err) if err.is_fatal_config_or_auth() => return Err(err),
            Err(err) => {
                if !quiet {
                    eprintln!(
                        "[git-untangle] warning: LLM commit message batch {} failed ({err})",
                        idx + 1
                    );
                }
            }
        }
    }

    Ok(drafted)
}

fn write_stdout(s: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    match out.write_all(s.as_bytes()).and_then(|()| out.flush()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e),
    }
}
