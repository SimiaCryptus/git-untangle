use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use commit_untangle_cli::chat::{
    build_initial_chat_messages, truncate_utf8_bytes, CapturedGitOutput,
};
use commit_untangle_core::model::{parse_and_classify, Operation};
use commit_untangle_core::schema::AnalysisArtifact;
use commit_untangle_git::tempdir::TempDir;
use commit_untangle_git::{GitBackend, GitCliBackend, TargetSpec};
use commit_untangle_llm::prompts::{
    build_cluster_refinement_request, build_commit_messages_request,
};
use commit_untangle_llm::{CassetteEntry, ChatMessage, ChatRequest, ChatRole, SemanticTask};

fn bin_git_untangle() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_git-untangle"))
}

fn bin_git_chat() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_git-chat"))
}

fn git(repo: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .current_dir(repo)
        .args(args)
        .env("GIT_AUTHOR_NAME", "Test Author")
        .env("GIT_AUTHOR_EMAIL", "author@example.com")
        .env("GIT_COMMITTER_NAME", "Test Author")
        .env("GIT_COMMITTER_EMAIL", "author@example.com")
        .output()
        .expect("git command should spawn");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

fn init_repo() -> TempDir {
    let tmp = TempDir::new("git-untangle-it").expect("tempdir");
    git(tmp.path(), &["init", "-b", "main"]);
    git(tmp.path(), &["config", "user.name", "Test Author"]);
    git(tmp.path(), &["config", "user.email", "author@example.com"]);
    tmp
}

#[test]
fn test_tangled_commit_rename_chmod_symlink_binary_and_dag_merges() {
    let repo = init_repo();
    let r = repo.path();

    // Base commit:
    // - src/legacy.rs (will be renamed to src/engine.rs AND edited AND bulk-replaced)
    // - scripts/deploy.sh (mode 100644 -> 100755 chmod)
    // - src/a.rs & src/b.rs (bulk replace `old_tok` -> `new_tok` + import reorder)
    // - CRLF + missing EOF newline file (`win.txt`)
    fs::create_dir_all(r.join("src")).unwrap();
    fs::create_dir_all(r.join("scripts")).unwrap();

    fs::write(
        r.join("src/legacy.rs"),
        "use crate::z;\nuse crate::a;\n\npub fn run() {\n    let x = old_tok();\n    println!(\"ok\");\n}\n",
    )
    .unwrap();
    fs::write(r.join("scripts/deploy.sh"), "#!/bin/sh\necho deploy\n").unwrap();
    fs::write(r.join("src/a.rs"), "pub fn fa() { old_tok(); }\n").unwrap();
    fs::write(r.join("src/b.rs"), "pub fn fb() { old_tok(); }\n").unwrap();
    fs::write(r.join("win.txt"), b"line1\r\nline2_no_eof").unwrap();

    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "Initial baseline"]);

    // Tangled commit mixing:
    // 1. Rename `src/legacy.rs` -> `src/engine.rs` + import reorder + bulk replace + whitespace + new call to `compute_hash`
    // 2. Executable permission `chmod +x scripts/deploy.sh` (with ZERO content change!)
    // 3. Symlink `latest-deploy` -> `scripts/deploy.sh`
    // 4. Binary file `assets/logo.bin` containing NUL bytes
    // 5. Generated lockfile `Cargo.lock`
    // 6. New module `src/hash.rs` defining `compute_hash` (creates syntactic def-before-use edge!)
    // 7. CRLF + unterminated EOF edit in `win.txt`
    git(r, &["mv", "src/legacy.rs", "src/engine.rs"]);
    fs::write(
        r.join("src/engine.rs"),
        "use crate::a;\nuse crate::z;\n\npub fn run() {\n    let x = new_tok();\n    let _h = compute_hash(x);\n    println!(  \"ok\"  );\n}\n",
    )
    .unwrap();

    #[cfg(unix)]
    {
        let mut perms = fs::metadata(r.join("scripts/deploy.sh"))
            .unwrap()
            .permissions();
        perms.set_mode(0o755);
        fs::set_permissions(r.join("scripts/deploy.sh"), perms).unwrap();

        std::os::unix::fs::symlink("scripts/deploy.sh", r.join("latest-deploy")).unwrap();
    }

    fs::create_dir_all(r.join("assets")).unwrap();
    fs::write(
        r.join("assets/logo.bin"),
        [0x89, b'P', b'N', b'G', 0x00, 0xff, 0x10],
    )
    .unwrap();
    fs::write(r.join("Cargo.lock"), "# generated\nversion = 3\n").unwrap();
    fs::write(
        r.join("src/hash.rs"),
        "pub fn compute_hash(v: usize) -> usize { v ^ 0x55 }\n",
    )
    .unwrap();
    fs::write(r.join("src/a.rs"), "pub fn fa() { new_tok(); }\n").unwrap();
    fs::write(r.join("src/b.rs"), "pub fn fb() { new_tok(); }\n").unwrap();
    fs::write(r.join("win.txt"), b"line1_edited\r\nline2_still_no_eof").unwrap();

    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "WIP: huge tangled commit"]);
    let tip_tree = git(r, &["rev-parse", "HEAD^{tree}"]);

    let out_dir = r.join("untangle-out");
    let out = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--no-llm",
            "--similarity",
            "1.1",
            "--emit",
            "branch,json,report,patches,viewer",
            "--branch",
            "untangled/head",
            "--output-dir",
            out_dir.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("git-untangle should execute");

    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let artifact: AnalysisArtifact =
        serde_json::from_slice(&out.stdout).expect("stdout must be valid AnalysisArtifact JSON");
    assert!(artifact.exact_reconstruction);
    assert_eq!(artifact.series.tree_matches, Some(true));
    assert_eq!(artifact.series.branch.as_deref(), Some("untangled/head"));
    assert!(artifact.operations.len() >= 5);
    assert!(
        !artifact.series.merge_commits.is_empty(),
        "independent branches should be joined by merge commit(s)"
    );

    // Verify Git tree SHA of the untangled branch matches original HEAD^{tree} BIT-FOR-BIT (including chmod +x, symlink, binary, CRLF!)
    let branch_tree = git(r, &["rev-parse", "untangled/head^{tree}"]);
    assert_eq!(
        branch_tree, tip_tree,
        "branch tree OID must match tip^{{tree}} identically"
    );

    // Verify generated artifacts exist and viewer embedded config is valid
    let entries: Vec<String> = fs::read_dir(&out_dir)
        .unwrap()
        .filter_map(|e| e.ok().and_then(|d| d.file_name().into_string().ok()))
        .collect();
    assert!(entries.iter().any(|n| n.ends_with("_graph.json")));
    assert!(entries.iter().any(|n| n.ends_with("_report.md")));
    assert!(entries.iter().any(|n| n.ends_with("_viewer.html")));
    assert!(entries.iter().any(|n| n.ends_with(".patch")));

    let viewer_file = entries
        .iter()
        .find(|n| n.ends_with("_viewer.html"))
        .unwrap();
    let viewer_html = fs::read_to_string(out_dir.join(viewer_file)).unwrap();
    assert!(viewer_html.contains("untangled/head"));
}

#[test]
fn test_git_chat_wrapper_passthrough_and_symmetric_range() {
    let repo = init_repo();
    let r = repo.path();

    fs::write(r.join("base.txt"), "base\n").unwrap();
    git(r, &["add", "base.txt"]);
    git(r, &["commit", "-m", "base"]);

    // Diverge `main` and `feature` to test symmetric `main...feature` merge-base resolution!
    git(r, &["checkout", "-b", "feature"]);
    fs::write(r.join("feat1.rs"), "pub fn alpha() -> i32 { 1 }\n").unwrap();
    git(r, &["add", "feat1.rs"]);
    git(r, &["commit", "-m", "feat part 1"]);

    fs::write(r.join("feat2.rs"), "pub fn beta() -> i32 { alpha() + 2 }\n").unwrap();
    git(r, &["add", "feat2.rs"]);
    git(r, &["commit", "-m", "feat part 2"]);
    let feature_tree = git(r, &["rev-parse", "feature^{tree}"]);

    git(r, &["checkout", "main"]);
    fs::write(r.join("main_only.txt"), "diverged on main\n").unwrap();
    git(r, &["add", "main_only.txt"]);
    git(r, &["commit", "-m", "main diverged"]);

    // 1. Test `git-chat` passthrough for standard Git command (`rev-parse --abbrev-ref HEAD`)
    let pass_out = Command::new(bin_git_chat())
        .current_dir(r)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .unwrap();
    assert!(pass_out.status.success());
    assert_eq!(String::from_utf8_lossy(&pass_out.stdout).trim(), "main");

    // 2. Test `git-chat untangle main...feature --no-llm`
    let untangle_out = Command::new(bin_git_chat())
        .current_dir(r)
        .args([
            "untangle",
            "main...feature",
            "--no-llm",
            "--branch",
            "untangled/feature-range",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        untangle_out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&untangle_out.stderr)
    );

    let artifact: AnalysisArtifact = serde_json::from_slice(&untangle_out.stdout).unwrap();
    let range = artifact.range.expect("range metadata present");
    assert!(range.symmetric);
    assert_eq!(range.commits.len(), 2);
    assert_eq!(artifact.series.tree_matches, Some(true));

    let untangled_tree = git(r, &["rev-parse", "untangled/feature-range^{tree}"]);
    assert_eq!(untangled_tree, feature_tree);
}

#[test]
fn test_config_layering_security_guard_dry_run_and_cassette_replay() {
    let repo = init_repo();
    let r = repo.path();

    fs::write(r.join("one.rs"), "pub fn one() {}\n").unwrap();
    git(r, &["add", "one.rs"]);
    git(r, &["commit", "-m", "c1"]);

    fs::write(r.join("one.rs"), "pub fn one() { let _ = 1; }\n").unwrap();
    fs::write(r.join("two.rs"), "pub fn two() { let _ = 2; }\n").unwrap();
    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "c2"]);

    // 1. Security check: storing an API key in `.git/config` must fail with EXIT_USAGE_OR_CONFIG (2)
    git(
        r,
        &["config", "--local", "untangle.apiKey", "sk-secret-leak"],
    );
    let bad_cfg = Command::new(bin_git_untangle())
        .current_dir(r)
        .args(["HEAD", "--no-llm"])
        .output()
        .unwrap();
    assert_eq!(bad_cfg.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad_cfg.stderr).contains("refusing API credential"));
    git(r, &["config", "--local", "--unset", "untangle.apiKey"]);

    // 2. Dry-run check: must not create branch
    let dry = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--no-llm",
            "--dry-run",
            "--branch",
            "should-not-exist",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(dry.status.code(), Some(0));
    let refs = git(r, &["branch", "--list", "should-not-exist"]);
    assert!(refs.is_empty(), "--dry-run must not create branch");

    // 3. Root commit precondition check: untangling initial commit returns EXIT_PRECONDITION (3)
    let root_out = Command::new(bin_git_untangle())
        .current_dir(r)
        .args(["HEAD~1", "--no-llm"])
        .output()
        .unwrap();
    assert_eq!(root_out.status.code(), Some(3));

    // 4. Test cassette replay for BOTH cluster refinement AND LLM-authored commit messages
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let (cluster_req, commit_msg_req) = rt.block_on(async {
        let (backend, _) = GitCliBackend::discover(r).unwrap();
        let target = backend
            .resolve_target(&TargetSpec {
                spec: "HEAD".to_owned(),
                ..Default::default()
            })
            .await
            .unwrap();
        let deltas = backend.extract_deltas(&target).await.unwrap();
        let parsed = parse_and_classify(deltas, 3);
        let batch = vec![vec!["i1".to_owned()], vec!["i2".to_owned()]];
        let c_req = build_cluster_refinement_request(&batch, &target.message, &parsed.items);

        let mut merged_op = Operation::new(
            "op1",
            "Initialize numeric bindings in one and two",
            "Merged via cassette replay.",
            "feature",
            vec!["i1".to_owned(), "i2".to_owned()],
        );
        merged_op.confidence = 0.75;
        let m_req = build_commit_messages_request(&[merged_op], &target.message, &parsed.items);
        (c_req, m_req)
    });
    let entry1 = CassetteEntry {
        key: cluster_req.cache_key(),
        task: SemanticTask::RefineClusters,
        prompt_version: "v1".to_owned(),
        response: serde_json::json!({
            "actions": [
                {"type": "merge_clusters", "clusters": ["c1", "c2"], "reason": "Unified update"}
            ],
            "labels": {
                "c1+c2": {
                    "name": "Initialize numeric bindings in one and two",
                    "description": "Merged via cassette replay.",
                    "category": "feature"
                }
            }
        }),
    };
    let entry2 = CassetteEntry {
        key: commit_msg_req.cache_key(),
        task: SemanticTask::WriteCommitMessages,
        prompt_version: "v1".to_owned(),
        response: serde_json::json!({
            "messages": {
                "op1": {
                    "subject": "feat: add numeric bindings across one and two",
                    "body": "Introduces placeholder numeric bindings in both modules for downstream tests.",
                    "item_subjects": {
                        "i1": "feat(one): bind placeholder constant 1",
                        "i2": "feat(two): add module two with binding 2"
                    }
                }
            }
        }),
    };
    let cassette_path = r.join("cassette.jsonl");
    fs::write(
        &cassette_path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&entry1).unwrap(),
            serde_json::to_string(&entry2).unwrap()
        ),
    )
    .unwrap();

    let replay_out = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--similarity",
            "1.5",
            "--replay-llm",
            cassette_path.to_str().unwrap(),
            "--emit",
            "branch",
            "--branch",
            "untangled/replayed",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        replay_out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&replay_out.stderr)
    );
    let replayed: AnalysisArtifact = serde_json::from_slice(&replay_out.stdout).unwrap();
    assert_eq!(replayed.operations.len(), 1);
    assert_eq!(
        replayed.operations[0].name,
        "feat: add numeric bindings across one and two"
    );
    assert_eq!(replayed.series.tree_matches, Some(true));

    // Verify the actual emitted Git commits use the LLM-authored item subjects & body + trailers
    let log_msgs = git(
        r,
        &["log", "--format=%s%n%b---", "HEAD~1..untangled/replayed"],
    );
    assert!(log_msgs.contains("feat(one): bind placeholder constant 1"));
    assert!(log_msgs.contains("feat(two): add module two with binding 2"));
    assert!(log_msgs.contains("Introduces placeholder numeric bindings in both modules"));
    assert!(log_msgs.contains("Commit-Analyzer-Operation: op1"));
}

#[test]
fn test_diamond_dependencies_and_pathspec_filtering() {
    let repo = init_repo();
    let r = repo.path();

    fs::create_dir_all(r.join("src")).unwrap();
    fs::create_dir_all(r.join("docs")).unwrap();
    fs::write(r.join("src/app.rs"), "pub fn main_entry() -> i32 { 0 }\n").unwrap();
    fs::write(r.join("docs/guide.md"), "# Guide\n").unwrap();
    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "init"]);

    // Commit adds:
    // - `src/left.rs` defining `left_helper`
    // - `src/right.rs` defining `right_helper`
    // - `src/app.rs` referencing BOTH `left_helper` AND `right_helper` -> multi-parent prerequisite merge!
    // - `docs/guide.md` unrelated doc update (excluded when `-- src/` pathspec is passed!)
    fs::write(
        r.join("src/left.rs"),
        "pub fn left_helper() -> i32 { 10 }\n",
    )
    .unwrap();
    fs::write(
        r.join("src/right.rs"),
        "pub fn right_helper() -> i32 { 32 }\n",
    )
    .unwrap();
    fs::write(
        r.join("src/app.rs"),
        "pub fn main_entry() -> i32 { left_helper() + right_helper() }\n",
    )
    .unwrap();
    fs::write(r.join("docs/guide.md"), "# Guide\n\nUpdated docs.\n").unwrap();
    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "diamond + docs"]);

    let out = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--no-llm",
            "--similarity",
            "1.5",
            "--branch",
            "untangled/src-only",
            "--json",
            "--",
            "src/",
        ])
        .output()
        .unwrap();

    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let artifact: AnalysisArtifact = serde_json::from_slice(&out.stdout).unwrap();
    assert!(artifact.exact_reconstruction);
    assert_eq!(artifact.series.tree_matches, Some(true));
    assert_eq!(artifact.operations.len(), 3);

    // `app.rs` operation must have BOTH `left.rs` and `right.rs` operations in `merge_parents`,
    // producing a prerequisite merge commit and leaving 1 sink!
    let app_op = artifact
        .operations
        .iter()
        .find(|o| o.files.iter().any(|f| f == "src/app.rs"))
        .expect("app.rs operation");
    assert_eq!(app_op.merge_parents.len(), 2);
    assert_eq!(artifact.series.merge_commits.len(), 1);

    // Verify `docs/guide.md` on `untangled/src-only` stayed at its base version while `src/` matches HEAD!
    let docs_on_branch = git(r, &["show", "untangled/src-only:docs/guide.md"]);
    assert_eq!(docs_on_branch, "# Guide");
}

#[test]
fn test_verify_closed_loop_repair_tiers_and_exhaustion() {
    let repo = init_repo();
    let r = repo.path();

    // Base commit: valid state
    fs::write(r.join("a_consumer.txt"), "STATE=base\n").unwrap();
    fs::write(r.join("z_provider.txt"), "// empty\n").unwrap();
    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "base"]);

    // Commit updates a_consumer (which comes first alphabetically!) to require a symbol defined in z_provider,
    // without using the symbol token in a_consumer so syntactic Stage 3 does NOT add an edge.
    // Our synthetic build command detects the broken intermediate state and prints `missing symbol runtime_hook_v2`.
    fs::write(r.join("a_consumer.txt"), "STATE=v2_enabled\n").unwrap();
    fs::write(r.join("z_provider.txt"), "pub fn runtime_hook_v2() {}\n").unwrap();
    git(r, &["add", "-A"]);
    git(r, &["commit", "-m", "enable v2 hook"]);
    let tip_tree = git(r, &["rev-parse", "HEAD^{tree}"]);

    #[cfg(unix)]
    let verify_cmd = "if grep -q v2_enabled a_consumer.txt && ! grep -q runtime_hook_v2 z_provider.txt; then echo 'error: missing symbol runtime_hook_v2' >&2; exit 1; fi";
    #[cfg(windows)]
    let verify_cmd = "findstr /C:v2_enabled a_consumer.txt >nul && (findstr /C:runtime_hook_v2 z_provider.txt >nul || (echo error: missing symbol runtime_hook_v2 1>&2 & exit /b 1)) || exit /b 0";

    // 1. Tier 1 automatic repair should add `z_provider -> a_consumer` and pass!
    let out = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--no-llm",
            "--similarity",
            "1.5",
            "--verify",
            verify_cmd,
            "--branch",
            "untangled/verified",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let artifact: AnalysisArtifact = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(artifact.verification.status, "passed");
    assert!(
        artifact
            .edges
            .iter()
            .any(|e| e.kind == "build-verification"),
        "expected build-verification edge"
    );
    assert_eq!(artifact.operations[0].files, vec!["z_provider.txt"]);
    assert_eq!(artifact.operations[1].files, vec!["a_consumer.txt"]);
    assert_eq!(
        git(r, &["rev-parse", "untangled/verified^{tree}"]),
        tip_tree
    );

    // 2. Retry budget exhaustion (`--max-retries 0`) must return EXIT_VERIFY_EXHAUSTED (4)
    let exhausted = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--no-llm",
            "--similarity",
            "1.5",
            "--verify",
            verify_cmd,
            "--max-retries",
            "0",
            "--emit",
            "none",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(exhausted.status.code(), Some(4));
    let exh_art: AnalysisArtifact = serde_json::from_slice(&exhausted.stdout).unwrap();
    assert!(exh_art
        .verification
        .status
        .contains("retry budget exhausted"));

    // 3. Assumption A0 violated (parent fails to build) skips verification cleanly with exit 0
    let a0 = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD", "--no-llm", "--verify", "exit 1", "--emit", "none", "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(a0.status.code(), Some(0));
    let a0_art: AnalysisArtifact = serde_json::from_slice(&a0.stdout).unwrap();
    assert!(a0_art.verification.status.contains("assumption A0"));

    // 4. Tier 4 fallback merge when build fails with no identifiable symbol
    #[cfg(unix)]
    let generic_fail = "if grep -q v2_enabled a_consumer.txt && ! grep -q runtime_hook_v2 z_provider.txt; then echo 'generic build error' >&2; exit 1; fi";
    #[cfg(windows)]
    let generic_fail = "findstr /C:v2_enabled a_consumer.txt >nul && (findstr /C:runtime_hook_v2 z_provider.txt >nul || (echo generic build error 1>&2 & exit /b 1)) || exit /b 0";
    let merged = Command::new(bin_git_untangle())
        .current_dir(r)
        .args([
            "HEAD",
            "--no-llm",
            "--similarity",
            "1.5",
            "--verify",
            generic_fail,
            "--emit",
            "none",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(merged.status.code(), Some(0));
    let merged_art: AnalysisArtifact = serde_json::from_slice(&merged.stdout).unwrap();
    assert_eq!(merged_art.verification.status, "passed");
    assert_eq!(
        merged_art.operations.len(),
        1,
        "fallback merge should combine inseparable operations into 1"
    );
}

#[test]
fn test_packaging_completions_and_manpages() {
    for shell in ["bash", "zsh", "fish", "powershell"] {
        let out = Command::new(bin_git_untangle())
            .args(["--generate-completions", shell])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0), "completions for {shell}");
        let s = String::from_utf8_lossy(&out.stdout);
        assert!(s.contains("git-untangle"));
        assert!(s.contains("no-llm"));
    }

    let man = Command::new(bin_git_untangle())
        .arg("--generate-manpage")
        .output()
        .unwrap();
    assert_eq!(man.status.code(), Some(0));
    let man_text = String::from_utf8_lossy(&man.stdout);
    assert!(man_text.contains(".TH GIT-UNTANGLE 1"));
    assert!(man_text.contains("EXIT STATUS"));

    let chat_man = Command::new(bin_git_chat())
        .arg("--generate-manpage")
        .output()
        .unwrap();
    assert_eq!(chat_man.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&chat_man.stdout).contains(".TH GIT-CHAT 1"));
}

#[test]
fn test_git_chat_interactive_session_and_configurable_byte_limit() {
    use std::io::Write;
    use std::process::Stdio;

    let repo = init_repo();
    let r = repo.path();

    // Create a large file (> 2000 bytes) so `git show HEAD` exceeds a 512-byte self-limit.
    let big_body: String = (0..80)
        .map(|i| format!("line_{i:02}: the quick brown fox jumps over the lazy dog\n"))
        .collect();
    fs::write(r.join("big.txt"), &big_body).unwrap();
    git(r, &["add", "big.txt"]);
    git(r, &["commit", "-m", "add large text fixture"]);

    // Compute exact expected captured output for `git show HEAD` truncated at 512 bytes
    let raw_show = Command::new("git")
        .current_dir(r)
        .args(["show", "HEAD"])
        .output()
        .unwrap();
    let combined = String::from_utf8_lossy(&raw_show.stdout).into_owned();
    assert!(combined.len() > 1500, "git show output should exceed 512B");
    let max_bytes = 512usize;
    let (prefix, truncated) = truncate_utf8_bytes(&combined, max_bytes);
    assert!(truncated);
    let mut context_text = prefix.to_owned();
    context_text.push_str(&format!(
        "\n[... output truncated at {max_bytes} bytes; total {} bytes ...]",
        combined.len()
    ));

    let captured = CapturedGitOutput {
        command_display: "git show HEAD".to_owned(),
        exit_code: 0,
        context_text,
        total_bytes: combined.len(),
        truncated: true,
    };

    // Prepare a 2-turn chat cassette (turn 1 via stdin, turn 2 follow-up)
    let mut msgs1 = build_initial_chat_messages(&captured, max_bytes);
    msgs1.push(ChatMessage {
        role: ChatRole::User,
        content: "Was the output truncated?".to_owned(),
    });
    let req1 = ChatRequest {
        messages: msgs1.clone(),
        temperature: 0.2,
    };
    let reply1 = "Yes, git show output was truncated to 512 bytes.";

    let mut msgs2 = msgs1;
    msgs2.push(ChatMessage {
        role: ChatRole::Assistant,
        content: reply1.to_owned(),
    });
    msgs2.push(ChatMessage {
        role: ChatRole::User,
        content: "What file was added?".to_owned(),
    });
    let req2 = ChatRequest {
        messages: msgs2,
        temperature: 0.2,
    };
    let reply2 = "The commit added `big.txt`.";

    let cassette_path = r.join("chat-cassette.jsonl");
    let e1 = CassetteEntry {
        key: req1.cache_key(),
        task: SemanticTask::InteractiveChat,
        prompt_version: "v1".to_owned(),
        response: serde_json::json!({ "reply": reply1 }),
    };
    let e2 = CassetteEntry {
        key: req2.cache_key(),
        task: SemanticTask::InteractiveChat,
        prompt_version: "v1".to_owned(),
        response: serde_json::json!({ "reply": reply2 }),
    };
    fs::write(
        &cassette_path,
        format!(
            "{}\n{}\n",
            serde_json::to_string(&e1).unwrap(),
            serde_json::to_string(&e2).unwrap()
        ),
    )
    .unwrap();

    // Configure limit via git config `untangle.chatMaxBytes` to verify config layering works too
    git(r, &["config", "--local", "untangle.chatMaxBytes", "512"]);

    let mut child = Command::new(bin_git_chat())
        .current_dir(r)
        .args([
            "--replay-llm",
            cassette_path.to_str().unwrap(),
            "show",
            "HEAD",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    {
        let stdin = child.stdin.as_mut().unwrap();
        writeln!(stdin, "Was the output truncated?").unwrap();
        writeln!(stdin, "What file was added?").unwrap();
        writeln!(stdin, "/quit").unwrap();
    }

    let out = child.wait_with_output().unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Direct git output echoed AND both interactive chat answers printed
    assert!(stdout.contains("add large text fixture"));
    assert!(stdout.contains(reply1));
    assert!(stdout.contains(reply2));
    assert!(stderr.contains("limit: 512 bytes, truncated"));
}
