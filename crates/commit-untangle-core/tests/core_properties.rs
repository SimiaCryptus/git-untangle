use std::collections::BTreeSet;

use commit_untangle_core::bytes::RawBytes;
use commit_untangle_core::classify::{is_generated_path, is_reorder, substitution_of};
use commit_untangle_core::graph::{ChangeGraph, EdgeKind, MechanicalPlacement};
use commit_untangle_core::hunk::{apply_hunks, parse_zero_context_diff, ZeroContextHunk};
use commit_untangle_core::model::{parse_and_classify, FileMode, ItemKind, Operation, RawFileDelta};
use commit_untangle_core::proposal::{
    apply_cluster_proposal, verify_partition, ClusterAction, ClusterRefinementProposal,
    DependencyProposal, OperationLabel, PlannedEdge,
};
use commit_untangle_core::replay::{file_state, verify_exact_reconstruction};
use commit_untangle_core::{build_deterministic_plan, DeterministicPlanConfig};
use indexmap::IndexMap;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Synthesizes a valid `git diff -U0` byte stream from an original line list and non-overlapping edits.
fn synthesize_u0_diff(
    old_lines: &[Vec<u8>],
    edits: &[(usize, usize, Vec<Vec<u8>>)],
) -> (RawBytes, RawBytes, RawBytes) {
    let mut old_raw = Vec::new();
    for l in old_lines {
        old_raw.extend_from_slice(l);
    }

    let mut new_raw = Vec::new();
    let mut diff_out = Vec::new();
    diff_out.extend_from_slice(b"diff --git a/test.txt b/test.txt\n--- a/test.txt\n+++ b/test.txt\n");

    let mut pos = 0usize;
    let mut new_line_cursor = 1usize;

    for (start, old_count, replacement) in edits {
        for l in &old_lines[pos..*start] {
            new_raw.extend_from_slice(l);
            new_line_cursor += 1;
        }

        let old_start_hdr = if *old_count == 0 { *start } else { *start + 1 };
        let new_count = replacement.len();
        let new_start_hdr = if new_count == 0 {
            new_line_cursor.saturating_sub(1)
        } else {
            new_line_cursor
        };

        let old_part = if *old_count == 1 {
            format!("{old_start_hdr}")
        } else {
            format!("{old_start_hdr},{old_count}")
        };
        let new_part = if new_count == 1 {
            format!("{new_start_hdr}")
        } else {
            format!("{new_start_hdr},{new_count}")
        };
        diff_out.extend_from_slice(format!("@@ -{old_part} +{new_part} @@\n").as_bytes());

        for ol in &old_lines[*start..(*start + *old_count)] {
            diff_out.push(b'-');
            diff_out.extend_from_slice(ol);
            if !ol.ends_with(b"\n") {
                diff_out.push(b'\n');
                diff_out.extend_from_slice(b"\\ No newline at end of file\n");
            }
        }
        for nl in replacement {
            new_raw.extend_from_slice(nl);
            diff_out.push(b'+');
            diff_out.extend_from_slice(nl);
            if !nl.ends_with(b"\n") {
                diff_out.push(b'\n');
                diff_out.extend_from_slice(b"\\ No newline at end of file\n");
            }
        }

        new_line_cursor += new_count;
        pos = *start + *old_count;
    }

    for l in &old_lines[pos..] {
        new_raw.extend_from_slice(l);
    }

    (RawBytes(old_raw), RawBytes(new_raw), RawBytes(diff_out))
}

fn gen_line(rng: &mut ChaCha8Rng, is_last: bool) -> Vec<u8> {
    let len = rng.gen_range(0..24);
    let mut line = Vec::with_capacity(len + 2);
    for _ in 0..len {
        let choice = rng.gen_range(0..10);
        let b = match choice {
            0..=6 => rng.gen_range(b'a'..=b'z'),
            7 => b' ',
            8 => b'_',
            _ => rng.gen_range(0x80u8..=0xfeu8), // arbitrary non-UTF8 high bytes
        };
        line.push(b);
    }
    if is_last && rng.gen_bool(0.35) {
        // Omit trailing newline at EOF (and occasionally end with bare '\r')
        if rng.gen_bool(0.2) {
            line.push(b'\r');
        }
        if line.is_empty() {
            line.push(b'x');
        }
    } else if rng.gen_bool(0.3) {
        line.extend_from_slice(b"\r\n");
    } else {
        line.push(b'\n');
    }
    line
}

#[test]
fn prop_10000_hunk_roundtrip_and_subset_commutativity() {
    let mut rng = ChaCha8Rng::seed_from_u64(0x5EED_2026_1127);

    // 10,000 randomized property test cases covering LF, CRLF, bare CR, missing EOF newline,
    // pure insertions, pure deletions, non-UTF8 bytes, and permuted hunk subsets.
    for case_idx in 0..10_000 {
        let num_lines = rng.gen_range(0..18);
        let mut old_lines = Vec::with_capacity(num_lines);
        for i in 0..num_lines {
            old_lines.push(gen_line(&mut rng, i + 1 == num_lines));
        }

        let mut edits = Vec::new();
        let mut cursor = 0usize;
        while cursor <= num_lines {
            if rng.gen_bool(0.45) {
                let max_del = num_lines - cursor;
                let old_count = if max_del == 0 {
                    0
                } else {
                    rng.gen_range(0..=max_del.min(4))
                };
                let new_count = if old_count == 0 {
                    rng.gen_range(1..=3)
                } else {
                    rng.gen_range(0..=3)
                };
                // Only allow missing-EOF-newline in replacement's final line if edit reaches EOF
                let reaches_eof = cursor + old_count == num_lines;
                let mut rep = Vec::with_capacity(new_count);
                for r_i in 0..new_count {
                    let can_omit_nl = reaches_eof && (r_i + 1 == new_count);
                    let mut l = gen_line(&mut rng, can_omit_nl);
                    if !can_omit_nl && !l.ends_with(b"\n") {
                        l.push(b'\n');
                    }
                    rep.push(l);
                }
                // Note: if `old_lines` has an unterminated final line and we insert *after* it
                // (`cursor == num_lines` and `!old_lines.last().ends_with('\n')`), Git represents
                // that by replacing the last line rather than pure insertion at `num_lines`.
                let invalid_after_unterminated_eof = cursor == num_lines
                    && old_count == 0
                    && old_lines.last().is_some_and(|l| !l.ends_with(b"\n"));
                // Similarly, deleting the tail of a file when the surviving prefix is always '\n'-terminated is valid.
                if !invalid_after_unterminated_eof && (old_count > 0 || new_count > 0) {
                    edits.push((cursor, old_count, rep));
                    cursor += old_count.max(1);
                    continue;
                }
            }
            cursor += 1;
        }

        let (old_raw, new_raw, diff_raw) = synthesize_u0_diff(&old_lines, &edits);
        let hunks = parse_zero_context_diff(&diff_raw)
            .unwrap_or_else(|e| panic!("case {case_idx} failed to parse diff: {e}"));

        assert_eq!(hunks.len(), edits.len(), "case {case_idx} hunk count");

        // Invariant I3 (empty subset -> base)
        let base_replayed = apply_hunks(&old_raw, &[]).unwrap();
        assert_eq!(base_replayed, old_raw, "case {case_idx} empty subset mismatch");

        // Invariant I3 (full set -> exact tip)
        let all_refs: Vec<&ZeroContextHunk> = hunks.iter().collect();
        let full_replayed = apply_hunks(&old_raw, &all_refs)
            .unwrap_or_else(|e| panic!("case {case_idx} full apply failed: {e}"));
        assert_eq!(
            full_replayed, new_raw,
            "case {case_idx} byte-exact roundtrip mismatch"
        );

        // Invariant I2 (subset commutativity under random subset & random order)
        if !hunks.is_empty() {
            let mut subset: Vec<&ZeroContextHunk> = hunks
                .iter()
                .filter(|_| rng.gen_bool(0.5))
                .collect();
            let forward = apply_hunks(&old_raw, &subset).unwrap();
            subset.shuffle(&mut rng);
            let shuffled = apply_hunks(&old_raw, &subset).unwrap();
            assert_eq!(forward, shuffled, "case {case_idx} subset permutation mismatch");
        }
    }
}

#[test]
fn prop_adversarial_cluster_proposals_conserve_partition() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE_BABE_0001);

    for case_idx in 0..1_000 {
        let item_count = rng.gen_range(1..35);
        let deltas: Vec<RawFileDelta> = (0..item_count)
            .map(|i| RawFileDelta {
                status: 'A',
                old_path: None,
                new_path: Some(format!("src/mod_{i}.rs")),
                old_mode: None,
                new_mode: Some(FileMode::Regular),
                old_content: None,
                new_content: Some(RawBytes::from(format!("pub fn func_{i}() {{}}\n"))),
                zero_context_diff: None,
            })
            .collect();

        let parsed = parse_and_classify(deltas, 3);
        let expected_ids: BTreeSet<String> = parsed.items.keys().cloned().collect();

        let mut initial: Vec<Vec<String>> = Vec::new();
        for id in &expected_ids {
            if initial.is_empty() || rng.gen_bool(0.4) {
                initial.push(vec![id.clone()]);
            } else {
                let idx = rng.gen_range(0..initial.len());
                initial[idx].push(id.clone());
            }
        }

        let mut actions = Vec::new();
        let num_actions = rng.gen_range(0..12);
        for _ in 0..num_actions {
            match rng.gen_range(0..3) {
                0 => actions.push(ClusterAction::MergeClusters {
                    clusters: vec![
                        format!("c{}", rng.gen_range(0..=initial.len() + 2)),
                        format!("c{}", rng.gen_range(0..=initial.len() + 2)),
                    ],
                    reason: "test merge".to_owned(),
                }),
                1 => {
                    let g1: Vec<String> = expected_ids
                        .iter()
                        .filter(|_| rng.gen_bool(0.25))
                        .cloned()
                        .chain(std::iter::once("i99999".to_owned()))
                        .collect();
                    let g2: Vec<String> = expected_ids
                        .iter()
                        .filter(|_| rng.gen_bool(0.25))
                        .cloned()
                        .collect();
                    actions.push(ClusterAction::SplitCluster {
                        cluster: format!("c{}", rng.gen_range(1..=initial.len() + 1)),
                        groups: vec![g1, g2],
                        reason: None,
                    });
                }
                _ => actions.push(ClusterAction::MoveItems {
                    items: vec![
                        format!("i{}", rng.gen_range(1..=item_count + 5)),
                        "bogus".to_owned(),
                    ],
                    from_cluster: format!("c{}", rng.gen_range(1..=initial.len() + 1)),
                    to_cluster: format!("c{}", rng.gen_range(1..=initial.len() + 1)),
                }),
            }
        }

        let mut labels = IndexMap::new();
        labels.insert(
            "c1".to_owned(),
            OperationLabel {
                name: "Refined cluster".to_owned(),
                description: "Desc".to_owned(),
                category: Some("INVALID_CAT".to_owned()),
                suspicious: true,
                suspicion_reason: Some("looks odd".to_owned()),
            },
        );

        let proposal = ClusterRefinementProposal { actions, labels };
        let ops = apply_cluster_proposal(&initial, &proposal, &parsed.items)
            .unwrap_or_else(|e| panic!("case {case_idx} proposal failed: {e}"));
        verify_partition(&expected_ids, &ops)
            .unwrap_or_else(|e| panic!("case {case_idx} partition violated: {e}"));
    }
}

#[test]
fn prop_random_graphs_contract_cycles_and_toposort() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEAD_BEEF_4242);

    for case_idx in 0..1_000 {
        let n_ops = rng.gen_range(1..25);
        let deltas: Vec<RawFileDelta> = (0..n_ops)
            .map(|i| RawFileDelta {
                status: 'A',
                old_path: None,
                new_path: Some(format!("f_{i}.rs")),
                old_mode: None,
                new_mode: Some(FileMode::Regular),
                old_content: None,
                new_content: Some(RawBytes::from(format!("fn a_{i}() {{}}\n"))),
                zero_context_diff: None,
            })
            .collect();
        let parsed = parse_and_classify(deltas, 3);
        let expected_items: BTreeSet<String> = parsed.items.keys().cloned().collect();

        let ops: Vec<Operation> = parsed
            .items
            .keys()
            .enumerate()
            .map(|(idx, item_id)| {
                Operation::new(
                    format!("op{}", idx + 1),
                    format!("Op {}", idx + 1),
                    "desc",
                    "feature",
                    vec![item_id.clone()],
                )
            })
            .collect();

        let mut graph = ChangeGraph::new(ops);
        let edge_attempts = rng.gen_range(0..(n_ops * 3));
        for _ in 0..edge_attempts {
            let a = format!("op{}", rng.gen_range(1..=n_ops));
            let b = format!("op{}", rng.gen_range(1..=n_ops));
            graph.add_edge(&a, &b, EdgeKind::DefinitionBeforeUse, "rand", true);
        }

        // Contract arbitrary cycles via Tarjan SCC
        graph.merge_cycles();

        // Attempt random semantic edges with cycle rejection
        let mut sem_edges = Vec::new();
        for _ in 0..10 {
            let a = format!("op{}", rng.gen_range(1..=n_ops));
            let b = format!("op{}", rng.gen_range(1..=n_ops));
            sem_edges.push(PlannedEdge {
                before: a,
                after: b,
                reason: "sem".to_owned(),
            });
        }
        let (_, _, ranks) = graph.apply_dependency_proposal(&DependencyProposal {
            edges: sem_edges,
            suggested_order: vec!["op3".to_owned(), "op1".to_owned()],
        });

        let ordered = graph
            .topological_order(MechanicalPlacement::First, &ranks, &parsed.items)
            .unwrap_or_else(|e| panic!("case {case_idx} failed topological sort after SCC merge: {e}"));

        verify_partition(&expected_items, &ordered)
            .unwrap_or_else(|e| panic!("case {case_idx} lost items during cycle contraction: {e}"));

        // Verify transitive reduction produces valid ancestor ideals and non-empty sinks
        let tr = graph.transitive_reduction(&ordered);
        assert!(
            !tr.sinks.is_empty(),
            "case {case_idx} non-empty DAG must have >=1 sink"
        );
        for op in &ordered {
            let r = &tr.reduced_parents[&op.id];
            for p in r {
                assert!(tr.ancestors[&op.id].contains(p));
            }
        }
    }
}

#[test]
fn test_mechanical_classifications_and_multibyte_utf8_substitutions() {
    // Reorder check
    let reorder_hunk = ZeroContextHunk {
        start: 0,
        old_count: 3,
        new_start: 1,
        old_lines: vec![
            RawBytes::from("use std::io;\n"),
            RawBytes::from("use std::collections::HashMap;\n"),
            RawBytes::from("use std::fmt;\n"),
        ],
        new_lines: vec![
            RawBytes::from("use std::collections::HashMap;\n"),
            RawBytes::from("use std::fmt;\n"),
            RawBytes::from("use std::io;\n"),
        ],
    };
    assert!(is_reorder(&reorder_hunk));

    // Multibyte UTF-8 surrounding an identifier substitution must never panic on char boundaries
    let utf8_sub_hunk = ZeroContextHunk {
        start: 2,
        old_count: 2,
        new_start: 3,
        old_lines: vec![
            RawBytes::from("// Café résumé: old_client_name 🚀\n"),
            RawBytes::from("let x = old_client_name.call(); // 日本語\n"),
        ],
        new_lines: vec![
            RawBytes::from("// Café résumé: new_client_handle 🚀\n"),
            RawBytes::from("let x = new_client_handle.call(); // 日本語\n"),
        ],
    };
    assert_eq!(
        substitution_of(&utf8_sub_hunk),
        Some(("old_client_name".to_owned(), "new_client_handle".to_owned()))
    );

    // Generated paths
    assert!(is_generated_path("Cargo.lock"));
    assert!(is_generated_path("web/node_modules/pkg/index.js"));
    assert!(is_generated_path("api/proto/service.pb.go"));
    assert!(!is_generated_path("src/lock_manager.rs"));
}

#[test]
fn test_end_to_end_deterministic_plan_rename_mode_bulk_and_def_use() {
    // File 0: Renamed + whitespace + semantic caller of `compute_checksum`
    let old_f0 = RawBytes::from("fn run() {\n    let a = old_api();\n    println!(\"hi\");\n}\n");
    let new_f0 = RawBytes::from(
        "fn run() {\n    let a = new_api();\n    let c = compute_checksum(a);\n    println!( \"hi\" );\n}\n",
    );
    let diff_f0 = RawBytes::from(
        "@@ -2,2 +2,3 @@\n-    let a = old_api();\n-    println!(\"hi\");\n+    let a = new_api();\n+    let c = compute_checksum(a);\n+    println!( \"hi\" );\n",
    );
    // Wait: let's split whitespace onto a separate hunk so whitespace & semantic are distinct hunks!
    let old_f0_clean = RawBytes::from(
        "fn run() {\n    let a = old_api();\n    keep_one();\n    keep_two();\n    println!(\"hi\");\n}\n",
    );
    let new_f0_clean = RawBytes::from(
        "fn run() {\n    let a = new_api();\n    keep_one();\n    keep_two();\n    println!(  \"hi\"  );\n}\n",
    );
    let diff_f0_clean = RawBytes::from(
        "@@ -2 +2 @@\n-    let a = old_api();\n+    let a = new_api();\n@@ -5 +5 @@\n-    println!(\"hi\");\n+    println!(  \"hi\"  );\n",
    );
    let _ = (old_f0, new_f0, diff_f0);

    // File 1 & File 2: 2 more occurrences of `old_api` -> `new_api` (total 3 => BULK_REPLACE!)
    let old_f1 = RawBytes::from("pub fn f1() { old_api(); }\n");
    let new_f1 = RawBytes::from("pub fn f1() { new_api(); }\n");
    let diff_f1 = RawBytes::from("@@ -1 +1 @@\n-pub fn f1() { old_api(); }\n+pub fn f1() { new_api(); }\n");

    let old_f2 = RawBytes::from("pub fn f2() { old_api(); }\n");
    let new_f2 = RawBytes::from("pub fn f2() { new_api(); }\n");
    let diff_f2 = RawBytes::from("@@ -1 +1 @@\n-pub fn f2() { old_api(); }\n+pub fn f2() { new_api(); }\n");

    // File 3: Defines `compute_DigestValue` and File 4 calls `compute_DigestValue` -> def-before-use edge!
    let new_f3 = RawBytes::from("pub fn compute_digest_value(input: usize) -> usize { input + 1 }\n");
    let old_f4 = RawBytes::from("pub fn caller() -> usize { 0 }\n");
    let new_f4 = RawBytes::from("pub fn caller() -> usize { compute_digest_value(41) }\n");
    let diff_f4 = RawBytes::from(
        "@@ -1 +1 @@\n-pub fn caller() -> usize { 0 }\n+pub fn caller() -> usize { compute_digest_value(41) }\n",
    );

    let deltas = vec![
        RawFileDelta {
            status: 'R',
            old_path: Some("src/old_runner.rs".to_owned()),
            new_path: Some("src/runner.rs".to_owned()),
            old_mode: Some(FileMode::Regular),
            new_mode: Some(FileMode::Executable),
            old_content: Some(old_f0_clean),
            new_content: Some(new_f0_clean),
            zero_context_diff: Some(diff_f0_clean),
        },
        RawFileDelta {
            status: 'M',
            old_path: Some("src/a.rs".to_owned()),
            new_path: Some("src/a.rs".to_owned()),
            old_mode: Some(FileMode::Regular),
            new_mode: Some(FileMode::Regular),
            old_content: Some(old_f1),
            new_content: Some(new_f1),
            zero_context_diff: Some(diff_f1),
        },
        RawFileDelta {
            status: 'M',
            old_path: Some("src/b.rs".to_owned()),
            new_path: Some("src/b.rs".to_owned()),
            old_mode: Some(FileMode::Regular),
            new_mode: Some(FileMode::Regular),
            old_content: Some(old_f2),
            new_content: Some(new_f2),
            zero_context_diff: Some(diff_f2),
        },
        RawFileDelta {
            status: 'A',
            old_path: None,
            new_path: Some("src/digest.rs".to_owned()),
            old_mode: None,
            new_mode: Some(FileMode::Regular),
            old_content: None,
            new_content: Some(new_f3),
            zero_context_diff: None,
        },
        RawFileDelta {
            status: 'M',
            old_path: Some("src/caller.rs".to_owned()),
            new_path: Some("src/caller.rs".to_owned()),
            old_mode: Some(FileMode::Regular),
            new_mode: Some(FileMode::Regular),
            old_content: Some(old_f4),
            new_content: Some(new_f4),
            zero_context_diff: Some(diff_f4),
        },
    ];

    let parsed = parse_and_classify(deltas, 3);
    assert!(verify_exact_reconstruction(&parsed.files, &parsed.items).is_empty());

    // Verify orthogonal move + mode_change + bulk_replace + whitespace
    assert!(parsed.items.values().any(|i| i.kind == ItemKind::Move));
    assert!(parsed.items.values().any(|i| i.kind == ItemKind::ModeChange));
    assert!(parsed.items.values().any(|i| i.kind == ItemKind::Whitespace));
    assert_eq!(
        parsed
            .items
            .values()
            .filter(|i| i.kind == ItemKind::BulkReplace)
            .count(),
        3
    );

    // Applying ONLY the move item moves `src/old_runner.rs` -> `src/runner.rs` with original content and Regular mode!
    let move_id = parsed.files[0].move_item_id.clone().unwrap();
    let only_move: BTreeSet<String> = [move_id].into_iter().collect();
    let moved_state = file_state(&parsed.files[0], &parsed.items, &only_move)
        .unwrap()
        .unwrap();
    assert_eq!(moved_state.path, "src/runner.rs");
    assert_eq!(moved_state.mode, FileMode::Regular);
    assert_eq!(
        moved_state.content,
        parsed.files[0].old_content.clone().unwrap()
    );

    // Run deterministic plan with high similarity threshold so digest.rs and caller.rs stay separate ops
    // and verify syntactic `definition-before-use` edge orders `digest.rs` before `caller.rs`!
    let plan = build_deterministic_plan(
        &parsed.files,
        &parsed.items,
        &DeterministicPlanConfig {
            similarity_threshold: 1.2,
            max_cluster_size: 40,
            mechanical_placement: MechanicalPlacement::First,
        },
    )
    .unwrap();

    // Move & ModeChange must come first (tier 0)
    assert!(plan.order[0].is_pure_move_or_mode(&parsed.items));

    let def_pos = plan
        .order
        .iter()
        .position(|op| op.files(&parsed.items).contains(&"src/digest.rs".to_owned()))
        .unwrap();
    let use_pos = plan
        .order
        .iter()
        .position(|op| op.files(&parsed.items).contains(&"src/caller.rs".to_owned()))
        .unwrap();
    assert!(
        def_pos < use_pos,
        "definition op ({def_pos}) must precede caller op ({use_pos})"
    );
}
