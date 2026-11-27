use std::sync::{Arc, Mutex};

use commit_untangle_core::bytes::RawBytes;
use commit_untangle_core::model::{parse_and_classify, FileMode, Operation, RawFileDelta};
use commit_untangle_core::proposal::{
    BuildDiagnosisProposal, ClusterRefinementProposal, CommitMessageProposal,
};
use commit_untangle_llm::prompts::{
    build_commit_message_batch_request, item_summary, plan_commit_message_batches,
    DEFAULT_BATCH_CHAR_BUDGET, MAX_ITEMS_PER_COMMIT_MSG_BATCH,
};
use commit_untangle_llm::{
    extract_json_value, generate_typed, ChatMessage, ChatModel, ChatRequest, ChatRole, LlmError,
    ModelSpec, ProviderKind, ProviderModel, RecordingModel, ReplayModel, SemanticModel,
    SemanticTask, StructuredRequest, ToolCall, ToolChatMessage, ToolChatRequest, ToolDefinition,
};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

async fn spawn_mock_http(responses: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(Vec::new()));
    let cap_clone = Arc::clone(&captured);

    tokio::spawn(async move {
        for body in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            let mut buf = vec![0u8; 32768];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            let req_text = String::from_utf8_lossy(&buf[..n]).into_owned();
            cap_clone.lock().unwrap().push(req_text);

            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });

    (format!("http://{addr}"), captured)
}

fn sample_request(task: SemanticTask) -> StructuredRequest {
    StructuredRequest {
        task,
        prompt_version: "v1".to_owned(),
        system_prompt: "System".to_owned(),
        user_prompt: "User".to_owned(),
        json_schema: json!({"type": "object"}),
        temperature: 0.1,
    }
}

#[test]
fn test_model_spec_parsing_and_json_extraction() {
    assert_eq!(
        ModelSpec::parse("anthropic::claude-sonnet-4-5").unwrap(),
        ModelSpec {
            provider: ProviderKind::Anthropic,
            model: "claude-sonnet-4-5".to_owned(),
        }
    );
    assert_eq!(
        ModelSpec::parse("gemini-2.5-pro").unwrap(),
        ModelSpec {
            provider: ProviderKind::Gemini,
            model: "gemini-2.5-pro".to_owned(),
        }
    );
    assert_eq!(
        ModelSpec::parse("ollama::qwen2.5-coder:32b").unwrap(),
        ModelSpec {
            provider: ProviderKind::Ollama,
            model: "qwen2.5-coder:32b".to_owned(),
        }
    );

    let fenced = "Here is the plan:\n```json\n{\"actions\":[],\"labels\":{}}\n```\nDone.";
    assert_eq!(
        extract_json_value(fenced).unwrap(),
        json!({"actions": [], "labels": {}})
    );
}

#[test]
fn test_anthropic_gemini_openai_retry_and_ollama_cassette_wire_conformance() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        // 1. Anthropic native Messages API + markdown-fenced JSON
        let anthropic_body = json!({
            "content": [{
                "type": "text",
                "text": "```json\n{\"actions\":[{\"type\":\"merge_clusters\",\"clusters\":[\"c1\",\"c2\"],\"reason\":\"coupled\"}],\"labels\":{}}\n```"
            }]
        })
        .to_string();
        let (base, reqs) = spawn_mock_http(vec![anthropic_body]).await;
        let model = ProviderModel::new(
            ModelSpec::parse("anthropic::claude-sonnet-4-5").unwrap(),
            Some("test-anth-key".to_owned()),
            base,
        )
        .unwrap();
        let prop: ClusterRefinementProposal =
            generate_typed(&model, sample_request(SemanticTask::RefineClusters))
                .await
                .unwrap();
        assert_eq!(prop.actions.len(), 1);
        let raw_req = reqs.lock().unwrap()[0].clone();
        assert!(raw_req.starts_with("POST /v1/messages "));
        assert!(raw_req.to_ascii_lowercase().contains("x-api-key: test-anth-key"));
        assert!(raw_req.to_ascii_lowercase().contains("anthropic-version: 2023-06-01"));

        // 2. Gemini native GenerateContent API
        let gemini_body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "{\"required_operation_ids\":[\"op2\"],\"explanation\":\"op2 defines missing type\"}"
                    }]
                }
            }]
        })
        .to_string();
        let (base, reqs) = spawn_mock_http(vec![gemini_body]).await;
        let model = ProviderModel::new(
            ModelSpec::parse("gemini::gemini-2.5-pro").unwrap(),
            Some("test-gem-key".to_owned()),
            base,
        )
        .unwrap();
        let diag: BuildDiagnosisProposal =
            generate_typed(&model, sample_request(SemanticTask::DiagnoseBuildFailure))
                .await
                .unwrap();
        assert_eq!(diag.required_operation_ids, vec!["op2"]);
        let raw_req = reqs.lock().unwrap()[0].clone();
        assert!(raw_req.starts_with("POST /v1beta/models/gemini-2.5-pro:generateContent "));
        assert!(raw_req.to_ascii_lowercase().contains("x-goog-api-key: test-gem-key"));

        // 3. OpenAI Chat Completions + 1-shot repair retry on malformed first reply
        let bad_first = json!({
            "choices": [{ "message": { "content": "Sorry, I cannot comply with JSON right now." } }]
        })
        .to_string();
        let good_second = json!({
            "choices": [{ "message": { "content": "{\"edges\":[],\"suggested_order\":[\"op1\"]}" } }]
        })
        .to_string();
        let (base, reqs) = spawn_mock_http(vec![bad_first, good_second]).await;
        let model = ProviderModel::new(
            ModelSpec::parse("openai::gpt-4.1").unwrap(),
            Some("test-oai-key".to_owned()),
            base,
        )
        .unwrap();
        let val = model
            .generate_json(sample_request(SemanticTask::InferDependencies))
            .await
            .unwrap();
        assert_eq!(val["suggested_order"], json!(["op1"]));
        assert_eq!(reqs.lock().unwrap().len(), 2, "must perform 1-shot repair retry");

        // 4. Ollama native /api/chat + RecordingModel -> ReplayModel cassette roundtrip
        let ollama_body = json!({
            "message": { "content": "{\"actions\":[],\"labels\":{}}" }
        })
        .to_string();
        let (base, reqs) = spawn_mock_http(vec![ollama_body]).await;
        let ollama = ProviderModel::new(
            ModelSpec::parse("ollama::qwen2.5-coder:32b").unwrap(),
            None,
            base,
        )
        .unwrap();
        let cassette_path = std::env::temp_dir().join(format!(
            "git-untangle-cassette-test-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&cassette_path);
        let recorder = RecordingModel::new(ollama, &cassette_path);
        let req = sample_request(SemanticTask::RefineClusters);
        let live_val = recorder.generate_json(req.clone()).await.unwrap();
        assert!(reqs.lock().unwrap()[0].starts_with("POST /api/chat "));

        let replayer = ReplayModel::from_jsonl_file(&cassette_path).unwrap();
        let replayed_val = replayer.generate_json(req).await.unwrap();
        assert_eq!(live_val, replayed_val);
        let _ = std::fs::remove_file(&cassette_path);

        // 5. Multi-turn ChatModel over Ollama + RecordingModel -> ReplayModel chat cassette
        let chat_body = json!({
            "message": { "content": "The diff renames two helper functions." }
        })
        .to_string();
        let (base, reqs) = spawn_mock_http(vec![chat_body]).await;
        let ollama_chat = ProviderModel::new(
            ModelSpec::parse("ollama::qwen2.5-coder:32b").unwrap(),
            None,
            base,
        )
        .unwrap();
        let chat_cassette = std::env::temp_dir().join(format!(
            "git-untangle-chat-cassette-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&chat_cassette);
        let chat_recorder = RecordingModel::new(ollama_chat, &chat_cassette);
        let chat_req = ChatRequest {
            messages: vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are git-chat.".to_owned(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "What changed?".to_owned(),
                },
            ],
            temperature: 0.2,
        };
        let reply = chat_recorder.chat(chat_req.clone()).await.unwrap();
        assert_eq!(reply, "The diff renames two helper functions.");
        assert!(reqs.lock().unwrap()[0].starts_with("POST /api/chat "));

        let chat_replayer = ReplayModel::from_jsonl_file(&chat_cassette).unwrap();
        assert_eq!(chat_replayer.chat(chat_req).await.unwrap(), reply);
        let _ = std::fs::remove_file(&chat_cassette);
    });
}

#[test]
fn test_commit_message_lenient_deserialization_slice_planning_and_fatal_classification() {
    // 1. Null `item_subjects`, bracketed IDs, and nested `items` objects must all deserialize cleanly
    let raw_llm_json = json!({
        "messages": {
            "[op1]": {
                "subject": "feat(llm): add multi-provider structured prompt adapters",
                "body": "Introduces provider HTTP adapters and prompt builders.",
                "items": {
                    "[i1]": {
                        "subject": "feat(llm): add Anthropic and Gemini HTTP client",
                        "body": "Implements JSON extraction and retry loop for Anthropic and Gemini."
                    },
                    "i2": {
                        "subject": "feat(llm): add structured prompt builders",
                        "body": "Adds batching and item summaries for semantic operations."
                    }
                }
            },
            "op3": {
                "subject": "chore: add commit-untangle-core manifest",
                "body": "Declares workspace dependencies for the core crate.",
                "item_subjects": null,
                "item_bodies": null
            }
        }
    });
    let parsed_prop: CommitMessageProposal = serde_json::from_value(raw_llm_json).unwrap();
    assert_eq!(parsed_prop.messages.len(), 2);
    let op1 = &parsed_prop.messages["op1"];
    assert_eq!(
        op1.item_subjects["i1"],
        "feat(llm): add Anthropic and Gemini HTTP client"
    );
    assert!(op1.item_bodies["i1"].contains("Anthropic and Gemini"));
    assert!(parsed_prop.messages["op3"].item_subjects.is_empty());

    // 2. `AddFile` item_summary includes actual file lines + oversized operation slices cover every item
    let total_files = MAX_ITEMS_PER_COMMIT_MSG_BATCH * 2 + 3;
    let deltas: Vec<RawFileDelta> = (0..total_files)
        .map(|idx| RawFileDelta {
            status: 'A',
            old_path: None,
            new_path: Some(format!("crates/mod_{idx}/Cargo.toml")),
            old_mode: None,
            new_mode: Some(FileMode::Regular),
            old_content: None,
            new_content: Some(RawBytes::from(format!(
                "[package]\nname = \"mod-{idx}\"\nversion = \"0.1.0\"\n"
            ))),
            zero_context_diff: None,
        })
        .collect();
    let parsed = parse_and_classify(deltas, 3);
    let first_item = parsed.items.values().next().unwrap();
    let summary = item_summary(first_item, 4);
    assert!(
        summary.contains("name = \"mod-0\""),
        "AddFile summary must preview file contents: {summary}"
    );

    let big_op = Operation::new(
        "op1",
        "Add workspace manifests",
        "Adds manifests",
        "config",
        parsed.items.keys().cloned().collect(),
    );
    let batches = plan_commit_message_batches(&[big_op], &parsed.items, DEFAULT_BATCH_CHAR_BUDGET);
    assert_eq!(
        batches.len(),
        3,
        "27 items should split into 3 slices of <=12"
    );
    let last_req = build_commit_message_batch_request(&batches[2], "ctx", &parsed.items);
    assert!(
        last_req.user_prompt.contains(&format!("[i{total_files}]")),
        "final slice must include the last item"
    );

    // 3. HTTP 404 model-not-found is classified as fatal with a model hint
    let err_404 = LlmError::ProviderResponse {
        status: 404,
        body: "{\"type\":\"error\",\"error\":{\"type\":\"not_found_error\",\"message\":\"model: claude-nonexistent\"}}".to_owned(),
    };
    assert!(err_404.is_fatal_config_or_auth());
    assert!(err_404.diagnostic_hint().unwrap().contains("--model"));
}

#[test]
fn test_anthropic_deprecated_temperature_auto_omits_and_retries() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().unwrap();
        let captured = Arc::new(Mutex::new(Vec::<String>::new()));
        let cap_clone = Arc::clone(&captured);

        tokio::spawn(async move {
            let responses = vec![
                (
                    "400 Bad Request",
                    json!({
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "`temperature` is deprecated for this model."
                        }
                    })
                    .to_string(),
                ),
                (
                    "200 OK",
                    json!({
                        "content": [{ "type": "text", "text": "{\"messages\":{\"op1\":{\"subject\":\"feat: add adapter\",\"body\":\"Adds adapter.\"}}}" }],
                        "stop_reason": "end_turn"
                    })
                    .to_string(),
                ),
                (
                    "200 OK",
                    json!({
                        "content": [{ "type": "text", "text": "{\"messages\":{\"op2\":{\"subject\":\"chore: update lockfile\",\"body\":\"Updates lockfile.\"}}}" }],
                        "stop_reason": "end_turn"
                    })
                    .to_string(),
                ),
            ];
            for (status, body) in responses {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = vec![0u8; 32768];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                cap_clone
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&buf[..n]).into_owned());
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(resp.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });

        let model = ProviderModel::new(
            ModelSpec::parse("anthropic::claude-haiku-5-5").unwrap(),
            Some("test-anth-key".to_owned()),
            format!("http://{addr}"),
        )
        .unwrap();

        let p1: CommitMessageProposal =
            generate_typed(&model, sample_request(SemanticTask::WriteCommitMessages))
                .await
                .unwrap();
        assert_eq!(p1.messages["op1"].subject, "feat: add adapter");

        // Second call on the same ProviderModel should omit `temperature` on the first try
        let p2: CommitMessageProposal =
            generate_typed(&model, sample_request(SemanticTask::WriteCommitMessages))
                .await
                .unwrap();
        assert_eq!(p2.messages["op2"].subject, "chore: update lockfile");

        let reqs = captured.lock().unwrap();
        assert_eq!(reqs.len(), 3);
        assert!(reqs[0].contains("\"temperature\""));
        assert!(!reqs[1].contains("\"temperature\""));
        assert!(!reqs[2].contains("\"temperature\""));
    });
}

#[test]
fn test_chat_with_tools_wire_conformance_across_providers_and_cassette() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let tools = vec![ToolDefinition {
            name: "git_search_commits".to_owned(),
            description: "Search commits".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        }];

        let base_req = ToolChatRequest {
            messages: vec![
                ToolChatMessage::System {
                    content: "You are git-chat.".to_owned(),
                },
                ToolChatMessage::User {
                    content: "Find auth commit".to_owned(),
                },
            ],
            tools: tools.clone(),
            temperature: 0.2,
        };

        // 1. Anthropic tool_use + tool_result roundtrip
        let anth_r1 = json!({
            "content": [
                { "type": "text", "text": "Let me search." },
                { "type": "tool_use", "id": "tu_1", "name": "git_search_commits", "input": { "query": "auth" } }
            ]
        })
        .to_string();
        let anth_r2 = json!({
            "content": [{ "type": "text", "text": "Found commit abc123." }]
        })
        .to_string();
        let (base, reqs) = spawn_mock_http(vec![anth_r1, anth_r2]).await;
        let anth = ProviderModel::new(
            ModelSpec::parse("anthropic::claude-sonnet-4-5").unwrap(),
            Some("test-key".to_owned()),
            base,
        )
        .unwrap();
        let res1 = anth.chat_with_tools(base_req.clone()).await.unwrap();
        assert_eq!(res1.tool_calls.len(), 1);
        assert_eq!(res1.tool_calls[0].id, "tu_1");
        assert_eq!(res1.tool_calls[0].name, "git_search_commits");
        assert_eq!(res1.tool_calls[0].arguments["query"], "auth");

        let mut follow_req = base_req.clone();
        follow_req.messages.push(ToolChatMessage::Assistant {
            content: res1.content,
            tool_calls: res1.tool_calls,
        });
        follow_req.messages.push(ToolChatMessage::ToolResult {
            tool_call_id: "tu_1".to_owned(),
            tool_name: "git_search_commits".to_owned(),
            content: "abc123 feat: add auth".to_owned(),
            is_error: false,
        });
        let res2 = anth.chat_with_tools(follow_req).await.unwrap();
        assert!(res2.tool_calls.is_empty());
        assert_eq!(res2.content, "Found commit abc123.");
        assert!(reqs.lock().unwrap()[1].contains("\"tool_result\""));

        // 2. OpenAI function tool_calls
        let oai_r1 = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "git_search_commits",
                            "arguments": "{\"query\":\"auth\"}"
                        }
                    }]
                }
            }]
        })
        .to_string();
        let (base, _reqs) = spawn_mock_http(vec![oai_r1]).await;
        let oai = ProviderModel::new(
            ModelSpec::parse("openai::gpt-4.1").unwrap(),
            Some("test-key".to_owned()),
            base,
        )
        .unwrap();
        let oai_res = oai.chat_with_tools(base_req.clone()).await.unwrap();
        assert_eq!(
            oai_res.tool_calls,
            vec![ToolCall {
                id: "call_1".to_owned(),
                name: "git_search_commits".to_owned(),
                arguments: json!({"query": "auth"}),
            }]
        );

        // 3. Gemini functionCall + cassette record/replay
        let gem_r1 = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "git_search_commits",
                            "args": { "query": "auth" }
                        }
                    }]
                }
            }]
        })
        .to_string();
        let (base, _reqs) = spawn_mock_http(vec![gem_r1]).await;
        let gem = ProviderModel::new(
            ModelSpec::parse("gemini::gemini-2.5-pro").unwrap(),
            Some("test-key".to_owned()),
            base,
        )
        .unwrap();
        let cassette_path = std::env::temp_dir().join(format!(
            "git-untangle-tool-chat-cassette-{}.jsonl",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&cassette_path);
        let recorder = RecordingModel::new(gem, &cassette_path);
        let live_res = recorder.chat_with_tools(base_req.clone()).await.unwrap();
        assert_eq!(live_res.tool_calls.len(), 1);
        assert_eq!(live_res.tool_calls[0].name, "git_search_commits");

        let replayer = ReplayModel::from_jsonl_file(&cassette_path).unwrap();
        let replayed = replayer.chat_with_tools(base_req).await.unwrap();
        assert_eq!(replayed, live_res);
        let _ = std::fs::remove_file(&cassette_path);
    });
}
