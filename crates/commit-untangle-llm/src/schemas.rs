use serde_json::{json, Value};

/// JSON Schema for `ClusterRefinementProposal`.
pub fn cluster_refinement_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": { "const": "merge_clusters" },
                                "clusters": { "type": "array", "items": { "type": "string" } },
                                "reason": { "type": "string" }
                            },
                            "required": ["type", "clusters"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": { "const": "split_cluster" },
                                "cluster": { "type": "string" },
                                "groups": {
                                    "type": "array",
                                    "items": { "type": "array", "items": { "type": "string" } }
                                },
                                "reason": { "type": "string" }
                            },
                            "required": ["type", "cluster", "groups"]
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": { "const": "move_items" },
                                "items": { "type": "array", "items": { "type": "string" } },
                                "from_cluster": { "type": "string" },
                                "to_cluster": { "type": "string" }
                            },
                            "required": ["type", "items", "from_cluster", "to_cluster"]
                        }
                    ]
                }
            },
            "labels": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "name": { "type": "string" },
                        "description": { "type": "string" },
                        "category": {
                            "type": "string",
                            "enum": ["refactor", "feature", "fix", "test", "docs", "config", "build", "cleanup", "formatting", "generated"]
                        },
                        "suspicious": { "type": "boolean" },
                        "suspicion_reason": { "type": ["string", "null"] }
                    },
                    "required": ["name", "description", "category"]
                }
            }
        },
        "required": ["actions", "labels"]
    })
}

/// JSON Schema for `DependencyProposal`.
pub fn dependency_proposal_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "before": { "type": "string" },
                        "after": { "type": "string" },
                        "reason": { "type": "string" }
                    },
                    "required": ["before", "after", "reason"]
                }
            },
            "suggested_order": {
                "type": "array",
                "items": { "type": "string" }
            }
        },
        "required": ["edges", "suggested_order"]
    })
}

/// JSON Schema for `BuildDiagnosisProposal`.
pub fn build_diagnosis_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "required_operation_ids": {
                "type": "array",
                "items": { "type": "string" }
            },
            "explanation": { "type": "string" }
        },
        "required": ["required_operation_ids", "explanation"]
    })
}

/// JSON Schema for `CommitMessageProposal`.
pub fn commit_messages_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "messages": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "subject": { "type": "string" },
                        "body": { "type": "string" },
                        "item_subjects": {
                            "type": "object",
                            "additionalProperties": { "type": "string" }
                        }
                    },
                    "required": ["subject", "body"]
                }
            }
        },
        "required": ["messages"]
    })
}
