use std::collections::{BTreeSet, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

static TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_]*").expect("valid token regex"));

static DEF_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(
            r"\b(?:fun|def|class|interface|object|struct|enum|trait|type|typealias|val|var|let|const|function|func|fn|record|module|namespace|macro)\s+(?:<[^>]*>\s*)?(?:mut\s+)?(?:[A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)",
        )
        .expect("valid def regex 1"),
        Regex::new(
            r"\b(?:public|private|protected|internal|static|final|override|abstract|virtual)\s+(?:[\w<>\[\],.?]+\s+)*([A-Za-z_]\w*)\s*\(",
        )
        .expect("valid def regex 2"),
    ]
});

static KEYWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "abstract",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "boolean",
        "break",
        "byte",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "def",
        "default",
        "del",
        "delete",
        "do",
        "double",
        "elif",
        "else",
        "enum",
        "except",
        "export",
        "extends",
        "false",
        "final",
        "finally",
        "float",
        "for",
        "from",
        "fun",
        "func",
        "function",
        "global",
        "if",
        "implements",
        "import",
        "in",
        "instanceof",
        "int",
        "interface",
        "internal",
        "is",
        "lambda",
        "let",
        "long",
        "new",
        "nil",
        "none",
        "None",
        "not",
        "null",
        "object",
        "or",
        "override",
        "package",
        "pass",
        "private",
        "protected",
        "public",
        "raise",
        "return",
        "self",
        "short",
        "static",
        "string",
        "String",
        "struct",
        "super",
        "switch",
        "this",
        "throw",
        "throws",
        "true",
        "True",
        "False",
        "try",
        "type",
        "typeof",
        "val",
        "var",
        "void",
        "when",
        "while",
        "with",
        "yield",
        "Int",
        "Unit",
        "Any",
        "List",
        "Map",
        "Set",
        "use",
        "mut",
        "pub",
        "impl",
        "fn",
        "the",
        "end",
        "then",
        "crate",
        "super",
        "where",
        "match",
        "Option",
        "Result",
        "Some",
        "Err",
        "Vec",
        "str",
        "usize",
        "bool",
    ]
    .into_iter()
    .collect()
});

#[derive(Clone, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalInfo {
    pub added_tokens: BTreeSet<String>,
    pub removed_tokens: BTreeSet<String>,
    pub defined_added: BTreeSet<String>,
    pub defined_removed: BTreeSet<String>,
    pub new_refs: BTreeSet<String>,
    pub dropped_refs: BTreeSet<String>,
    pub signature: BTreeSet<String>,
}

impl LexicalInfo {
    pub fn analyze(removed_text: &str, added_text: &str) -> Self {
        let added_tokens = tokens(added_text);
        let removed_tokens = tokens(removed_text);
        let defined_added = definitions(added_text);
        let defined_removed = definitions(removed_text);

        let new_refs: BTreeSet<String> = added_tokens
            .difference(&removed_tokens)
            .filter(|t| is_significant(t))
            .cloned()
            .collect();

        let dropped_refs: BTreeSet<String> = removed_tokens
            .difference(&added_tokens)
            .filter(|t| is_significant(t))
            .cloned()
            .collect();

        let diff_union: BTreeSet<String> = new_refs.union(&dropped_refs).cloned().collect();
        let signature = if !diff_union.is_empty() {
            diff_union
        } else {
            added_tokens
                .union(&removed_tokens)
                .filter(|t| is_significant(t))
                .cloned()
                .collect()
        };

        Self {
            added_tokens,
            removed_tokens,
            defined_added,
            defined_removed,
            new_refs,
            dropped_refs,
            signature,
        }
    }
}

pub fn tokens(text: &str) -> BTreeSet<String> {
    if text.is_empty() {
        return BTreeSet::new();
    }
    TOKEN_RE
        .find_iter(text)
        .map(|m| m.as_str().to_owned())
        .collect()
}

pub fn definitions(text: &str) -> BTreeSet<String> {
    if text.is_empty() {
        return BTreeSet::new();
    }
    let mut out = BTreeSet::new();
    for re in DEF_REGEXES.iter() {
        for caps in re.captures_iter(text) {
            if let Some(m) = caps.get(1) {
                let sym = m.as_str();
                if !KEYWORDS.contains(sym) {
                    out.insert(sym.to_owned());
                }
            }
        }
    }
    out
}

pub fn is_significant(token: &str) -> bool {
    token.len() >= 3 && !KEYWORDS.contains(token)
}

pub fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}
