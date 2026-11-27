use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::cluster::truncate_chars;
use crate::error::CoreError;
use crate::model::{ChangeItem, Operation};
use crate::proposal::DependencyProposal;
use crate::symbols::is_significant;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MechanicalPlacement {
    #[default]
    First,
    Last,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    #[serde(rename = "definition-before-use")]
    DefinitionBeforeUse,
    #[serde(rename = "removal-after-last-use")]
    RemovalAfterLastUse,
    #[serde(rename = "semantic")]
    Semantic,
    #[serde(rename = "build-verification")]
    BuildVerification,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DefinitionBeforeUse => "definition-before-use",
            Self::RemovalAfterLastUse => "removal-after-last-use",
            Self::Semantic => "semantic",
            Self::BuildVerification => "build-verification",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub before: String,
    pub after: String,
    pub kind: EdgeKind,
    pub reason: String,
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct ChangeGraph {
    pub ops: Vec<Operation>,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Eq, PartialEq)]
struct HeapEntry {
    tier: u8,
    llm_rank: usize,
    position: u64,
    id: String,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior on BinaryHeap
        other
            .tier
            .cmp(&self.tier)
            .then_with(|| other.llm_rank.cmp(&self.llm_rank))
            .then_with(|| other.position.cmp(&self.position))
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl ChangeGraph {
    pub fn new(ops: Vec<Operation>) -> Self {
        Self {
            ops,
            edges: Vec::new(),
        }
    }

    pub fn op(&self, id: &str) -> Option<&Operation> {
        self.ops.iter().find(|o| o.id == id)
    }

    pub fn op_mut(&mut self, id: &str) -> Option<&mut Operation> {
        self.ops.iter_mut().find(|o| o.id == id)
    }

    fn successors(&self) -> HashMap<&str, Vec<&str>> {
        let mut map: HashMap<&str, Vec<&str>> = HashMap::new();
        for e in &self.edges {
            map.entry(e.before.as_str())
                .or_default()
                .push(e.after.as_str());
        }
        map
    }

    pub fn reachable(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        let succ = self.successors();
        let mut seen = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(from);
        while let Some(cur) = queue.pop_front() {
            if !seen.insert(cur) {
                continue;
            }
            if let Some(neighbors) = succ.get(cur) {
                for &n in neighbors {
                    if n == to {
                        return true;
                    }
                    queue.push_back(n);
                }
            }
        }
        false
    }

    pub fn add_edge(
        &mut self,
        before: &str,
        after: &str,
        kind: EdgeKind,
        reason: impl Into<String>,
        allow_cycle: bool,
    ) -> bool {
        if before == after {
            return false;
        }
        if self.op(before).is_none() || self.op(after).is_none() {
            return false;
        }
        if self
            .edges
            .iter()
            .any(|e| e.before == before && e.after == after)
        {
            return false;
        }
        if !allow_cycle && self.reachable(after, before) {
            return false;
        }
        self.edges.push(Edge {
            before: before.to_owned(),
            after: after.to_owned(),
            kind,
            reason: reason.into(),
        });
        true
    }

    /// Merges multiple operations into `ids[0]`, redirecting incident edges and removing self-loops
    /// and duplicate parallel edges.
    pub fn merge(&mut self, ids: &[String], note: &str) -> Result<String, CoreError> {
        let Some(target_id) = ids.first().cloned() else {
            return Err(CoreError::UnknownOperation("<empty>".to_owned()));
        };
        if self.op(&target_id).is_none() {
            return Err(CoreError::UnknownOperation(target_id));
        }

        let mut seen_others = BTreeSet::new();
        let others: Vec<Operation> = ids
            .iter()
            .skip(1)
            .filter(|&id| id != &target_id && seen_others.insert(id.clone()))
            .filter_map(|id| {
                let pos = self.ops.iter().position(|o| &o.id == id)?;
                Some(self.ops.remove(pos))
            })
            .collect();

        let other_ids: HashSet<String> = others.iter().map(|o| o.id.clone()).collect();

        if let Some(target) = self.op_mut(&target_id) {
            for other in others {
                target.item_ids.extend(other.item_ids);
                target.name = truncate_chars(&format!("{} + {}", target.name, other.name), 160);
                target.suspicious = target.suspicious || other.suspicious;
                if target.suspicion_reason.is_none() {
                    target.suspicion_reason = other.suspicion_reason;
                }
                target.confidence = target.confidence.min(other.confidence);
                if target.category != other.category {
                    target.category = "mixed".to_owned();
                }
                target.rule = None;
            }
            if !note.trim().is_empty() {
                target.description = format!("{}\n\n{}", target.description.trim(), note.trim())
                    .trim()
                    .to_owned();
            }
        }

        for edge in &mut self.edges {
            if other_ids.contains(&edge.before) {
                edge.before = target_id.clone();
            }
            if other_ids.contains(&edge.after) {
                edge.after = target_id.clone();
            }
        }
        self.edges.retain(|e| e.before != e.after);
        let mut unique_pairs = HashSet::new();
        self.edges
            .retain(|e| unique_pairs.insert((e.before.clone(), e.after.clone())));

        Ok(target_id)
    }

    /// Tarjan's Strongly Connected Components ($O(|V| + |E|)$).
    pub fn strongly_connected(&self) -> Vec<Vec<String>> {
        struct Tarjan<'a> {
            succ: HashMap<&'a str, Vec<&'a str>>,
            index: usize,
            idx: HashMap<&'a str, usize>,
            low: HashMap<&'a str, usize>,
            on_stack: HashSet<&'a str>,
            stack: Vec<&'a str>,
            result: Vec<Vec<String>>,
        }

        impl<'a> Tarjan<'a> {
            fn strong(&mut self, v: &'a str) {
                self.idx.insert(v, self.index);
                self.low.insert(v, self.index);
                self.index += 1;
                self.stack.push(v);
                self.on_stack.insert(v);

                if let Some(neighbors) = self.succ.get(v).cloned() {
                    for w in neighbors {
                        if !self.idx.contains_key(w) {
                            self.strong(w);
                            let lw = self.low[w];
                            let lv = self.low.get_mut(v).unwrap();
                            *lv = (*lv).min(lw);
                        } else if self.on_stack.contains(w) {
                            let iw = self.idx[w];
                            let lv = self.low.get_mut(v).unwrap();
                            *lv = (*lv).min(iw);
                        }
                    }
                }

                if self.low[v] == self.idx[v] {
                    let mut comp = Vec::new();
                    while let Some(w) = self.stack.pop() {
                        self.on_stack.remove(w);
                        comp.push(w.to_owned());
                        if w == v {
                            break;
                        }
                    }
                    self.result.push(comp);
                }
            }
        }

        let mut state = Tarjan {
            succ: self.successors(),
            index: 0,
            idx: HashMap::new(),
            low: HashMap::new(),
            on_stack: HashSet::new(),
            stack: Vec::new(),
            result: Vec::new(),
        };

        for op in &self.ops {
            if !state.idx.contains_key(op.id.as_str()) {
                state.strong(op.id.as_str());
            }
        }

        state.result
    }

    /// Contracts every non-trivial SCC (`size > 1`) into a single operation until the graph is a strict DAG.
    pub fn merge_cycles(&mut self) -> Vec<Vec<String>> {
        let mut merged_groups = Vec::new();
        loop {
            let sccs: Vec<Vec<String>> = self
                .strongly_connected()
                .into_iter()
                .filter(|c| c.len() > 1)
                .collect();
            if sccs.is_empty() {
                return merged_groups;
            }
            for mut comp in sccs {
                comp.sort_by_key(|id| {
                    self.ops
                        .iter()
                        .position(|o| &o.id == id)
                        .unwrap_or(usize::MAX)
                });
                let names: Vec<String> = comp
                    .iter()
                    .filter_map(|id| self.op(id).map(|o| o.name.clone()))
                    .collect();
                let note = format!(
                    "(Merged because of a dependency cycle between: {})",
                    names.join("; ")
                );
                let _ = self.merge(&comp, &note);
                merged_groups.push(comp);
            }
        }
    }

    /// Infers deterministic `definition-before-use` and `removal-after-last-use` edges.
    pub fn add_deterministic_edges(&mut self, items_by_id: &IndexMap<String, ChangeItem>) {
        let mut definers: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut removers: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for op in &self.ops {
            let added = op.defined_added(items_by_id);
            let removed = op.defined_removed(items_by_id);
            for sym in &added {
                if is_significant(sym) {
                    definers.entry(sym.clone()).or_default().push(op.id.clone());
                }
            }
            for sym in removed.difference(&added) {
                if is_significant(sym) {
                    removers.entry(sym.clone()).or_default().push(op.id.clone());
                }
            }
        }

        struct OpLexicalSnapshot {
            id: String,
            defs_added: BTreeSet<String>,
            refs_added: BTreeSet<String>,
            refs_removed: BTreeSet<String>,
        }

        let snapshot: Vec<OpLexicalSnapshot> = self
            .ops
            .iter()
            .map(|op| OpLexicalSnapshot {
                id: op.id.clone(),
                defs_added: op.defined_added(items_by_id),
                refs_added: op.refs_added(items_by_id),
                refs_removed: op.refs_removed(items_by_id),
            })
            .collect();

        for snap in snapshot {
            let b_id = snap.id;
            for sym in snap.refs_added.difference(&snap.defs_added) {
                if let Some(defs) = definers.get(sym) {
                    if defs.len() <= 50 {
                        for a_id in defs {
                            if a_id != &b_id {
                                self.add_edge(
                                    a_id,
                                    &b_id,
                                    EdgeKind::DefinitionBeforeUse,
                                    format!(
                                        "`{sym}` is introduced by {a_id} and referenced by {b_id}"
                                    ),
                                    true,
                                );
                            }
                        }
                    }
                }
            }

            for sym in &snap.refs_removed {
                if let Some(rems) = removers.get(sym) {
                    if rems.len() <= 50 {
                        for a_id in rems {
                            if a_id != &b_id {
                                self.add_edge(
                                    &b_id,
                                    a_id,
                                    EdgeKind::RemovalAfterLastUse,
                                    format!(
                                        "{b_id} removes references to `{sym}`, whose definition is deleted by {a_id}"
                                    ),
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Applies an LLM `DependencyProposal` with strict cycle rejection (`allow_cycle = false`).
    /// Returns `(added_edges, rejected_edges, llm_order_ranks)`.
    pub fn apply_dependency_proposal(
        &mut self,
        proposal: &DependencyProposal,
    ) -> (usize, usize, IndexMap<String, usize>) {
        let mut added = 0usize;
        let mut rejected = 0usize;
        for edge in &proposal.edges {
            let b = edge.before.trim();
            let a = edge.after.trim();
            if self.op(b).is_some() && self.op(a).is_some() {
                let reason = if edge.reason.trim().is_empty() {
                    "LLM-inferred semantic dependency".to_owned()
                } else {
                    edge.reason.trim().to_owned()
                };
                if self.add_edge(b, a, EdgeKind::Semantic, reason, false) {
                    added += 1;
                } else {
                    rejected += 1;
                }
            }
        }

        let mut llm_ranks = IndexMap::new();
        for id in &proposal.suggested_order {
            let trimmed = id.trim();
            if self.op(trimmed).is_some() && !llm_ranks.contains_key(trimmed) {
                let idx = llm_ranks.len();
                llm_ranks.insert(trimmed.to_owned(), idx);
            }
        }
        (added, rejected, llm_ranks)
    }

    /// Computes a deterministic topological linearization using Kahn's algorithm with a priority queue.
    pub fn topological_order(
        &self,
        placement: MechanicalPlacement,
        llm_ranks: &IndexMap<String, usize>,
        items_by_id: &IndexMap<String, ChangeItem>,
    ) -> Result<Vec<Operation>, CoreError> {
        let mut indeg: HashMap<&str, usize> = self.ops.iter().map(|o| (o.id.as_str(), 0)).collect();
        for e in &self.edges {
            if let Some(d) = indeg.get_mut(e.after.as_str()) {
                *d += 1;
            }
        }
        let succ = self.successors();
        let op_map: HashMap<&str, &Operation> =
            self.ops.iter().map(|o| (o.id.as_str(), o)).collect();

        let make_entry = |op: &Operation| -> HeapEntry {
            let tier = if op.is_pure_move_or_mode(items_by_id) {
                0
            } else if op.category == "generated" {
                4
            } else if op.is_mechanical(items_by_id) {
                match placement {
                    MechanicalPlacement::First => 1,
                    MechanicalPlacement::Last => 3,
                }
            } else {
                2
            };
            let llm_rank = llm_ranks.get(&op.id).copied().unwrap_or(usize::MAX);
            HeapEntry {
                tier,
                llm_rank,
                position: op.position(items_by_id),
                id: op.id.clone(),
            }
        };

        let mut heap = BinaryHeap::new();
        for op in &self.ops {
            if indeg[op.id.as_str()] == 0 {
                heap.push(make_entry(op));
            }
        }

        let mut ordered = Vec::with_capacity(self.ops.len());
        while let Some(entry) = heap.pop() {
            let op = op_map[entry.id.as_str()];
            ordered.push(op.clone());
            if let Some(neighbors) = succ.get(entry.id.as_str()) {
                for &s in neighbors {
                    if let Some(d) = indeg.get_mut(s) {
                        *d -= 1;
                        if *d == 0 {
                            heap.push(make_entry(op_map[s]));
                        }
                    }
                }
            }
        }

        if ordered.len() != self.ops.len() {
            return Err(CoreError::GraphCycle {
                visited: ordered.len(),
                total: self.ops.len(),
            });
        }
        Ok(ordered)
    }

    /// Computes strict ancestor closures, transitively reduced direct parents, and sink operation IDs
    /// for a given topological `order`.
    pub fn transitive_reduction(&self, order: &[Operation]) -> TransitiveReduction {
        let order_index: HashMap<&str, usize> = order
            .iter()
            .enumerate()
            .map(|(i, o)| (o.id.as_str(), i))
            .collect();

        let mut direct_preds: HashMap<&str, Vec<&str>> = HashMap::new();
        for e in &self.edges {
            if order_index.contains_key(e.before.as_str())
                && order_index.contains_key(e.after.as_str())
            {
                direct_preds
                    .entry(e.after.as_str())
                    .or_default()
                    .push(e.before.as_str());
            }
        }

        let mut ancestors: IndexMap<String, BTreeSet<String>> = IndexMap::new();
        let mut reduced_parents: IndexMap<String, Vec<String>> = IndexMap::new();

        for op in order {
            let preds = direct_preds
                .get(op.id.as_str())
                .cloned()
                .unwrap_or_default();
            let mut anc = BTreeSet::new();
            for &p in &preds {
                if let Some(p_anc) = ancestors.get(p) {
                    anc.extend(p_anc.iter().cloned());
                }
                anc.insert(p.to_owned());
            }

            let mut reduced: Vec<String> = preds
                .iter()
                .copied()
                .filter(|&d| {
                    !preds.iter().any(|&other| {
                        other != d && ancestors.get(other).is_some_and(|o_anc| o_anc.contains(d))
                    })
                })
                .map(str::to_owned)
                .collect();
            reduced.sort_by_key(|id| order_index.get(id.as_str()).copied().unwrap_or(usize::MAX));
            reduced.dedup();

            ancestors.insert(op.id.clone(), anc);
            reduced_parents.insert(op.id.clone(), reduced);
        }

        let sinks: Vec<String> = order
            .iter()
            .map(|o| o.id.clone())
            .filter(|id| !ancestors.values().any(|anc| anc.contains(id)))
            .collect();

        TransitiveReduction {
            ancestors,
            reduced_parents,
            sinks,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TransitiveReduction {
    pub ancestors: IndexMap<String, BTreeSet<String>>,
    pub reduced_parents: IndexMap<String, Vec<String>>,
    pub sinks: Vec<String>,
}
