"""Event tree objects and direct-mDST tree construction helpers.

The core rule is explicit: MC information can define training topology and
labels, but reconstructed mother kinematics are always recomputed from retained
daughter four-vectors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Sequence

from hypertagging.preprocessing.pid_filter import PidFilter, tokenize_pdg


@dataclass
class FourVector:
    """Simple HEP four-vector with invariant mass helper."""

    px: float
    py: float
    pz: float
    energy: float

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
            self.energy + other.energy,
        )

    @classmethod
    def sum(cls, vectors: Iterable["FourVector"]) -> "FourVector":
        total = cls(0.0, 0.0, 0.0, 0.0)
        for vector in vectors:
            total = total + vector
        return total

    @property
    def mass(self) -> float:
        mass2 = self.energy * self.energy - self.px * self.px - self.py * self.py - self.pz * self.pz
        return math.sqrt(max(mass2, 0.0))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.px, self.py, self.pz, self.energy)


@dataclass
class TreeNode:
    """One retained or copied node in an event-level reconstruction tree."""

    node_id: int
    pdg: int
    charge: float
    p4: FourVector
    parent_id: int | None = None
    daughter_ids: list[int] = field(default_factory=list)
    level: int = -1
    reco_id: str | None = None
    mc_id: int | None = None
    source_node_id: int | None = None
    copied_from: int | None = None
    flags: set[str] = field(default_factory=set)
    prod_time: float = 0.0
    vertex: tuple[float, float, float] = (0.0, 0.0, 0.0)
    n_daughters: int = 0
    mc_p4: FourVector | None = None

    @property
    def token(self) -> int:
        return tokenize_pdg(self.pdg)

    def feature(self) -> list[float]:
        """Return legacy ordered features, including tokenized PDG as column 0."""

        return [
            float(self.token),
            float(self.p4.mass),
            float(self.charge),
            float(self.p4.energy),
            float(self.prod_time),
            float(self.vertex[0]),
            float(self.vertex[1]),
            float(self.vertex[2]),
            float(self.p4.px),
            float(self.p4.py),
            float(self.p4.pz),
            float(self.n_daughters),
        ]


@dataclass
class EventTree:
    """A possibly truth-guided reco-kinematic event tree."""

    event_id: int
    nodes: dict[int, TreeNode] = field(default_factory=dict)
    root_ids: list[int] = field(default_factory=list)
    metadata: dict[str, int | float | str] = field(default_factory=dict)

    def add_node(self, node: TreeNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id {node.node_id}")
        self.nodes[node.node_id] = node

    def next_node_id(self) -> int:
        return (max(self.nodes) + 1) if self.nodes else 0

    def children(self, node_id: int) -> list[TreeNode]:
        return [self.nodes[child_id] for child_id in self.nodes[node_id].daughter_ids]


@dataclass(frozen=True)
class RecoRecord:
    """Basf2-independent reconstructed object record."""

    reco_id: str
    pdg: int
    charge: float
    p4: FourVector
    mc_id: int | None = None
    flags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class MCRecord:
    """Basf2-independent truth record used for topology and labels."""

    mc_id: int
    pdg: int
    charge: float
    mother_id: int | None
    p4: FourVector | None = None
    name: str | None = None
    is_primary: bool | None = None
    prod_time: float = 0.0
    vertex: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class BuildSummary:
    """Counters for preprocessing diagnostics."""

    events: int = 0
    nodes_before_pruning: int = 0
    nodes_after_pruning: int = 0
    copied_nodes: int = 0
    unmatched_reco: int = 0
    failed_events: int = 0
    timings: Counter[str] = field(default_factory=Counter)
    pid_summary: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "events": self.events,
            "nodes_before_pruning": self.nodes_before_pruning,
            "nodes_after_pruning": self.nodes_after_pruning,
            "copied_nodes": self.copied_nodes,
            "unmatched_reco": self.unmatched_reco,
            "failed_events": self.failed_events,
            "timings": dict(self.timings),
            "pid_summary": self.pid_summary,
        }


def build_truth_guided_tree(
    *,
    event_id: int,
    mc_records: Sequence[MCRecord],
    reco_records: Sequence[RecoRecord],
    pid_filter: PidFilter | None = None,
    include_unmatched_reco: bool = True,
) -> EventTree:
    """Build a retained tree from truth topology and reconstructed leaves.

    Retained MC nodes provide topology/labels. Nodes with matched reconstructed
    records use reco p4 directly. Retained mother p4 is recomputed after
    contraction from retained daughters and never copied from MC truth.
    """

    pid_filter = pid_filter or PidFilter()
    tree = EventTree(event_id=event_id)
    retained_by_mc: dict[int, int] = {}
    reco_by_mc: dict[int, list[RecoRecord]] = defaultdict(list)

    for reco in reco_records:
        if reco.mc_id is not None:
            reco_by_mc[int(reco.mc_id)].append(reco)

    for mc in mc_records:
        decision = pid_filter.decide(pdg=mc.pdg, name=mc.name, is_primary=mc.is_primary)
        if not decision.keep:
            continue
        p4 = reco_by_mc.get(mc.mc_id, [None])[0].p4 if reco_by_mc.get(mc.mc_id) else FourVector(0.0, 0.0, 0.0, 0.0)
        flags = set()
        reco_id = None
        if reco_by_mc.get(mc.mc_id):
            reco_id = reco_by_mc[mc.mc_id][0].reco_id
        else:
            flags.add("truth_topology_only")
        node = TreeNode(
            node_id=tree.next_node_id(),
            pdg=mc.pdg,
            charge=mc.charge,
            p4=p4,
            reco_id=reco_id,
            mc_id=mc.mc_id,
            flags=flags,
            prod_time=mc.prod_time,
            vertex=mc.vertex,
            mc_p4=mc.p4,
        )
        tree.add_node(node)
        retained_by_mc[mc.mc_id] = node.node_id

    for mc in mc_records:
        child_id = retained_by_mc.get(mc.mc_id)
        if child_id is None:
            continue
        retained_mother = _nearest_retained_mother(mc.mother_id, {m.mc_id: m for m in mc_records}, retained_by_mc)
        if retained_mother is None:
            tree.root_ids.append(child_id)
            continue
        parent_id = retained_by_mc[retained_mother]
        tree.nodes[child_id].parent_id = parent_id
        tree.nodes[parent_id].daughter_ids.append(child_id)

    matched_reco_ids = {node.reco_id for node in tree.nodes.values() if node.reco_id is not None}
    if include_unmatched_reco:
        for reco in reco_records:
            if reco.reco_id in matched_reco_ids:
                continue
            decision = pid_filter.decide(pdg=reco.pdg, is_primary=True)
            if not decision.keep:
                continue
            node = TreeNode(
                node_id=tree.next_node_id(),
                pdg=reco.pdg,
                charge=reco.charge,
                p4=reco.p4,
                reco_id=reco.reco_id,
                mc_id=reco.mc_id,
                flags=set(reco.flags) | {"unmatched_reco"},
            )
            tree.add_node(node)
            tree.root_ids.append(node.node_id)

    _drop_empty_truth_leaves(tree)
    recompute_mother_p4_from_daughters(tree)
    return tree


def _nearest_retained_mother(
    mother_id: int | None,
    mc_by_id: dict[int, MCRecord],
    retained_by_mc: dict[int, int],
) -> int | None:
    while mother_id is not None:
        if mother_id in retained_by_mc:
            return mother_id
        mother = mc_by_id.get(mother_id)
        mother_id = mother.mother_id if mother is not None else None
    return None


def _drop_empty_truth_leaves(tree: EventTree) -> None:
    """Remove retained truth-only leaves that have no reco evidence."""

    changed = True
    while changed:
        changed = False
        for node_id, node in list(tree.nodes.items()):
            if node.daughter_ids or "truth_topology_only" not in node.flags:
                continue
            if node.parent_id is not None and node.parent_id in tree.nodes:
                parent = tree.nodes[node.parent_id]
                parent.daughter_ids = [child_id for child_id in parent.daughter_ids if child_id != node_id]
            if node_id in tree.root_ids:
                tree.root_ids.remove(node_id)
            del tree.nodes[node_id]
            changed = True


def recompute_mother_p4_from_daughters(tree: EventTree) -> None:
    """Recursively set every mother p4 to the sum of retained daughter p4."""

    ordered = _topological_children_first(tree)
    for node_id in ordered:
        node = tree.nodes[node_id]
        node.n_daughters = len(node.daughter_ids)
        if node.daughter_ids:
            node.p4 = FourVector.sum(tree.nodes[child_id].p4 for child_id in node.daughter_ids)


def _topological_children_first(tree: EventTree) -> list[int]:
    visited: set[int] = set()
    visiting: set[int] = set()
    ordered: list[int] = []

    def visit(node_id: int) -> None:
        if node_id in visiting:
            raise ValueError(f"cycle detected at node {node_id}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for child_id in tree.nodes[node_id].daughter_ids:
            if child_id not in tree.nodes:
                raise ValueError(f"node {node_id} references missing daughter {child_id}")
            visit(child_id)
        visiting.remove(node_id)
        visited.add(node_id)
        ordered.append(node_id)

    for node_id in list(tree.nodes):
        visit(node_id)
    return ordered


def copy_shared_daughters(tree: EventTree) -> int:
    """Copy nodes when one child is referenced by multiple parents."""

    parents_by_child: dict[int, list[int]] = defaultdict(list)
    for parent in tree.nodes.values():
        for child_id in parent.daughter_ids:
            parents_by_child[child_id].append(parent.node_id)

    copied = 0
    for child_id, parent_ids in list(parents_by_child.items()):
        if len(parent_ids) <= 1:
            continue
        for parent_id in parent_ids[1:]:
            new_id = _clone_subtree(tree, child_id)
            daughters = tree.nodes[parent_id].daughter_ids
            tree.nodes[parent_id].daughter_ids = [new_id if existing == child_id else existing for existing in daughters]
            tree.nodes[new_id].parent_id = parent_id
            copied += 1
    return copied


def _clone_subtree(tree: EventTree, node_id: int) -> int:
    source = tree.nodes[node_id]
    new_id = tree.next_node_id()
    clone = TreeNode(
        node_id=new_id,
        pdg=source.pdg,
        charge=source.charge,
        p4=source.p4,
        parent_id=source.parent_id,
        daughter_ids=[],
        level=source.level,
        reco_id=source.reco_id,
        mc_id=source.mc_id,
        source_node_id=source.source_node_id if source.source_node_id is not None else source.node_id,
        copied_from=source.node_id,
        flags=set(source.flags) | {"copied"},
        prod_time=source.prod_time,
        vertex=source.vertex,
        n_daughters=source.n_daughters,
        mc_p4=source.mc_p4,
    )
    tree.add_node(clone)
    for child_id in source.daughter_ids:
        child_clone_id = _clone_subtree(tree, child_id)
        tree.nodes[child_clone_id].parent_id = new_id
        clone.daughter_ids.append(child_clone_id)
    return new_id


def validate_tree(tree: EventTree, *, p4_tolerance: float = 1e-8) -> dict[str, float | int]:
    """Validate DAG, parent/daughter links, levels, copies, and p4 sums."""

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for node in tree.nodes.values():
        for child_id in node.daughter_ids:
            if child_id not in tree.nodes:
                raise ValueError(f"node {node.node_id} references missing daughter {child_id}")
            if tree.nodes[child_id].parent_id != node.node_id:
                raise ValueError(f"child {child_id} parent mismatch")
            if node.level <= tree.nodes[child_id].level:
                raise ValueError(f"level violation parent {node.node_id} child {child_id}")
        if node.parent_id is not None and node.parent_id not in tree.nodes:
            raise ValueError(f"node {node.node_id} references missing parent {node.parent_id}")
        if node.copied_from is not None and node.copied_from not in tree.nodes:
            raise ValueError(f"copied node {node.node_id} has missing source {node.copied_from}")
        if node.daughter_ids:
            summed = FourVector.sum(tree.nodes[child_id].p4 for child_id in node.daughter_ids)
            for stored, expected in zip(node.p4.as_tuple(), summed.as_tuple()):
                diff = abs(stored - expected)
                rel = diff / max(abs(expected), p4_tolerance)
                max_abs_diff = max(max_abs_diff, diff)
                max_rel_diff = max(max_rel_diff, rel)
                if diff > p4_tolerance and rel > p4_tolerance:
                    raise ValueError(
                        f"mother {node.node_id} p4 differs from daughter sum: "
                        f"stored={node.p4.as_tuple()} summed={summed.as_tuple()}"
                    )
    _topological_children_first(tree)
    return {"nodes": len(tree.nodes), "max_abs_p4_diff": max_abs_diff, "max_rel_p4_diff": max_rel_diff}


def build_trees(
    events: Iterable[tuple[int, Sequence[MCRecord], Sequence[RecoRecord]]],
    *,
    pid_filter: PidFilter | None = None,
) -> tuple[list[EventTree], BuildSummary]:
    """Build and validate many event trees with timing counters."""

    pid_filter = pid_filter or PidFilter()
    summary = BuildSummary()
    trees: list[EventTree] = []
    for event_id, mc_records, reco_records in events:
        start = time.perf_counter()
        summary.events += 1
        summary.nodes_before_pruning += len(mc_records) + len(reco_records)
        try:
            tree = build_truth_guided_tree(
                event_id=event_id,
                mc_records=mc_records,
                reco_records=reco_records,
                pid_filter=pid_filter,
            )
            summary.copied_nodes += copy_shared_daughters(tree)
            recompute_mother_p4_from_daughters(tree)
            summary.nodes_after_pruning += len(tree.nodes)
            summary.unmatched_reco += sum("unmatched_reco" in node.flags for node in tree.nodes.values())
            trees.append(tree)
        except Exception:
            summary.failed_events += 1
            raise
        finally:
            summary.timings["tree_build_seconds"] += time.perf_counter() - start
    summary.pid_summary = pid_filter.summary.as_dict()
    return trees, summary
