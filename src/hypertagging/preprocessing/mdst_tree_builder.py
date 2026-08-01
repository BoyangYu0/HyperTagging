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
    node_kind: str = "unknown"
    candidate_confidence: float | None = None
    track_features: dict[str, float] = field(default_factory=dict)
    cluster_features: dict[str, float] = field(default_factory=dict)
    klm_features: dict[str, float] = field(default_factory=dict)
    raw_pdg: int | None = None
    input_pid_token: int | None = None
    pid_target_token: int | None = None
    truth_pdg: int | None = None
    truth_pid_token: int | None = None
    reco_charge: float | None = None
    truth_charge: float | None = None
    energy_source: str = "legacy_unspecified"
    leaf_kinematics_mode: str = "legacy_unspecified"
    track_energy_hypotheses: dict[str, float] = field(default_factory=dict)
    track_energy_availability: dict[str, bool] = field(default_factory=dict)
    pid_likelihoods: dict[str, float] = field(default_factory=dict)
    pid_likelihood_availability: dict[str, bool] = field(default_factory=dict)
    pid_likelihood_status: dict[str, str] = field(default_factory=dict)
    pid_detector_availability: dict[str, bool] = field(default_factory=dict)
    track_fit_hypothesis: str | None = None
    track_fit_selection_method: str = "not_applicable"
    track_fit_available: bool = False
    track_fit_fallback_reason: str | None = None
    track_fit_policy_diagnostics: dict[str, float | str | bool] = field(default_factory=dict)
    reco_object_id: str | None = None
    associated_reco_id: str | None = None
    recursive_leaf_source_ids: list[str] = field(default_factory=list)
    full_truth_daughter_count: int = 0
    retained_daughter_count: int = 0
    reconstructed_daughter_count: int = 0
    complete_truth_decay: bool = False
    complete_reconstructable_decay: bool = False
    partial_missing_daughters: bool = False
    contracted_intermediate: bool = False
    valid_reconstruction_target: bool = False

    def __post_init__(self) -> None:
        if self.raw_pdg is None:
            self.raw_pdg = self.pdg
        if self.input_pid_token is None:
            self.input_pid_token = tokenize_pdg(self.raw_pdg)
        if self.pid_target_token is None:
            self.pid_target_token = tokenize_pdg(self.pdg)
        if self.truth_pdg is None and self.mc_id is not None:
            self.truth_pdg = self.pdg
        if self.truth_pid_token is None and self.truth_pdg is not None:
            self.truth_pid_token = tokenize_pdg(self.truth_pdg)
        if self.reco_charge is None:
            self.reco_charge = self.charge
        if self.reco_object_id is None:
            self.reco_object_id = self.reco_id

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
    full_truth_channel_record: dict[str, object] = field(default_factory=dict)
    full_truth_channel_record_cc: dict[str, object] = field(default_factory=dict)

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
    node_kind: str = "unknown"
    candidate_confidence: float | None = None
    track_features: dict[str, float] = field(default_factory=dict)
    cluster_features: dict[str, float] = field(default_factory=dict)
    klm_features: dict[str, float] = field(default_factory=dict)
    raw_pdg: int | None = None
    input_pid_token: int | None = None
    pid_target_token: int | None = None
    truth_pdg: int | None = None
    truth_pid_token: int | None = None
    reco_charge: float | None = None
    truth_charge: float | None = None
    energy_source: str = "legacy_unspecified"
    leaf_kinematics_mode: str = "legacy_unspecified"
    track_energy_hypotheses: dict[str, float] = field(default_factory=dict)
    track_energy_availability: dict[str, bool] = field(default_factory=dict)
    pid_likelihoods: dict[str, float] = field(default_factory=dict)
    pid_likelihood_availability: dict[str, bool] = field(default_factory=dict)
    pid_likelihood_status: dict[str, str] = field(default_factory=dict)
    pid_detector_availability: dict[str, bool] = field(default_factory=dict)
    track_fit_hypothesis: str | None = None
    track_fit_selection_method: str = "not_applicable"
    track_fit_available: bool = False
    track_fit_fallback_reason: str | None = None
    track_fit_policy_diagnostics: dict[str, float | str | bool] = field(default_factory=dict)
    reco_quality_score: float | None = None
    relation_class: str = "unclassified"
    underlying_reco_id: str | None = None
    associated_reco_id: str | None = None

    def __post_init__(self) -> None:
        raw_pdg = self.pdg if self.raw_pdg is None else self.raw_pdg
        object.__setattr__(self, "raw_pdg", raw_pdg)
        if self.input_pid_token is None:
            object.__setattr__(self, "input_pid_token", tokenize_pdg(raw_pdg))
        if self.pid_target_token is None:
            target_pdg = self.truth_pdg if self.truth_pdg is not None else raw_pdg
            object.__setattr__(self, "pid_target_token", tokenize_pdg(target_pdg))
        if self.truth_pid_token is None and self.truth_pdg is not None:
            object.__setattr__(self, "truth_pid_token", tokenize_pdg(self.truth_pdg))
        if self.reco_charge is None:
            object.__setattr__(self, "reco_charge", self.charge)
        if self.underlying_reco_id is None:
            object.__setattr__(self, "underlying_reco_id", self.reco_id)


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
    unique_match: int = 0
    duplicate_track: int = 0
    split_cluster: int = 0
    ambiguous_relation: int = 0
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
            "unique_match": self.unique_match,
            "duplicate_track": self.duplicate_track,
            "split_cluster": self.split_cluster,
            "ambiguous_relation": self.ambiguous_relation,
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
    for candidates in reco_by_mc.values():
        candidates.sort(key=_reco_primary_key)

    for mc in mc_records:
        decision = pid_filter.decide(pdg=mc.pdg, name=mc.name, is_primary=mc.is_primary)
        if not decision.keep:
            continue
        primary = reco_by_mc.get(mc.mc_id, [None])[0] if reco_by_mc.get(mc.mc_id) else None
        p4 = primary.p4 if primary is not None else FourVector(0.0, 0.0, 0.0, 0.0)
        flags = set()
        reco_id = None
        if primary is not None:
            reco_id = primary.reco_id
            flags.add(_match_class(reco_by_mc[mc.mc_id]))
        else:
            flags.add("truth_topology_only")
        node = TreeNode(
            node_id=tree.next_node_id(),
            pdg=mc.pdg,
            charge=primary.reco_charge if primary is not None else mc.charge,
            p4=p4,
            reco_id=reco_id,
            mc_id=mc.mc_id,
            flags=flags,
            prod_time=mc.prod_time,
            vertex=mc.vertex,
            mc_p4=mc.p4,
            node_kind=primary.node_kind if primary is not None else "composite",
            candidate_confidence=(
                primary.candidate_confidence if primary is not None else None
            ),
            track_features=(
                dict(primary.track_features) if primary is not None else {}
            ),
            cluster_features=(
                dict(primary.cluster_features) if primary is not None else {}
            ),
            klm_features=(
                dict(primary.klm_features) if primary is not None else {}
            ),
            raw_pdg=primary.raw_pdg if primary is not None else mc.pdg,
            input_pid_token=primary.input_pid_token if primary is not None else 0,
            pid_target_token=tokenize_pdg(mc.pdg),
            truth_pdg=mc.pdg,
            truth_pid_token=tokenize_pdg(mc.pdg),
            reco_charge=primary.reco_charge if primary is not None else None,
            truth_charge=mc.charge,
            energy_source=primary.energy_source if primary is not None else "truth_topology_no_reco_p4",
            leaf_kinematics_mode=(
                primary.leaf_kinematics_mode if primary is not None else "truth_topology_only"
            ),
            track_energy_hypotheses=(
                dict(primary.track_energy_hypotheses) if primary is not None else {}
            ),
            track_energy_availability=(
                dict(primary.track_energy_availability) if primary is not None else {}
            ),
            pid_likelihoods=dict(primary.pid_likelihoods) if primary is not None else {},
            pid_likelihood_availability=(
                dict(primary.pid_likelihood_availability) if primary is not None else {}
            ),
            pid_likelihood_status=(
                dict(primary.pid_likelihood_status) if primary is not None else {}
            ),
            pid_detector_availability=(
                dict(primary.pid_detector_availability) if primary is not None else {}
            ),
            track_fit_hypothesis=(
                primary.track_fit_hypothesis if primary is not None else None
            ),
            track_fit_selection_method=(
                primary.track_fit_selection_method
                if primary is not None
                else "not_applicable"
            ),
            track_fit_available=(
                primary.track_fit_available if primary is not None else False
            ),
            track_fit_fallback_reason=(
                primary.track_fit_fallback_reason if primary is not None else None
            ),
            track_fit_policy_diagnostics=(
                dict(primary.track_fit_policy_diagnostics) if primary is not None else {}
            ),
            reco_object_id=primary.underlying_reco_id if primary is not None else None,
            associated_reco_id=primary.associated_reco_id if primary is not None else None,
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
        if mc.mother_id != retained_mother:
            tree.nodes[parent_id].contracted_intermediate = True
            tree.nodes[child_id].flags.add("contracted_intermediate_path")
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
            has_retained_truth_match = (
                reco.mc_id is not None and int(reco.mc_id) in retained_by_mc
            )
            relation_flags = (
                {_match_class(reco_by_mc[int(reco.mc_id)]), "competing_reco"}
                if has_retained_truth_match
                else {"unmatched_reco"}
            )
            node = TreeNode(
                node_id=tree.next_node_id(),
                pdg=reco.pdg,
                charge=reco.charge,
                p4=reco.p4,
                reco_id=reco.reco_id,
                mc_id=reco.mc_id,
                flags=set(reco.flags) | relation_flags,
                node_kind=reco.node_kind,
                candidate_confidence=reco.candidate_confidence,
                track_features=dict(reco.track_features),
                cluster_features=dict(reco.cluster_features),
                klm_features=dict(reco.klm_features),
                raw_pdg=reco.raw_pdg,
                input_pid_token=reco.input_pid_token,
                pid_target_token=reco.pid_target_token,
                truth_pdg=reco.truth_pdg,
                truth_pid_token=reco.truth_pid_token,
                reco_charge=reco.reco_charge,
                truth_charge=reco.truth_charge,
                energy_source=reco.energy_source,
                leaf_kinematics_mode=reco.leaf_kinematics_mode,
                track_energy_hypotheses=dict(reco.track_energy_hypotheses),
                track_energy_availability=dict(reco.track_energy_availability),
                pid_likelihoods=dict(reco.pid_likelihoods),
                pid_likelihood_availability=dict(reco.pid_likelihood_availability),
                pid_likelihood_status=dict(reco.pid_likelihood_status),
                pid_detector_availability=dict(reco.pid_detector_availability),
                track_fit_hypothesis=reco.track_fit_hypothesis,
                track_fit_selection_method=reco.track_fit_selection_method,
                track_fit_available=reco.track_fit_available,
                track_fit_fallback_reason=reco.track_fit_fallback_reason,
                track_fit_policy_diagnostics=dict(reco.track_fit_policy_diagnostics),
                reco_object_id=reco.underlying_reco_id,
                associated_reco_id=reco.associated_reco_id,
            )
            tree.add_node(node)
            tree.root_ids.append(node.node_id)

    # This snapshot is computed before reco-efficiency removal.  It contains
    # topology/PID labels only; channel construction never reads node p4.
    from hypertagging.preprocessing.channels import event_channel_record

    tree.full_truth_channel_record = event_channel_record(tree)
    tree.full_truth_channel_record_cc = event_channel_record(
        tree,
        charge_conjugate_normalize=True,
    )
    _drop_empty_truth_leaves(tree)
    recompute_mother_p4_from_daughters(tree)
    _annotate_reconstruction_contract(tree, mc_records)
    return tree


def _reco_primary_key(record: RecoRecord) -> tuple[float, str]:
    """Reco-only, deterministic primary-candidate policy.

    Larger finite fit/candidate quality is preferred.  Stable underlying
    provenance breaks ties.  Neither MC PID nor truth kinematics is consulted.
    """

    quality = record.reco_quality_score
    if quality is None:
        quality = record.candidate_confidence
    finite_quality = float(quality) if quality is not None and math.isfinite(quality) else -math.inf
    return (-finite_quality, str(record.underlying_reco_id or record.reco_id))


def _match_class(records: Sequence[RecoRecord]) -> str:
    if len(records) == 1:
        return "unique_match"
    kinds = {record.node_kind for record in records}
    if kinds == {"track"}:
        return "duplicate_track"
    if kinds == {"ecl_cluster"}:
        return "split_cluster"
    return "ambiguous_relation"


def _annotate_reconstruction_contract(tree: EventTree, mc_records: Sequence[MCRecord]) -> None:
    """Attach completeness and recursive reconstructed-source metadata."""

    truth_children = Counter(record.mother_id for record in mc_records if record.mother_id is not None)
    for node in tree.nodes.values():
        node.full_truth_daughter_count = int(truth_children[node.mc_id]) if node.mc_id is not None else 0
        node.retained_daughter_count = len(node.daughter_ids)
        node.reconstructed_daughter_count = sum(
            tree.nodes[child_id].reco_id is not None or bool(tree.nodes[child_id].daughter_ids)
            for child_id in node.daughter_ids
        )
        if node.daughter_ids:
            node.complete_truth_decay = (
                node.full_truth_daughter_count > 0
                and node.retained_daughter_count == node.full_truth_daughter_count
            )
            node.complete_reconstructable_decay = (
                node.reconstructed_daughter_count == node.retained_daughter_count
            )
            node.partial_missing_daughters = (
                node.full_truth_daughter_count > node.retained_daughter_count
                or node.reconstructed_daughter_count < node.retained_daughter_count
            )
            node.valid_reconstruction_target = (
                node.retained_daughter_count >= 2 and node.complete_reconstructable_decay
            )

    for node_id in _topological_children_first(tree):
        node = tree.nodes[node_id]
        if node.daughter_ids:
            sources = {
                source
                for child_id in node.daughter_ids
                for source in tree.nodes[child_id].recursive_leaf_source_ids
            }
        else:
            source = node.reco_object_id or node.reco_id
            sources = {source} if source else set()
            if node.associated_reco_id:
                sources.add(node.associated_reco_id)
        node.recursive_leaf_source_ids = sorted(sources)


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
        node_kind=source.node_kind,
        candidate_confidence=source.candidate_confidence,
        track_features=dict(source.track_features),
        cluster_features=dict(source.cluster_features),
        klm_features=dict(source.klm_features),
        raw_pdg=source.raw_pdg,
        input_pid_token=source.input_pid_token,
        pid_target_token=source.pid_target_token,
        truth_pdg=source.truth_pdg,
        truth_pid_token=source.truth_pid_token,
        reco_charge=source.reco_charge,
        truth_charge=source.truth_charge,
        energy_source=source.energy_source,
        leaf_kinematics_mode=source.leaf_kinematics_mode,
        track_energy_hypotheses=dict(source.track_energy_hypotheses),
        track_energy_availability=dict(source.track_energy_availability),
        pid_likelihoods=dict(source.pid_likelihoods),
        pid_likelihood_availability=dict(source.pid_likelihood_availability),
        pid_likelihood_status=dict(source.pid_likelihood_status),
        pid_detector_availability=dict(source.pid_detector_availability),
        track_fit_hypothesis=source.track_fit_hypothesis,
        track_fit_selection_method=source.track_fit_selection_method,
        track_fit_available=source.track_fit_available,
        track_fit_fallback_reason=source.track_fit_fallback_reason,
        track_fit_policy_diagnostics=dict(source.track_fit_policy_diagnostics),
        reco_object_id=source.reco_object_id,
        associated_reco_id=source.associated_reco_id,
        recursive_leaf_source_ids=list(source.recursive_leaf_source_ids),
        full_truth_daughter_count=source.full_truth_daughter_count,
        retained_daughter_count=source.retained_daughter_count,
        reconstructed_daughter_count=source.reconstructed_daughter_count,
        complete_truth_decay=source.complete_truth_decay,
        complete_reconstructable_decay=source.complete_reconstructable_decay,
        partial_missing_daughters=source.partial_missing_daughters,
        contracted_intermediate=source.contracted_intermediate,
        valid_reconstruction_target=source.valid_reconstruction_target,
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
            summary.unique_match += sum("unique_match" in node.flags for node in tree.nodes.values())
            summary.duplicate_track += sum("duplicate_track" in node.flags for node in tree.nodes.values())
            summary.split_cluster += sum("split_cluster" in node.flags for node in tree.nodes.values())
            summary.ambiguous_relation += sum(
                "ambiguous_relation" in node.flags for node in tree.nodes.values()
            )
            trees.append(tree)
        except Exception:
            summary.failed_events += 1
            raise
        finally:
            summary.timings["tree_build_seconds"] += time.perf_counter() - start
    summary.pid_summary = pid_filter.summary.as_dict()
    return trees, summary
