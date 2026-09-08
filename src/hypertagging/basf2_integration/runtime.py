"""Truth-free ONNX inference and full-depth hierarchical beam search.

This module deliberately depends only on NumPy and the Python standard
library.  basf2 imports ONNX Runtime lazily when a model bundle is opened; no
training package, PyTorch checkpoint, pickle, or project virtual environment
is loaded in event processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import itertools
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from hypertagging.deployment.onnx_contract import (
    FEATURE_NAMES,
    MODEL_OUTPUT_NAMES,
    file_sha256,
    load_bundle_manifest,
)


NODE_KIND_UNKNOWN = 0
NODE_KIND_TRACK = 1
NODE_KIND_ECL = 2
NODE_KIND_COMPOSITE = 3
NODE_KIND_OTHER = 4
NODE_KIND_KLM = 5

LEAF_MODE_RAW_TRACK = 0
LEAF_MODE_FIXED = 1
LEAF_MODE_ECL = 2
LEAF_MODE_KLM = 3
LEAF_MODE_COMPOSITE = 4

COMPOSITE_TYPE_INPUT_FIXED = 0
COMPOSITE_TYPE_PREDICTED = 2

CHARGED_STABLE_MASSES_GEV: dict[int, float] = {
    11: 0.00051099895,
    13: 0.1056583755,
    211: 0.13957039,
    321: 0.493677,
    2212: 0.93827208816,
}
CHARGED_STABLE_CHARGES: dict[int, int] = {
    11: -1, -11: 1,
    13: -1, -13: 1,
    211: 1, -211: -1,
    321: 1, -321: -1,
    2212: 1, -2212: -1,
}


@dataclass(frozen=True)
class Node:
    """One detector leaf or model-created composite in a beam hypothesis."""

    node_id: int
    input_token: int
    current_token: int
    pdg: int
    p4: tuple[float, float, float, float]
    charge: float
    level: int
    kind_id: int
    leaf_mode_id: int
    parent_id: int = -1
    daughter_ids: tuple[int, ...] = ()
    source_keys: frozenset[str] = frozenset()
    candidate_confidence: float = 0.0
    object_score: float = 0.0
    query_id: int = -1
    common_features: tuple[float, ...] = ()
    common_availability: tuple[bool, ...] = ()
    track_features: tuple[float, ...] = ()
    track_availability: tuple[bool, ...] = ()
    cluster_features: tuple[float, ...] = ()
    cluster_availability: tuple[bool, ...] = ()
    klm_features: tuple[float, ...] = ()
    klm_availability: tuple[bool, ...] = ()
    source_object: object | None = field(default=None, compare=False, repr=False)
    source_kind: str = ""

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be non-negative")
        if self.level < 0:
            raise ValueError("node level must be non-negative")
        if len(self.p4) != 4 or not all(math.isfinite(float(value)) for value in self.p4):
            raise ValueError("node p4 must contain four finite values")
        if not math.isfinite(float(self.charge)):
            raise ValueError("node charge must be finite")
        if self.level == 0 and not self.source_keys:
            raise ValueError("detector leaves require reconstructed source provenance")
        if self.level > 0 and len(self.daughter_ids) < 2:
            raise ValueError("reconstructed composites require at least two daughters")


@dataclass(frozen=True)
class Proposal:
    query_id: int
    mother_token: int
    daughter_ids: tuple[int, ...]
    object_score: float
    type_probability: float
    pointer_quality: float
    learned_confidence: float
    confidence: float


@dataclass(frozen=True)
class Hypothesis:
    nodes: tuple[Node, ...]
    score: float = 0.0
    accepted_by_level: tuple[tuple[int, ...], ...] = ()
    completed_root_id: int | None = None
    empty_levels: int = 0

    def node_by_id(self) -> dict[int, Node]:
        return {node.node_id: node for node in self.nodes}

    def fingerprint(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            sorted(
                (
                    node.level,
                    node.current_token,
                    tuple(sorted(node.source_keys)),
                    tuple(sorted(node.daughter_ids)),
                    node.parent_id,
                )
                for node in self.nodes
            )
        )


@dataclass(frozen=True)
class ModelOutputs:
    object_logits: np.ndarray
    type_logits: np.ndarray
    pointer_logits: np.ndarray
    cardinality_logits: np.ndarray
    confidence_logits: np.ndarray
    leaf_pid_logits: np.ndarray
    current_p4: np.ndarray


@dataclass(frozen=True)
class ReconstructionResult:
    best: Hypothesis
    beam: tuple[Hypothesis, ...]
    stop_reason: str
    levels_processed: int
    model_manifest_sha256: str

    @property
    def completed(self) -> bool:
        return self.best.completed_root_id is not None


class LevelScorer(Protocol):
    manifest: Mapping[str, Any]
    manifest_sha256: str

    def score(self, level: int, inputs: Mapping[str, np.ndarray]) -> ModelOutputs:
        ...


class OnnxModelBundle:
    """Load one hash-checked ONNX session per reconstruction level."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        intra_op_threads: int = 1,
        session_factory: Callable[[str, object], object] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_bundle_manifest(self.manifest_path)
        self.manifest_sha256 = file_sha256(self.manifest_path)
        self.bundle_dir = self.manifest_path.parent
        compatible = self.manifest.get("runtime", {}).get("basf2_releases", [])
        active_release = os.environ.get("BELLE2_RELEASE", "")
        if compatible and active_release and active_release not in compatible:
            raise RuntimeError(
                f"ONNX bundle is not declared compatible with basf2 {active_release!r}; "
                f"allowed releases: {compatible}"
            )
        if session_factory is None:
            try:
                import onnxruntime as ort  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError(
                    "onnxruntime is required for basf2 inference; use a CVMFS "
                    "release that provides it (the current light release does)"
                ) from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, int(intra_op_threads))
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            def session_factory(filename: str, _options: object) -> object:
                return ort.InferenceSession(
                    filename,
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )

        self._sessions: dict[int, object] = {}
        for level_text, entry in self.manifest["models"].items():
            level = int(level_text)
            model_path = (self.bundle_dir / entry["file"]).resolve()
            if model_path.parent != self.bundle_dir:
                raise ValueError("ONNX model path escapes its bundle directory")
            actual = file_sha256(model_path)
            if actual != entry["sha256"]:
                raise ValueError(
                    f"ONNX model hash mismatch for level {level}: "
                    f"{actual} != {entry['sha256']}"
                )
            session = session_factory(str(model_path), None)
            session_inputs = [item.name for item in session.get_inputs()]
            session_outputs = [item.name for item in session.get_outputs()]
            if session_inputs != entry["runtime_inputs"]:
                raise ValueError(
                    f"level {level} ONNX inputs differ from manifest: "
                    f"{session_inputs} != {entry['runtime_inputs']}"
                )
            if session_outputs != list(MODEL_OUTPUT_NAMES):
                raise ValueError(
                    f"level {level} ONNX outputs differ from contract: "
                    f"{session_outputs} != {list(MODEL_OUTPUT_NAMES)}"
                )
            self._sessions[level] = session

    def score(self, level: int, inputs: Mapping[str, np.ndarray]) -> ModelOutputs:
        session = self._sessions.get(int(level))
        if session is None:
            raise ValueError(f"ONNX bundle has no scorer for level {level}")
        names = [item.name for item in session.get_inputs()]
        missing = [name for name in names if name not in inputs]
        if missing:
            raise ValueError(f"missing ONNX inputs for level {level}: {missing}")
        values = session.run(list(MODEL_OUTPUT_NAMES), {name: inputs[name] for name in names})
        outputs = ModelOutputs(*[np.asarray(value) for value in values])
        _validate_model_outputs(outputs, self.manifest["contract"])
        return outputs


class BeamSearchReconstructor:
    """Run current-forest, source-exclusive reconstruction to full depth."""

    def __init__(self, scorer: LevelScorer) -> None:
        self.scorer = scorer
        self.manifest = scorer.manifest
        self.contract = self.manifest["contract"]
        self.policy = self.manifest["reconstruction_policy"]
        self.pid_tokens = tuple(int(pdg) for pdg in self.contract["pid_tokens"])
        self.token_by_pdg = {pdg: index for index, pdg in enumerate(self.pid_tokens)}
        self.root_tokens = frozenset(int(value) for value in self.policy["root_tokens"])
        self.allowed_static = frozenset(
            int(value) for value in self.policy["static_allowed_mother_tokens"]
        )
        self.allowed_by_level = {
            int(level): frozenset(int(value) for value in values)
            for level, values in self.policy.get("allowed_mother_types_by_level", [])
        }
        self.token_charge = tuple(float(value) for value in self.policy["token_charge"])
        if len(self.token_charge) != len(self.pid_tokens):
            raise ValueError("token charge vector does not match PID vocabulary")

    def reconstruct(self, leaves: Sequence[Node]) -> ReconstructionResult:
        if not leaves:
            raise ValueError("full reconstruction requires at least one FSP")
        if any(node.level != 0 for node in leaves):
            raise ValueError("initial reconstruction input must contain level-zero FSPs only")
        if len({node.node_id for node in leaves}) != len(leaves):
            raise ValueError("initial FSP node IDs must be unique")
        if sorted(node.node_id for node in leaves) != list(range(len(leaves))):
            raise ValueError("initial FSP node IDs must be dense and zero based")
        if len(leaves) > int(self.contract["max_nodes"]):
            raise ValueError("FSP multiplicity exceeds ONNX bundle max_nodes")
        beam = [Hypothesis(nodes=tuple(sorted(leaves, key=lambda node: node.node_id)))]
        completed_pool: dict[tuple[tuple[object, ...], ...], Hypothesis] = {}
        levels_processed = 0
        for level in (int(value) for value in self.contract["levels"]):
            levels_processed = level
            expanded: list[Hypothesis] = []
            for hypothesis in beam:
                if hypothesis.completed_root_id is not None:
                    expanded.append(hypothesis)
                    continue
                inputs = self.pack_hypothesis(hypothesis, level)
                outputs = self.scorer.score(level, inputs)
                pid_adjusted = self._apply_hard_leaf_pid(hypothesis, outputs)
                proposals = self._decode_proposals(pid_adjusted, outputs, level)
                if not proposals:
                    expanded.append(
                        replace(
                            pid_adjusted,
                            accepted_by_level=pid_adjusted.accepted_by_level + ((),),
                            empty_levels=pid_adjusted.empty_levels + 1,
                        )
                    )
                    continue
                proposal_sets = self._conflict_free_proposal_sets(
                    pid_adjusted, proposals
                )
                for proposal_set in proposal_sets:
                    if len(pid_adjusted.nodes) + len(proposal_set) > int(
                        self.contract["max_nodes"]
                    ):
                        continue
                    expanded.append(self._append_proposals(pid_adjusted, proposal_set, level))
            if not expanded:
                break
            deduplicated: dict[tuple[tuple[object, ...], ...], Hypothesis] = {}
            for hypothesis in expanded:
                fingerprint = hypothesis.fingerprint()
                previous = deduplicated.get(fingerprint)
                if previous is None or _hypothesis_sort_key(hypothesis) < _hypothesis_sort_key(previous):
                    deduplicated[fingerprint] = hypothesis
            for fingerprint, hypothesis in deduplicated.items():
                if hypothesis.completed_root_id is None:
                    continue
                previous = completed_pool.get(fingerprint)
                if (
                    previous is None
                    or _hypothesis_sort_key(hypothesis)
                    < _hypothesis_sort_key(previous)
                ):
                    completed_pool[fingerprint] = hypothesis
            completed_pool = dict(
                sorted(completed_pool.items(), key=lambda item: _hypothesis_sort_key(item[1]))[
                    : int(self.policy["beam_width"])
                ]
            )
            beam = sorted(deduplicated.values(), key=_hypothesis_sort_key)[
                : int(self.policy["beam_width"])
            ]
            if beam and all(item.completed_root_id is not None for item in beam):
                break
        completed = list(completed_pool.values())
        candidates = completed or beam
        if not candidates:
            raise RuntimeError("beam search produced no hypotheses")
        best = sorted(candidates, key=_hypothesis_sort_key)[0]
        reason = "configured_root_reconstructed" if completed else "max_level_reached_without_root"
        return ReconstructionResult(
            best=best,
            beam=tuple(beam),
            stop_reason=reason,
            levels_processed=levels_processed,
            model_manifest_sha256=self.scorer.manifest_sha256,
        )

    def pack_hypothesis(self, hypothesis: Hypothesis, level: int) -> dict[str, np.ndarray]:
        """Compact one tree state into the exact fixed-width ONNX contract."""

        nodes = tuple(sorted(hypothesis.nodes, key=lambda node: node.node_id))
        max_nodes = int(self.contract["max_nodes"])
        max_sources = int(self.contract["max_sources"])
        if len(nodes) > max_nodes:
            raise ValueError("beam hypothesis exceeds ONNX max_nodes")
        positions = {node.node_id: index for index, node in enumerate(nodes)}
        source_keys = sorted(set().union(*(node.source_keys for node in nodes)))
        if len(source_keys) > max_sources:
            raise ValueError("beam hypothesis exceeds ONNX max_sources")
        source_positions = {key: index for index, key in enumerate(source_keys)}

        result: dict[str, np.ndarray] = {}
        for block in ("common", "track", "cluster", "klm", "composite"):
            width = len(FEATURE_NAMES[block])
            result[f"{block}_features"] = np.zeros((1, max_nodes, width), dtype=np.float32)
            result[f"{block}_availability"] = np.zeros((1, max_nodes, width), dtype=np.bool_)
        result["daughter_input_pid_histogram"] = np.zeros(
            (1, max_nodes, len(self.pid_tokens)), dtype=np.float32
        )
        result["daughter_input_pid_histogram_available"] = np.zeros(
            (1, max_nodes), dtype=np.bool_
        )
        for name, fill in (
            ("node_kind_ids", 0),
            ("leaf_kinematics_mode_ids", 0),
            ("pid_labels", 0),
            ("level_ids", -1),
            ("parent_ids", -1),
            ("node_ids", -1),
            ("reco_ids", -1),
            ("source_node_ids", -1),
            ("copied_from", -1),
            ("runtime_composite_type_source_ids", COMPOSITE_TYPE_INPUT_FIXED),
        ):
            result[name] = np.full((1, max_nodes), fill, dtype=np.int64)
        result["p4"] = np.zeros((1, max_nodes, 4), dtype=np.float32)
        result["charge"] = np.zeros((1, max_nodes), dtype=np.float32)
        result["daughter_adjacency"] = np.zeros(
            (1, max_nodes, max_nodes), dtype=np.bool_
        )
        result["node_mask"] = np.zeros((1, max_nodes), dtype=np.bool_)
        result["active"] = np.zeros((1, max_nodes), dtype=np.bool_)
        result["copied"] = np.zeros((1, max_nodes), dtype=np.bool_)
        result["recursive_leaf_source_mask"] = np.zeros(
            (1, max_nodes, max_sources), dtype=np.bool_
        )

        for position, node in enumerate(nodes):
            common, common_available = _common_block(node)
            result["common_features"][0, position] = common
            result["common_availability"][0, position] = common_available
            for block in ("track", "cluster", "klm"):
                values = getattr(node, f"{block}_features")
                available = getattr(node, f"{block}_availability")
                if values:
                    result[f"{block}_features"][0, position] = np.asarray(values, dtype=np.float32)
                if available:
                    result[f"{block}_availability"][0, position] = np.asarray(available, dtype=np.bool_)
            if node.level > 0:
                daughters = [nodes[positions[daughter_id]] for daughter_id in node.daughter_ids]
                composite, composite_available = _composite_block(node, daughters)
                result["composite_features"][0, position] = composite
                result["composite_availability"][0, position] = composite_available
                for daughter in daughters:
                    result["daughter_input_pid_histogram"][
                        0, position, daughter.current_token
                    ] += 1.0
                result["daughter_input_pid_histogram_available"][0, position] = True
            result["node_kind_ids"][0, position] = node.kind_id
            result["leaf_kinematics_mode_ids"][0, position] = node.leaf_mode_id
            result["pid_labels"][0, position] = node.input_token
            result["level_ids"][0, position] = node.level
            result["p4"][0, position] = np.asarray(node.p4, dtype=np.float32)
            result["charge"][0, position] = node.charge
            result["parent_ids"][0, position] = node.parent_id
            result["node_ids"][0, position] = node.node_id
            result["reco_ids"][0, position] = node.node_id if node.level == 0 else -1
            result["source_node_ids"][0, position] = node.node_id
            result["node_mask"][0, position] = True
            result["active"][0, position] = True
            if node.level > 0:
                result["runtime_composite_type_source_ids"][0, position] = COMPOSITE_TYPE_PREDICTED
            for daughter_id in node.daughter_ids:
                result["daughter_adjacency"][0, position, positions[daughter_id]] = True
            for source_key in node.source_keys:
                result["recursive_leaf_source_mask"][
                    0, position, source_positions[source_key]
                ] = True

        for block in ("track", "cluster"):
            normalization = self.manifest["normalization"][block]
            mean = np.asarray(normalization["mean"], dtype=np.float32)
            std = np.asarray(normalization["standard_deviation"], dtype=np.float32)
            if np.any(~np.isfinite(std)) or np.any(std <= 0):
                raise ValueError(f"bundle {block} normalizer has non-positive scale")
            available = result[f"{block}_availability"]
            normalized = (result[f"{block}_features"] - mean) / std
            result[f"{block}_features"] = np.where(
                available, normalized, np.zeros_like(normalized)
            ).astype(np.float32)

        allowed, bias = self._type_constraints(level)
        result["allowed_type_mask"] = allowed
        result["type_logit_bias"] = bias
        pointer_valid = np.zeros((1, max_nodes), dtype=np.bool_)
        valid_leaf_kinds = set(int(value) for value in self.policy["valid_leaf_node_kinds"])
        valid_composite_kinds = set(
            int(value) for value in self.policy["valid_composite_node_kinds"]
        )
        for position, node in enumerate(nodes):
            kind_valid = node.kind_id in valid_leaf_kinds | valid_composite_kinds
            if (
                self.policy.get("allow_fixed_hypothesis_unknown_kind", False)
                and node.leaf_mode_id == LEAF_MODE_FIXED
            ):
                kind_valid = True
            pointer_valid[0, position] = (
                kind_valid and node.parent_id < 0 and node.level < int(level)
            )
        result["pointer_validity_mask"] = pointer_valid
        return result

    def _type_constraints(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        allowed = np.zeros((len(self.pid_tokens),), dtype=np.bool_)
        allowed[list(self.allowed_static)] = True
        if self.policy.get("initial_state_policy") == "upsilon4s":
            for token in (23, 40):
                if token < allowed.size:
                    allowed[token] = False
        observed = self.allowed_by_level.get(int(level), frozenset())
        mode = self.policy.get("empirical_type_prior_mode", "off")
        bias = np.zeros((len(self.pid_tokens),), dtype=np.float32)
        if observed and mode == "hard":
            observed_mask = np.zeros_like(allowed)
            observed_mask[list(observed)] = True
            allowed &= observed_mask
        elif observed and mode == "soft":
            observed_mask = np.zeros_like(allowed)
            observed_mask[list(observed)] = True
            bias[allowed & ~observed_mask] = -float(
                self.policy.get("empirical_type_soft_penalty", 2.0)
            )
        if not allowed.any():
            raise ValueError(f"reconstruction policy rejects all types at level {level}")
        return allowed, bias

    def _apply_hard_leaf_pid(
        self, hypothesis: Hypothesis, outputs: ModelOutputs
    ) -> Hypothesis:
        logits = outputs.leaf_pid_logits[0]
        adjusted_by_id: dict[int, Node] = {}
        for position, node in enumerate(sorted(hypothesis.nodes, key=lambda item: item.node_id)):
            if node.leaf_mode_id != LEAF_MODE_RAW_TRACK:
                adjusted_by_id[node.node_id] = node
                continue
            allowed = [
                token
                for token, pdg in enumerate(self.pid_tokens)
                if CHARGED_STABLE_CHARGES.get(pdg) == int(round(node.charge))
            ]
            if not allowed:
                raise ValueError(f"raw track node {node.node_id} has unsupported charge")
            token = max(allowed, key=lambda item: (float(logits[position, item]), -item))
            pdg = self.pid_tokens[token]
            mass = CHARGED_STABLE_MASSES_GEV[abs(pdg)]
            px, py, pz, _energy = node.p4
            energy = math.sqrt(px * px + py * py + pz * pz + mass * mass)
            adjusted_by_id[node.node_id] = replace(
                node,
                current_token=token,
                pdg=pdg,
                p4=(px, py, pz, energy),
            )
        # The rollout construction contract is hard leaf PID followed by exact
        # recursive daughter sums.  Rebuild all earlier composites in level
        # order in case a later level changes a track's PID prediction.
        for node in sorted(
            hypothesis.nodes, key=lambda item: (item.level, item.node_id)
        ):
            if node.level == 0:
                continue
            daughters = [adjusted_by_id[node_id] for node_id in node.daughter_ids]
            adjusted_by_id[node.node_id] = replace(
                node,
                p4=tuple(
                    sum(float(daughter.p4[axis]) for daughter in daughters)
                    for axis in range(4)
                ),
                charge=float(sum(daughter.charge for daughter in daughters)),
                source_keys=frozenset().union(
                    *(daughter.source_keys for daughter in daughters)
                ),
            )
        return replace(
            hypothesis,
            nodes=tuple(adjusted_by_id[node.node_id] for node in hypothesis.nodes),
        )

    def _decode_proposals(
        self,
        hypothesis: Hypothesis,
        outputs: ModelOutputs,
        level: int,
    ) -> list[Proposal]:
        nodes = tuple(sorted(hypothesis.nodes, key=lambda node: node.node_id))
        valid_leaf_kinds = {
            int(value) for value in self.policy["valid_leaf_node_kinds"]
        }
        valid_composite_kinds = {
            int(value) for value in self.policy["valid_composite_node_kinds"]
        }
        eligible = [
            position
            for position, node in enumerate(nodes)
            if (
                node.parent_id < 0
                and node.level < level
                and (
                    node.kind_id in valid_leaf_kinds | valid_composite_kinds
                    or (
                        self.policy.get(
                            "allow_fixed_hypothesis_unknown_kind", False
                        )
                        and node.leaf_mode_id == LEAF_MODE_FIXED
                    )
                )
            )
        ]
        proposals: list[Proposal] = []
        object_probabilities = _sigmoid(outputs.object_logits[0])
        learned_confidences = _sigmoid(outputs.confidence_logits[0])
        for query_id, object_score in enumerate(object_probabilities.tolist()):
            if object_score < float(self.policy["object_threshold"]):
                continue
            pointer_probabilities = _sigmoid(outputs.pointer_logits[0, query_id])
            cardinality = int(np.argmax(outputs.cardinality_logits[0, query_id]))
            daughter_positions = self._select_daughters(
                nodes,
                eligible,
                pointer_probabilities,
                cardinality,
            )
            if len(daughter_positions) < int(self.policy["minimum_daughters"]):
                continue
            type_logits = outputs.type_logits[0, query_id].astype(
                np.float64, copy=True
            )
            # The exported decoder has already consumed type_logit_bias.  The
            # host repeats only the hard allow-list check; adding the soft bias
            # here would count the empirical prior twice.
            allowed, _bias = self._type_constraints(level)
            constrained_type_logits = type_logits
            constrained_type_logits[~allowed] = -np.inf
            mother_token = int(np.argmax(constrained_type_logits))
            type_probabilities = _softmax(constrained_type_logits[allowed])
            allowed_positions = np.flatnonzero(allowed)
            allowed_index = int(np.flatnonzero(allowed_positions == mother_token)[0])
            type_probability = float(type_probabilities[allowed_index])
            configured_type_threshold = self.policy.get("type_probability_threshold")
            if configured_type_threshold is not None and type_probability < float(
                configured_type_threshold
            ):
                continue
            daughter_nodes = [nodes[position] for position in daughter_positions]
            if (
                mother_token in self.root_tokens
                and self.policy.get("root_requires_all_sources", False)
            ):
                all_sources = frozenset().union(
                    *(node.source_keys for node in nodes if node.level == 0)
                )
                proposed_sources = frozenset().union(
                    *(node.source_keys for node in daughter_nodes)
                )
                if proposed_sources != all_sources:
                    continue
            if self.policy.get("mother_charge_compatibility") in {
                "hard", "soft_train_hard_rollout"
            }:
                actual_charge = sum(node.charge for node in daughter_nodes)
                expected_charge = self.token_charge[mother_token]
                if abs(actual_charge - expected_charge) > float(
                    self.policy.get("mother_charge_tolerance", 1e-6)
                ):
                    continue
            mother_p4 = tuple(
                sum(float(node.p4[axis]) for node in daughter_nodes)
                for axis in range(4)
            )
            if not self._physical_valid(mother_p4):
                continue
            pointer_quality = float(
                np.mean(pointer_probabilities[np.asarray(daughter_positions, dtype=np.int64)])
            )
            learned = float(learned_confidences[query_id])
            confidence = (
                learned
                if self.policy.get("use_learned_confidence", False)
                else float(object_score) * type_probability * pointer_quality
            )
            if confidence < float(self.policy["confidence_threshold"]):
                continue
            proposals.append(
                Proposal(
                    query_id=query_id,
                    mother_token=mother_token,
                    daughter_ids=tuple(nodes[position].node_id for position in daughter_positions),
                    object_score=float(object_score),
                    type_probability=type_probability,
                    pointer_quality=pointer_quality,
                    learned_confidence=learned,
                    confidence=confidence,
                )
            )
        proposals.sort(key=lambda item: (-item.confidence, item.query_id, item.daughter_ids))
        return proposals[: int(self.policy["max_proposals_per_hypothesis"])]

    def _select_daughters(
        self,
        nodes: Sequence[Node],
        eligible: Sequence[int],
        probabilities: np.ndarray,
        cardinality: int,
    ) -> tuple[int, ...]:
        minimum = float(self.policy["pointer_threshold"])
        if self.policy.get("daughter_cardinality_policy") != "predicted":
            cardinality = len(eligible)
            insufficient = "reduce"
        else:
            insufficient = self.policy.get("cardinality_insufficient_policy", "invalid")
        if cardinality > len(eligible):
            if insufficient == "invalid":
                return ()
            cardinality = len(eligible)
        selected: list[int] = []
        ordered = sorted(
            (position for position in eligible if probabilities[position] >= minimum),
            key=lambda position: (-float(probabilities[position]), position),
        )
        for position in ordered:
            if any(nodes[position].source_keys & nodes[other].source_keys for other in selected):
                continue
            selected.append(position)
            if len(selected) >= cardinality:
                break
        if len(selected) != cardinality and insufficient == "invalid":
            return ()
        return tuple(sorted(selected))

    def _physical_valid(self, p4: Sequence[float]) -> bool:
        configured = dict(self.policy.get("loose_physical_constraints", []))
        if not configured:
            return True
        px, py, pz, energy = (float(value) for value in p4)
        momentum = math.sqrt(px * px + py * py + pz * pz)
        mass = math.sqrt(max(energy * energy - momentum * momentum, 0.0))
        checks = {
            "minimum_mother_energy": energy >= configured.get("minimum_mother_energy", -math.inf),
            "minimum_mother_mass": mass >= configured.get("minimum_mother_mass", -math.inf),
            "maximum_mother_mass": mass <= configured.get("maximum_mother_mass", math.inf),
            "maximum_mother_momentum": momentum <= configured.get("maximum_mother_momentum", math.inf),
        }
        return all(checks[name] for name in configured)

    def _conflict_free_proposal_sets(
        self,
        hypothesis: Hypothesis,
        proposals: Sequence[Proposal],
    ) -> tuple[tuple[Proposal, ...], ...]:
        candidates: list[tuple[float, tuple[Proposal, ...]]] = []
        node_sources = {
            node.node_id: node.source_keys for node in hypothesis.nodes
        }
        # Direct daughter IDs are not sufficient here: two detector objects
        # can be distinct nodes but share a reconstructed source (for example
        # an associated ECL/KLM pair).  Compare the recursive source union of
        # every proposed mother, just as the offline rollout does.
        for count in range(1, len(proposals) + 1):
            for chosen in itertools.combinations(proposals, count):
                daughter_ids: set[int] = set()
                query_ids: set[int] = set()
                source_keys: set[str] = set()
                valid = True
                for proposal in chosen:
                    proposal_sources = set().union(
                        *(node_sources[node_id] for node_id in proposal.daughter_ids)
                    )
                    if proposal.query_id in query_ids or daughter_ids.intersection(
                        proposal.daughter_ids
                    ) or source_keys.intersection(proposal_sources):
                        valid = False
                        break
                    query_ids.add(proposal.query_id)
                    daughter_ids.update(proposal.daughter_ids)
                    source_keys.update(proposal_sources)
                # A completed hypothesis publishes exactly one root.  Keep
                # root queries as singleton alternatives so another root (or
                # an unrelated side proposal) cannot inflate its beam score
                # or overwrite the selected root ID.
                if valid and any(
                    proposal.mother_token in self.root_tokens
                    for proposal in chosen
                ):
                    valid = len(chosen) == 1
                if valid:
                    candidates.append((sum(item.confidence for item in chosen), chosen))
        candidates.sort(
            key=lambda item: (
                -item[0],
                tuple(proposal.query_id for proposal in item[1]),
                tuple(proposal.daughter_ids for proposal in item[1]),
            )
        )
        return tuple(
            chosen for _score, chosen in candidates[: int(self.policy["beam_width"])]
        )

    def _append_proposals(
        self,
        hypothesis: Hypothesis,
        proposals: Sequence[Proposal],
        level: int,
    ) -> Hypothesis:
        nodes_by_id = hypothesis.node_by_id()
        next_id = max(nodes_by_id) + 1
        parent_updates: dict[int, int] = {}
        appended: list[Node] = []
        completed_root_id = hypothesis.completed_root_id
        for offset, proposal in enumerate(sorted(proposals, key=lambda item: item.query_id)):
            mother_id = next_id + offset
            daughters = [nodes_by_id[node_id] for node_id in proposal.daughter_ids]
            for daughter in daughters:
                if daughter.parent_id >= 0 or daughter.node_id in parent_updates:
                    raise RuntimeError("beam proposal reuses an already-parented forest root")
                parent_updates[daughter.node_id] = mother_id
            p4 = tuple(sum(node.p4[axis] for node in daughters) for axis in range(4))
            charge = sum(node.charge for node in daughters)
            sources = frozenset().union(*(node.source_keys for node in daughters))
            mother = Node(
                node_id=mother_id,
                input_token=proposal.mother_token,
                current_token=proposal.mother_token,
                pdg=self.pid_tokens[proposal.mother_token],
                p4=tuple(float(value) for value in p4),
                charge=float(charge),
                level=int(level),
                kind_id=NODE_KIND_COMPOSITE,
                leaf_mode_id=LEAF_MODE_COMPOSITE,
                daughter_ids=tuple(sorted(proposal.daughter_ids)),
                source_keys=sources,
                candidate_confidence=proposal.confidence,
                object_score=proposal.object_score,
                query_id=proposal.query_id,
            )
            appended.append(mother)
            if proposal.mother_token in self.root_tokens:
                completed_root_id = mother_id
        updated = [
            replace(node, parent_id=parent_updates[node.node_id])
            if node.node_id in parent_updates
            else node
            for node in hypothesis.nodes
        ]
        accepted_ids = tuple(node.node_id for node in appended)
        return Hypothesis(
            nodes=tuple(updated + appended),
            score=hypothesis.score + sum(proposal.confidence for proposal in proposals),
            accepted_by_level=hypothesis.accepted_by_level + (accepted_ids,),
            completed_root_id=completed_root_id,
            empty_levels=hypothesis.empty_levels,
        )


def _common_block(node: Node) -> tuple[np.ndarray, np.ndarray]:
    if node.common_features:
        values = np.asarray(node.common_features, dtype=np.float32)
        available = np.asarray(node.common_availability, dtype=np.bool_)
        if values.shape != (len(FEATURE_NAMES["common"]),) or available.shape != values.shape:
            raise ValueError("node common feature block has the wrong width")
        values = values.copy()
        available = available.copy()
    else:
        values = np.zeros((len(FEATURE_NAMES["common"]),), dtype=np.float32)
        available = np.zeros_like(values, dtype=np.bool_)
    px, py, pz, energy = node.p4
    values[:6] = (
        px,
        py,
        pz,
        energy,
        math.sqrt(max(energy * energy - px * px - py * py - pz * pz, 0.0)),
        node.charge,
    )
    values[6:11] = (
        node.current_token,
        node.level,
        1.0,
        0.0,
        len(node.daughter_ids),
    )
    values[11] = node.candidate_confidence
    available[[0, 1, 2, 3, 4, 5, 10]] = True
    available[[6, 7, 8, 9]] = False
    available[11] = node.candidate_confidence > 0.0
    return values, available


def _composite_block(
    node: Node, daughters: Sequence[Node]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(FEATURE_NAMES["composite"]),), dtype=np.float32)
    available = np.zeros_like(values, dtype=np.bool_)
    values[:6] = (*node.p4, node.charge, len(daughters))
    daughter_confidences = [item.candidate_confidence for item in daughters]
    values[6] = float(np.mean(daughter_confidences)) if daughter_confidences else 0.0
    values[7] = min(daughter_confidences) if daughter_confidences else 0.0
    values[8] = 0.0
    available[:9] = True
    return values, available


def _validate_model_outputs(outputs: ModelOutputs, contract: Mapping[str, Any]) -> None:
    max_nodes = int(contract["max_nodes"])
    n_queries = int(contract["n_queries"])
    n_types = len(contract["pid_tokens"])
    max_cardinality = int(contract["max_cardinality"])
    expected = {
        "object_logits": (1, n_queries),
        "type_logits": (1, n_queries, n_types),
        "pointer_logits": (1, n_queries, max_nodes),
        "cardinality_logits": (1, n_queries, max_cardinality + 1),
        "confidence_logits": (1, n_queries),
        "leaf_pid_logits": (1, max_nodes, n_types),
        "current_p4": (1, max_nodes, 4),
    }
    for name, shape in expected.items():
        value = getattr(outputs, name)
        if value.shape != shape:
            raise ValueError(f"ONNX output {name} shape {value.shape} != {shape}")
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"ONNX output {name} is not floating point")
        if not np.isfinite(value).all():
            raise ValueError(f"ONNX output {name} contains non-finite values")


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _softmax(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    shifted = value - np.max(value)
    exponential = np.exp(shifted)
    return exponential / exponential.sum()


def _hypothesis_sort_key(hypothesis: Hypothesis) -> tuple[object, ...]:
    return (
        -float(hypothesis.score),
        -int(hypothesis.completed_root_id is not None),
        hypothesis.empty_levels,
        hypothesis.fingerprint(),
    )


__all__ = [
    "BeamSearchReconstructor",
    "Hypothesis",
    "LEAF_MODE_COMPOSITE",
    "LEAF_MODE_ECL",
    "LEAF_MODE_FIXED",
    "LEAF_MODE_KLM",
    "LEAF_MODE_RAW_TRACK",
    "ModelOutputs",
    "NODE_KIND_COMPOSITE",
    "NODE_KIND_ECL",
    "NODE_KIND_KLM",
    "NODE_KIND_OTHER",
    "NODE_KIND_TRACK",
    "NODE_KIND_UNKNOWN",
    "Node",
    "OnnxModelBundle",
    "Proposal",
    "ReconstructionResult",
]
