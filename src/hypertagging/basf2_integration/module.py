"""Additive basf2 module for ONNX-guided full-decay reconstruction.

The module consumes only reconstructed :class:`Belle2.ParticleList` objects.
It never opens a training checkpoint and never asks a reconstructed object for
simulation matching.  The selected ONNX bundle and the beam-search policy in
its manifest therefore completely determine the hierarchy built at runtime.

Only the completed root of the best beam hypothesis is published, in a new
configurable ParticleList.  Existing input lists and their Particle objects
are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from hypertagging.basf2_integration.runtime import (
    BeamSearchReconstructor,
    LEAF_MODE_ECL,
    LEAF_MODE_FIXED,
    LEAF_MODE_KLM,
    LEAF_MODE_RAW_TRACK,
    NODE_KIND_ECL,
    NODE_KIND_KLM,
    NODE_KIND_OTHER,
    NODE_KIND_TRACK,
    Node,
    OnnxModelBundle,
    ReconstructionResult,
)
from hypertagging.deployment.onnx_contract import FEATURE_NAMES


try:  # Keep ordinary CPython imports useful for documentation and unit tests.
    import basf2 as _basf2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only outside basf2
    _basf2 = None


_Basf2Module = _basf2.Module if _basf2 is not None else object

_CHARGED_STABLE_PDGS = (11, 13, 211, 321, 2212)
_CHARGED_STABLE_NAMES = ("electron", "muon", "pion", "kaon", "proton")
_CHARGED_STABLE_MASSES_GEV = (
    0.00051099895,
    0.1056583755,
    0.13957039,
    0.493677,
    0.93827208816,
)
_PION_MASS_GEV = 0.13957039


class HyperTaggingFullDecayModule(_Basf2Module):  # type: ignore[misc, valid-type]
    """Run truth-free, full-depth beam reconstruction inside a basf2 path.

    Parameters
    ----------
    manifest_path:
        Hash-checked ONNX bundle manifest produced by the deployment exporter.
    input_particle_lists:
        Existing final-state ParticleLists.  Candidates with daughters are
        ignored, and repeated detector sources across lists are deduplicated.
    output_particle_list:
        A new list receiving zero or one completed best-hypothesis root.
        It must not name any input or standard/offline reconstruction list.
    write_out:
        Register the new list for output.  The safe default keeps it transient.
    intra_op_threads:
        CPU thread count for each ONNX Runtime session.
    """

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        input_particle_lists: Sequence[str],
        output_particle_list: str = "Upsilon(4S):HyperTagging",
        write_out: bool = False,
        intra_op_threads: int = 1,
    ) -> None:
        if _basf2 is None:
            raise RuntimeError(
                "HyperTaggingFullDecayModule must be constructed inside a basf2 "
                "release providing the basf2 Python module"
            )
        super().__init__()
        inputs = tuple(str(name) for name in input_particle_lists)
        if not inputs or any(not name for name in inputs):
            raise ValueError("input_particle_lists must contain at least one list name")
        if len(set(inputs)) != len(inputs):
            raise ValueError("input_particle_lists contains duplicate names")
        if output_particle_list in inputs:
            raise ValueError(
                "output ParticleList must be distinct from every input list"
            )
        if intra_op_threads < 1:
            raise ValueError("intra_op_threads must be positive")

        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self.input_particle_lists = inputs
        self.output_particle_list = str(output_particle_list)
        self.write_out = bool(write_out)
        self.intra_op_threads = int(intra_op_threads)

        self._events = 0
        self._events_with_inputs = 0
        self._completed = 0
        self._input_candidates = 0
        self._deduplicated_candidates = 0
        self._scorer: OnnxModelBundle | None = None
        self._reconstructor: BeamSearchReconstructor | None = None
        self._Belle2: Any = None
        self._PxPyPzEVector: Any = None

    def initialize(self) -> None:
        """Register DataStore inputs/outputs and load hash-checked sessions."""

        from ROOT import Belle2  # type: ignore[import-not-found]
        from ROOT.Math import PxPyPzEVector  # type: ignore[import-not-found]

        self._Belle2 = Belle2
        self._PxPyPzEVector = PxPyPzEVector
        self._pid_detector_sets = tuple(
            Belle2.Const.PIDDetectorSet(getattr(Belle2.Const, name))
            for name in ("SVD", "CDC", "TOP", "ARICH", "ECL", "KLM")
        )
        self._scorer = OnnxModelBundle(
            self.manifest_path,
            intra_op_threads=self.intra_op_threads,
        )
        self._reconstructor = BeamSearchReconstructor(self._scorer)
        self._token_by_pdg = {
            int(pdg): token
            for token, pdg in enumerate(self._scorer.manifest["contract"]["pid_tokens"])
        }

        self._input_lists = []
        for list_name in self.input_particle_lists:
            store = Belle2.PyStoreObj(Belle2.ParticleList.Class(), list_name)
            store.isRequired()
            self._input_lists.append((list_name, store))

        # Refuse to merge into or recreate any list registered by RootInput or
        # an earlier module.  `isOptional` is a non-mutating existence probe
        # during initialization; an absent name can then be freshly registered.
        output_store = Belle2.PyStoreObj(
            Belle2.ParticleList.Class(), self.output_particle_list
        )
        if output_store.isOptional():
            raise RuntimeError(
                "refusing to overwrite an existing ParticleList: "
                f"{self.output_particle_list}"
            )
        flags = (
            Belle2.DataStore.c_WriteOut
            if self.write_out
            else Belle2.DataStore.c_DontWriteOut
        )

        descriptor = Belle2.DecayDescriptor()
        if not descriptor.init(self.output_particle_list):
            raise ValueError(
                "invalid output ParticleList name: "
                f"{self.output_particle_list}"
            )
        self._output_pdg = int(descriptor.getMother().getPDGCode())
        root_pdgs = {
            int(self._scorer.manifest["contract"]["pid_tokens"][token])
            for token in self._scorer.manifest["reconstruction_policy"][
                "root_tokens"
            ]
        }
        if self._output_pdg not in root_pdgs:
            raise ValueError(
                "output ParticleList PDG is not a configured model root: "
                f"{self._output_pdg} not in {sorted(root_pdgs)}"
            )

        # Register only the additive output list.  ParticleListHelper also
        # re-registers the shared Particles array and can silently change its
        # persistence flag; doing that would mutate the surrounding offline
        # analysis configuration.
        if not output_store.registerInDataStore(flags):
            raise RuntimeError(
                f"could not register output ParticleList {self.output_particle_list}"
            )
        self._output_store = output_store
        self._particles = Belle2.PyStoreArray(Belle2.Particle.Class())
        self._particles.isRequired()

        # Particle::addExtraInfo can create the per-event object, but its type
        # must first be registered during initialize.
        self._extra_info = Belle2.PyStoreObj(Belle2.ParticleExtraInfoMap.Class())
        if not self._extra_info.isOptional():
            self._extra_info.registerInDataStore(flags)

        _basf2.B2INFO(
            "HyperTagging ONNX full-decay module initialized: "
            f"manifest={self._scorer.manifest_sha256}, "
            f"inputs={self.input_particle_lists}, output={self.output_particle_list}"
        )

    def event(self) -> None:
        """Reconstruct one event and publish at most one completed root."""

        assert self._reconstructor is not None
        self._events += 1
        if not self._output_store.create():
            raise RuntimeError(
                f"could not create output ParticleList {self.output_particle_list}"
            )
        self._output_store.obj().initialize(
            self._output_pdg, self.output_particle_list
        )
        if not self._extra_info.isValid() and not self._extra_info.create():
            raise RuntimeError("could not create ParticleExtraInfoMap")

        leaves = self._collect_particle_list_leaves()
        if not leaves:
            return
        self._events_with_inputs += 1
        result = self._reconstructor.reconstruct(leaves)
        if not result.completed:
            return
        root_id = result.best.completed_root_id
        if root_id is None:  # Narrows the optional value for type checkers.
            return
        nodes = result.best.node_by_id()
        root = nodes[root_id]
        expected_pdg = self._output_pdg
        if root.pdg != expected_pdg:
            raise RuntimeError(
                "best ONNX root PDG does not match output ParticleList: "
                f"{root.pdg} != {expected_pdg} ({self.output_particle_list})"
            )

        stored_root = self._materialize_subtree(root_id, nodes, {})
        self._decorate_root(stored_root, result)
        self._output_store.addParticle(stored_root)
        if int(self._output_store.getListSize()) != 1:
            raise RuntimeError(
                "HyperTagging output list violated its unique-root contract"
            )
        self._completed += 1

    def terminate(self) -> None:
        """Report compact job-level reconstruction counters."""

        _basf2.B2INFO(
            "HyperTagging ONNX full-decay summary: "
            f"events={self._events}, events_with_inputs={self._events_with_inputs}, "
            f"completed={self._completed}, input_candidates={self._input_candidates}, "
            f"deduplicated_candidates={self._deduplicated_candidates}"
        )

    def _collect_particle_list_leaves(self) -> tuple[Node, ...]:
        leaves: list[Node] = []
        seen_sources: set[str] = set()
        candidates: list[tuple[int, int, int, object, str, frozenset[str], str]] = []
        for list_order, (list_name, store) in enumerate(self._input_lists):
            if not store.isValid():
                raise RuntimeError(
                    f"required input ParticleList is unavailable: {list_name}"
                )
            for position in range(int(store.getListSize())):
                particle = store.getParticle(position)
                if not particle or int(particle.getNDaughters()) != 0:
                    continue
                self._input_candidates += 1
                primary_key, source_keys, source_kind = self._particle_source_keys(
                    particle
                )
                candidates.append(
                    (
                        int(particle.getArrayIndex()),
                        list_order,
                        position,
                        particle,
                        primary_key,
                        source_keys,
                        source_kind,
                    )
                )

        # ParticleList stores may group particle and anti-particle candidates
        # separately.  Global Particles indices restore a deterministic event
        # order independent of that list-internal presentation.
        candidates.sort(key=lambda item: item[:3])
        for (
            _array_index,
            _list_order,
            _position,
            particle,
            primary_key,
            source_keys,
            source_kind,
        ) in candidates:
            if primary_key in seen_sources:
                self._deduplicated_candidates += 1
                continue
            seen_sources.add(primary_key)
            leaves.append(
                self._particle_to_node(
                    particle,
                    node_id=len(leaves),
                    source_keys=source_keys,
                    source_kind=source_kind,
                )
            )

        maximum = int(self._scorer.manifest["contract"]["max_nodes"])  # type: ignore[union-attr]
        if len(leaves) > maximum:
            raise RuntimeError(
                f"event has {len(leaves)} unique FSPs but ONNX bundle supports {maximum} nodes"
            )
        return tuple(leaves)

    def _particle_source_keys(
        self, particle: object
    ) -> tuple[str, frozenset[str], str]:
        source = int(particle.getParticleSource())  # type: ignore[attr-defined]
        source_names = {
            int(self._Belle2.Particle.c_Track): "Track",
            int(self._Belle2.Particle.c_ECLCluster): "ECLCluster",
            int(self._Belle2.Particle.c_KLMCluster): "KLMCluster",
        }
        source_kind = source_names.get(source, "Particle")
        if source_kind == "Particle":
            index = int(particle.getArrayIndex())  # type: ignore[attr-defined]
        else:
            index = int(particle.getMdstArrayIndex())  # type: ignore[attr-defined]
        primary_key = f"{source_kind}:{index}"
        source_keys = {primary_key}

        # KLM and ECL candidates may describe overlapping detector evidence.
        # Preserve the reconstructed association in the recursive source mask
        # so no proposal (or pair of proposals) can reuse it.  This mirrors the
        # schema-v4 `associated_reco_id` contract without consulting truth.
        if source_kind == "KLMCluster":
            cluster = _optional_call(particle, ("getKLMCluster",))
            associated_ecl = _first_related_named(
                cluster, ("ECLClusters", "ECLCluster")
            )
            if associated_ecl is not None:
                associated_index = _optional_finite(associated_ecl, ("getArrayIndex",))
                if associated_index is not None and associated_index >= 0:
                    source_keys.add(f"ECLCluster:{int(associated_index)}")
        return primary_key, frozenset(source_keys), source_kind

    def _particle_to_node(
        self,
        particle: object,
        *,
        node_id: int,
        source_keys: frozenset[str],
        source_kind: str,
    ) -> Node:
        pdg = int(particle.getPDGCode())  # type: ignore[attr-defined]
        candidate_confidence = _optional_finite(particle, ("getPValue",))
        token = int(self._token_by_pdg.get(pdg, 0))
        node_kwargs: dict[str, Any] = {}

        if source_kind == "Track":
            track = _optional_call(particle, ("getTrack",))
            fit = _select_max_p_value_track_fit(
                track,
                pion_hypothesis=self._Belle2.Const.ChargedStable(211),
            )
            if fit is None:
                raise RuntimeError(
                    "Track Particle has no fit accepted by "
                    "max_p_value_then_pion_fallback-v1: "
                    f"{sorted(source_keys)}"
                )
            momentum = fit.getMomentum()
            px = _required_vector_component(momentum, ("X", "Px"), "track px")
            py = _required_vector_component(momentum, ("Y", "Py"), "track py")
            pz = _required_vector_component(momentum, ("Z", "Pz"), "track pz")
            charge_value = _optional_finite(fit, ("getChargeSign",))
            if charge_value is None:
                raise RuntimeError(
                    f"selected Track fit has no finite charge: {source_keys}"
                )
            charge = charge_value
            momentum2 = px * px + py * py + pz * pz
            energy = math.sqrt(momentum2 + _PION_MASS_GEV * _PION_MASS_GEV)
            candidate_confidence = _optional_finite(fit, ("getPValue",))
            token = 0
            pdg = 0
            kind_id = NODE_KIND_TRACK
            mode_id = LEAF_MODE_RAW_TRACK
            node_kwargs.update(
                self._track_feature_blocks(particle, track, fit, momentum2)
            )
        else:
            p4 = particle.get4Vector()  # type: ignore[attr-defined]
            px, py, pz, energy = (
                float(p4.Px()),
                float(p4.Py()),
                float(p4.Pz()),
                float(p4.E()),
            )
            charge = float(particle.getCharge())  # type: ignore[attr-defined]
            if source_kind == "ECLCluster":
                kind_id = NODE_KIND_ECL
                mode_id = LEAF_MODE_ECL
                node_kwargs.update(
                    self._ecl_feature_blocks(particle, (px, py, pz, energy))
                )
            elif source_kind == "KLMCluster":
                kind_id = NODE_KIND_KLM
                mode_id = LEAF_MODE_KLM
                node_kwargs.update(self._klm_feature_blocks(particle))
            else:
                kind_id = NODE_KIND_OTHER
                mode_id = LEAF_MODE_FIXED

        if not all(math.isfinite(value) for value in (px, py, pz, energy, charge)):
            raise RuntimeError(
                f"non-finite reconstructed FSP in source {sorted(source_keys)}"
            )
        if candidate_confidence is None or candidate_confidence < 0.0:
            candidate_confidence = 0.0

        return Node(
            node_id=node_id,
            input_token=token,
            current_token=token,
            pdg=pdg,
            p4=(px, py, pz, energy),
            charge=charge,
            level=0,
            kind_id=kind_id,
            leaf_mode_id=mode_id,
            source_keys=source_keys,
            candidate_confidence=float(candidate_confidence),
            source_object=particle,
            source_kind=source_kind,
            **node_kwargs,
        )

    def _track_feature_blocks(
        self,
        particle: object,
        track: object,
        fit: object,
        momentum2: float,
    ) -> dict[str, tuple[float, ...] | tuple[bool, ...]]:
        values: dict[str, float | None] = {
            "fit_p_value": _optional_finite(fit, ("getPValue",)),
            "d0": _optional_finite(fit, ("getD0",)),
            "z0": _optional_finite(fit, ("getZ0",)),
            "phi0": _optional_finite(fit, ("getPhi0",)),
            "omega": _optional_finite(fit, ("getOmega",)),
            "tan_lambda": _optional_finite(fit, ("getTanLambda",)),
        }

        # Match the training producer exactly: PIDLikelihood values are usable
        # only when the Track relation exists and at least one configured PID
        # detector reports data. Empty relations can otherwise return finite
        # zero log-likelihoods that look valid but were masked during training.
        likelihood = _first_related_named(
            track, ("PIDLikelihoods", "PIDLikelihood")
        )
        likelihood_available = _pid_likelihood_detector_available(
            likelihood, self._pid_detector_sets
        )
        for name, pdg, mass in zip(
            _CHARGED_STABLE_NAMES,
            _CHARGED_STABLE_PDGS,
            _CHARGED_STABLE_MASSES_GEV,
        ):
            log_likelihood: float | None = None
            if likelihood is not None and likelihood_available:
                try:
                    candidate = float(
                        likelihood.getLogL(self._Belle2.Const.ChargedStable(pdg))
                    )
                except (RuntimeError, TypeError, ValueError):
                    candidate = math.nan
                if math.isfinite(candidate):
                    log_likelihood = candidate
            values[f"pid_log_likelihood_{name}"] = log_likelihood
            values[f"energy_hypothesis_{name}"] = math.sqrt(momentum2 + mass * mass)
        block, available = _ordered_feature_block(values, FEATURE_NAMES["track"])
        return {"track_features": block, "track_availability": available}

    def _ecl_feature_blocks(
        self,
        particle: object,
        p4: tuple[float, float, float, float],
    ) -> dict[str, tuple[float, ...] | tuple[bool, ...]]:
        px, py, pz, energy = p4
        cluster = _optional_call(particle, ("getECLCluster",))
        momentum = math.sqrt(px * px + py * py + pz * pz)
        values: dict[str, float | None] = {
            "cluster_energy": energy,
            "theta": math.acos(max(-1.0, min(1.0, pz / momentum))) if momentum else 0.0,
            "phi": math.atan2(py, px),
            "photon_hypothesis": 1.0,
        }
        if cluster:
            values.update(
                {
                    "time": _optional_finite(cluster, ("getTime",)),
                    "e9_over_e21": _optional_finite(
                        cluster, ("getE9oE21", "getE9OverE21")
                    ),
                    "n_crystals": _optional_finite(
                        cluster,
                        ("getNumberOfCrystals", "getNumberOfConnectedCrystals"),
                    ),
                    "min_track_distance": _optional_finite(
                        cluster, ("getMinTrackDistance",)
                    ),
                    "track_matched": _optional_boolean(cluster, ("isTrack",)),
                }
            )
        block, available = _ordered_feature_block(values, FEATURE_NAMES["cluster"])
        return {"cluster_features": block, "cluster_availability": available}

    def _klm_feature_blocks(
        self, particle: object
    ) -> dict[str, tuple[float, ...] | tuple[bool, ...]]:
        cluster = _optional_call(particle, ("getKLMCluster",))
        values: dict[str, float | None] = {}
        if cluster:
            position = _optional_call(cluster, ("getClusterPosition",))
            if position is not None:
                values.update(
                    {
                        "x": _vector_component(position, ("X", "x")),
                        "y": _vector_component(position, ("Y", "y")),
                        "z": _vector_component(position, ("Z", "z")),
                    }
                )
            values.update(
                {
                    "energy": _optional_finite(cluster, ("getEnergy",)),
                    "momentum_magnitude": _optional_finite(
                        cluster, ("getMomentumMag",)
                    ),
                    "time": _optional_finite(cluster, ("getTime",)),
                    "layers": _optional_finite(cluster, ("getLayers",)),
                    "innermost_layer": _optional_finite(
                        cluster, ("getInnermostLayer",)
                    ),
                    "associated_ecl_cluster": _optional_finite(
                        cluster, ("getAssociatedEclClusterFlag",)
                    ),
                }
            )
        block, available = _ordered_feature_block(values, FEATURE_NAMES["klm"])
        return {"klm_features": block, "klm_availability": available}

    def _materialize_subtree(
        self,
        node_id: int,
        nodes: Mapping[int, Node],
        materialized: dict[int, object],
    ) -> object:
        existing = materialized.get(node_id)
        if existing is not None:
            return existing
        node = nodes[node_id]
        daughters = [
            self._materialize_subtree(daughter_id, nodes, materialized)
            for daughter_id in node.daughter_ids
        ]
        particle = self._particles.appendNew()
        if not particle:
            raise RuntimeError("could not append HyperTagging Particle to DataStore")
        momentum = self._PxPyPzEVector(*node.p4)
        if node.level == 0 and node.source_object is not None:
            # Copy, rather than reuse or mutate, the input Particle.  ROOT's
            # generated assignment hook preserves its MDST source/relation
            # provenance in the new StoreArray element.
            particle.__assign__(node.source_object)
        else:
            # The Lorentz-vector constructor initializes the correct flavor
            # type from PDG; appendDaughter below changes the source to
            # c_Composite while retaining that flavor classification.
            particle.__assign__(self._Belle2.Particle(momentum, int(node.pdg)))
        particle.setPDGCode(int(node.pdg))
        particle.set4Vector(momentum)
        for daughter in daughters:
            particle.appendDaughter(daughter)
        materialized[node_id] = particle
        return particle

    def _decorate_root(self, particle: object, result: ReconstructionResult) -> None:
        manifest_prefix = int(result.model_manifest_sha256[:13], 16)
        particle.addExtraInfo("HyperTaggingScore", float(result.best.score))
        particle.addExtraInfo("HyperTaggingCompleted", 1.0)
        particle.addExtraInfo("HyperTaggingLevels", float(result.levels_processed))
        particle.addExtraInfo("HyperTaggingBeamSize", float(len(result.beam)))
        # Thirteen hexadecimal digits fit exactly in an IEEE-754 double and
        # provide a compact join key to the full hash logged at initialize.
        particle.addExtraInfo("HyperTaggingManifestHash52", float(manifest_prefix))


def add_hypertagging_full_decay(
    path: object,
    *,
    manifest_path: str | Path,
    input_particle_lists: Sequence[str],
    output_particle_list: str = "Upsilon(4S):HyperTagging",
    write_out: bool = False,
    intra_op_threads: int = 1,
) -> HyperTaggingFullDecayModule:
    """Construct and append :class:`HyperTaggingFullDecayModule` to ``path``."""

    module = HyperTaggingFullDecayModule(
        manifest_path=manifest_path,
        input_particle_lists=input_particle_lists,
        output_particle_list=output_particle_list,
        write_out=write_out,
        intra_op_threads=intra_op_threads,
    )
    path.add_module(module)  # type: ignore[attr-defined]
    return module


def _ordered_feature_block(
    values: Mapping[str, float | None], names: Sequence[str]
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    block: list[float] = []
    available: list[bool] = []
    for name in names:
        value = values.get(name)
        valid = value is not None and math.isfinite(float(value))
        block.append(float(value) if valid else 0.0)
        available.append(valid)
    return tuple(block), tuple(available)


def _select_max_p_value_track_fit(
    track: object | None, *, pion_hypothesis: object
) -> object | None:
    """Implement the training producer's max-p-value/pion-fallback policy."""

    if track is None:
        return None
    collection = getattr(track, "getTrackFitResults", None)
    if collection is not None:
        try:
            candidates: list[tuple[float, str, object]] = []
            for pair in collection():
                hypothesis = getattr(pair, "first", None)
                fit = getattr(pair, "second", None)
                if fit is None:
                    try:
                        hypothesis, fit = pair[0], pair[1]
                    except (IndexError, TypeError):
                        continue
                p_value = _optional_finite(fit, ("getPValue",))
                if fit and p_value is not None:
                    candidates.append(
                        (p_value, _charged_hypothesis_name(hypothesis), fit)
                    )
            if candidates:
                return max(candidates, key=lambda item: (item[0], item[1]))[2]
        except (RuntimeError, TypeError, ValueError):
            pass

    best = getattr(track, "getTrackFitResultWithBestPValue", None)
    if best is not None:
        try:
            result = best()
        except (RuntimeError, TypeError, ValueError):
            result = None
        if result:
            return result

    closest = getattr(track, "getTrackFitResultWithClosestMass", None)
    if closest is not None:
        try:
            result = closest(pion_hypothesis)
        except (RuntimeError, TypeError, ValueError):
            result = None
        if result:
            return result
    return None


def _charged_hypothesis_name(hypothesis: object | None) -> str:
    if hypothesis is None:
        return "unknown"
    pdg = _optional_finite(hypothesis, ("getPDGCode",))
    if pdg is not None:
        by_pdg = dict(zip(_CHARGED_STABLE_PDGS, _CHARGED_STABLE_NAMES))
        return by_pdg.get(abs(int(pdg)), f"pdg_{abs(int(pdg))}")
    return str(hypothesis)


def _optional_call(obj: object, names: Sequence[str]) -> object | None:
    for name in names:
        method = getattr(obj, name, None)
        if method is None:
            continue
        try:
            return method()
        except (RuntimeError, TypeError, ValueError):
            continue
    return None


def _optional_finite(obj: object, names: Sequence[str]) -> float | None:
    value = _optional_call(obj, names)
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _optional_boolean(obj: object, names: Sequence[str]) -> float | None:
    value = _optional_call(obj, names)
    return None if value is None else float(bool(value))


def _pid_likelihood_detector_available(
    likelihood: object | None, detector_sets: Sequence[object]
) -> bool:
    if likelihood is None:
        return False
    is_available = getattr(likelihood, "isAvailable", None)
    if is_available is None:
        return False
    for detector_set in detector_sets:
        try:
            if bool(is_available(detector_set)):
                return True
        except (RuntimeError, TypeError, ValueError):
            continue
    return False


def _first_related_named(
    obj: object | None, relation_names: Sequence[str]
) -> object | None:
    """Return a reconstructed relation across supported release API variants."""

    if obj is None:
        return None
    for relation_name in relation_names:
        for method_name in ("getRelatedTo", "getRelatedFrom", "getRelated"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                related = method(relation_name)
            except (RuntimeError, TypeError, ValueError):
                continue
            if related:
                return related
        for method_name in ("getRelationsWith", "getRelationsTo", "getRelationsFrom"):
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                related_collection = method(relation_name)
            except (RuntimeError, TypeError, ValueError):
                continue
            try:
                return next(iter(related_collection))
            except (StopIteration, TypeError):
                continue
    return None


def _vector_component(vector: object, names: Sequence[str]) -> float | None:
    return _optional_finite(vector, names)


def _required_vector_component(
    vector: object, names: Sequence[str], description: str
) -> float:
    value = _vector_component(vector, names)
    if value is None:
        raise RuntimeError(f"selected fit has no finite {description}")
    return value


__all__ = ["HyperTaggingFullDecayModule", "add_hypertagging_full_decay"]
