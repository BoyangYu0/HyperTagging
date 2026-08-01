"""Thin basf2-dependent event source for direct mDST preprocessing.

This module deliberately imports basf2/ROOT only inside functions so the rest
of the preprocessing package remains importable and testable in a normal
``uv`` Python environment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from hypertagging.preprocessing.export_dataset import export_trees
from hypertagging.preprocessing.schema_v2 import SCHEMA_VERSION_V1, SCHEMA_VERSION_V2, export_trees_v2
from hypertagging.preprocessing.schema_v3 import SCHEMA_VERSION_V3, export_trees_v3
from hypertagging.preprocessing.schema_v4 import (
    SCHEMA_VERSION_V4,
    ParquetEventWriter,
)
from hypertagging.preprocessing.levelize_tree import assign_levels
from hypertagging.preprocessing.mdst_tree_builder import (
    FourVector,
    MCRecord,
    RecoRecord,
    build_trees,
    validate_tree,
)
from hypertagging.preprocessing.pid_filter import PidFilter, tokenize_pdg
from hypertagging.reconstruction.kinematics import (
    CANONICAL_TRACK_HYPOTHESIS,
    CHARGED_STABLE_NAMES,
    CHARGED_STABLE_PDGS,
    PARTICLE_MASSES_GEV,
)


@dataclass(frozen=True)
class Basf2PreprocessConfig:
    """Configuration for the basf2 direct-mDST steering module."""

    input_files: tuple[str, ...]
    output: Path
    max_events: int | None = None
    entry_sequences: tuple[str, ...] | None = None
    debug_event: int | None = None
    particle_arrays: tuple[str, ...] = ()
    include_tracks: bool = True
    include_ecl_clusters: bool = True
    include_klm_clusters: bool = True
    allow_mc_leaf_kinematics_for_debug: bool = False
    schema_version: str = SCHEMA_VERSION_V4
    charge_conjugate_normalize: bool = False
    event_buffer_size: int = 128
    row_group_size: int = 128
    leaf_kinematics_mode: str = "raw_track_predicted_pid"


@dataclass(frozen=True)
class TrackFitSelection:
    """Data-independent fit choice and its auditable provenance."""

    fit: object | None
    hypothesis: str | None
    method: str
    available: bool
    fallback_reason: str | None = None


def run_basf2_preprocessing(config: Basf2PreprocessConfig) -> Path:
    """Run direct-mDST preprocessing inside a basf2 Python process."""

    if config.leaf_kinematics_mode == "fixed_hypothesis_candidate":
        if not config.particle_arrays:
            raise ValueError(
                "fixed_hypothesis_candidate requires at least one explicit Particle array"
            )
        if config.include_tracks:
            raise ValueError(
                "fixed_hypothesis_candidate cannot silently include raw Tracks; pass --no-tracks"
            )
    import basf2 as b2  # type: ignore[import-not-found]
    import modularAnalysis as ma  # type: ignore[import-not-found]

    main = b2.create_path()
    input_kwargs: dict[str, object] = {
        "filelist": list(config.input_files),
        "environmentType": "default",
        "path": main,
    }
    if config.entry_sequences is not None:
        if len(config.entry_sequences) != len(config.input_files):
            raise ValueError("entry_sequences must have one value per input file")
        input_kwargs["entrySequences"] = list(config.entry_sequences)
    ma.inputMdstList(**input_kwargs)
    class DirectMdstCollector(_DirectMdstCollector, b2.Module):  # type: ignore[misc, valid-type]
        pass

    collector = DirectMdstCollector(config)
    main.add_module(collector)
    try:
        b2.process(path=main, max_event=config.max_events or 0)
        return collector.write_output()
    except Exception:
        if collector._v4_writer is not None:
            collector._v4_writer.abort()
        raise


class _DirectMdstCollector:
    """basf2 module-like collector.

    It avoids the historical loop over many prebuilt ParticleLists.  It reads
    DataStore arrays for MC topology and reconstructed candidates/objects.
    """

    def __init__(self, config: Basf2PreprocessConfig) -> None:
        super().__init__()  # type: ignore[misc]
        self.config = config
        self.events: list[tuple[int, Sequence[MCRecord], Sequence[RecoRecord]]] = []
        self.event_metadata: list[dict[str, int | str]] = []
        self._event_count = 0
        self.collection_stats: Counter[str] = Counter()
        self._v4_writer: ParquetEventWriter | None = None
        self._v4_pid_filter: PidFilter | None = None

    def initialize(self) -> None:
        from ROOT import Belle2  # type: ignore[import-not-found]

        self.event_info = Belle2.PyStoreObj("EventMetaData")
        self.mc_particles = Belle2.PyStoreArray("MCParticles")
        self.particle_arrays = [Belle2.PyStoreArray(name) for name in self.config.particle_arrays]
        self.tracks = Belle2.PyStoreArray("Tracks")
        self.ecl_clusters = Belle2.PyStoreArray("ECLClusters")
        self.klm_clusters = Belle2.PyStoreArray("KLMClusters")
        self._charged_stable = Belle2.Const.ChargedStable
        self._cluster_utils = Belle2.ClusterUtils()
        self._photon_hypothesis = Belle2.ECLCluster.EHypothesisBit.c_nPhotons
        self._pid_detector_sets = {
            name.lower(): Belle2.Const.PIDDetectorSet(getattr(Belle2.Const, name))
            for name in ("SVD", "CDC", "TOP", "ARICH", "ECL", "KLM")
        }
        if self.config.schema_version == SCHEMA_VERSION_V4:
            self._v4_pid_filter = PidFilter()
            self._v4_writer = ParquetEventWriter(
                self.config.output,
                event_buffer_size=self.config.event_buffer_size,
                row_group_size=self.config.row_group_size,
                metadata={
                    "source_file": (
                        self.config.input_files[0]
                        if len(self.config.input_files) == 1
                        else list(self.config.input_files)
                    ),
                    "entry_start": None,
                    "entry_stop_exclusive": None,
                    "leaf_kinematics_mode": self.config.leaf_kinematics_mode,
                    "charge_conjugate_normalization": (
                        self.config.charge_conjugate_normalize
                    ),
                },
            )

    def event(self) -> None:
        event_id = int(self.event_info.getEvent()) if self.event_info else self._event_count
        if self.config.debug_event is not None and event_id != self.config.debug_event:
            self._event_count += 1
            return
        mc_records = self._collect_mc_records()
        reco_records = self._collect_reco_records()
        if not reco_records and self.config.allow_mc_leaf_kinematics_for_debug:
            reco_records = self._debug_reco_from_truth_leaves(mc_records)
        if not reco_records:
            raise RuntimeError(
                "No reconstructed Particle/Track/ECLCluster records were found in this mDST event. "
                "Use an mDST containing reconstruction relations or pass "
                "--allow-mc-leaf-kinematics-for-debug only for synthetic debugging."
            )
        experiment = int(self.event_info.getExperiment()) if self.event_info else -1
        run = int(self.event_info.getRun()) if self.event_info else -1
        production = int(self.event_info.getProduction()) if self.event_info else -1
        source_file = _event_source_file(self.event_info, self.config.input_files)
        metadata = {
                "experiment": experiment,
                "run": run,
                "production": production,
                "event_uid": f"{experiment}:{run}:{event_id}:{production}",
                "source_file": source_file,
                "source_category": _source_category(source_file),
                "source_file_resolution": (
                    "event_metadata"
                    if len(self.config.input_files) > 1 and source_file
                    else ("single_input" if len(self.config.input_files) == 1 else "unresolved")
                ),
            }
        if self._v4_writer is not None:
            assert self._v4_pid_filter is not None
            trees, _summary = build_trees(
                [(event_id, mc_records, reco_records)],
                pid_filter=self._v4_pid_filter,
            )
            tree = trees[0]
            tree.metadata.update(metadata)
            assign_levels(tree)
            validate_tree(tree)
            self._v4_writer.write_tree(
                tree,
                charge_conjugate_normalize=self.config.charge_conjugate_normalize,
            )
        else:
            self.events.append((event_id, mc_records, reco_records))
            self.event_metadata.append(metadata)
        self._event_count += 1

    def write_output(self) -> Path:
        if self._v4_writer is not None:
            assert self._v4_pid_filter is not None
            self._v4_writer.metadata["preprocessing_configuration"] = {
                "collection": dict(sorted(self.collection_stats.items())),
                "pid_filter": self._v4_pid_filter.summary.as_dict(),
                "input_files": list(self.config.input_files),
                "entry_sequences": (
                    None
                    if self.config.entry_sequences is None
                    else list(self.config.entry_sequences)
                ),
                "event_buffer_size": self.config.event_buffer_size,
                "row_group_size": self.config.row_group_size,
            }
            return self._v4_writer.close()
        pid_filter = PidFilter()
        trees, summary = build_trees(self.events, pid_filter=pid_filter)
        for tree, metadata in zip(trees, self.event_metadata):
            tree.metadata.update(metadata)
            assign_levels(tree)
            validate_tree(tree)
        summary_record = summary.as_dict()
        summary_record["collection"] = dict(sorted(self.collection_stats.items()))
        summary_record["input_files"] = list(self.config.input_files)
        summary_record["entry_sequences"] = (
            None if self.config.entry_sequences is None else list(self.config.entry_sequences)
        )
        if self.config.schema_version == SCHEMA_VERSION_V1:
            return export_trees(trees, self.config.output, summary=summary_record)
        if self.config.schema_version == SCHEMA_VERSION_V2:
            return export_trees_v2(
                trees,
                self.config.output,
                summary=summary_record,
                charge_conjugate_normalize=self.config.charge_conjugate_normalize,
            )
        if self.config.schema_version == SCHEMA_VERSION_V3:
            return export_trees_v3(
                trees,
                self.config.output,
                summary=summary_record,
                charge_conjugate_normalize=self.config.charge_conjugate_normalize,
            )
        raise ValueError(f"Unsupported output schema: {self.config.schema_version}")

    def _collect_mc_records(self) -> list[MCRecord]:
        records: list[MCRecord] = []
        for particle in self.mc_particles:
            mother = particle.getMother()
            p4 = particle.get4Vector()
            vertex = particle.getProductionVertex()
            records.append(
                MCRecord(
                    mc_id=int(particle.getArrayIndex()),
                    pdg=int(particle.getPDG()),
                    charge=float(particle.getCharge()),
                    mother_id=None if not mother else int(mother.getArrayIndex()),
                    p4=FourVector(float(p4.Px()), float(p4.Py()), float(p4.Pz()), float(particle.getEnergy())),
                    name=str(particle.getName()),
                    is_primary=bool(particle.isPrimaryParticle()),
                    prod_time=float(particle.getProductionTime()),
                    vertex=(float(vertex.x()), float(vertex.y()), float(vertex.z())),
                )
            )
        return records

    def _collect_reco_records(self) -> list[RecoRecord]:
        records: list[RecoRecord] = []
        for store_array in self.particle_arrays:
            records.extend(self._collect_particle_array(store_array))
        if self.config.include_tracks:
            records.extend(self._collect_tracks())
        if self.config.include_ecl_clusters:
            records.extend(self._collect_ecl_clusters())
        if self.config.include_klm_clusters:
            records.extend(self._collect_klm_clusters())
        dedup: dict[str, RecoRecord] = {}
        for record in records:
            # Particle candidates and raw Tracks can refer to the same
            # underlying Track.  Prefer the first explicitly configured
            # candidate and deduplicate by provenance, not display ID.
            dedup.setdefault(str(record.underlying_reco_id or record.reco_id), record)
        self.collection_stats["reco_provenance_duplicates"] += len(records) - len(dedup)
        return list(dedup.values())

    def _collect_particle_array(self, store_array: object) -> list[RecoRecord]:
        records: list[RecoRecord] = []
        try:
            iterator = iter(store_array)  # type: ignore[arg-type]
        except TypeError:
            return records
        for particle in iterator:
            try:
                if hasattr(particle, "getNDaughters") and int(particle.getNDaughters()) > 0:
                    continue
                p4 = particle.get4Vector()
                mc = _related_mc(particle)
                pdg = int(particle.getPDGCode() if hasattr(particle, "getPDGCode") else particle.getPDG())
                underlying_reco_id = _particle_underlying_reco_id(particle)
                records.append(
                    RecoRecord(
                        reco_id=f"Particle:{particle.getArrayIndex()}",
                        pdg=pdg,
                        charge=float(particle.getCharge()),
                        p4=FourVector(float(p4.Px()), float(p4.Py()), float(p4.Pz()), float(p4.E())),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                        node_kind="unknown",
                        candidate_confidence=_optional_float(particle, ("getPValue",)),
                        raw_pdg=pdg,
                        input_pid_token=tokenize_pdg(pdg),
                        truth_pdg=None if mc is None else int(mc.getPDG()),
                        truth_pid_token=None if mc is None else tokenize_pdg(int(mc.getPDG())),
                        truth_charge=None if mc is None else float(mc.getCharge()),
                        energy_source="basf2_particle_candidate_hypothesis",
                        leaf_kinematics_mode="fixed_hypothesis_candidate",
                        reco_quality_score=_optional_float(particle, ("getPValue",)),
                        underlying_reco_id=underlying_reco_id,
                    )
                )
            except Exception as exc:
                self.collection_stats[f"particle_errors:{type(exc).__name__}"] += 1
                continue
        self.collection_stats["particle_records"] += len(records)
        return records

    def _collect_tracks(self) -> list[RecoRecord]:
        records: list[RecoRecord] = []
        for track in self.tracks:
            try:
                mc = _related_mc(track)
                fit_selection = _select_data_independent_track_fit(
                    track,
                    pion_hypothesis=self._charged_stable(211),
                )
                fit = fit_selection.fit
                if not fit:
                    self.collection_stats["tracks_without_fit"] += 1
                    continue
                momentum = fit.getMomentum()
                charge = float(fit.getChargeSign())
                px, py, pz = float(momentum.X()), float(momentum.Y()), float(momentum.Z())
                p2 = px * px + py * py + pz * pz
                energy_hypotheses = {
                    name: math.sqrt(p2 + PARTICLE_MASSES_GEV[pdg] ** 2)
                    for name, pdg in zip(CHARGED_STABLE_NAMES, CHARGED_STABLE_PDGS)
                }
                (
                    likelihoods,
                    likelihood_availability,
                    likelihood_status,
                    detector_availability,
                ) = self._track_pid_likelihoods(track)
                truth_pdg, truth_charge = _mc_truth_fields(mc)
                fit_quality = _optional_float(fit, ("getPValue",))
                records.append(
                    RecoRecord(
                        reco_id=f"Track:{track.getArrayIndex()}",
                        pdg=0,
                        charge=charge,
                        p4=FourVector(px, py, pz, energy_hypotheses[CANONICAL_TRACK_HYPOTHESIS]),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                        node_kind="track",
                        candidate_confidence=fit_quality,
                        track_features=_available_values(
                            {
                                "fit_p_value": _optional_float(fit, ("getPValue",)),
                                "d0": _optional_float(fit, ("getD0",)),
                                "z0": _optional_float(fit, ("getZ0",)),
                                "phi0": _optional_float(fit, ("getPhi0",)),
                                "omega": _optional_float(fit, ("getOmega",)),
                                "tan_lambda": _optional_float(fit, ("getTanLambda",)),
                            }
                        ),
                        raw_pdg=0,
                        input_pid_token=0,
                        pid_target_token=0 if truth_pdg is None else tokenize_pdg(truth_pdg),
                        truth_pdg=truth_pdg,
                        truth_pid_token=None if truth_pdg is None else tokenize_pdg(truth_pdg),
                        reco_charge=charge,
                        truth_charge=truth_charge,
                        energy_source=f"canonical_{CANONICAL_TRACK_HYPOTHESIS}_mass_hypothesis",
                        leaf_kinematics_mode="raw_track_predicted_pid",
                        track_energy_hypotheses=energy_hypotheses,
                        track_energy_availability={
                            name: True for name in CHARGED_STABLE_NAMES
                        },
                        pid_likelihoods=likelihoods,
                        pid_likelihood_availability=likelihood_availability,
                        pid_likelihood_status=likelihood_status,
                        pid_detector_availability=detector_availability,
                        track_fit_hypothesis=fit_selection.hypothesis,
                        track_fit_selection_method=fit_selection.method,
                        track_fit_available=fit_selection.available,
                        track_fit_fallback_reason=fit_selection.fallback_reason,
                        reco_quality_score=fit_quality,
                        underlying_reco_id=f"Track:{track.getArrayIndex()}",
                    )
                )
            except Exception as exc:
                self.collection_stats[f"track_errors:{type(exc).__name__}"] += 1
                continue
        self.collection_stats["track_records"] += len(records)
        return records

    def _track_pid_likelihoods(
        self, track: object
    ) -> tuple[
        dict[str, float],
        dict[str, bool],
        dict[str, str],
        dict[str, bool],
    ]:
        """Read only verified generic-mDST PIDLikelihood relations/accessors."""

        pid_likelihood = _related_named(track, "PIDLikelihoods") or _related_named(
            track, "PIDLikelihood"
        )
        values: dict[str, float] = {}
        availability: dict[str, bool] = {}
        status: dict[str, str] = {}
        detector_availability: dict[str, bool] = {}
        detector_sets = getattr(self, "_pid_detector_sets", {})
        if pid_likelihood:
            is_available = getattr(pid_likelihood, "isAvailable", None)
            for detector_name, detector_set in detector_sets.items():
                try:
                    detector_availability[detector_name] = bool(
                        is_available(detector_set)
                    )
                except Exception:
                    detector_availability[detector_name] = False
        for name, pdg in zip(CHARGED_STABLE_NAMES, CHARGED_STABLE_PDGS):
            hypothesis = self._charged_stable(pdg)
            available = False
            value: float | None = None
            if not pid_likelihood:
                status[name] = "relation_missing"
            elif getattr(pid_likelihood, "getLogL", None) is None:
                status[name] = "method_unavailable_in_release"
            elif detector_sets and not any(detector_availability.values()):
                status[name] = "detector_likelihood_unavailable"
            else:
                try:
                    candidate = float(pid_likelihood.getLogL(hypothesis))
                except Exception:
                    candidate = math.nan
                    status[name] = "method_unavailable_in_release"
                if math.isfinite(candidate):
                    value = candidate
                    available = True
                    status[name] = "valid_likelihood_value"
                elif name not in status:
                    status[name] = "detector_likelihood_unavailable"
            availability[name] = available
            if value is not None:
                values[name] = value
        return values, availability, status, detector_availability

    def _collect_ecl_clusters(self) -> list[RecoRecord]:
        records: list[RecoRecord] = []
        for cluster in self.ecl_clusters:
            try:
                if cluster.isTrack():
                    self.collection_stats["ecl_track_matched_skipped"] += 1
                    continue
                if not cluster.hasHypothesis(self._photon_hypothesis):
                    self.collection_stats["ecl_without_photon_hypothesis"] += 1
                    continue
                mc = _related_mc(cluster)
                truth_pdg, truth_charge = _mc_truth_fields(mc)
                momentum = self._cluster_utils.Get4MomentumFromCluster(cluster, self._photon_hypothesis)
                records.append(
                    RecoRecord(
                        reco_id=f"ECLCluster:{cluster.getArrayIndex()}",
                        pdg=22,
                        charge=0.0,
                        p4=FourVector(
                            float(momentum.Px()),
                            float(momentum.Py()),
                            float(momentum.Pz()),
                            float(momentum.E()),
                        ),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                        node_kind="ecl_cluster",
                        raw_pdg=22,
                        input_pid_token=tokenize_pdg(22),
                        pid_target_token=tokenize_pdg(22 if truth_pdg is None else truth_pdg),
                        truth_pdg=truth_pdg,
                        truth_pid_token=None if truth_pdg is None else tokenize_pdg(truth_pdg),
                        reco_charge=0.0,
                        truth_charge=truth_charge,
                        energy_source="ecl_cluster_photon_hypothesis",
                        leaf_kinematics_mode="fixed_hypothesis_candidate",
                        underlying_reco_id=f"ECLCluster:{cluster.getArrayIndex()}",
                        cluster_features=_available_values(
                            {
                                "cluster_energy": float(momentum.E()),
                                "theta": _theta_from_xyz(
                                    float(momentum.Px()),
                                    float(momentum.Py()),
                                    float(momentum.Pz()),
                                ),
                                "phi": math.atan2(float(momentum.Py()), float(momentum.Px())),
                                "time": _optional_float(cluster, ("getTime",)),
                                "e9_over_e21": _optional_float(
                                    cluster,
                                    ("getE9oE21", "getE9OverE21"),
                                ),
                                "n_crystals": _optional_float(
                                    cluster,
                                    ("getNumberOfCrystals", "getNumberOfConnectedCrystals"),
                                ),
                                "min_track_distance": _optional_float(
                                    cluster,
                                    ("getMinTrackDistance",),
                                ),
                                "photon_hypothesis": 1.0,
                                "track_matched": 0.0,
                            }
                        ),
                    )
                )
            except Exception as exc:
                self.collection_stats[f"ecl_errors:{type(exc).__name__}"] += 1
                continue
        self.collection_stats["ecl_records"] += len(records)
        return records

    def _collect_klm_clusters(self) -> list[RecoRecord]:
        """Collect reconstructed KLM clusters with an explicit KLM node kind."""

        records: list[RecoRecord] = []
        for cluster in self.klm_clusters:
            try:
                momentum = cluster.getMomentum()
                px = float(momentum.Px())
                py = float(momentum.Py())
                pz = float(momentum.Pz())
                energy = float(
                    momentum.E()
                    if hasattr(momentum, "E")
                    else cluster.getEnergy()
                )
                mc = _related_mc(cluster)
                truth_pdg, truth_charge = _mc_truth_fields(mc)
                position = cluster.getClusterPosition()
                records.append(
                    RecoRecord(
                        reco_id=f"KLMCluster:{cluster.getArrayIndex()}",
                        pdg=130,
                        charge=0.0,
                        p4=FourVector(px, py, pz, energy),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                        node_kind="klm_cluster",
                        raw_pdg=130,
                        input_pid_token=tokenize_pdg(130),
                        pid_target_token=tokenize_pdg(
                            130 if truth_pdg is None else truth_pdg
                        ),
                        truth_pdg=truth_pdg,
                        truth_pid_token=(
                            None if truth_pdg is None else tokenize_pdg(truth_pdg)
                        ),
                        reco_charge=0.0,
                        truth_charge=truth_charge,
                        energy_source="klm_cluster_reconstructed_momentum",
                        leaf_kinematics_mode="klm_cluster",
                        klm_features=_available_values(
                            {
                                "energy": _optional_float(cluster, ("getEnergy",)),
                                "momentum_magnitude": _optional_float(
                                    cluster, ("getMomentumMag",)
                                ),
                                "x": float(position.X()),
                                "y": float(position.Y()),
                                "z": float(position.Z()),
                                "time": _optional_float(cluster, ("getTime",)),
                                "layers": _optional_float(cluster, ("getLayers",)),
                                "innermost_layer": _optional_float(
                                    cluster, ("getInnermostLayer",)
                                ),
                                "associated_ecl_cluster": _optional_float(
                                    cluster, ("getAssociatedEclClusterFlag",)
                                ),
                            }
                        ),
                        underlying_reco_id=f"KLMCluster:{cluster.getArrayIndex()}",
                    )
                )
            except Exception as exc:
                self.collection_stats[f"klm_errors:{type(exc).__name__}"] += 1
                continue
        self.collection_stats["klm_records"] += len(records)
        return records

    def _debug_reco_from_truth_leaves(self, mc_records: Sequence[MCRecord]) -> list[RecoRecord]:
        mother_ids = {record.mother_id for record in mc_records if record.mother_id is not None}
        records: list[RecoRecord] = []
        for mc in mc_records:
            if mc.mc_id in mother_ids or mc.p4 is None:
                continue
            records.append(
                RecoRecord(
                    reco_id=f"DebugMCLeaf:{mc.mc_id}",
                    pdg=mc.pdg,
                    charge=mc.charge,
                    p4=mc.p4,
                    mc_id=mc.mc_id,
                    flags=frozenset({"debug_mc_leaf_kinematics"}),
                    node_kind="unknown",
                )
            )
        return records


def _related_mc(obj: object) -> object | None:
    for method_name in ("getRelatedTo", "getRelated"):
        method = getattr(obj, method_name, None)
        if method is None:
            continue
        for relation_name in ("MCParticles", "MCParticle"):
            try:
                related = method(relation_name)
            except TypeError:
                continue
            if related:
                return related
    return None


def _related_named(obj: object, relation_name: str) -> object | None:
    for method_name in ("getRelatedTo", "getRelatedFrom", "getRelated"):
        method = getattr(obj, method_name, None)
        if method is None:
            continue
        try:
            related = method(relation_name)
        except (TypeError, RuntimeError):
            continue
        if related:
            return related
    return None


def _mc_truth_fields(mc: object | None) -> tuple[int | None, float | None]:
    if mc is None:
        return None, None
    try:
        pdg = int(mc.getPDG())
    except Exception:
        pdg = None
    try:
        charge = float(mc.getCharge())
    except Exception:
        charge = None
    return pdg, charge


def _select_data_independent_track_fit(
    track: object, *, pion_hypothesis: object
) -> TrackFitSelection:
    """Select the maximum-p-value reconstructed fit without consulting MC.

    Release 08-03-00 exposes getTrackFitResults as charged-hypothesis/fit
    pairs but has no getTrackFitResultWithBestPValue method. Newer API
    variants are supported explicitly, with a deterministic pion closest-mass
    fallback.
    """

    fit_results = getattr(track, "getTrackFitResults", None)
    collection_failure: str | None = None
    if fit_results is not None:
        try:
            candidates: list[tuple[float, str, object]] = []
            for pair in fit_results():
                hypothesis = getattr(
                    pair,
                    "first",
                    pair[0] if isinstance(pair, tuple) else None,
                )
                fit = getattr(
                    pair,
                    "second",
                    pair[1] if isinstance(pair, tuple) else None,
                )
                if not fit:
                    continue
                p_value = _optional_float(fit, ("getPValue",))
                if p_value is None:
                    continue
                candidates.append(
                    (p_value, _charged_hypothesis_name(hypothesis), fit)
                )
            if candidates:
                _p_value, hypothesis, fit = max(
                    candidates, key=lambda item: (item[0], item[1])
                )
                return TrackFitSelection(
                    fit=fit,
                    hypothesis=hypothesis,
                    method="getTrackFitResults_max_p_value",
                    available=True,
                )
            collection_failure = "no_finite_pvalue_fit_in_collection"
        except Exception as exc:
            collection_failure = f"fit_collection_error:{type(exc).__name__}"
    else:
        collection_failure = "getTrackFitResults_unavailable"

    best = getattr(track, "getTrackFitResultWithBestPValue", None)
    if best is not None:
        try:
            result = best()
        except Exception as exc:
            result = None
            collection_failure = f"best_pvalue_accessor_error:{type(exc).__name__}"
        if result:
            return TrackFitSelection(
                fit=result,
                hypothesis=_fit_result_hypothesis_name(result),
                method="getTrackFitResultWithBestPValue",
                available=True,
                fallback_reason=collection_failure,
            )

    closest = getattr(track, "getTrackFitResultWithClosestMass", None)
    if closest is not None:
        try:
            result = closest(pion_hypothesis)
        except Exception as exc:
            return TrackFitSelection(
                fit=None,
                hypothesis=None,
                method="unavailable",
                available=False,
                fallback_reason=f"closest_mass_accessor_error:{type(exc).__name__}",
            )
        if result:
            return TrackFitSelection(
                fit=result,
                hypothesis="pion",
                method="getTrackFitResultWithClosestMass_pion",
                available=True,
                fallback_reason=collection_failure,
            )
    return TrackFitSelection(
        fit=None,
        hypothesis=None,
        method="unavailable",
        available=False,
        fallback_reason=collection_failure or "no_supported_fit_accessor",
    )


def _data_independent_track_fit(
    track: object, *, pion_hypothesis: object
) -> object | None:
    """Compatibility wrapper returning only the selected fit object."""

    return _select_data_independent_track_fit(
        track, pion_hypothesis=pion_hypothesis
    ).fit


def _charged_hypothesis_name(hypothesis: object | None) -> str:
    if hypothesis is None:
        return "unknown"
    pdg_method = getattr(hypothesis, "getPDGCode", None)
    if pdg_method is not None:
        try:
            pdg = abs(int(pdg_method()))
            return dict(
                zip(map(abs, CHARGED_STABLE_PDGS), CHARGED_STABLE_NAMES)
            ).get(pdg, f"pdg_{pdg}")
        except Exception:
            pass
    return str(hypothesis)


def _fit_result_hypothesis_name(fit: object) -> str:
    particle_type = getattr(fit, "getParticleType", None)
    if particle_type is None:
        return "unknown"
    try:
        return _charged_hypothesis_name(particle_type())
    except Exception:
        return "unknown"


def _particle_underlying_reco_id(particle: object) -> str:
    track = _related_named(particle, "Track")
    if track is not None and hasattr(track, "getArrayIndex"):
        return f"Track:{int(track.getArrayIndex())}"
    cluster = _related_named(particle, "ECLCluster")
    if cluster is not None and hasattr(cluster, "getArrayIndex"):
        return f"ECLCluster:{int(cluster.getArrayIndex())}"
    return f"Particle:{int(particle.getArrayIndex())}"


def _mass_from_pdg(pdg: int) -> float:
    masses = {
        11: 0.00051099895,
        13: 0.1056583755,
        211: 0.13957039,
        321: 0.493677,
        2212: 0.93827208816,
    }
    return masses.get(abs(int(pdg)), masses[211])


def _optional_float(obj: object, method_names: tuple[str, ...]) -> float | None:
    """Read a genuinely available scalar without assigning a missing sentinel."""

    for method_name in method_names:
        method = getattr(obj, method_name, None)
        if method is None:
            continue
        try:
            value = float(method())
        except Exception:
            continue
        if math.isfinite(value):
            return value
    return None


def _available_values(values: dict[str, float | None]) -> dict[str, float]:
    return {name: float(value) for name, value in values.items() if value is not None and math.isfinite(value)}


def _theta_from_xyz(px: float, py: float, pz: float) -> float | None:
    norm = (px * px + py * py + pz * pz) ** 0.5
    if norm == 0.0:
        return None
    return math.acos(max(-1.0, min(1.0, pz / norm)))


def _source_category(path: str) -> str:
    known = ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar", "taupair")
    parts = Path(path).parts
    return next((part for part in parts if part in known), "")


def _event_source_file(event_info: object, input_files: tuple[str, ...]) -> str:
    if len(input_files) == 1:
        return input_files[0]
    for accessor in ("getParentLfn", "getInputFileName"):
        method = getattr(event_info, accessor, None)
        if method is None:
            continue
        try:
            value = str(method())
        except Exception:
            continue
        if value:
            exact = next(
                (
                    candidate
                    for candidate in input_files
                    if candidate == value or Path(candidate).name == Path(value).name
                ),
                None,
            )
            return exact or value
    return ""
