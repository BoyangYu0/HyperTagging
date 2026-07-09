"""Thin basf2-dependent event source for direct mDST preprocessing.

This module deliberately imports basf2/ROOT only inside functions so the rest
of the preprocessing package remains importable and testable in a normal
``uv`` Python environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hypertagging.preprocessing.export_dataset import export_trees
from hypertagging.preprocessing.levelize_tree import assign_levels
from hypertagging.preprocessing.mdst_tree_builder import (
    FourVector,
    MCRecord,
    RecoRecord,
    build_trees,
    validate_tree,
)
from hypertagging.preprocessing.pid_filter import PidFilter


@dataclass(frozen=True)
class Basf2PreprocessConfig:
    """Configuration for the basf2 direct-mDST steering module."""

    input_files: tuple[str, ...]
    output: Path
    max_events: int | None = None
    debug_event: int | None = None
    particle_arrays: tuple[str, ...] = ("Particles",)
    include_tracks: bool = True
    include_ecl_clusters: bool = True
    allow_mc_leaf_kinematics_for_debug: bool = False


def run_basf2_preprocessing(config: Basf2PreprocessConfig) -> Path:
    """Run direct-mDST preprocessing inside a basf2 Python process."""

    import basf2 as b2  # type: ignore[import-not-found]
    import modularAnalysis as ma  # type: ignore[import-not-found]

    main = b2.create_path()
    ma.inputMdst(environmentType="default", filename=list(config.input_files), path=main)
    class DirectMdstCollector(_DirectMdstCollector, b2.Module):  # type: ignore[misc, valid-type]
        pass

    collector = DirectMdstCollector(config)
    main.add_module(collector)
    b2.process(path=main, max_event=config.max_events or 0)
    return collector.write_output()


class _DirectMdstCollector:
    """basf2 module-like collector.

    It avoids the historical loop over many prebuilt ParticleLists.  It reads
    DataStore arrays for MC topology and reconstructed candidates/objects.
    """

    def __init__(self, config: Basf2PreprocessConfig) -> None:
        super().__init__()  # type: ignore[misc]
        self.config = config
        self.events: list[tuple[int, Sequence[MCRecord], Sequence[RecoRecord]]] = []
        self._event_count = 0

    def initialize(self) -> None:
        from ROOT import Belle2  # type: ignore[import-not-found]

        self.event_info = Belle2.PyStoreObj("EventMetaData")
        self.mc_particles = Belle2.PyStoreArray("MCParticles")
        self.particle_arrays = [Belle2.PyStoreArray(name) for name in self.config.particle_arrays]
        self.tracks = Belle2.PyStoreArray("Tracks")
        self.ecl_clusters = Belle2.PyStoreArray("ECLClusters")

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
        self.events.append((event_id, mc_records, reco_records))
        self._event_count += 1

    def write_output(self) -> Path:
        pid_filter = PidFilter()
        trees, summary = build_trees(self.events, pid_filter=pid_filter)
        for tree in trees:
            assign_levels(tree)
            validate_tree(tree)
        return export_trees(trees, self.config.output, summary=summary.as_dict())

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
        dedup: dict[str, RecoRecord] = {}
        for record in records:
            dedup.setdefault(record.reco_id, record)
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
                records.append(
                    RecoRecord(
                        reco_id=f"Particle:{particle.getArrayIndex()}",
                        pdg=pdg,
                        charge=float(particle.getCharge()),
                        p4=FourVector(float(p4.Px()), float(p4.Py()), float(p4.Pz()), float(p4.E())),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                    )
                )
            except Exception:
                continue
        return records

    def _collect_tracks(self) -> list[RecoRecord]:
        records: list[RecoRecord] = []
        for track in self.tracks:
            try:
                mc = _related_mc(track)
                pdg = int(mc.getPDG()) if mc is not None else 211
                fit = track.getTrackFitResultWithClosestMass(abs(pdg))
                momentum = fit.getMomentum()
                charge = float(fit.getChargeSign())
                mass = _mass_from_pdg(pdg)
                px, py, pz = float(momentum.X()), float(momentum.Y()), float(momentum.Z())
                energy = (px * px + py * py + pz * pz + mass * mass) ** 0.5
                records.append(
                    RecoRecord(
                        reco_id=f"Track:{track.getArrayIndex()}",
                        pdg=pdg,
                        charge=charge,
                        p4=FourVector(px, py, pz, energy),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                    )
                )
            except Exception:
                continue
        return records

    def _collect_ecl_clusters(self) -> list[RecoRecord]:
        records: list[RecoRecord] = []
        for cluster in self.ecl_clusters:
            try:
                mc = _related_mc(cluster)
                pdg = int(mc.getPDG()) if mc is not None else 22
                energy = float(cluster.getEnergy())
                momentum = cluster.getMomentum()
                records.append(
                    RecoRecord(
                        reco_id=f"ECLCluster:{cluster.getArrayIndex()}",
                        pdg=pdg,
                        charge=0.0,
                        p4=FourVector(float(momentum.X()), float(momentum.Y()), float(momentum.Z()), energy),
                        mc_id=None if mc is None else int(mc.getArrayIndex()),
                    )
                )
            except Exception:
                continue
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


def _mass_from_pdg(pdg: int) -> float:
    masses = {
        11: 0.00051099895,
        13: 0.1056583755,
        211: 0.13957039,
        321: 0.493677,
        2212: 0.93827208816,
    }
    return masses.get(abs(int(pdg)), masses[211])
