#!/usr/bin/env python3
"""One-event mDST smoke for Track PID feature-availability parity."""

from __future__ import annotations

import os
from pathlib import Path

import basf2
import modularAnalysis as ma
from ROOT import Belle2

from hypertagging.basf2_integration.module import (
    HyperTaggingFullDecayModule,
    _first_related_named,
    _pid_likelihood_detector_available,
)
from hypertagging.deployment.onnx_contract import FEATURE_NAMES


POSITIVE_LIST = "pi+:HyperTaggingTrackPositiveSmoke"
NEGATIVE_LIST = "pi-:HyperTaggingTrackNegativeSmoke"
PHOTON_LIST = "gamma:HyperTaggingTrackSmoke"
OUTPUT_LIST = "Upsilon(4S):HyperTaggingTrackSmoke"


def _terminals(particle: object) -> list[object]:
    if int(particle.getNDaughters()) == 0:
        return [particle]
    result: list[object] = []
    for index in range(int(particle.getNDaughters())):
        result.extend(_terminals(particle.getDaughter(index)))
    return result


class TrackContractFullDecayModule(HyperTaggingFullDecayModule):
    """Check the deployed mask against the producer's detector gate."""

    def _collect_particle_list_leaves(self):  # type: ignore[no-untyped-def]
        leaves = super()._collect_particle_list_leaves()
        tracks = [node for node in leaves if node.source_kind == "Track"]
        if len(tracks) != 2 or {int(round(node.charge)) for node in tracks} != {-1, 1}:
            raise RuntimeError(
                "track smoke requires one reconstructed track per charge: "
                f"n_tracks={len(tracks)}, charges={[node.charge for node in tracks]}, "
                f"source_kinds={[node.source_kind for node in leaves]}"
            )
        likelihood_positions = tuple(
            FEATURE_NAMES["track"].index(f"pid_log_likelihood_{name}")
            for name in ("electron", "muon", "pion", "kaon", "proton")
        )
        for node in tracks:
            track = node.source_object.getTrack()
            likelihood = _first_related_named(
                track, ("PIDLikelihoods", "PIDLikelihood")
            )
            expected = _pid_likelihood_detector_available(
                likelihood, self._pid_detector_sets
            )
            actual = {
                bool(node.track_availability[position])
                for position in likelihood_positions
            }
            if actual != {expected}:
                raise RuntimeError(
                    "deployed PID-likelihood mask differs from detector availability: "
                    f"expected {expected}, got {sorted(actual)}"
                )
        return leaves


class AssertTrackSourcesPreserved(basf2.Module):
    def __init__(self) -> None:
        super().__init__()
        self.events_checked = 0

    def event(self) -> None:
        output = Belle2.PyStoreObj(Belle2.ParticleList.Class(), OUTPUT_LIST)
        if not output.isValid() or int(output.getListSize()) != 1:
            raise RuntimeError("track smoke expected one completed output root")
        terminals = _terminals(output.getParticle(0))
        source_counts = {
            int(Belle2.Particle.c_Track): 0,
            int(Belle2.Particle.c_ECLCluster): 0,
        }
        track_pdgs: set[int] = set()
        source_keys: set[tuple[int, int]] = set()
        for particle in terminals:
            source = int(particle.getParticleSource())
            if source not in source_counts:
                raise RuntimeError(f"unexpected terminal source kind {source}")
            source_counts[source] += 1
            source_key = (source, int(particle.getMdstArrayIndex()))
            if source_key in source_keys:
                raise RuntimeError(f"reused reconstructed source {source_key}")
            source_keys.add(source_key)
            if source == int(Belle2.Particle.c_Track):
                track_pdgs.add(int(particle.getPDGCode()))
        if sorted(source_counts.values()) != [2, 2] or track_pdgs != {-211, 211}:
            raise RuntimeError(
                "track smoke lost source/PID construction: "
                f"counts={source_counts}, track_pdgs={sorted(track_pdgs)}"
            )
        self.events_checked += 1

    def terminate(self) -> None:
        if self.events_checked != 1:
            raise RuntimeError(f"expected one track event, got {self.events_checked}")
        basf2.B2INFO(
            "HyperTagging real-track smoke passed: detector-gated PID features "
            "and two unique Track sources survived materialization"
        )


manifest = Path(os.environ["HYPERTAGGING_ONNX_BUNDLE"]).resolve()
mdst = Path(os.environ["HYPERTAGGING_SMOKE_MDST"]).resolve()
if not manifest.is_file() or not mdst.is_file():
    raise RuntimeError("real-track smoke inputs are unavailable")

path = basf2.create_path()
ma.inputMdst(str(mdst), path=path)
ma.fillParticleList(POSITIVE_LIST, "charge > 0", writeOut=False, path=path)
ma.rankByHighest(POSITIVE_LIST, "p", numBest=1, path=path)
ma.fillParticleList(NEGATIVE_LIST, "charge < 0", writeOut=False, path=path)
ma.rankByHighest(NEGATIVE_LIST, "p", numBest=1, path=path)
ma.fillParticleList(PHOTON_LIST, "", writeOut=False, path=path)
ma.rankByHighest(PHOTON_LIST, "E", numBest=2, path=path)
path.add_module(
    TrackContractFullDecayModule(
        manifest_path=manifest,
        input_particle_lists=(POSITIVE_LIST, NEGATIVE_LIST, PHOTON_LIST),
        output_particle_list=OUTPUT_LIST,
        write_out=False,
    )
)
path.add_module(AssertTrackSourcesPreserved())
basf2.process(path=path, max_event=1)
