#!/usr/bin/env python3
"""One-event reconstructed-mDST smoke for detector-source preservation."""

from __future__ import annotations

import os
from pathlib import Path

import basf2
import modularAnalysis as ma
from ROOT import Belle2

from hypertagging.basf2_integration.module import HyperTaggingFullDecayModule


INPUT_LIST = "gamma:HyperTaggingMdstSmoke"
OUTPUT_LIST = "Upsilon(4S):HyperTaggingMdstSmoke"


def _terminals(particle: object) -> list[object]:
    if int(particle.getNDaughters()) == 0:
        return [particle]
    output: list[object] = []
    for index in range(int(particle.getNDaughters())):
        output.extend(_terminals(particle.getDaughter(index)))
    return output


class AssertMdstSourcesPreserved(basf2.Module):
    def __init__(self) -> None:
        super().__init__()
        self.events_checked = 0

    def event(self) -> None:
        output = Belle2.PyStoreObj(Belle2.ParticleList.Class(), OUTPUT_LIST)
        if not output.isValid() or int(output.getListSize()) != 1:
            raise RuntimeError("real-mDST smoke expected one completed output root")
        root = output.getParticle(0)
        terminals = _terminals(root)
        if len(terminals) != 4:
            raise RuntimeError(f"expected four terminal photons, got {len(terminals)}")
        source_kind = int(Belle2.Particle.c_ECLCluster)
        actual_kinds = [int(particle.getParticleSource()) for particle in terminals]
        if actual_kinds != [source_kind] * 4:
            raise RuntimeError(
                "materialized terminals lost their ECL source type: "
                f"{actual_kinds}"
            )
        mdst_indices = [int(particle.getMdstArrayIndex()) for particle in terminals]
        if len(set(mdst_indices)) != 4:
            raise RuntimeError(
                f"materialized terminals reused ECL sources: {mdst_indices}"
            )
        self.events_checked += 1

    def terminate(self) -> None:
        if self.events_checked != 1:
            raise RuntimeError(
                f"expected one reconstructed-mDST event, got {self.events_checked}"
            )
        basf2.B2INFO(
            "HyperTagging real-mDST smoke passed: four unique ECL sources "
            "survived ONNX hierarchy materialization"
        )


manifest = Path(os.environ["HYPERTAGGING_ONNX_BUNDLE"]).resolve()
mdst = Path(os.environ["HYPERTAGGING_SMOKE_MDST"]).resolve()
if not manifest.is_file() or not mdst.is_file():
    raise RuntimeError("real-mDST smoke inputs are unavailable")

path = basf2.create_path()
ma.inputMdst(str(mdst), path=path)
ma.fillParticleList(INPUT_LIST, "", writeOut=False, path=path)
ma.rankByHighest(INPUT_LIST, "E", numBest=4, path=path)
path.add_module(
    HyperTaggingFullDecayModule(
        manifest_path=manifest,
        input_particle_lists=(INPUT_LIST,),
        output_particle_list=OUTPUT_LIST,
        write_out=False,
    )
)
path.add_module(AssertMdstSourcesPreserved())
basf2.process(path=path, max_event=1)
