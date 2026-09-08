#!/usr/bin/env python3
"""One-event basf2 smoke test for the additive ONNX full-decay module.

Run this steering file through ``tests/run_basf2_onnx_full_decay_smoke.sh``.
The wrapper creates the deterministic test bundle and activates the pinned
CVMFS light release before invoking basf2.
"""

from __future__ import annotations

import os
from pathlib import Path

import basf2
from ROOT import Belle2
from ROOT.Math import PxPyPzEVector

from hypertagging.basf2_integration.module import HyperTaggingFullDecayModule


INPUT_LIST = "gamma:HyperTaggingToy"
OUTPUT_LIST = "Upsilon(4S):HyperTagging"


class ToyParticleListProducer(basf2.Module):
    """Create four reconstruction-only FSPs; no MC collection is registered."""

    def initialize(self) -> None:
        self.particles = Belle2.ParticleListHelper(INPUT_LIST, False)

    def event(self) -> None:
        self.particles.create()
        momenta = (
            (0.10, 0.00, 0.00, 1.00),
            (-0.10, 0.00, 0.00, 1.10),
            (0.00, 0.15, 0.00, 1.20),
            (0.00, -0.15, 0.00, 1.30),
        )
        for values in momenta:
            self.particles.addParticle(PxPyPzEVector(*values))


def _terminal_particle_indices(particle: object) -> list[int]:
    number_of_daughters = int(particle.getNDaughters())
    if number_of_daughters == 0:
        return [int(particle.getArrayIndex())]
    terminal: list[int] = []
    for index in range(number_of_daughters):
        terminal.extend(_terminal_particle_indices(particle.getDaughter(index)))
    return terminal


class AssertUniqueCompletedRoot(basf2.Module):
    def __init__(self) -> None:
        super().__init__()
        self.events_checked = 0

    def event(self) -> None:
        output = Belle2.PyStoreObj(Belle2.ParticleList.Class(), OUTPUT_LIST)
        if not output.isValid():
            raise RuntimeError(f"missing HyperTagging output list {OUTPUT_LIST}")
        if output.getListSize() != 1:
            raise RuntimeError(
                f"expected exactly one completed root, got {output.getListSize()}"
            )
        root = output.getParticle(0)
        if int(root.getPDGCode()) != 300553:
            raise RuntimeError(
                f"model-directed root has PDG {root.getPDGCode()}, expected 300553"
            )
        terminals = _terminal_particle_indices(root)
        if len(terminals) != 4 or len(set(terminals)) != 4:
            raise RuntimeError(
                "materialized root must own each of the four source particles once; "
                f"got terminal indices {terminals}"
            )
        self.events_checked += 1

    def terminate(self) -> None:
        if self.events_checked != 1:
            raise RuntimeError(
                f"expected to check one event, checked {self.events_checked}"
            )
        basf2.B2INFO(
            "HyperTagging ONNX smoke passed: one unique Upsilon(4S) root "
            "from four truth-free toy FSPs"
        )


manifest = Path(os.environ["HYPERTAGGING_ONNX_BUNDLE"]).resolve()
if not manifest.is_file():
    raise RuntimeError(f"toy ONNX manifest does not exist: {manifest}")

path = basf2.create_path()
path.add_module("EventInfoSetter", evtNumList=[1])
path.add_module(ToyParticleListProducer())
path.add_module(
    HyperTaggingFullDecayModule(
        manifest_path=manifest,
        input_particle_lists=(INPUT_LIST,),
        output_particle_list=OUTPUT_LIST,
        write_out=False,
    )
)
path.add_module(AssertUniqueCompletedRoot())
basf2.process(path)

