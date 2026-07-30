import math

from hypertagging.preprocessing.basf2_mdst import Basf2PreprocessConfig, _DirectMdstCollector


class _Momentum:
    def X(self): return 0.3
    def Y(self): return -0.2
    def Z(self): return 0.4


class _Fit:
    def getMomentum(self): return _Momentum()
    def getChargeSign(self): return -1
    def getPValue(self): return 0.8


class _MC:
    def __init__(self, pdg): self.pdg = pdg
    def getPDG(self): return self.pdg
    def getArrayIndex(self): return 17
    def getCharge(self): return -1


class _Track:
    def __init__(self, truth_pdg): self.mc = None if truth_pdg is None else _MC(truth_pdg)
    def getArrayIndex(self): return 4
    def getRelatedTo(self, name): return self.mc if name in {"MCParticles", "MCParticle"} else None
    def getTrackFitResultWithBestPValue(self): return _Fit()


def _record(tmp_path, truth_pdg):
    collector = _DirectMdstCollector(
        Basf2PreprocessConfig(("input.root",), tmp_path / f"{truth_pdg}.parquet")
    )
    collector.tracks = [_Track(truth_pdg)]
    collector._charged_stable = lambda pdg: pdg
    return collector._collect_tracks()[0]


def test_changing_truth_pid_does_not_change_reconstructed_track_inputs(tmp_path):
    kaon = _record(tmp_path, -321)
    pion = _record(tmp_path, -211)
    for field in (
        "input_pid_token",
        "reco_charge",
        "p4",
        "track_features",
        "track_energy_hypotheses",
        "energy_source",
        "leaf_kinematics_mode",
    ):
        assert getattr(kaon, field) == getattr(pion, field)
    assert kaon.truth_pid_token != pion.truth_pid_token
    assert math.isclose(kaon.p4.energy, kaon.track_energy_hypotheses["pion"])


def test_mc_absent_raw_track_has_same_input_contract(tmp_path):
    present = _record(tmp_path, -321)
    absent = _record(tmp_path, None)
    assert present.p4 == absent.p4
    assert present.input_pid_token == absent.input_pid_token == 0
    assert present.reco_charge == absent.reco_charge == -1
    assert absent.truth_pdg is None and absent.truth_pid_token is None
