import math
from pathlib import Path
import sys
import types

from hypertagging.preprocessing.export_dataset import export_trees, load_processed
from hypertagging.preprocessing.levelize_tree import adjacent_level_samples, assign_levels, nodes_by_level
from hypertagging.preprocessing.basf2_mdst import (
    Basf2PreprocessConfig,
    _DirectMdstCollector,
    run_basf2_preprocessing,
)
from hypertagging.preprocessing.mdst_tree_builder import (
    EventTree,
    FourVector,
    MCRecord,
    RecoRecord,
    TreeNode,
    build_truth_guided_tree,
    copy_shared_daughters,
    recompute_mother_p4_from_daughters,
    validate_tree,
)
from hypertagging.preprocessing.pid_filter import PidFilter, tokenize_pdg
from scripts.preprocess_mdst import _find_repo_root


def _toy_records():
    mc = [
        MCRecord(0, 300553, 0.0, None, FourVector(0, 0, 0, 10.58), "Upsilon(4S)", True),
        MCRecord(1, 521, 1.0, 0, FourVector(0.1, 0, 0, 5.29), "B+", True),
        MCRecord(2, 421, 0.0, 1, FourVector(0.4, 0, 0, 2.0), "D0", True),
        MCRecord(3, 321, 1.0, 2, FourVector(0.2, 0, 0, 0.6), "K+", True),
        MCRecord(4, -211, -1.0, 2, FourVector(0.1, 0.1, 0, 0.4), "pi-", True),
        MCRecord(5, 113, 0.0, 1, FourVector(0, 0, 0, 0.7), "rho0", True),
        MCRecord(6, 22, 0.0, 5, FourVector(0, 0.2, 0, 0.2), "gamma", True),
    ]
    reco = [
        RecoRecord("trk-k", 321, 1.0, FourVector(0.25, 0.0, 0.0, 0.65), 3),
        RecoRecord("trk-pi", -211, -1.0, FourVector(0.10, 0.15, 0.0, 0.35), 4),
        RecoRecord("ecl-gamma", 22, 0.0, FourVector(0.0, 0.20, 0.0, 0.20), 6),
    ]
    return mc, reco


def test_truth_guided_tree_prunes_pid_and_recomputes_mother_p4():
    pid_filter = PidFilter()
    mc, reco = _toy_records()

    tree = build_truth_guided_tree(event_id=12, mc_records=mc, reco_records=reco, pid_filter=pid_filter)
    assign_levels(tree)
    stats = validate_tree(tree)

    pdgs = {node.pdg for node in tree.nodes.values()}
    assert 113 not in pdgs
    assert 22 in pdgs
    assert stats["max_abs_p4_diff"] == 0.0
    d0 = next(node for node in tree.nodes.values() if node.pdg == 421)
    daughters = [tree.nodes[child_id] for child_id in d0.daughter_ids]
    assert math.isclose(d0.p4.energy, sum(child.p4.energy for child in daughters))
    assert not math.isclose(d0.p4.energy, 2.0)
    assert pid_filter.summary.dropped["name_not_allowed"] == 1


def test_levelisation_and_adjacent_level_links():
    mc, reco = _toy_records()
    tree = build_truth_guided_tree(event_id=12, mc_records=mc, reco_records=reco)
    assign_levels(tree)

    grouped = nodes_by_level(tree)
    assert 0 in grouped
    assert max(grouped) >= 2
    samples = adjacent_level_samples(tree)
    assert samples
    assert all(len(sample.links) == len(sample.input_node_ids) for sample in samples)


def test_copy_shared_daughters_clones_subtree():
    tree = EventTree(event_id=1)
    tree.add_node(TreeNode(0, 521, 1, FourVector(0, 0, 0, 0), daughter_ids=[2]))
    tree.add_node(TreeNode(1, -521, -1, FourVector(0, 0, 0, 0), daughter_ids=[2]))
    tree.add_node(TreeNode(2, 211, 1, FourVector(1, 0, 0, 1.2), parent_id=0))
    tree.root_ids = [0, 1]

    copied = copy_shared_daughters(tree)
    recompute_mother_p4_from_daughters(tree)
    assign_levels(tree)
    validate_tree(tree)

    assert copied == 1
    copied_nodes = [node for node in tree.nodes.values() if node.copied_from == 2]
    assert len(copied_nodes) == 1
    assert copied_nodes[0].p4.energy == tree.nodes[2].p4.energy


def test_export_contains_canonical_and_legacy_views(tmp_path):
    mc, reco = _toy_records()
    tree = build_truth_guided_tree(event_id=12, mc_records=mc, reco_records=reco)
    assign_levels(tree)
    output = export_trees([tree], tmp_path / "processed.parquet", summary={"events": 1})

    payload = load_processed(output).to_list()[0]
    assert payload["schema_version"] == "direct-mdst-tree-v1"
    assert payload["events"][0]["event_id"] == 12
    assert payload["events"][0]["event_uid"] == "12"
    assert payload["legacy_levels"]
    first_level = payload["legacy_levels"][0]
    assert "feature" in first_level
    assert "motherIndex" in first_level
    assert first_level["feature"][0][0] in {tokenize_pdg(node["pdg"]) for node in payload["events"][0]["nodes"]}


def test_preprocess_script_finds_repo_without_dunder_file():
    repo_root = Path(__file__).resolve().parents[1]

    assert _find_repo_root(None, cwd=repo_root) == repo_root


def test_generic_mdst_config_does_not_request_udst_particle_array(tmp_path):
    config = Basf2PreprocessConfig(("input.root",), tmp_path / "output.parquet")

    assert config.particle_arrays == ()
    assert config.include_tracks
    assert config.include_ecl_clusters


def test_basf2_runner_passes_multiple_files_to_input_mdst_list(monkeypatch, tmp_path):
    calls = []

    class StopAfterInput(Exception):
        pass

    def input_mdst_list(**kwargs):
        calls.append(kwargs)
        raise StopAfterInput

    fake_basf2 = types.SimpleNamespace(create_path=lambda: object())
    fake_modular_analysis = types.SimpleNamespace(inputMdstList=input_mdst_list)
    monkeypatch.setitem(sys.modules, "basf2", fake_basf2)
    monkeypatch.setitem(sys.modules, "modularAnalysis", fake_modular_analysis)
    config = Basf2PreprocessConfig(("first.root", "second.root"), tmp_path / "output.parquet")

    try:
        run_basf2_preprocessing(config)
    except StopAfterInput:
        pass
    else:
        raise AssertionError("inputMdstList test sentinel was not raised")

    assert calls == [
        {
            "filelist": ["first.root", "second.root"],
            "environmentType": "default",
            "path": calls[0]["path"],
        }
    ]


def test_basf2_runner_passes_one_entry_sequence_per_file(monkeypatch, tmp_path):
    calls = []

    class StopAfterInput(Exception):
        pass

    def input_mdst_list(**kwargs):
        calls.append(kwargs)
        raise StopAfterInput

    monkeypatch.setitem(sys.modules, "basf2", types.SimpleNamespace(create_path=lambda: object()))
    monkeypatch.setitem(
        sys.modules,
        "modularAnalysis",
        types.SimpleNamespace(inputMdstList=input_mdst_list),
    )
    config = Basf2PreprocessConfig(
        ("first.root", "second.root"),
        tmp_path / "output.parquet",
        entry_sequences=("0:9", "20:29"),
    )

    try:
        run_basf2_preprocessing(config)
    except StopAfterInput:
        pass
    else:
        raise AssertionError("inputMdstList test sentinel was not raised")

    assert calls[0]["entrySequences"] == ["0:9", "20:29"]


def test_track_collector_uses_charged_stable_hypothesis(tmp_path):
    requested_pdgs = []

    class Momentum:
        def X(self):
            return 0.3

        def Y(self):
            return -0.2

        def Z(self):
            return 0.4

    class Fit:
        def getMomentum(self):
            return Momentum()

        def getChargeSign(self):
            return -1

    class MC:
        def getPDG(self):
            return -321

        def getArrayIndex(self):
            return 17

    class Track:
        def getArrayIndex(self):
            return 4

        def getRelatedTo(self, _name):
            return MC()

        def getTrackFitResultWithClosestMass(self, hypothesis):
            assert hypothesis == ("charged", 321)
            return Fit()

    collector = _DirectMdstCollector(Basf2PreprocessConfig(("input.root",), tmp_path / "output.parquet"))
    collector.tracks = [Track()]
    collector._charged_stable = lambda pdg: requested_pdgs.append(pdg) or ("charged", pdg)

    records = collector._collect_tracks()

    assert requested_pdgs == [321]
    assert len(records) == 1
    assert records[0].pdg == -321
    assert records[0].mc_id == 17


def test_ecl_collector_uses_cluster_utils_and_skips_track_matches(tmp_path):
    photon_hypothesis = object()

    class Momentum:
        def Px(self):
            return 0.1

        def Py(self):
            return 0.2

        def Pz(self):
            return 0.3

        def E(self):
            return 0.5

    class ClusterUtils:
        def Get4MomentumFromCluster(self, cluster, hypothesis):
            assert cluster.getArrayIndex() == 8
            assert hypothesis is photon_hypothesis
            return Momentum()

    class MC:
        def getArrayIndex(self):
            return 23

    class Cluster:
        def __init__(self, *, matched_to_track):
            self.matched_to_track = matched_to_track

        def getArrayIndex(self):
            return 8 if not self.matched_to_track else 9

        def isTrack(self):
            return self.matched_to_track

        def hasHypothesis(self, hypothesis):
            return hypothesis is photon_hypothesis

        def getRelatedTo(self, _name):
            return MC()

    collector = _DirectMdstCollector(Basf2PreprocessConfig(("input.root",), tmp_path / "output.parquet"))
    collector.ecl_clusters = [Cluster(matched_to_track=False), Cluster(matched_to_track=True)]
    collector._cluster_utils = ClusterUtils()
    collector._photon_hypothesis = photon_hypothesis

    records = collector._collect_ecl_clusters()

    assert len(records) == 1
    assert records[0].pdg == 22
    assert records[0].mc_id == 23
    assert records[0].p4.as_tuple() == (0.1, 0.2, 0.3, 0.5)
    assert collector.collection_stats["ecl_track_matched_skipped"] == 1
