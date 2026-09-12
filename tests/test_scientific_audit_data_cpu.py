"""Scientific regressions for the September 2026 detector/tree audit."""

from copy import deepcopy

import pytest
import torch

from hypertagging.data.capacity import dataset_capacity_statistics
from hypertagging.data.heterogeneous import heterogeneous_event_from_record
from hypertagging.preprocessing.mdst_tree_builder import (
    FourVector, MCRecord, RecoRecord, build_truth_guided_tree,
)
from hypertagging.preprocessing.schema_v3 import (
    V3_COMMON_FEATURE_NAMES, V3_COMPOSITE_FEATURE_NAMES,
)
from hypertagging.preprocessing.schema_v4 import export_trees_v4, iter_event_records_v4


def _partial_tree():
    # Both neutral truth mothers retain charge +1 after a negative track is
    # lost.  The upper mother must propagate that reconstructed charge too.
    mc = [
        MCRecord(0, 511, 0., None),
        MCRecord(1, 421, 0., 0),
        MCRecord(2, 211, 1., 1),
        MCRecord(3, 22, 0., 1),
        MCRecord(4, -211, -1., 1),
        MCRecord(5, 22, 0., 0),
    ]
    reco = [
        RecoRecord('Track:0', 211, 1., FourVector(.2, 0., 0., .3), mc_id=2, node_kind='track'),
        RecoRecord('ECLCluster:0', 22, 0., FourVector(0., .1, 0., .1), mc_id=3, node_kind='ecl_cluster'),
        RecoRecord('ECLCluster:1', 22, 0., FourVector(0., 0., .1, .1), mc_id=5, node_kind='ecl_cluster'),
    ]
    return build_truth_guided_tree(event_id=1, mc_records=mc, reco_records=reco)


def test_producer_partial_mother_charge_is_recursive_daughter_sum(tmp_path):
    tree = _partial_tree()
    for node in tree.nodes.values():
        if node.daughter_ids:
            assert node.charge == node.reco_charge == 1.
            assert node.truth_charge == 0.
            assert node.p4.as_tuple() == FourVector.sum(
                tree.nodes[child].p4 for child in node.daughter_ids
            ).as_tuple()
    path = export_trees_v4([tree], tmp_path / 'partial.parquet')
    record = next(iter_event_records_v4(path))
    for node in record['nodes']:
        if node['daughter_ids']:
            assert node['common_features']['charge'] == 1.
            assert node['composite_features']['summed_charge'] == 1.
            assert node['truth_charge'] == 0.


def test_existing_shard_charge_is_rebuilt_without_mutating_truth_or_record(tmp_path):
    path = export_trees_v4([_partial_tree()], tmp_path / 'old.parquet')
    record = next(iter_event_records_v4(path))
    for node in record['nodes']:
        if node['daughter_ids']:
            node['charge'] = node['reco_charge'] = 0.
            node['common_features']['charge'] = 0.
            node['composite_features']['summed_charge'] = -7.
    before = deepcopy(record)
    event = heterogeneous_event_from_record(record)
    mothers = event.daughter_adjacency.any(dim=-1)
    assert torch.equal(event.charge[mothers], torch.ones(int(mothers.sum())))
    assert torch.equal(event.common_features[mothers, V3_COMMON_FEATURE_NAMES.index('charge')], event.charge[mothers])
    assert torch.equal(event.composite_features[mothers, V3_COMPOSITE_FEATURE_NAMES.index('summed_charge')], event.charge[mothers])
    assert record == before
    for mother in mothers.nonzero(as_tuple=False).flatten():
        assert event.charge[mother] == event.charge[event.daughter_adjacency[mother]].sum()


def test_diagnostic_capacity_counts_ineligible_retained_mothers(tmp_path):
    path = export_trees_v4([_partial_tree()], tmp_path / 'capacity.parquet')
    record = next(iter_event_records_v4(path))
    for node in record['nodes']:
        node['valid_reconstruction_target'] = False
    event = heterogeneous_event_from_record(record)
    diagnostic = dataset_capacity_statistics([event], target_policy='diagnostic_all')
    primary = dataset_capacity_statistics([event], target_policy='complete_only')
    assert diagnostic.maximum_mothers_per_level == {1: 1, 2: 1}
    assert diagnostic.daughter_cardinality_counts == {2: 2}
    assert primary.maximum_mothers_per_level == {1: 0, 2: 0}
    with pytest.raises(ValueError, match='unknown reconstruction target policy'):
        dataset_capacity_statistics([], target_policy='misspelled')


def test_verification_cli_streams_native_v4_and_checks_charge(tmp_path, capsys):
    from scripts import verify_preprocessing

    path = export_trees_v4([_partial_tree(), _partial_tree()], tmp_path / 'verify.parquet')
    assert verify_preprocessing.main([
        '--input', str(path), '--all-events', '--check-tree', '--check-p4', '--check-charge', '--check-pid'
    ]) == 0
    assert 'validated events=2' in capsys.readouterr().out
    record = next(iter_event_records_v4(path))
    next(node for node in record['nodes'] if node['daughter_ids'])['charge'] = -3.
    with pytest.raises(ValueError, match='charge mismatch'):
        verify_preprocessing.check_charge(record, tolerance=1e-8)


def test_verification_cli_keeps_legacy_payload_support(tmp_path, capsys):
    from scripts import verify_preprocessing
    from hypertagging.preprocessing.export_dataset import export_trees

    path = export_trees([_partial_tree()], tmp_path / 'legacy.parquet')
    assert verify_preprocessing.main([
        '--input', str(path), '--all-events', '--check-tree', '--check-p4', '--check-charge'
    ]) == 0
    assert 'validated events=1' in capsys.readouterr().out


def test_sidecar_empirical_type_prior_uses_training_sources_only(tmp_path):
    from hypertagging.data.dataset_index import (
        build_dataset_index, build_dataset_index_from_sidecars, load_dataset_index,
    )
    from hypertagging.preprocessing.schema_v4 import ParquetEventWriter

    seed = export_trees_v4([_partial_tree()], tmp_path / 'seed.parquet')
    record = next(iter_event_records_v4(seed))
    paths = []
    for split, token in [('train', 15), ('validation', 16)]:
        event = deepcopy(record)
        event['event_uid'] = split
        event['source_file'] = split + '.root'
        for node in event['nodes']:
            if node['daughter_ids']:
                node['pid_target_token'] = token
        path = tmp_path / (split + '.parquet')
        with ParquetEventWriter(path, metadata={'source_file': event['source_file']}) as writer:
            writer.write_event(event)
        paths.append(path)
    overrides = {'train.root': 'train', 'validation.root': 'validation'}
    full = load_dataset_index(build_dataset_index(
        paths, tmp_path / 'full.json', source_split_overrides=overrides,
    ))
    sidecar = load_dataset_index(build_dataset_index_from_sidecars(
        paths, tmp_path / 'sidecar.json', source_split_overrides=overrides,
    ))
    assert full['allowed_types_by_level'] == sidecar['allowed_types_by_level'] == {'1': [15], '2': [15]}
