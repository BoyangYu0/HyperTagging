import torch
import pytest
from scripts.diagnose_phase78_partition import (
    pair_features,
    pair_targets,
    conditional_counts,
    head_capture,
    split_rows,
)
from hypertagging.models.direct_membership import DirectMembershipHead


def test_pair_features_symmetric_and_labels_permutation_invariant():
    h = torch.randn(7, 128)
    ij = torch.triu_indices(7, 7, 1)
    y = torch.tensor([1, 1, 2, 2, 0, -1, 0])
    assert torch.equal(pair_features(h, ij), pair_features(h, ij.flip(0)))
    a = pair_targets(y, ij)
    b = pair_targets(torch.where(y > 0, 3 - y, y), ij)
    assert all(torch.equal(x, z) for x, z in zip(a, b))
    assert int(a[1].sum()) == 6 and int(a[3].sum()) == 6
    assert not bool((a[1] & a[2]).any())


def test_conditional_is_unordered_and_majority_null_explicit():
    y = torch.tensor([1, 1, 2, 0, -1])
    z = torch.tensor([[0.0, 2, 1]] * 5)
    a = conditional_counts(z, y)
    assert a == conditional_counts(z, torch.where(y > 0, 3 - y, y))
    assert a["conditional_optimal_correct"] == a["majority_slot_null_correct"] == 2
    assert a["B_nodes"] == 3


def test_trace_reproduces_head_and_node_permutation():
    torch.manual_seed(8)
    head = DirectMembershipHead(128, 256).eval()
    x = torch.randn(1, 7, 128)
    m = torch.ones(1, 7, dtype=torch.bool)
    with torch.no_grad():
        h, q, z = head_capture(head, x, m)
        native, _ = head(x, m)
        assert torch.equal(z, native[0])
        order = torch.tensor([4, 2, 0, 6, 3, 1, 5])
        _, q2, z2 = head_capture(head, x[:, order], m)
        torch.testing.assert_close(q, q2)
        torch.testing.assert_close(z[order], z2)


def test_partition_is_identity_only_disjoint_and_rejects_duplicates():
    rows = [{"uid": str(i), "category": "charged"} for i in range(256)]
    a = split_rows(rows)
    assert a == split_rows(list(reversed(rows)))
    assert list(a.values()).count("fit") == 192
    with pytest.raises(ValueError):
        split_rows(rows[:-1])
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        split_rows(rows)


def test_shared_detector_sources_excluded_without_dropping_nodes():
    from scripts.diagnose_phase78_partition import source_pairs

    sources = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool)
    ij, count = source_pairs(sources)
    assert count == 1 and ij.tolist() == [[0, 1], [2, 2]]
    assert sources.shape[0] == 3
