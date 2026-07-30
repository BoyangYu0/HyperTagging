import torch

from hypertagging.reconstruction.level_rollout import CompositeProposal, resolve_exclusive_proposals


def test_higher_level_shared_original_leaf_conflicts_but_disjoint_does_not():
    # Nodes 3 and 4 are composites: both recursively contain source column 0.
    recursive = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=torch.bool,
    )
    proposals = [
        CompositeProposal(0, 1, (3,), 0.9, 0.9),
        CompositeProposal(1, 1, (4,), 0.8, 0.8),
        CompositeProposal(2, 1, (2,), 0.7, 0.7),
    ]
    accepted = resolve_exclusive_proposals(
        proposals,
        recursive_leaf_source_mask=recursive,
    )
    assert [proposal.query_id for proposal in accepted] == [0, 2]
