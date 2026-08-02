import json

from hypertagging.data.notebook_fixtures import notebook_fixture_trees
from hypertagging.preprocessing.channels import channel_count_array


def test_channel_representation_json_round_trip_preserves_depth_and_multiplicity():
    tree = notebook_fixture_trees()[0]
    representation = channel_count_array(tree, 1)
    restored = json.loads(json.dumps(representation, sort_keys=True))
    assert restored == representation
    assert restored["depth_pid_counts"]
    assert restored["branch_multiplicities"]
    assert restored["branch_multiplicity_counts"]
    assert sum(
        record["count"] for record in restored["branch_multiplicity_counts"]
    ) == len(restored["branch_multiplicities"])
