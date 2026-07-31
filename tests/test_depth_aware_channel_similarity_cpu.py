from hypertagging.preprocessing.channels import ChannelSimilarityWeights, structured_channel_similarity


def test_depth_pid_and_intermediate_weights_change_similarity():
    left = {
        "pid_counts": [{"token": 8, "count": 1}],
        "depth_pid_counts": [{"depth": 1, "token": 8, "count": 1}],
        "selected_intermediate_counts": [{"token": 8, "count": 1}],
        "branch_multiplicities": [2],
    }
    right = {
        **left,
        "depth_pid_counts": [{"depth": 3, "token": 8, "count": 1}],
        "selected_intermediate_counts": [],
    }
    pid_only = structured_channel_similarity(
        left, right, weights=ChannelSimilarityWeights(
            w_pid=1, w_depth_pid=0, w_multiplicity=0, w_intermediate=0,
        )
    )
    depth_aware = structured_channel_similarity(left, right)
    assert pid_only == 1.0
    assert depth_aware < pid_only

