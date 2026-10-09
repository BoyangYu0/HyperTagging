import copy
import torch
from scripts.diagnose_phase76_membership import stage_errors, detached_trace
from hypertagging.models.direct_membership import direct_membership_loss
from hypertagging.models.assembly_development import AssemblyMembershipDecoder


def test_matching_unassigned_and_source_error_denominators():
    stage = {
        "probabilities": [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.9, 0.05, 0.05]],
        "presence": [0.9, 0.9],
    }
    c, t, g = stage_errors(stage, [1, 2, 0])
    assert c["source_classification_errors"] == 0 and c["raw_exact"] == 2
    assert c["valid_nodes"] == 3 and c["predicted_unassigned"] == 1
    shifted = copy.deepcopy(stage)
    shifted["probabilities"][0] = [0.8, 0.1, 0.1]
    c, t, g = stage_errors(shifted, [1, 2, 0])
    assert c["source_classification_errors"] == 1 and c["missing_to_unassigned"] == 1


def test_presence_is_separate_from_raw_exact_and_unknown_is_unavailable():
    stage = {
        "probabilities": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "presence": [0.1, 0.1],
    }
    c, _, _ = stage_errors(stage, [1, 2])
    assert c["raw_exact"] == 2 and c["exact_hidden_by_presence"] == 2
    c, t, _ = stage_errors(stage, [-1, -1])
    assert c["valid_nodes"] == 0 and not t


def test_loss_permutation_and_masked_unknown_gradients():
    torch.manual_seed(9)
    logits = torch.randn(2, 4, 3, requires_grad=True)
    objects = torch.randn(2, 2, requires_grad=True)
    targets = torch.tensor([[1, 2, 0, -1], [0, 0, 0, 0]])
    mask = torch.ones(2, 4, dtype=torch.bool)
    a = direct_membership_loss(logits, objects, targets, mask)
    b = direct_membership_loss(
        logits, objects, torch.where(targets > 0, 3 - targets, targets), mask
    )
    assert torch.equal(a, b)
    a.backward()
    assert logits.grad[0, 3].abs().sum() == 0
    assert logits.grad[1].abs().sum() > 0 and objects.grad.abs().sum() > 0


def test_trace_is_detached_output_only_and_does_not_change_forward():
    torch.manual_seed(4)
    m = AssemblyMembershipDecoder().eval()
    h = torch.randn(5, 128)
    sources = torch.eye(5, dtype=torch.bool)
    out = m(h, sources)
    before = out["logits"].detach().clone()
    trace = detached_trace(out)
    assert torch.equal(before, m(h, sources)["logits"])
    assert set(trace) == {"stages", "states"} and isinstance(
        trace["stages"]["proposal"]["probabilities"][0][0], float
    )
    assert not any(k in trace for k in ("targets", "supervision", "category"))


def test_trial_identity_survives_opposite_slot_permutations():
    a = {
        "probabilities": [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "presence": [1.0, 1.0],
    }
    b = copy.deepcopy(a)
    b["probabilities"] = [list(reversed(p[1:])) for p in a["probabilities"]]
    b["probabilities"] = [[0.0] + p for p in b["probabilities"]]
    _, at, _ = stage_errors(a, [1, 1, 2])
    _, bt, _ = stage_errors(b, [1, 1, 2])
    assert [(t["target_id"], t["size"]) for t in at] == [(1, 2), (2, 1)]
    assert at == bt


def test_background_only_merges_are_not_called_B_contamination():
    from scripts.diagnose_phase76_membership import joined_trace

    targets = torch.tensor([1, 1, 2, 2, 0, 0, 0, 0])
    stage = {
        "probabilities": torch.nn.functional.one_hot(targets, 3).float().tolist(),
        "presence": [1.0, 1.0],
    }
    trace = {
        "stages": {"proposal": stage, "refinement": stage},
        "states": [
            {"groups": [[i] for i in range(8)], "pairs": [], "logits": []},
            {"groups": [[0, 4], [1, 5], [2, 3], [6, 7]], "pairs": [], "logits": []},
        ],
    }
    row = {
        "targets": targets,
        "supervision": {"b_groups": [{0, 1}, {2, 3}], "node_sets": [], "parents": []},
    }
    _, m = joined_trace(trace, row)
    assert m["merged_groups"] == 4 and m["contains_unassigned_merges"] == 3
    assert m["mixed_B_and_unassigned_merges"] == 2 and m["pure_unassigned_merges"] == 1
    assert (
        m["B_source_nodes"] == 4 and m["final_B_nodes_in_mixed_unassigned_context"] == 2
    )
