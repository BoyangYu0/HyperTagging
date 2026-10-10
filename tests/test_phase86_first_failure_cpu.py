"""Causal audit regressions: delayed recovery, constraints, joint support failures."""

from dataclasses import asdict

from hypertagging.preprocessing.pid_filter import PDG_TOKENS
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from hypertagging.reconstruction.level_rollout import RolloutConfig
from scripts.phase86_first_failure import legal_group, lifecycle_target, query_support


def fixture():
    pid = PDG_TOKENS.index(111)
    policy = ReconstructionConstraintPolicy()
    state = dict(
        node_mask=[True, True],
        node_ids=[0, 1],
        level_ids=[0, 0],
        pid_labels=[0, 0],
        parent_ids=[-1, -1],
        node_kind_ids=[0, 0],
        recursive_leaf_source_mask=[[True, False], [False, True]],
        source_conflict_matrix=[[False, False], [False, False]],
        p4=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        charge=[0.0, 0.0],
        daughter_adjacency=[[False, False], [False, False]],
    )
    # Use accepted node kinds independent of representation enum order.
    state["node_kind_ids"] = [policy.valid_leaf_node_kinds[0]] * 2
    ty = [0.0] * len(PDG_TOKENS)
    ty[pid] = 1.0
    d = dict(
        state=state,
        hard_decode_context_mask=[True, True],
        context_mask=[True, True],
        pointer_validity_mask=[True, True],
        forest_pointer_validity_mask=[True, True],
        raw_proposals=[],
    )
    step = dict(
        height=1,
        decode_trace=d,
        probabilities=dict(
            object=[0.9],
            type=[ty],
            pointer=[[0.8, 0.8]],
            cardinality=[[0.0, 0.0, 1.0]],
            confidence=[0.9],
        ),
        accepted=[],
    )
    config = asdict(RolloutConfig(constraint_policy=policy))
    return pid, policy, step, config


def test_current_charge_constraint_is_separate_from_source_cover():
    pid, policy, step, _ = fixture()
    step["decode_trace"]["state"]["charge"] = [1.0, 1.0]
    assert "mother_charge_incompatible" in legal_group(
        step["decode_trace"]["state"], [0, 1], pid, policy, 1
    )


def test_query_retains_multiple_failures_without_changing_proposals():
    _, policy, step, config = fixture()
    step["probabilities"]["object"] = [0.1]
    step["probabilities"]["pointer"] = [[0.1, 0.9]]
    step["probabilities"]["cardinality"] = [[0.0, 1.0, 0.0]]
    rows = query_support(step, [0, 1], policy, config)
    assert {
        "object_rejection",
        "cardinality_support",
        "pointer_threshold_support",
        "pointer_ranking_or_source_selection",
    } <= set(rows[0]["failures"])
    assert step["decode_trace"]["raw_proposals"] == []


def test_delayed_round_proposal_counts_as_all_round_recovery():
    pid, policy, step, config = fixture()
    step["height"] = 3
    p = dict(daughter_positions=[0, 1], mother_type=pid, query_id=0)
    step["decode_trace"]["raw_proposals"] = [p]
    step["accepted"] = [p]
    rounds, loss = lifecycle_target(
        [frozenset([0]), frozenset([1])], pid, dict(config=config, steps=[step]), policy
    )
    assert (
        rounds[0]["accepted_correct_type"] == 1
        and rounds[0]["legal_correct_type_group"]
    )
    assert loss is None


def test_one_root_split_is_only_an_oracle_opportunity():
    pid, policy, step, config = fixture()
    s = step["decode_trace"]["state"]
    s.update(
        node_mask=[True] * 3,
        node_ids=[0, 1, 2],
        level_ids=[0, 0, 1],
        pid_labels=[0, 0, pid],
        parent_ids=[2, 2, -1],
        node_kind_ids=[policy.valid_leaf_node_kinds[0]] * 2
        + [policy.valid_composite_node_kinds[0]],
        recursive_leaf_source_mask=[[True, False], [False, True], [True, True]],
        source_conflict_matrix=[[False] * 3 for _ in range(3)],
        p4=[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]],
        charge=[0.0] * 3,
        daughter_adjacency=[[False] * 3, [False] * 3, [True, True, False]],
    )
    step["height"] = 2
    step["decode_trace"]["context_mask"] = [True] * 3
    step["decode_trace"]["forest_pointer_validity_mask"] = [False, False, True]
    rounds, loss = lifecycle_target(
        [frozenset([0]), frozenset([1])], pid, dict(config=config, steps=[step]), policy
    )
    assert loss is not None and rounds[0]["single_root_split_oracle"]
    assert rounds[0]["accepted_correct_type"] == 0
    assert s["parent_ids"] == [2, 2, -1]


def joined_fixture():
    """Four disjoint retained trees: legal, charge-incompatible, unary, partial."""
    from copy import deepcopy
    from types import SimpleNamespace
    import torch

    pid, policy, step, config = fixture()
    groups = [[0, 1], [2, 3], [4], [5, 6]]
    leaves = 7
    count = leaves + len(groups)
    state = step['decode_trace']['state']
    state.update(
        node_mask=[True] * leaves,
        node_ids=list(range(leaves)),
        level_ids=[0] * leaves,
        pid_labels=[0] * leaves,
        parent_ids=[-1] * leaves,
        node_kind_ids=[policy.valid_leaf_node_kinds[0]] * leaves,
        recursive_leaf_source_mask=torch.eye(leaves, dtype=torch.bool).tolist(),
        source_conflict_matrix=torch.zeros(leaves, leaves, dtype=torch.bool).tolist(),
        p4=[[0., 0., 0., 1.]] * leaves,
        charge=[0.] * leaves,
        daughter_adjacency=torch.zeros(leaves, leaves, dtype=torch.bool).tolist(),
    )
    for key in ('context_mask', 'hard_decode_context_mask', 'pointer_validity_mask', 'forest_pointer_validity_mask'):
        step['decode_trace'][key] = [True] * leaves
    step['probabilities']['object'] = [0.1]
    step['probabilities']['pointer'] = [[0.8] * leaves]
    step.update(state_before=deepcopy(state), eligible_positions=list(range(leaves)), proposals=[], appended_node_ids=[])
    sources = torch.zeros(1, count, leaves, dtype=torch.bool)
    sources[0, :leaves] = torch.eye(leaves, dtype=torch.bool)
    adjacency = torch.zeros(1, count, count, dtype=torch.bool)
    for i, daughters in enumerate(groups):
        sources[0, leaves + i, daughters] = True
        adjacency[0, leaves + i, daughters] = True
    labels = torch.tensor([[0] * leaves + [pid, PDG_TOKENS.index(521), pid, pid]])
    truth = dict(
        node_mask=torch.ones(1, count, dtype=torch.bool),
        level_ids=torch.tensor([[0] * leaves + [1] * len(groups)]),
        pid_labels=labels.clone(), pid_target_labels=labels,
        daughter_adjacency=adjacency, recursive_leaf_source_mask=sources,
        valid_reconstruction_target=torch.tensor([[False] * leaves + [True] * len(groups)]),
        recursive_reconstructable_complete=torch.tensor([[True] * (count - 1) + [False]]),
    )
    projection = SimpleNamespace(
        audit=SimpleNamespace(original_fsp_positions=(tuple(range(leaves)),)),
        batch={'recursive_leaf_source_mask': sources[:, :leaves]},
    )
    trace = dict(config=config, steps=[step], final_state=deepcopy(state),
                 truth_used_for_generation=False, mode='predicted')
    return truth, projection, trace


def test_incompatible_eligible_targets_stay_in_primary_denominator():
    from scripts.phase86_first_failure import evaluate_first_failure

    truth, projection, trace = joined_fixture()
    result = evaluate_first_failure(truth, projection, trace,
                                    target_policy='complete_only', minimum_daughters=2)
    # The same accounting used for the historical 455 must retain every eligible
    # target, including target-population incompatibilities, without renormalizing.
    assert result['policy_eligible_targets'] == 3
    assert sum(result['first_failure_counts'].values()) == 3
    assert len(result['targets']) == 4
    assert [row['first_failure'] for row in result['targets']] == [
        'object_rejection', 'available_group_constraint_incompatible',
        'target_cardinality_incompatible', 'outside_policy',
    ]
    assert sum(row.get('legal_group_any_round', False) for row in result['targets']) == 1


def test_posthoc_target_mutation_changes_labels_but_cannot_mutate_trace():
    from copy import deepcopy
    from scripts.phase86_first_failure import evaluate_first_failure

    truth, projection, trace = joined_fixture()
    before = deepcopy(trace)
    baseline = evaluate_first_failure(truth, projection, trace,
                                      target_policy='complete_only', minimum_daughters=2)
    changed = {key: value.clone() for key, value in truth.items()}
    changed['pid_target_labels'][0, 7] = PDG_TOKENS.index(521)
    altered = evaluate_first_failure(changed, projection, trace,
                                     target_policy='complete_only', minimum_daughters=2)
    assert baseline['targets'][0]['first_failure'] != altered['targets'][0]['first_failure']
    assert trace == before
    assert baseline['policy_eligible_targets'] == altered['policy_eligible_targets'] == 3


def test_terminal_wrong_commitment_records_creation_round_without_later_step():
    from copy import deepcopy

    pid, policy, step, config = fixture()
    step['appended_node_ids'] = [2]
    final = deepcopy(step['decode_trace']['state'])
    final.update(node_mask=[True] * 3, node_ids=[0, 1, 2],
                 parent_ids=[2, 2, -1],
                 recursive_leaf_source_mask=[[True, False], [False, True], [True, True]])
    # Wrong type: a terminal merge consumes the required daughter roots but did
    # not produce the target; its creation round must survive the posthoc join.
    wrong = dict(query_id=0, mother_type=PDG_TOKENS.index(511), daughter_positions=[0, 1])
    step['accepted'] = [wrong]
    step['decode_trace']['raw_proposals'] = [wrong]
    _, loss = lifecycle_target([frozenset([0]), frozenset([1])], pid,
                               dict(config=config, steps=[step], final_state=final), policy)
    assert loss['observation_phase'] == 'terminal'
    assert loss['consuming_commitments'] == [dict(node_id=2, commitment_round=1)]
    assert loss['had_prior_legal_group']


def test_successful_terminal_assembly_is_not_a_delay_repair_opportunity():
    from copy import deepcopy
    from scripts.phase86_first_failure import evaluate_first_failure

    truth, projection, trace = joined_fixture()
    step = trace['steps'][0]
    pid = int(truth['pid_target_labels'][0, 7])
    proposal = dict(query_id=0, mother_type=pid, daughter_positions=[0, 1],
                    object_score=0.9, confidence=0.9)
    step['probabilities']['object'] = [0.9]
    step['accepted'] = [proposal]
    step['proposals'] = [proposal]
    step['decode_trace']['raw_proposals'] = [proposal]
    step['appended_node_ids'] = [7]
    final = deepcopy(trace['final_state'])
    final['node_mask'].append(True)
    final['node_ids'].append(7)
    final['parent_ids'][:2] = [7, 7]
    final['parent_ids'].append(-1)
    final['recursive_leaf_source_mask'].append([True, True] + [False] * 5)
    trace['final_state'] = final
    result = evaluate_first_failure(truth, projection, trace,
                                    target_policy='complete_only', minimum_daughters=2)
    row = result['targets'][0]
    assert row['first_failure'] == 'accepted_correct_all_rounds'
    assert row['delay_oracle_opportunity'] is False
    assert row['single_split_oracle_opportunity'] is False


def test_dependency_join_counts_terminally_formed_child_as_formed():
    from copy import deepcopy
    import torch
    from scripts.phase86_first_failure import evaluate_first_failure

    truth, projection, trace = joined_fixture()
    pid = int(truth['pid_target_labels'][0, 7])
    count = truth['node_mask'].shape[1]
    adjacency = torch.zeros(1, count + 1, count + 1, dtype=torch.bool)
    adjacency[:, :count, :count] = truth['daughter_adjacency']
    adjacency[0, count, [7, 8]] = True
    truth['daughter_adjacency'] = adjacency
    truth['recursive_leaf_source_mask'] = torch.cat([
        truth['recursive_leaf_source_mask'],
        truth['recursive_leaf_source_mask'][:, [7]] | truth['recursive_leaf_source_mask'][:, [8]],
    ], dim=1)
    for key, value in [('node_mask', True), ('level_ids', 2), ('pid_labels', pid),
                       ('pid_target_labels', pid), ('valid_reconstruction_target', True),
                       ('recursive_reconstructable_complete', True)]:
        truth[key] = torch.cat([truth[key], truth[key].new_tensor([[value]])], dim=1)
    step = trace['steps'][0]
    proposal = dict(query_id=0, mother_type=pid, daughter_positions=[0, 1],
                    object_score=0.9, confidence=0.9)
    step.update(accepted=[proposal], proposals=[proposal], appended_node_ids=[7])
    step['decode_trace']['raw_proposals'] = [proposal]
    final = deepcopy(trace['final_state'])
    final['node_mask'].append(True)
    final['node_ids'].append(7)
    final['parent_ids'][:2] = [7, 7]
    final['parent_ids'].append(-1)
    final['recursive_leaf_source_mask'].append([True, True] + [False] * 5)
    trace['final_state'] = final
    result = evaluate_first_failure(truth, projection, trace,
                                    target_policy='complete_only', minimum_daughters=2)
    parent = next(row for row in result['targets'] if row['node_position'] == count)
    missing = parent['first_failure_dependency_trace']['never_formed_dependencies']
    assert [row['node_position'] for row in missing] == [8]
    assert result['policy_eligible_targets'] == 4


def test_head_cardinality_capacity_is_not_legal_proposal_reachability():
    pid, policy, step, config = fixture()
    state = step['decode_trace']['state']
    state.update(node_mask=[True] * 3, node_ids=[0, 1, 2], level_ids=[0] * 3,
                 pid_labels=[0] * 3, parent_ids=[-1] * 3,
                 node_kind_ids=[policy.valid_leaf_node_kinds[0]] * 3,
                 recursive_leaf_source_mask=[[True, False, False], [False, True, False], [False, False, True]],
                 source_conflict_matrix=[[False] * 3 for _ in range(3)],
                 p4=[[0., 0., 0., 1.]] * 3, charge=[0.] * 3,
                 daughter_adjacency=[[False] * 3 for _ in range(3)])
    for key in ('context_mask', 'hard_decode_context_mask', 'pointer_validity_mask', 'forest_pointer_validity_mask'):
        step['decode_trace'][key] = [True] * 3
    step['probabilities']['pointer'] = [[0.8] * 3]
    # The cardinality head supports only 0,1,2. Complete source coverage and
    # charge/ontology compatibility cannot make a three-daughter proposal legal.
    rows, _ = lifecycle_target([frozenset([i]) for i in range(3)], pid,
                               dict(config=config, steps=[step]), policy)
    assert rows[0]['exact_roots_present']
    assert not rows[0]['legal_correct_type_group']
    assert not rows[0]['single_root_split_oracle']


def test_existing_root_excluded_by_policy_is_not_called_never_formed():
    pid, policy, step, config = fixture()
    step['decode_trace']['context_mask'][0] = False
    step['decode_trace']['forest_pointer_validity_mask'][0] = False
    step['decode_trace']['hard_decode_context_mask'][0] = False
    rows, _ = lifecycle_target([frozenset([0]), frozenset([1])], pid,
                               dict(config=config, steps=[step]), policy)
    assert rows[0]['raw_exact_roots_present']
    assert not rows[0]['exact_roots_present']
    assert rows[0]['never_formed_daughter_sets'] == []
    assert rows[0]['root_policy_unavailable_sets'] == [[0]]


def test_bounded_alias_enumeration_preserves_unknown_negative():
    pid, policy, step, config = fixture()
    count = 34
    state = step['decode_trace']['state']
    state.update(node_mask=[True] * count, node_ids=list(range(count)), level_ids=[0] * count,
                 pid_labels=[0] * count, parent_ids=[-1] * count,
                 node_kind_ids=[policy.valid_leaf_node_kinds[0]] * count,
                 recursive_leaf_source_mask=[[True, False]] * 17 + [[False, True]] * 17,
                 source_conflict_matrix=[[False] * count for _ in range(count)],
                 p4=[[0., 0., 0., 1.]] * count,
                 charge=([1.] * 16 + [0.]) * 2,
                 daughter_adjacency=[[False] * count for _ in range(count)])
    for key in ('context_mask', 'hard_decode_context_mask', 'pointer_validity_mask', 'forest_pointer_validity_mask'):
        step['decode_trace'][key] = [True] * count
    step['probabilities']['pointer'] = [[0.8] * count]
    rows, _ = lifecycle_target([frozenset([0]), frozenset([1])], pid,
                               dict(config=config, steps=[step]), policy)
    assert rows[0]['alias_combinations'] == 289
    assert rows[0]['alias_combinations_truncated']
    assert not rows[0]['legal_correct_type_group']
    # The only neutral pair lies after the finite cap. The audit must expose
    # uncertainty rather than assert all source-equivalent choices incompatible.
    assert rows[0]['legality_status'] == 'UNRESOLVED_ALIAS_TRUNCATION'


def test_group_policy_validity_does_not_certify_daughter_tree_or_pid():
    """Distinct valid daughter trees with equal sources share group-policy status."""
    from copy import deepcopy

    pid, policy, template, config = fixture()
    outcomes = []
    # Both root representations cover sources0,1,2, but one has an additional
    # internal mother and a different root PID. This audit deliberately does
    # not verify either daughter tree against truth.
    for nested in (False, True):
        step = deepcopy(template)
        children = [[], [], [], [], [0, 1] if nested else [0, 1, 2]]
        if nested:
            children.append([4, 2])
        count = len(children)
        parents = [-1] * count
        memberships = [{i} for i in range(4)]
        levels = [0] * 4
        for node in range(4, count):
            memberships.append(set().union(*(memberships[c] for c in children[node])))
            levels.append(1 + max(levels[c] for c in children[node]))
            for child in children[node]:
                parents[child] = node
        state = dict(
            node_mask=[True] * count, node_ids=list(range(count)), level_ids=levels,
            pid_labels=[0] * 4 + [PDG_TOKENS.index(511) if nested else pid] * (count - 4),
            parent_ids=parents,
            node_kind_ids=[policy.valid_leaf_node_kinds[0]] * 4 + [policy.valid_composite_node_kinds[0]] * (count - 4),
            recursive_leaf_source_mask=[[i in group for i in range(4)] for group in memberships],
            source_conflict_matrix=[[bool(a & b) and i != j for j, b in enumerate(memberships)] for i, a in enumerate(memberships)],
            p4=[[0., 0., 0., float(len(group))] for group in memberships], charge=[0.] * count,
            daughter_adjacency=[[i in group for i in range(count)] for group in children],
        )
        step['height'] = 3
        step['decode_trace']['state'] = state
        step['decode_trace']['context_mask'] = [True] * count
        for key in ('hard_decode_context_mask', 'pointer_validity_mask', 'forest_pointer_validity_mask'):
            step['decode_trace'][key] = [parent < 0 for parent in parents]
        step['probabilities']['pointer'] = [[0.8] * count]
        rows, _ = lifecycle_target([frozenset([0, 1, 2]), frozenset([3])], pid,
                                   dict(config=config, steps=[step]), policy)
        assert rows[0]['exact_roots_present']
        outcomes.append(rows[0]['legal_correct_type_group'])
    assert outcomes == [True, True]


def test_explicit_measurement_scope_preserves_eligible_failure_counts():
    from scripts.phase86_first_failure import evaluate_first_failure

    truth, projection, trace = joined_fixture()
    result = evaluate_first_failure(truth, projection, trace,
                                    target_policy='complete_only', minimum_daughters=2)
    scope = result['measurement_scope']
    assert scope['daughter_identity'] == 'exact_detector_source_set_equality_only'
    assert scope['recursive_daughter_topology_verified'] is False
    assert scope['recursive_daughter_pid_verified'] is False
    assert scope['native_deep_reachability_verified'] is False
    assert result['policy_eligible_targets'] == 3
    assert result['first_failure_counts'] == {
        'object_rejection': 1,
        'available_group_constraint_incompatible': 1,
        'target_cardinality_incompatible': 1,
    }
