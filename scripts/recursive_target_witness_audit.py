"""Bounded post-generation recursive identity audit. No inference or fitting.

Source keys identify evaluator leaves; detector resource conflicts remain a
separate policy predicate. None of this module is called by reconstruction.
"""

from __future__ import annotations

from itertools import islice, product
import math


class GraphInvalid(ValueError):
    pass


class RecursiveGraph:
    """Memoized signatures preserving composite class, unary edges and PID."""

    def __init__(self, children, composite, leaf_keys, pids, *, limit=10000):
        self.children = tuple(tuple(x) for x in children)
        pids = [p if p is not None else -1 for p in pids]
        self.composite, self.leaf_keys, self.pids = composite, leaf_keys, pids
        n = len(children)
        if n > limit:
            raise GraphInvalid("node_bound_exceeded")
        if not all(len(x) == n for x in (composite, leaf_keys, pids)):
            raise GraphInvalid("axis_mismatch")
        self.topology, self.with_pid, self.height, self.leaves = {}, {}, {}, {}
        self.descendants = {}
        # Iterative DFS avoids a hidden Python recursion-depth bound.
        for root in range(n):
            stack, visiting = [(root, False)], set()
            while stack:
                node, finish = stack.pop()
                if node in self.topology:
                    continue
                if not 0 <= node < n:
                    raise GraphInvalid("dangling_child")
                ds = self.children[node]
                if finish:
                    visiting.remove(node)
                    if ds and not composite[node]:
                        raise GraphInvalid("leaf_has_children")
                    if composite[node]:
                        top = ("composite", tuple(sorted(self.topology[d] for d in ds)))
                        pid = (
                            "composite",
                            pids[node],
                            tuple(sorted(self.with_pid[d] for d in ds)),
                        )
                        leaves = tuple(sorted(k for d in ds for k in self.leaves[d]))
                    else:
                        if leaf_keys[node] is None:
                            raise GraphInvalid("missing_leaf_key")
                        top = ("leaf", leaf_keys[node])
                        pid = ("leaf", pids[node], leaf_keys[node])
                        leaves = (leaf_keys[node],)
                    self.topology[node], self.with_pid[node] = top, pid
                    self.height[node] = (
                        1 + max((self.height[d] for d in ds), default=-1)
                        if composite[node]
                        else 0
                    )
                    self.leaves[node] = leaves
                    self.descendants[node] = frozenset([node]).union(
                        *(self.descendants[d] for d in ds)
                    )
                else:
                    if node in visiting:
                        raise GraphInvalid("cycle")
                    visiting.add(node)
                    stack.append((node, True))
                    stack.extend((d, False) for d in reversed(ds))

    def pid_available(self, node):
        return all(
            self.pids[d] is not None and self.pids[d] > 0
            for d in self.descendants[node]
        )


def bounded_tuples(candidate_lists, *, cap=256):
    """Count rejected/noninjective tuples too; never materialize a product."""
    if not 0 < cap <= 256:
        raise ValueError("tuple bound changed")
    lists = [tuple(x) for x in candidate_lists]
    total = math.prod(map(len, lists)) if lists else 0
    return islice(product(*lists), min(cap, total)), total > cap


def inherited_blockers(graph, local):
    result = {}
    for node in range(len(graph.children)):
        blockers = [d for d in graph.descendants[node] if local.get(d)]
        lowest = [
            d
            for d in blockers
            if not any(x != d and x in graph.descendants[d] for x in blockers)
        ]
        result[node] = sorted(lowest, key=lambda d: (graph.height[d], d))
    return result


def matches(graph, positions, target_graph, daughters, *, pid=False):
    a = graph.with_pid if pid else graph.topology
    b = target_graph.with_pid if pid else target_graph.topology
    if pid and (
        not all(graph.pid_available(p) for p in positions)
        or not all(target_graph.pid_available(p) for p in daughters)
    ):
        return None
    return sorted(a[p] for p in positions) == sorted(b[d] for d in daughters)


def observed_graph(state, leaf_key_by_node_id, *, unavailable_pid_nodes=()):
    ids = state["node_ids"]
    if (
        len(set(ids)) != len(ids)
        or len(state["daughter_adjacency"]) != len(ids)
        or any(len(row) != len(ids) for row in state["daughter_adjacency"])
    ):
        raise GraphInvalid("duplicate_node_id_or_adjacency_axis")
    children = [
        [j for j, v in enumerate(row) if v] for row in state["daughter_adjacency"]
    ]
    if any(sum(q in ds for ds in children) > 1 for q in range(len(ids))):
        raise GraphInvalid("multiple_parents")
    for child in range(len(ids)):
        parents = [p for p, ds in enumerate(children) if child in ds]
        if state["parent_ids"][child] != (parents[0] if parents else -1):
            raise GraphInvalid("parent_adjacency_inconsistent")
        if not state["node_mask"][child] and (parents or children[child]):
            raise GraphInvalid("inactive_graph_reference")
    composite = [h > 0 for h in state["level_ids"]]
    keys = [
        None if composite[p] else leaf_key_by_node_id.get(node)
        for p, node in enumerate(state["node_ids"])
    ]
    # Saved pid_labels is stale for raw-track predictions: never fill unknowns.
    pids = [
        None if node in unavailable_pid_nodes else state["pid_labels"][p]
        for p, node in enumerate(state["node_ids"])
    ]
    return RecursiveGraph(children, composite, keys, pids)


def terminal_adjacency(trace):
    """Rebuild only recorded accepted edges, cross-check every later snapshot."""
    edges = {}
    for step in trace["steps"]:
        state = step["decode_trace"]["state"]
        ids = state["node_ids"]
        for p, node in enumerate(ids):
            current = tuple(
                ids[d] for d, v in enumerate(state["daughter_adjacency"][p]) if v
            )
            if node in edges and edges[node] != current:
                raise GraphInvalid("later_adjacency_mismatch")
            edges[node] = current
        if len(step["accepted"]) != len(step["appended_node_ids"]):
            raise GraphInvalid("commitment_length_mismatch")
        for proposal, node in zip(step["accepted"], step["appended_node_ids"]):
            if node in edges:
                raise GraphInvalid("reused_appended_id")
            edges[node] = tuple(ids[d] for d in proposal["daughter_positions"])
    final = dict(trace["final_state"])
    ids = final["node_ids"]
    if set(ids) != set(edges):
        raise GraphInvalid("terminal_node_identity_mismatch")
    final["daughter_adjacency"] = [[d in edges[p] for d in ids] for p in ids]
    return final


def audit_event(truth, projection, trace, historical, policy, architecture=None):
    """Join authenticated original targets to already generated native states."""
    import torch
    from hypertagging.preprocessing.schema_v4 import LEAF_MODE_TO_ID
    from scripts.phase84_hierarchy_trace import source_rows

    if len(trace["steps"]) > 6:
        raise GraphInvalid("round_bound_exceeded")
    kept = list(projection.audit.original_fsp_positions[0])
    keys = list(projection.audit.evaluation_fsp_source_keys[0])
    source = truth["recursive_leaf_source_mask"][0].bool()
    columns = source[kept].any(0)
    if columns.nonzero().flatten().tolist() != historical["source_columns_kept"]:
        raise GraphInvalid("source_column_mismatch")
    first = trace["steps"][0]["decode_trace"]["state"]
    for name in ("node_ids", "recursive_leaf_source_mask", "source_conflict_matrix"):
        if first[name] != projection.batch[name][0].tolist():
            raise GraphInvalid("initial_projection_mismatch:" + name)
    key_map = dict(zip(first["node_ids"], keys))
    unknown_pid = {
        node
        for node, mode in zip(first["node_ids"], first["leaf_kinematics_mode_ids"])
        if mode == LEAF_MODE_TO_ID["raw_track_predicted_pid"]
    }
    n = int(truth["node_mask"][0].sum())
    ds = [
        [j for j, v in enumerate(row[:n]) if v]
        for row in truth["daughter_adjacency"][0, :n].tolist()
    ]
    heights = truth["level_ids"][0, :n].tolist()
    labels = truth.get("pid_target_labels", truth["pid_labels"])[0, :n].tolist()
    from hypertagging.evaluation.full_decay_metrics import canonical_fsp_membership

    canonical_memberships, canonical_keys = canonical_fsp_membership(truth)
    leaf_keys = [None] * n
    for p in range(n):
        if heights[p] == 0:
            indices = canonical_memberships[p].nonzero().flatten().tolist()
            if len(indices) != 1:
                raise GraphInvalid("canonical_leaf_membership_invalid")
            leaf_keys[p] = canonical_keys[indices[0]]
        if sum(p in children for children in ds) > 1:
            raise GraphInvalid("target_multiple_parents")
    if [leaf_keys[p] for p in kept] != keys:
        raise GraphInvalid("target_projection_evaluator_key_mismatch")
    target = RecursiveGraph(ds, [h > 0 for h in heights], leaf_keys, labels)
    target_sources = [
        frozenset(row.nonzero().flatten().tolist()) for row in source[:, columns]
    ]
    states = []
    for step in trace["steps"]:
        d = step["decode_trace"]
        s = d["state"]
        r = step["height"]
        g = observed_graph(s, key_map, unavailable_pid_nodes=unknown_pid)
        allowed, _ = policy.type_constraints(r, device=torch.device("cpu"))
        states.append((step, g, source_rows(s), allowed.tolist()))
    final = terminal_adjacency(trace)
    fg = observed_graph(final, key_map, unavailable_pid_nodes=unknown_pid)
    local, checks = {}, {}
    for p in range(n):
        reasons = []
        if heights[p] > 0:
            if len(ds[p]) < policy.minimum_daughters:
                reasons.append("minimum_arity")
            if target.height[p] > 6:
                reasons.append("intrinsic_height_exceeds_horizon")
            possible = []
            for r in range(max(1, target.height[p]), 7):
                a, _ = policy.type_constraints(r, device=torch.device("cpu"))
                possible.append(0 <= labels[p] < len(a) and bool(a[labels[p]]))
            if labels[p] > 0 and possible and not any(possible):
                reasons.append("hard_mother_ontology_all_remaining_rounds")
            if (
                architecture
                and trace["config"]["use_cardinality"]
                and policy.daughter_cardinality_policy == "predicted"
            ):
                per_round = dict(architecture.get("max_cardinality_by_level", []))
                capacities = [
                    per_round.get(r, architecture["max_cardinality"])
                    for r in range(max(1, target.height[p]), 7)
                ]
                if capacities and all(len(ds[p]) > cap for cap in capacities):
                    reasons.append("cardinality_capacity_all_remaining_rounds")
            if len(target.leaves[p]) != len(set(target.leaves[p])):
                reasons.append("duplicate_evaluator_leaf")
            if policy.reject_recursive_source_conflicts:
                supports = [set(source[q].nonzero().flatten().tolist()) for q in ds[p]]
                if sum(map(len, supports)) != len(set().union(*supports)):
                    reasons.append("recursive_detector_resource_conflict")
        elif p not in kept:
            reasons.append("detector_leaf_not_projected")
        local[p] = reasons
        checks[p] = dict(
            node_position=p,
            node_id=int(truth["node_ids"][0, p]),
            intrinsic_height=target.height[p],
            stored_height=heights[p],
            immediate_daughters=ds[p],
            pid_token=labels[p],
            local_blockers=reasons,
            measurement_physical_compatibility="unresolved_no_constructed_state",
            matched_training_exposure="unavailable",
        )
    blockers = inherited_blockers(target, local)
    rows, attempted = [], 0
    for old in historical["targets"]:
        p = old["node_position"]
        daughters = ds[p]
        row = dict(
            historical=old,
            node_position=p,
            intrinsic_height=target.height[p],
            all_blocking_descendants=[
                q for q in sorted(target.descendants[p]) if local[q]
            ],
            lowest_blocking_descendants=blockers[p],
            contract_status="proven_incompatible" if blockers[p] else "unresolved",
            contract_unknowns=[
                "query_node_capacity_global_schedule",
                "physical_measurement_and_unvisited_states",
            ],
            rounds=[],
            terminal_topology=any(
                fg.topology[q] == target.topology[p] for q in fg.topology
            ),
            terminal_recursive_pid=terminal_pid_match(fg, target, p),
        )
        for step, g, sources, allowed in states:
            d = step["decode_trace"]
            s = d["state"]
            r = step["height"]
            roots = [
                q
                for q, node in enumerate(s["node_ids"])
                if s["node_mask"][q]
                and s["parent_ids"][q] < 0
                and d["hard_decode_context_mask"][q]
                and d["forest_pointer_validity_mask"][q]
                and d["pointer_validity_mask"][q]
            ]
            lists = [
                sorted(
                    (q for q in roots if sources[q] == target_sources[child]),
                    key=lambda q: s["node_ids"][q],
                )
                for child in daughters
            ]
            stream, truncated = bounded_tuples(lists)
            found = dict(
                source=False, height=False, topology=False, recursive_pid=False
            )
            unknown = False
            constraints = {}
            count = 0
            witnesses = {}
            capacity = max(
                (len(c) - 1 for c in step["probabilities"]["cardinality"]), default=0
            )
            for positions in stream:
                count += 1
                attempted += 1
                why = []
                if len(set(positions)) != len(positions):
                    why.append("duplicate_node")
                if len(positions) < policy.minimum_daughters:
                    why.append("minimum_arity")
                if (
                    trace["config"]["use_cardinality"]
                    and policy.daughter_cardinality_policy == "predicted"
                    and len(positions) > capacity
                ):
                    why.append("cardinality_output_capacity")
                if policy.reject_recursive_source_conflicts:
                    if sum(len(sources[q]) for q in positions) != len(
                        set().union(*(sources[q] for q in positions))
                    ):
                        why.append("recursive_source_conflict")
                    if any(
                        s["source_conflict_matrix"][a][b]
                        for a in positions
                        for b in positions
                        if a != b
                    ):
                        why.append("detector_source_conflict")
                pid = labels[p]
                if not (0 <= pid < len(allowed) and allowed[pid]):
                    why.append("mother_ontology")
                expected = dict(trace["config"].get("mother_charge_by_token", [])).get(
                    pid, policy.expected_charge(pid)
                )
                if (
                    policy.mother_charge_compatibility
                    in {"hard", "soft_train_hard_rollout"}
                    and abs(
                        float(torch.tensor([s["charge"][q] for q in positions]).sum())
                        - expected
                    )
                    > policy.mother_charge_tolerance
                ):
                    why.append("saved_numerical_charge")
                if (
                    positions
                    and policy.loose_physical_constraints
                    and not policy.rollout_physical_valid(
                        torch.tensor([s["p4"][q] for q in positions]).sum(0)
                    )
                ):
                    why.append("saved_numerical_physics")
                for reason in why:
                    constraints[reason] = constraints.get(reason, 0) + 1
                if why:
                    continue
                height = all(
                    g.height[q] == target.height[child]
                    for q, child in zip(positions, daughters)
                )
                topology = height and bool(matches(g, positions, target, daughters))
                pid_match = (
                    matches(g, positions, target, daughters, pid=True)
                    if topology
                    else False
                )
                if pid_match is None:
                    unknown = True
                for name, value in [
                    ("source", True),
                    ("height", height),
                    ("topology", topology),
                    ("recursive_pid", pid_match is True),
                ]:
                    if value:
                        found[name] = True
                        witnesses.setdefault(name, list(positions))
            emissions = {}
            for name, proposals in [
                ("raw_hard_decoded", d["raw_proposals"]),
                ("local_retained", step["proposals"]),
                ("accepted", step["accepted"]),
            ]:
                observed = []
                for proposal in proposals:
                    positions = proposal["daughter_positions"]
                    src = sorted(
                        tuple(sorted(sources[q])) for q in positions
                    ) == sorted(tuple(sorted(target_sources[q])) for q in daughters)
                    topo = bool(matches(g, positions, target, daughters))
                    observed.append(
                        dict(
                            query_id=proposal["query_id"],
                            daughter_positions=positions,
                            source=src,
                            topology=topo,
                            mother_pid=proposal["mother_type"] == labels[p],
                            recursive_pid=matches(
                                g, positions, target, daughters, pid=True
                            )
                            if topo
                            else False,
                        )
                    )
                emissions[name] = observed
            missing = []
            for child, aliases in zip(daughters, lists):
                exact = [q for q in roots if g.topology[q] == target.topology[child]]
                present = [
                    q for q in g.topology if g.topology[q] == target.topology[child]
                ]
                if not exact:
                    missing.append(
                        dict(
                            child=child,
                            blockers=blockers[child],
                            source_aliases=aliases,
                            reason="consumed_or_ineligible"
                            if present
                            else "source_equivalent_wrong_topology"
                            if aliases
                            else "absent_structure",
                        )
                    )
            row["rounds"].append(
                dict(
                    round=r,
                    attempted_tuples=count,
                    truncated=truncated,
                    available=found,
                    witness_positions=witnesses,
                    recursive_pid_unavailable=unknown,
                    absence_status="unresolved_bound"
                    if truncated
                    else "observed_absence",
                    constraint_rejection_counts=constraints,
                    emissions=emissions,
                    missing_children=missing,
                    query_count=len(step["probabilities"]["object"]),
                    cardinality_output_capacity=capacity,
                )
            )
        row["available_any_round"] = {
            k: any(rr["available"][k] for rr in row["rounds"])
            for k in ("source", "height", "topology", "recursive_pid")
        }
        rows.append(row)
    return dict(
        targets=rows,
        descendant_checks=list(checks.values()),
        attempted_tuples=attempted,
        projection_audit=projection.audit.as_dict(),
        raw_track_pid_unavailable_node_ids=sorted(unknown_pid),
        target_graph_nodes=n,
        observed_states=len(states),
        availability_note="PID unknown for raw tracks; observed numerical policy is separate from physical measurement validity; no repair counterfactual executed",
    )


def terminal_pid_match(observed, target, node):
    matching = [
        q for q in observed.topology if observed.topology[q] == target.topology[node]
    ]
    if any(
        observed.pid_available(q)
        and target.pid_available(node)
        and observed.with_pid[q] == target.with_pid[node]
        for q in matching
    ):
        return True
    if not target.pid_available(node) or any(
        not observed.pid_available(q) for q in matching
    ):
        return None
    return False
