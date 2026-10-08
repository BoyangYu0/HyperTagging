"""Truth joins performed ONLY after a detached, truth-free search has ended."""
from __future__ import annotations

from collections import Counter, defaultdict

from hypertagging.evaluation.full_decay_metrics import _tree_view, _truth_b_roots, _truth_target_mask


def membership_possible(state: dict, target: frozenset[int]) -> bool:
    """Optimistic no-split bound, not proof of legal decoder reachability.

    Existing exact groups count even after a higher composite consumes them.
    Clean roots may still fail charge, type, cardinality or detector exclusivity.
    """
    groups = [frozenset(x) for x in state["memberships"]]
    if any(groups[p] == target for p in state["composites"]):
        return True
    clean = [groups[p] for p in state["roots"] if groups[p] <= target]
    return bool(target) and frozenset().union(*clean) == target


def group_present(state: dict, target: frozenset[int]) -> bool:
    return any(frozenset(state["memberships"][p]) == target for p in state["composites"])


def _proposal_group(record: dict, proposal: dict) -> frozenset[int]:
    return frozenset().union(*(record["state"]["memberships"][p] for p in proposal["daughters"]))


def analyze_search_trace(truth_batch, projection, trace, *, target_policy: str,
                         minimum_daughters: int, object_threshold: float,
                         pointer_threshold: float) -> dict:
    """Count targets once/event/level, existentially across evaluated states.

    Generated means visited and passed production constraints before local caps;
    unvisited subset combinations are not claimed to have been generated.
    Eligible daughter presence is optimistic parentless-context availability,
    before committed-source alias filtering and joint proposal constraints.
    """
    truth = _tree_view(truth_batch, 0, truth=True)
    keys = projection.evaluation_leaf_source_keys[0].tolist()
    key_to_dense = defaultdict(set)
    for dense, key in enumerate(keys):
        key_to_dense[int(key)].add(dense)

    def dense_members(node):
        sources = truth.source_set(node)
        # Ambiguous identity mappings must not become measured failures.
        if not sources or any(len(key_to_dense[k]) != 1 for k in sources):
            return None
        return frozenset(next(iter(key_to_dense[k])) for k in sources)

    mask = _truth_target_mask(truth_batch, 0, target_policy=target_policy,
                              minimum_daughters=minimum_daughters)
    decodes = [r for r in trace if r["stage"] == "decode"]
    final = next(r["states"] for r in reversed(trace) if r["stage"] == "final")
    counts = defaultdict(Counter)
    for node in truth.positions:
        if not bool(mask[node]):
            continue
        level = int(truth_batch["level_ids"][0, node])
        counts[str(level)]["truth_targets"] += 1
        daughters = [dense_members(child) for child in truth.children(node)]
        target = dense_members(node)
        if target is None or any(d is None for d in daughters):
            counts[str(level)]["identity_unavailable"] += 1
            continue
        counts[str(level)]["identity_available"] += 1
        daughter_key = tuple(sorted(tuple(sorted(d)) for d in daughters))
        flags = Counter()
        for r in decodes:
            if r["level"] != level:
                continue
            memberships = r["state"]["memberships"]
            positions = [[p for p in r["eligible"] if frozenset(memberships[p]) == d]
                         for d in daughters]
            if all(positions):
                flags["eligible_daughters_present"] = 1
                for q, obj in enumerate(r["object_probability"]):
                    if obj >= object_threshold and all(
                        any(r["pointer_probability"][q][p] >= pointer_threshold for p in ps)
                        for ps in positions
                    ):
                        flags["object_pointer_threshold_support"] = 1
            for stage in ("generated", "query_retained", "level_retained"):
                for proposal in r[stage]:
                    signature = tuple(sorted(tuple(sorted(memberships[p])) for p in proposal["daughters"]))
                    if signature == daughter_key:
                        flags[stage + "_exact_daughters_any_type"] = 1
                        if proposal["type"] == int(truth.pid[node]):
                            flags[stage + "_exact_daughters_and_type"] = 1
        flags["final_pool_membership"] = int(any(group_present(s, target) for s in final))
        counts[str(level)].update(flags)

    b_rows = []
    for root in _truth_b_roots(truth):
        target = dense_members(root)
        if target is None:
            b_rows.append({"available": False})
            continue
        generated = {stage: any(_proposal_group(r, p) == target for r in decodes for p in r[stage])
                     for stage in ("generated", "query_retained", "level_retained")}
        expanded = any(group_present(r["state"], target) for r in trace if r["stage"] == "expanded")
        top1 = group_present(final[0], target)
        pool = any(group_present(s, target) for s in final)
        retained = [r for r in trace if r["stage"] == "retained_level"]
        first_block = next((r["level"] for r in retained
                            if not any(membership_possible(s, target) for s in r["states"])), None)
        if top1:
            reason = "top1_membership_present"
        elif pool:
            reason = "pool_present_top1_missed"
        elif expanded:
            reason = "constructed_group_lost_from_final_pool"
        elif generated["level_retained"]:
            reason = "proposal_present_not_constructed"
        elif generated["query_retained"]:
            reason = "group_lost_at_level_cap"
        elif generated["generated"]:
            reason = "group_lost_at_query_cap"
        elif first_block is not None:
            reason = "never_proposed_and_clean_root_cover_lost"
        else:
            reason = "never_proposed_despite_optimistic_root_cover"
        groups = [frozenset(s["memberships"][p]) for s in final for p in s["composites"]]
        best = max(groups, key=lambda g: len(g & target) / len(g | target), default=frozenset())
        b_rows.append({"available": True, "leaf_count": len(target), "failure_class": reason,
                       "first_no_clean_root_cover_level": first_block,
                       "top1_membership_present": top1, "pool_membership_present": pool,
                       "proposal_stages": generated, "constructed_anywhere": expanded,
                       "best_pool_group_iou": len(best & target) / len(best | target),
                       "best_pool_missing": len(target - best), "best_pool_foreign": len(best - target)})
    return {"by_level": {k: dict(v) for k, v in counts.items()}, "b_units": b_rows,
            "bound_semantics": "clean_root_union_is_necessary_not_sufficient_no_split_condition"}


def summarize_search_survival(records: list[dict]) -> dict:
    levels = defaultdict(Counter)
    reasons, first = Counter(), Counter()
    available = unavailable = 0
    for record in records:
        for level, counts in record["by_level"].items():
            levels[level].update(counts)
        for unit in record["b_units"]:
            if not unit["available"]:
                unavailable += 1
                continue
            available += 1
            reasons[unit["failure_class"]] += 1
            first[str(unit["first_no_clean_root_cover_level"])] += 1
    return {"events": len(records), "available_b_units": available,
            "identity_unavailable_b_units": unavailable, "b_failure_classes": dict(reasons),
            "first_no_clean_root_cover_level": dict(first),
            "by_level": {k: dict(v) for k, v in levels.items()}}
