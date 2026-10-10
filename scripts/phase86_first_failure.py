"""Posthoc all-round first-failure audit; never called during candidate generation.

Hypothetical exact daughter groups test opportunity only. They are never supplied
back to inference, ranked as predictions, or counted as measured recovery.
Daughter identity here means detector-source-set equality only. Group-policy
validity does not verify recursive daughter topology/PID or native deep reachability.
"""

from __future__ import annotations

from collections import Counter
from itertools import islice, product

import torch

from hypertagging.models.mother_pointer import constrained_daughter_decode
from hypertagging.reconstruction.constraints import ReconstructionConstraintPolicy
from scripts.phase84_hierarchy_trace import (
    evaluate_native_trace,
    proposal_signature,
    source_rows,
)

MAX_ALIAS_COMBINATIONS = 256


def legal_group(state, positions, pid, policy, round_id, *, require_roots=True):
    """Check group policy on reconstructed nodes, not daughter truth correctness.

    Matching caller-supplied daughter source sets does not establish that their
    internal topology or PID is correct, or that a deep target is reachable.
    """
    reasons = []
    if len(positions) < policy.minimum_daughters:
        reasons.append("cardinality_below_minimum")
    if len(set(positions)) != len(positions):
        reasons.append("duplicate_daughter")
    batch = {k: torch.tensor(v)[None] for k, v in state.items()}
    if any(
        not bool(policy.pointer_validity_mask(batch, round_id)[0, p]) for p in positions
    ):
        reasons.append("pointer_policy_incompatible")
    if require_roots and any(state["parent_ids"][p] >= 0 for p in positions):
        reasons.append("daughter_not_root")
    sources = source_rows(state)
    if policy.reject_recursive_source_conflicts:
        if sum(len(sources[p]) for p in positions) != len(
            set().union(*(sources[p] for p in positions))
        ):
            reasons.append("recursive_source_conflict")
        conflict = state.get("source_conflict_matrix")
        if conflict and any(
            conflict[a][b] for a in positions for b in positions if a != b
        ):
            reasons.append("detector_source_conflict")
    allowed, _ = policy.type_constraints(round_id, device=torch.device("cpu"))
    if pid < 0 or pid >= len(allowed) or not allowed[pid]:
        reasons.append("mother_ontology_incompatible")
    elif policy.mother_charge_compatibility in {"hard", "soft_train_hard_rollout"}:
        if (
            abs(
                sum(state["charge"][p] for p in positions) - policy.expected_charge(pid)
            )
            > policy.mother_charge_tolerance
        ):
            reasons.append("mother_charge_incompatible")
    if positions and not policy.rollout_physical_valid(
        batch["p4"][0, list(positions)].sum(0)
    ):
        reasons.append("physical_kinematics_incompatible")
    return sorted(set(reasons))


def query_support(step, desired, policy, config):
    """Replay learned support using the canonical constrained pointer decoder.

    Simultaneous failures are retained. Best-query precedence is a deterministic
    descriptive factorization, not an identified causal training intervention.
    """
    d = step["decode_trace"]
    state = d["state"]
    probs = step["probabilities"]
    context = [p for p, v in enumerate(d["hard_decode_context_mask"]) if v]
    source_conflict = (
        state.get("source_conflict_matrix")
        if policy.reject_recursive_source_conflicts
        else None
    )
    conflict = (
        torch.tensor(source_conflict, dtype=torch.bool)[context][:, context]
        if source_conflict
        else torch.zeros(len(context), len(context), dtype=torch.bool)
    )
    rows = []
    for q, obj in enumerate(probs["object"]):
        pp = torch.tensor(probs["pointer"][q], dtype=torch.float32)[context]
        cardinality = max(
            range(len(probs["cardinality"][q])), key=probs["cardinality"][q].__getitem__
        )
        predicted = (
            config["use_cardinality"]
            and policy.daughter_cardinality_policy == "predicted"
        )
        cardinality_used = cardinality if predicted else len(context)
        selected, valid = constrained_daughter_decode(
            pp,
            cardinality=cardinality_used,
            pointer_mask=torch.ones_like(pp, dtype=torch.bool),
            source_conflict=conflict,
            min_probability=config["pointer_threshold"],
            insufficient_policy=policy.cardinality_insufficient_policy
            if predicted
            else "reduce",
        )
        positions = [context[i] for i in selected.nonzero().flatten().tolist()]
        selection_exact = set(positions) == set(desired)
        pid = max(range(len(probs["type"][q])), key=probs["type"][q].__getitem__)
        annotations = []
        if obj < config["object_threshold"]:
            annotations.append("object_rejection")
        if predicted and cardinality != len(desired):
            annotations.append("cardinality_support")
        if predicted and cardinality > len(context):
            annotations.append("cardinality_exceeds_context")
        if any(probs["pointer"][q][p] < config["pointer_threshold"] for p in desired):
            annotations.append("pointer_threshold_support")
        if not valid or not selection_exact:
            annotations.append("pointer_ranking_or_source_selection")
        annotations += legal_group(state, desired, pid, policy, step["height"])
        type_prob = probs["type"][q][pid]
        if (
            config["type_probability_threshold"] is not None
            and type_prob < config["type_probability_threshold"]
        ):
            annotations.append("type_probability_rejection")
        confidence = (
            probs["confidence"][q]
            if config["use_learned_confidence"]
            else obj
            * type_prob
            * (sum(probs["pointer"][q][p] for p in desired) / len(desired))
        )
        if confidence < config["confidence_threshold"]:
            annotations.append("confidence_rejection")
        rows.append(
            dict(
                query_id=q,
                selected_positions=positions,
                selection_exact=selection_exact,
                predicted_cardinality=cardinality,
                predicted_type=pid,
                confidence_semantics="hypothetical_exact_daughters_not_actual_if_selection_differs",
                failures=sorted(set(annotations)),
            )
        )
    return rows


def lifecycle_target(support, pid, trace, policy):
    """Track source-set groups and checked policy across observed rounds only.

    ``legal_correct_type_group`` means a source-matching group of existing roots
    passes the checked policy for the target mother PID. It does not certify
    recursive daughter topology/PID or native deep decoder reachability.
    """
    signature = tuple(sorted(tuple(sorted(s)) for s in support))
    config = trace["config"]
    rounds = []
    commitment_round = {
        node: step["height"]
        for step in trace["steps"]
        for node in step.get("appended_node_ids", [])
    }
    first_loss = None
    previously_legal = False
    accepted_correct_previously = False
    for step in trace["steps"]:
        d = step.get("decode_trace")
        if d is None:
            raise ValueError("Exact post-PID decode trace is required")
        state = d["state"]
        sources = source_rows(state)
        roots = [
            p
            for p, v in enumerate(state["node_mask"])
            if v and state["parent_ids"][p] < 0
        ]
        eligible = [
            p
            for p in roots
            if d["forest_pointer_validity_mask"][p] and d["context_mask"][p]
        ]
        options = [[p for p in eligible if sources[p] == s] for s in support]
        all_present = all(options)
        raw_options = [[p for p in roots if sources[p] == source] for source in support]
        raw_all_present = all(raw_options)
        capacity_ok = not (
            config["use_cardinality"]
            and policy.daughter_cardinality_policy == "predicted"
            and len(support) >= len(step["probabilities"]["cardinality"][0])
        )
        clean_cover = all(
            set().union(*(sources[p] for p in roots if sources[p] <= s)) == s
            for s in support
        )
        # A contaminated root is a direct witness of irreversible root consumption,
        # but lack of clean cover alone is never asserted to be legal reachability.
        contaminants = [
            p
            for p in roots
            if any(sources[p] & s and not sources[p] <= s for s in support)
        ]
        if accepted_correct_previously:
            # This target was already recovered. Its own parent (and later
            # ancestors) consuming its daughter roots is successful assembly.
            contaminants = []
        if (
            first_loss is None
            and not accepted_correct_previously
            and not clean_cover
            and contaminants
        ):
            first_loss = dict(
                round=step["height"],
                observation_round=step["height"],
                observation_phase="before_decode",
                node_ids=[state["node_ids"][p] for p in contaminants],
                had_prior_legal_group=previously_legal,
                consuming_commitments=[
                    dict(
                        node_id=state["node_ids"][p],
                        commitment_round=commitment_round.get(state["node_ids"][p]),
                    )
                    for p in contaminants
                ],
            )
        combinations = []
        total = 1
        for opt in options:
            total *= len(opt)
        for positions in islice(product(*options), MAX_ALIAS_COMBINATIONS):
            failures = legal_group(state, positions, pid, policy, step["height"])
            if not capacity_ok:
                failures.append("cardinality_capacity")
            allowed, _ = policy.type_constraints(
                step["height"], device=torch.device("cpu")
            )
            any_type = capacity_ok and any(
                not legal_group(state, positions, t, policy, step["height"])
                for t in torch.where(allowed)[0].tolist()
            )
            combinations.append(
                dict(
                    positions=list(positions),
                    correct_type_failures=failures,
                    any_type_legal=any_type,
                    queries=query_support(step, positions, policy, config),
                )
            )
        exact = [
            p for p in d["raw_proposals"] if proposal_signature(p, sources) == signature
        ]
        accepted = [
            p for p in step["accepted"] if proposal_signature(p, sources) == signature
        ]
        legal = any(not c["correct_type_failures"] for c in combinations)
        previously_legal |= legal
        never = [sorted(s) for s in support if not any(v == s for v in sources)]
        # Finite oracle: release immediate children of ONE parentless mother.
        # No ancestor exists to invalidate at this edit depth. This tests only
        # daughter group construction legality; no embeddings/scores are reused.
        repair = []
        repair_truncated = False
        adjacency = state.get("daughter_adjacency", [])
        if not all_present and capacity_ok and not accepted_correct_previously:
            for parent in contaminants:
                if not adjacency:
                    break
                children = [p for p, v in enumerate(adjacency[parent]) if v]
                restored = [p for p in roots if p != parent] + children
                opts = [[p for p in restored if sources[p] == s] for s in support]
                if not all(opts):
                    continue
                modified = dict(state)
                modified["parent_ids"] = list(state["parent_ids"])
                modified["node_mask"] = list(state["node_mask"])
                modified["node_mask"][parent] = False
                for child in children:
                    modified["parent_ids"][child] = -1
                repair_total = 1
                for option in opts:
                    repair_total *= len(option)
                repair_truncated |= repair_total > MAX_ALIAS_COMBINATIONS
                for positions in islice(product(*opts), MAX_ALIAS_COMBINATIONS):
                    if not legal_group(
                        modified, positions, pid, policy, step["height"]
                    ):
                        # Still require source disjointness from every other root.
                        used = set().union(*(sources[p] for p in positions))
                        if any(
                            used & sources[p] for p in restored if p not in positions
                        ):
                            continue
                        repair.append(
                            dict(
                                split_node_id=state["node_ids"][parent],
                                daughter_node_ids=[
                                    state["node_ids"][p] for p in positions
                                ],
                            )
                        )
                        break
        rounds.append(
            dict(
                round=step["height"],
                exact_roots_present=all_present,
                raw_exact_roots_present=raw_all_present,
                root_policy_unavailable_sets=[
                    sorted(source)
                    for source, raw, available in zip(support, raw_options, options)
                    if raw and not available
                ],
                cardinality_capacity_compatible=capacity_ok,
                clean_source_cover_necessary_only=clean_cover,
                never_formed_daughter_sets=never,
                consuming_root_ids=[state["node_ids"][p] for p in contaminants],
                alias_combinations=total,
                alias_combinations_truncated=total > MAX_ALIAS_COMBINATIONS,
                legal_correct_type_group=legal,
                legality_status="LEGAL_WITNESS"
                if legal
                else (
                    "UNRESOLVED_ALIAS_TRUNCATION"
                    if total > MAX_ALIAS_COMBINATIONS
                    else "NO_LEGAL_GROUP"
                ),
                combinations=combinations,
                generated_exact_group=len(exact),
                generated_correct_type=sum(p["mother_type"] == pid for p in exact),
                accepted_exact_group=len(accepted),
                accepted_correct_type=sum(p["mother_type"] == pid for p in accepted),
                single_root_split_oracle=repair,
                split_search_truncated=repair_truncated,
            )
        )
        accepted_correct_previously |= any(p["mother_type"] == pid for p in accepted)
    # A final commitment has no later predecode snapshot. Inspect its detached
    # terminal state too; the observation time is distinct from creation time.
    if (
        first_loss is None
        and not accepted_correct_previously
        and trace.get("final_state") is not None
    ):
        state = trace["final_state"]
        sources = source_rows(state)
        roots = [
            p
            for p, active in enumerate(state["node_mask"])
            if active and state["parent_ids"][p] < 0
        ]
        clean_cover = all(
            set().union(*(sources[p] for p in roots if sources[p] <= s)) == s
            for s in support
        )
        contaminants = [
            p
            for p in roots
            if any(sources[p] & s and not sources[p] <= s for s in support)
        ]
        if not clean_cover and contaminants:
            observed_round = trace["steps"][-1]["height"] if trace["steps"] else None
            first_loss = dict(
                round=observed_round,
                observation_round=observed_round,
                observation_phase="terminal",
                node_ids=[state["node_ids"][p] for p in contaminants],
                had_prior_legal_group=previously_legal,
                consuming_commitments=[
                    dict(
                        node_id=state["node_ids"][p],
                        commitment_round=commitment_round.get(state["node_ids"][p]),
                    )
                    for p in contaminants
                ],
            )
    return rounds, first_loss


def evaluate_first_failure(
    truth, projection, trace, *, target_policy, minimum_daughters
):
    legacy = evaluate_native_trace(
        truth,
        projection,
        trace,
        target_policy=target_policy,
        minimum_daughters=minimum_daughters,
    )
    policy = ReconstructionConstraintPolicy.from_dict(
        trace["config"]["constraint_policy"]
    )
    if trace["config"]["mother_charge_by_token"]:
        raise ValueError("Native charge overrides require explicit audit support")
    original = truth["recursive_leaf_source_mask"][0].bool()
    kept = torch.tensor(projection.audit.original_fsp_positions[0])
    columns = original[kept].any(0)
    compact = original[:, columns]
    rows = []
    counts = Counter()
    for old in legacy["targets"]:
        row = dict(old)
        row["legacy_first_error"] = row.pop("first_error")
        if not old["policy_eligible"]:
            row["first_failure"] = "outside_policy"
            rows.append(row)
            continue
        n = old["node_position"]
        daughters = torch.where(
            truth["daughter_adjacency"][0, n] & truth["node_mask"][0]
        )[0].tolist()
        support = [frozenset(torch.where(compact[p])[0].tolist()) for p in daughters]
        row["daughter_source_sets"] = [sorted(s) for s in support]
        row["daughter_node_positions"] = daughters
        missing = (
            bool(original[[n, *daughters]][:, ~columns].any())
            or not support
            or any(not s for s in support)
        )
        alias = bool(support) and sum(map(len, support)) != len(set().union(*support))
        if missing or alias or len(support) < minimum_daughters:
            reason = (
                "source_unavailable"
                if missing
                else (
                    "target_source_conflict"
                    if alias
                    else "target_cardinality_incompatible"
                )
            )
            rounds = []
            first_loss = None
        else:
            rounds, first_loss = lifecycle_target(
                support, old["pid_token"], trace, policy
            )
            present = [r for r in rounds if r["exact_roots_present"]]
            legal = [r for r in rounds if r["legal_correct_type_group"]]
            queries = [
                q for r in rounds for c in r["combinations"] for q in c["queries"]
            ]
            if any(r["accepted_correct_type"] for r in rounds):
                reason = "accepted_correct_all_rounds"
            elif any(r["generated_correct_type"] for r in rounds):
                reason = "correct_proposal_retention_loss"
            elif any(r["generated_exact_group"] for r in rounds):
                reason = "generated_group_wrong_mother_type"
            elif not legal and any(r["alias_combinations_truncated"] for r in rounds):
                reason = "unresolved_legality_alias_truncation"
            elif present and not legal:
                reason = "available_group_constraint_incompatible"
            elif first_loss and not legal:
                reason = "wrong_irreversible_merge_before_legal_group"
            elif not present and any(r["raw_exact_roots_present"] for r in rounds):
                reason = "required_roots_present_but_policy_ineligible"
            elif not present:
                reason = "required_root_never_formed_or_unavailable"
            elif queries:
                precedence = [
                    "object_rejection",
                    "mother_ontology_incompatible",
                    "mother_charge_incompatible",
                    "physical_kinematics_incompatible",
                    "pointer_policy_incompatible",
                    "cardinality_support",
                    "pointer_threshold_support",
                    "pointer_ranking_or_source_selection",
                    "type_probability_rejection",
                    "confidence_rejection",
                ]
                # Among legal target-group rounds choose a query passing the longest
                # ordered prefix; this preserves joint failure evidence below.
                qs = [
                    q
                    for r in legal
                    for c in r["combinations"]
                    if not c["correct_type_failures"]
                    for q in c["queries"]
                ]
                depths = [
                    next(
                        (i for i, k in enumerate(precedence) if k in q["failures"]),
                        len(precedence),
                    )
                    for q in qs
                ]
                best = max(depths, default=len(precedence))
                reason = (
                    precedence[best]
                    if best < len(precedence)
                    else "unresolved_decoder_support"
                )
            else:
                reason = "round_or_search_limit"
        row.update(
            first_failure=reason,
            rounds=rounds,
            first_clean_cover_loss=first_loss,
            annotations=sorted(
                {
                    f
                    for r in rounds
                    for c in r["combinations"]
                    for f in c["correct_type_failures"]
                }
                | {
                    f
                    for r in rounds
                    for c in r["combinations"]
                    for q in c["queries"]
                    for f in q["failures"]
                }
            ),
            legal_group_any_round=any(r["legal_correct_type_group"] for r in rounds),
            single_split_oracle_opportunity=(
                not any(r["accepted_correct_type"] for r in rounds)
                and any(r["single_root_split_oracle"] for r in rounds)
            ),
            split_opportunity_status=(
                "ALREADY_ACCEPTED"
                if any(r["accepted_correct_type"] for r in rounds)
                else "WITNESS"
                if any(r["single_root_split_oracle"] for r in rounds)
                else "UNRESOLVED_TRUNCATION"
                if any(r["split_search_truncated"] for r in rounds)
                else "NO_WITNESS_IN_DECLARED_EDIT_SCOPE"
            ),
            legality_status=(
                "LEGAL_WITNESS"
                if any(r["legal_correct_type_group"] for r in rounds)
                else "UNRESOLVED_ALIAS_TRUNCATION"
                if any(r["alias_combinations_truncated"] for r in rounds)
                else "NO_LEGAL_GROUP"
            ),
            delay_oracle_opportunity=bool(
                first_loss
                and first_loss["had_prior_legal_group"]
                and not any(r["accepted_correct_type"] for r in rounds)
            ),
        )
        counts[reason] += 1
        rows.append(row)
    by_node = {r["node_position"]: r for r in rows}

    def dependency(node, seen):
        if node in seen:
            raise ValueError("Cycle in truth-only diagnostic dependency join")
        row = by_node.get(node)
        if row is None:
            return dict(
                node_position=node,
                status="detector_leaf_or_unavailable_retained_target",
            )
        children = []
        for child in row.get("daughter_node_positions", []):
            target_sources = frozenset(torch.where(compact[child])[0].tolist())
            formed = any(
                target_sources in source_rows(step["decode_trace"]["state"])
                for step in trace["steps"]
            )
            if not formed and trace.get("final_state") is not None:
                formed = target_sources in source_rows(trace["final_state"])
            if not formed:
                children.append(dependency(child, seen | {node}))
        return dict(
            node_position=node,
            first_failure=row["first_failure"],
            first_clean_cover_loss=row.get("first_clean_cover_loss"),
            never_formed_dependencies=children,
        )

    for row in rows:
        if row["policy_eligible"]:
            row["first_failure_dependency_trace"] = dependency(
                row["node_position"], set()
            )
    return dict(
        version="phase86-first-failure-v1",
        measurement_scope={
            "daughter_identity": "exact_detector_source_set_equality_only",
            "legal_correct_type_group": (
                "Existing source-matching daughter roots pass checked group policy "
                "for the target mother PID; recursive daughter topology/PID and "
                "native deep decoder reachability are not verified."
            ),
            "recursive_daughter_topology_verified": False,
            "recursive_daughter_pid_verified": False,
            "native_deep_reachability_verified": False,
        },
        policy_eligible_targets=sum(counts.values()),
        first_failure_counts=dict(counts),
        targets=rows,
        limitations=[
            "All TRAIN and posthoc; fit/assessment split does not remove prior inspection.",
            "Source-set/policy group validity is not exact recursive daughter topology/PID correctness or verified native deep reachability.",
            "First failure precedence is descriptive, not evidence of a causal benefit from changing one factor.",
            "Complete_only denotes retained recursive target policy, not physical resonance completeness.",
            "Clean cover is necessary only. Hypothetical exact-group checks and split/delay opportunities are oracle bounds, never deployable gains.",
            "One split opportunity releases only immediate children of one parentless mother; downstream neural states and deep recovery are not simulated.",
            "Greedy trace has one daughter-set choice per query; beam endpoints unavailable.",
            "Teacher-context supervision is not measured by generated-state support; previous teacher gradient audit remains separate.",
        ],
    )
