"""Review the sole completed successor; never choose thresholds or submit jobs."""

from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import json
import hashlib
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import sha, binding, write  # noqa: E402


def paired_effects(left, right):
    import numpy as np
    from scipy.stats import beta

    if [r["uid"] for r in left] != [r["uid"] for r in right]:
        raise ValueError("Paired identity order differs")
    rng = np.random.default_rng(20261009)
    effects = {}
    for metric, den in [
        ("raw_exact_memberships", "nominal_b_trials"),
        ("accepted_exact_memberships", "nominal_b_trials"),
        ("continuum_accepted_events", "continuum_events"),
    ]:
        selected = [i for i, e in enumerate(left) if e["counts"][den] > 0]
        x = np.array(
            [[r[i]["counts"][metric] for i in selected] for r in (left, right)]
        )
        n = np.array([left[i]["counts"][den] for i in selected])
        cats = [left[i]["category"] for i in selected]
        strata = [np.where(np.array(cats) == cat)[0] for cat in sorted(set(cats))]
        delta = []
        for _ in range(2000):
            idx = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
            delta.append(float((x[1, idx] - x[0, idx]).sum() / n[idx].sum()))
        effects[metric] = {
            "control": int(x[0].sum()),
            "intervention": int(x[1].sum()),
            "denominator": int(n.sum()),
            "difference": float((x[1] - x[0]).sum() / n.sum()),
            "paired_collision_stratified_bootstrap95": np.quantile(
                delta, [0.025, 0.975]
            ).tolist(),
            "paired_collisions": len(selected),
            "discordant_collision_counts": {
                "intervention_greater": int((x[1] > x[0]).sum()),
                "control_greater": int((x[0] > x[1]).sum()),
                "equal": int((x[0] == x[1]).sum()),
            },
            "event_any_exact_binomial95": [],
        }
        for row in x:
            k = int((row > 0).sum())
            N = len(row)
            effects[metric]["event_any_exact_binomial95"].append(
                {
                    "successes": k,
                    "collisions": N,
                    "lower": float(beta.ppf(0.025, k, N - k + 1)) if k else 0.0,
                    "upper": float(beta.ppf(0.975, k + 1, N - k)) if k < N else 1.0,
                }
            )
    return effects


def paired_source_errors(left, right):
    """Collision-stratified uncertainty for aggregate source recall/precision."""
    import numpy as np

    if [(r["uid"], r["category"]) for r in left] != [
        (r["uid"], r["category"]) for r in right
    ]:
        raise ValueError("Paired diagnostic identity differs")
    selected = [
        i for i, row in enumerate(left) if row["category"] in ("charged", "mixed")
    ]
    cats = np.array([left[i]["category"] for i in selected])
    strata = [np.where(cats == cat)[0] for cat in sorted(set(cats))]
    rng = np.random.default_rng(20261009)
    output = {}
    for stage in ("proposal", "refinement"):
        counts = [
            [rows[i]["stages"][stage]["counts"] for i in selected]
            for rows in (left, right)
        ]

        def values(key):
            return np.array([[c.get(key, 0) for c in arm] for arm in counts])

        intersection, missing, extra = (
            values("intersection"),
            values("missing"),
            values("extra"),
        )
        metrics = {
            "source_recall": (intersection, intersection + missing),
            "source_precision": (intersection, intersection + extra),
            "missing_to_unassigned_fraction": (
                values("missing_to_unassigned"),
                intersection + missing,
            ),
            "missing_to_other_B_fraction": (
                values("missing_to_other_B"),
                intersection + missing,
            ),
        }
        output[stage] = {}
        for name, (numerator, denominator) in metrics.items():
            sums = denominator.sum(1)
            record = {
                "numerators": numerator.sum(1).tolist(),
                "denominators": sums.tolist(),
                "paired_collisions": len(selected),
            }
            if not selected or bool((sums == 0).any()):
                record.update(
                    difference=None,
                    paired_collision_stratified_bootstrap95=None,
                    status="UNAVAILABLE_ZERO_SUPPORT",
                )
            else:
                delta = []
                for _ in range(2000):
                    idx = np.concatenate(
                        [rng.choice(st, len(st), replace=True) for st in strata]
                    )
                    den = denominator[:, idx].sum(1)
                    if bool((den == 0).any()):
                        continue
                    ratio = numerator[:, idx].sum(1) / den
                    delta.append(float(ratio[1] - ratio[0]))
                ratio = numerator.sum(1) / sums
                record.update(
                    rates=ratio.tolist(),
                    difference=float(ratio[1] - ratio[0]),
                    paired_collision_stratified_bootstrap95=np.quantile(
                        delta, [0.025, 0.975]
                    ).tolist()
                    if delta
                    else None,
                    status="DESCRIPTIVE_PAIRED_ONE_SEED",
                )
            output[stage][name] = record
    return output


def sampling_accounting(rows, presentations, seed):
    rng = random.Random(seed)
    selected = [rows[rng.randrange(len(rows))] for _ in range(presentations)]
    counts = Counter(row["uid"] for row in selected)
    return {
        "eligible_unique_events": len(rows),
        "actual_distinct_presented": len(counts),
        "not_presented": len(rows) - len(counts),
        "presentations": presentations,
        "sampled_with_replacement": True,
        "minimum_presentations_among_seen": min(counts.values()),
        "maximum_presentations_among_seen": max(counts.values()),
        "presentations_by_category": dict(Counter(r["category"] for r in selected)),
        "data_order_sha256": hashlib.sha256(
            "".join(row["uid"] + "\n" for row in selected).encode()
        ).hexdigest(),
    }


def continuum_membership(row, event):
    """Post-hoc flat-group overlap with explicit retained roots, never physical tagging."""
    from hypertagging.evaluation.full_decay_metrics import (
        _tree_view,
        _root_positions,
        _component_structurally_valid,
    )

    view = _tree_view(row["full"], 0, truth=True)
    roots = [node for node in _root_positions(view) if view.children(node)]
    cc = Counter(
        events=1,
        events_without_explicit_component=int(not roots),
        component_trials=0,
        unavailable=0,
        raw_exact=0,
        accepted_exact=0,
    )
    for node in roots:
        target = row["supervision"]["node_sets"][node]
        available = bool(target) and _component_structurally_valid(view, node)
        cc["component_trials"] += 1
        cc["unavailable"] += int(not available)
        if available:
            cc["raw_exact"] += int(set(target) in [set(g) for g in event["raw_groups"]])
            cc["accepted_exact"] += int(
                set(target) in [set(g) for g in event["accepted_groups"]]
            )
    return cc


def review(root):

    c = json.loads((root / "successor-campaign-v1/contract.json").read_text())
    from scripts.run_phase74_development import verify_bindings

    verify_bindings(c, Path(c["source_root"]))
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Guarded CPU review required")
    import torch

    torch.set_num_threads(1)
    cache = torch.load(c["cache"]["path"], weights_only=False, map_location="cpu")
    cachemap = {row["uid"]: row for row in cache["train"] + cache["development"]}
    designated = json.loads((root / "development-cohort.json").read_text())
    training = json.loads((root / "training-cohort.json").read_text())
    base = root / "successor-campaign-v1/runs"
    arms = c["arms"]
    records = {}
    private = {
        "contract": binding(root / "successor-campaign-v1/contract.json"),
        "arms": {},
    }
    out = {
        "version": "phase76-terminal-review-v1",
        "stage": "development",
        "status": "COMPLETED",
        "source_sha": c["source_sha"],
        "arms": {},
        "automatic_successor": False,
        "primary_eligible": False,
        "selection_rule": "fixed_final_and_threshold0.5",
        "training_unique_events": 1536,
        "fresh_heldout_events": 600,
    }
    orders = defaultdict(set)
    for arm in arms:
        terminal = json.loads((base / arm / "terminal.json").read_text())
        assert terminal["status"] == "COMPLETED"
        for item in [terminal["summary"], *terminal["checkpoints"]]:
            assert sha(item["path"]) == item["sha256"]
        for item in terminal["checkpoints"]:
            checkpoint = torch.load(
                item["path"], weights_only=False, map_location="cpu"
            )
            stage = checkpoint["stage"]
            assert checkpoint["source_sha"] == c["source_sha"]
            assert checkpoint["settings"] == c["arm_settings"][arm]
            assert checkpoint["step"] == c["settings"][stage + "_updates"]
            assert checkpoint["data_cache"] == c["cache"]
            assert checkpoint["gradient_rule"] == "joint"
            assert checkpoint["pid_weights_frozen_downstream"]
        r = json.loads((base / arm / "summary.json").read_text())
        assert r["contract_sha256"] == sha(root / "successor-campaign-v1/contract.json")
        assert r["compute"]["event_presentations"] == 20000
        assert r["compute"]["encoder_passes"] == 40000
        assert r["compute"]["evaluation_event_views"] == 2160
        tiny_uids = {e["uid"] for e in r["tiny"]["events"]}
        evaluation_rows = (
            cache["train"]
            + cache["development"]
            + [row for row in cache["train"] if row["uid"] in tiny_uids]
        )
        public = {
            "compute": r["compute"],
            "evidence_hashes": {
                "summary_sha256": terminal["summary"]["sha256"],
                "checkpoint_sha256_by_stage": {
                    Path(b["path"]).stem.removesuffix("-final"): b["sha256"]
                    for b in terminal["checkpoints"]
                },
            },
            "compute_accounting": {
                "native_encoder_passes_scope": "fit_only",
                "fit_encoder_passes": 40000,
                "evaluation_encoder_passes": 4320,
                "total_encoder_passes": 44320,
                "evaluation_detector_node_pair_proxy": sum(
                    len(row["targets"]) ** 2 for row in evaluation_rows
                ),
                "flops": None,
                "flops_status": "NOT_MEASURED; node-pair counts and passes are proxies, no gradient projection/probes; partial-context ablation can change realized compute",
            },
            "histories": r["histories"],
            "sampling_accounting": {},
            "roles": {},
            "curves": {},
            "unavailable": r["heldout"]["unavailable"],
        }
        for stage, count in [("tiny", 1000), ("downstream", 1500)]:
            hist = r["histories"][stage]
            assert hist["updates"] == count and hist["presentations"] == 8 * count
            stage_rows = (
                [cachemap[e["uid"]] for e in r["tiny"]["events"]]
                if stage == "tiny"
                else cache["train"]
            )
            accounting = sampling_accounting(
                stage_rows, 8 * count, c["settings"]["seed"]
            )
            assert accounting["data_order_sha256"] == hist["data_order_sha256"]
            public["sampling_accounting"][stage] = accounting
            orders[stage].add(hist["data_order_sha256"])
            steps = [
                json.loads(line)
                for line in (base / arm / f"{stage}-metrics.jsonl")
                .read_text()
                .splitlines()
            ]
            assert [x["step"] for x in steps] == list(range(1, count + 1))
            public["curves"][stage] = steps
        for role in ("tiny", "train", "heldout"):
            identities = [e["uid"] for e in r[role]["events"]]
            assert len(identities) == len(set(identities))
            if role == "heldout":
                assert set(identities) == set(designated["event_uids"])
            elif role == "train":
                assert set(identities) == set(training["event_uids"])
            else:
                assert len(identities) == 24 and set(identities) <= set(
                    training["event_uids"]
                )
            cats = defaultdict(Counter)
            total = Counter()
            for e in r[role]["events"]:
                cats[e["category"]].update(e["counts"])
                total.update(e["counts"])
                cats[e["category"]].update(
                    processed=1,
                    failed=0,
                    event_any_raw=int(e["counts"]["raw_exact_memberships"] > 0),
                    event_any_accepted=int(
                        e["counts"]["accepted_exact_memberships"] > 0
                    ),
                    event_both_raw=int(e["counts"]["raw_exact_memberships"] == 2),
                    event_both_accepted=int(
                        e["counts"]["accepted_exact_memberships"] == 2
                    ),
                )
            assert dict(total) == r[role]["counts"]
            channels, sizes = defaultdict(Counter), defaultdict(Counter)
            positives = []
            compatibility = Counter()
            continuum_components = defaultdict(Counter)
            for event in r[role]["events"]:
                row = cachemap[event["uid"]]
                if row["category"] not in ("charged", "mixed"):
                    continuum_components[row["category"]].update(
                        continuum_membership(row, event)
                    )
                    continue
                for slot in (1, 2):
                    target = set((row["targets"] == slot).nonzero().flatten().tolist())
                    raw = int(target in [set(g) for g in event["raw_groups"]])
                    accepted = int(target in [set(g) for g in event["accepted_groups"]])
                    compatibility["target_trials"] += 1
                    compatibility["cardinality_incompatible"] += int(len(target) < 2)
                    charge = float(row["charge"][list(target)].sum())
                    compatibility["charge_incompatible"] += int(
                        min(abs(charge - q) for q in (-1, 0, 1)) > 1e-5
                    )
                    compatibility["raw_generation_failure"] += int(not raw)
                    compatibility["raw_exact_retention_rejection"] += int(
                        raw and not accepted
                    )
                    compatibility["accepted_after_source_filter"] += int(
                        accepted and not raw
                    )
                    channel = str(
                        int(
                            row["full"][
                                f"b{slot}_reconstructable_channel_ids"
                            ].flatten()[0]
                        )
                    )
                    for counter in (
                        channels[row["category"] + ":retained-channel-" + channel],
                        sizes[str(len(target))],
                    ):
                        counter.update(trials=1, raw=raw, accepted=accepted)
                    if raw or accepted:
                        matching = [
                            i
                            for i, g in enumerate(row["supervision"]["node_sets"])
                            if set(g) == target
                        ]
                        positives.append(
                            {
                                "category": row["category"],
                                "channel": channel,
                                "fsp_size": len(target),
                                "charge": float(row["charge"][list(target)].sum()),
                                "truth_generation_heights": [
                                    int(row["full"]["level_ids"][0, i])
                                    for i in matching
                                ],
                                "raw": raw,
                                "accepted": accepted,
                            }
                        )
            public["roles"][role] = {
                "target_compatibility_and_first_errors": dict(compatibility),
                "continuum_component_membership": {
                    "by_source_type": dict(continuum_components),
                    "scope": "explicit_top_level_retained_composites_source_membership_only_not_physical_tree_or_parton_ancestry",
                    "beam_pool": {
                        "numerator": None,
                        "denominator": None,
                        "reason": "No competing beam hypotheses",
                    },
                },
                "by_retained_channel": dict(channels),
                "by_truth_fsp_size": dict(sizes),
                "positive_cases": positives,
                "counts": dict(total),
                "tag_efficiency": {
                    "physical_B_top1": {
                        "numerator": None,
                        "denominator": None,
                        "reason": "Flat optional memberships have no physical hierarchy; shared physical-tree evaluator not applicable",
                    },
                    "inclusive_flat_membership_top1": {
                        "raw_numerator": total["raw_exact_memberships"],
                        "accepted_numerator": total["accepted_exact_memberships"],
                        "denominator": total["nominal_b_trials"],
                        "unavailable": total["unavailable_membership_trials"],
                    },
                    "event_any_raw": {
                        "numerator": sum(
                            cats.get(k, Counter())["event_any_raw"]
                            for k in ("charged", "mixed")
                        ),
                        "denominator": sum(
                            cats.get(k, Counter())["processed"]
                            for k in ("charged", "mixed")
                        ),
                    },
                    "event_any_accepted": {
                        "numerator": sum(
                            cats.get(k, Counter())["event_any_accepted"]
                            for k in ("charged", "mixed")
                        ),
                        "denominator": sum(
                            cats.get(k, Counter())["processed"]
                            for k in ("charged", "mixed")
                        ),
                    },
                    "event_both_raw": {
                        "numerator": sum(
                            cats.get(k, Counter())["event_both_raw"]
                            for k in ("charged", "mixed")
                        ),
                        "denominator": sum(
                            cats.get(k, Counter())["processed"]
                            for k in ("charged", "mixed")
                        ),
                    },
                    "event_both_accepted": {
                        "numerator": sum(
                            cats.get(k, Counter())["event_both_accepted"]
                            for k in ("charged", "mixed")
                        ),
                        "denominator": sum(
                            cats.get(k, Counter())["processed"]
                            for k in ("charged", "mixed")
                        ),
                    },
                    "continuum_fake_B_slots": {
                        "numerator": sum(
                            cats.get(k, Counter())["accepted_groups"]
                            for k in ("ccbar", "uubar", "ddbar", "ssbar")
                        ),
                        "denominator": 2 * total["continuum_events"],
                    },
                    "continuum_fake_B_events": {
                        "numerator": total["continuum_accepted_events"],
                        "denominator": total["continuum_events"],
                    },
                    "physical_FEI_comparison_ready": False,
                },
                "by_category": dict(cats),
                "relation_ignored_pairs": {"count": None, "status": "NOT_RETAINED"},
            }
        out["arms"][arm] = public
        private["arms"][arm] = {
            "terminal": binding(base / arm / "terminal.json"),
            "checkpoints": terminal["checkpoints"],
            "summary": terminal["summary"],
        }
        records[arm] = r
    assert all(len(x) == 1 for x in orders.values())
    left, right = [records[a]["heldout"]["events"] for a in arms]
    assert [e["uid"] for e in left] == [e["uid"] for e in right]
    assert len(left) == 600 and Counter(e["category"] for e in left) == dict.fromkeys(
        ("charged", "mixed", "ccbar", "uubar", "ddbar", "ssbar"), 100
    )
    effects = paired_effects(left, right)
    out["paired_effects"] = effects
    out["tiny_raw_gate"] = {
        a: records[a]["tiny"]["counts"]["raw_exact_memberships"]
        / records[a]["tiny"]["counts"]["nominal_b_trials"]
        >= 0.95
        for a in arms
    }
    out["uncertainty"] = (
        "One seed and sparse correlated trials; bootstrap degeneracy is not zero population uncertainty. Phase74/75 cohorts are excluded."
    )
    # Exact control replay is meaningful on unchanged train/tiny inputs only.
    parent = Path(c["parent"]) / "campaign-v1/runs/128-existing"
    replay = {}
    for stage in ("tiny", "downstream"):
        old = [
            json.loads(line)
            for line in (parent / f"{stage}-metrics.jsonl").read_text().splitlines()
        ]
        new = out["arms"]["partial_context_on"]["curves"][stage]
        replay[stage] = all(
            all(a[k] == b[k] for k in ("step", "loss", "components", "gradient_norm"))
            for a, b in zip(old, new)
        ) and len(old) == len(new)
    out["control_replay_training_history"] = replay
    state_replay = {}
    for stage in ("tiny", "downstream"):
        old = torch.load(
            parent / f"{stage}-final.pt", weights_only=False, map_location="cpu"
        )
        new = torch.load(
            base / "partial_context_on" / f"{stage}-final.pt",
            weights_only=False,
            map_location="cpu",
        )
        state_replay[stage] = all(
            torch.equal(old[key][name], tensor)
            for key in ("model_state_dict", "decoder_state_dict")
            for name, tensor in new[key].items()
        )
    out["control_replay_model_states"] = state_replay
    assert all(replay.values()) and all(state_replay.values()), (
        "Control replay differs; investigate before inference"
    )
    improvement = (
        effects["accepted_exact_memberships"][
            "paired_collision_stratified_bootstrap95"
        ][0]
        > 0
        and effects["raw_exact_memberships"]["paired_collision_stratified_bootstrap95"][
            0
        ]
        > 0
    )
    background = (
        effects["continuum_accepted_events"]["paired_collision_stratified_bootstrap95"][
            1
        ]
        <= 0
    )
    out["preregistered_endpoint"] = {
        "tiny_raw_pass": all(out["tiny_raw_gate"].values()),
        "heldout_gain": improvement,
        "background_control": background,
        "passed": all(out["tiny_raw_gate"].values()) and improvement and background,
        "primary_in_this_pass": False,
    }
    write(root / "phase76-review-private.json", private)
    write(root / "phase76-review-public.json", out)
    print(
        json.dumps(
            {
                "tiny_gate": out["tiny_raw_gate"],
                "paired_effects": effects,
                "control_replay": replay,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    review(a.root.resolve())
