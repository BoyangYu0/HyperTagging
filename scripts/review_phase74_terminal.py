"""Authenticate terminal development evidence and join truth only after predictions."""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import sha, binding, write, read  # noqa: E402
from scripts.run_phase74_development import verify_bindings  # noqa: E402
from scripts.summarize_phase74_development import summarize  # noqa: E402


def review(parent, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU scheduler required")
    import torch
    import numpy as np
    from hypertagging.training.capacity_development import (
        fresh_model,
        fresh_decoder,
        assembly_loss,
    )

    torch.set_num_threads(1)
    c = json.loads((parent / "campaign-v1/contract.json").read_text())
    verify_bindings(c, Path(c["source_root"]))
    cache = torch.load(c["cache"]["path"], weights_only=False, map_location="cpu")
    cachemap = {r["uid"]: r for r in cache["train"] + cache["development"]}
    development = read(
        json.loads((parent / "data-admission.json").read_text())["development"]
    )
    train = read(json.loads((parent / "data-admission.json").read_text())["training"])
    exclusions = read(
        json.loads((parent / "data-admission.json").read_text())["exclusion_union"]
    )
    assert not set(development["event_uids"]) & (
        set(train["event_uids"]) | set(exclusions["event_uids"])
    )
    private = {
        "arms": {},
        "source_sha": c["source_sha"],
        "contract": binding(parent / "campaign-v1/contract.json"),
    }
    public = {
        "version": "phase74-terminal-review-v1",
        "stage": "development",
        "arms": {},
        "factorial": summarize(parent / "campaign-v1/runs"),
        "primary_eligible": False,
        "sealed_test_access": False,
    }
    orders = defaultdict(set)
    for arm in c["arms"]:
        root = parent / "campaign-v1/runs" / arm
        terminal = json.loads((root / "terminal.json").read_text())
        record = read(terminal["summary"])
        receipt = json.loads((parent / "campaign-v1" / f"{arm}-exit.json").read_text())
        assert terminal["status"] == "COMPLETED" and receipt["exit_status"] == 0
        assert record["contract_sha256"] == sha(parent / "campaign-v1/contract.json")
        stages = {}
        history = {}
        for stage, count in [
            ("pretraining", 1000),
            ("tiny", 1000),
            ("downstream", 1500),
        ]:
            path = root / f"{stage}-final.pt"
            item = next(x for x in terminal["checkpoints"] if x["path"] == str(path))
            assert sha(path) == item["sha256"]
            ck = torch.load(path, weights_only=False, map_location="cpu")
            assert (
                ck["step"] == count
                and ck["source_sha"] == c["source_sha"]
                and ck["data_cache"] == c["cache"]
                and ck["settings"] == c["settings"]
            )
            orders[stage].add(ck["data_order_sha256"])
            rows = [
                json.loads(line)
                for line in (root / f"{stage}-metrics.jsonl").read_text().splitlines()
            ]
            assert [r["step"] for r in rows] == list(range(1, count + 1))
            assert all(
                np.isfinite(r["loss"]) and np.isfinite(r["gradient_norm"]) for r in rows
            )
            history[stage] = rows
            stages[stage] = {
                "checkpoint": binding(path),
                "updates": count,
                "presentations": count * 8,
                "data_order_sha256": ck["data_order_sha256"],
            }
        pub = {
            "compute": record["compute"],
            "roles": {},
            "histories": history,
            "unavailable": record["heldout"]["unavailable"],
        }
        proven = []
        for role in ("tiny", "train", "heldout"):
            ev = record[role]["events"]
            uids = [r["uid"] for r in ev]
            assert len(uids) == len(set(uids))
            if role == "heldout":
                assert set(uids) == set(development["event_uids"])
            if role == "train":
                assert set(uids) == set(train["event_uids"])
            aggregate = Counter()
            cats = defaultdict(Counter)
            channels = defaultdict(Counter)
            sizes = defaultdict(Counter)
            failures = Counter()
            for e in ev:
                row = cachemap[e["uid"]]
                assert e["category"] == row["category"]
                aggregate.update(e["counts"])
                cats[e["category"]].update(e["counts"])
                raw = [frozenset(g) for g in e["raw_groups"]]
                accepted = [frozenset(g) for g in e["accepted_groups"]]
                used = set()
                for g in accepted:
                    for i in g:
                        sources = set(row["sources"][i].nonzero().flatten().tolist())
                        assert not sources & used
                        used |= sources
                any_raw = any_accepted = 0
                if row["category"] in ("charged", "mixed"):
                    for slot in (1, 2):
                        target = frozenset(
                            (row["targets"] == slot).nonzero().flatten().tolist()
                        )
                        rawhit = int(target in raw)
                        hit = int(target in accepted)
                        any_raw += rawhit
                        any_accepted += hit
                        full = row["full"]
                        channel = int(
                            full[f"b{slot}_reconstructable_channel_ids"].flatten()[0]
                        )
                        label = f"{row['category']}:retained-channel-{channel}"
                        for counter in (channels[label], sizes[str(len(target))]):
                            counter.update(trials=1, raw=rawhit, accepted=hit)
                        overlap = max(
                            (
                                len(target & g) / len(target | g) if target | g else 0
                                for g in raw
                            ),
                            default=0,
                        )
                        failures[
                            "exact"
                            if rawhit
                            else "near_jaccard_ge_0.8"
                            if overlap >= 0.8
                            else "far_jaccard_lt_0.8"
                        ] += 1
                        if rawhit or hit:
                            matching = [
                                i
                                for i, g in enumerate(row["supervision"]["node_sets"])
                                if g == target
                            ]
                            entry = {
                                "role": role,
                                "uid": e["uid"],
                                "category": row["category"],
                                "slot": slot,
                                "channel": channel,
                                "fsp_size": len(target),
                                "raw": rawhit,
                                "accepted": hit,
                                "fsp_positions": sorted(target),
                                "matching_truth_nodes": matching,
                                "truth_generation_heights": [
                                    int(full["level_ids"][0, i]) for i in matching
                                ],
                                "charge": float(row["charge"][list(target)].sum()),
                            }
                            proven.append(entry)
                    assert (
                        any_raw == e["counts"]["raw_exact_memberships"]
                        and any_accepted == e["counts"]["accepted_exact_memberships"]
                    )
                cats[e["category"]].update(
                    event_any_raw=int(any_raw > 0),
                    event_any_accepted=int(any_accepted > 0),
                    event_both_raw=int(any_raw == 2),
                    event_both_accepted=int(any_accepted == 2),
                    processed=1,
                    failed=0,
                )
            assert dict(aggregate) == record[role]["counts"]
            pub["roles"][role] = {
                "counts": dict(aggregate),
                "by_category": dict(cats),
                "by_retained_channel": dict(channels),
                "by_truth_fsp_size": dict(sizes),
                "membership_error_bins": dict(failures),
                "nominal_distinct_collisions": len(ev),
                "failure_count": 0,
                "physical_tag_efficiency": {
                    "numerator": None,
                    "denominator": None,
                    "status": "UNAVAILABLE_NO_PHYSICAL_TREE",
                },
                "continuum_component_recovery": {
                    "numerator": None,
                    "denominator": None,
                    "status": "UNAVAILABLE_FLAT_HEAD_ONLY_PREDICTS_OPTIONAL_B_GROUPS",
                },
            }
        pub["positive_cases"] = [
            {
                k: v
                for k, v in e.items()
                if k not in ("uid", "fsp_positions", "matching_truth_nodes")
            }
            for e in proven
        ]
        # Mechanism diagnosis on first16 UID-sorted original train events/category,
        # not heldout tuning. Compare encoder gradients at fixed final checkpoints.
        ck = torch.load(
            root / "downstream-final.pt", weights_only=False, map_location="cpu"
        )
        width = int(arm.split("-")[0])
        model = fresh_model(width, cache["runtime_normalizer"])
        decoder = fresh_decoder()
        model.load_state_dict(ck["model_state_dict"])
        decoder.load_state_dict(ck["decoder_state_dict"])
        model.eval()
        decoder.eval()
        diagnostic = []
        for cat in ("charged", "mixed"):
            for row in [r for r in cache["train"] if r["category"] == cat][:16]:
                _, parts, support = assembly_loss(model, decoder, row)
                params = tuple(model.encoder.parameters())
                a = torch.autograd.grad(
                    parts["membership"], params, retain_graph=True, allow_unused=True
                )
                z = torch.autograd.grad(
                    parts["within_b_relation"], params, allow_unused=True
                )
                dot = sum(
                    float((x * y).sum())
                    for x, y in zip(a, z)
                    if x is not None and y is not None
                )
                an = sum(float((x * x).sum()) for x in a if x is not None) ** 0.5
                zn = sum(float((x * x).sum()) for x in z if x is not None) ** 0.5
                diagnostic.append(
                    {
                        "uid": row["uid"],
                        "category": cat,
                        "dot": dot,
                        "membership_norm": an,
                        "relation_norm": zn,
                        "cosine": dot / max(an * zn, 1e-20),
                        "membership_loss": float(parts["membership"].detach()),
                        "relation_loss": float(parts["within_b_relation"].detach()),
                    }
                )
        pub["gradient_diagnostic"] = {
            "training_events": 32,
            "negative_dot_events": sum(r["dot"] < 0 for r in diagnostic),
            "mean_cosine": float(np.mean([r["cosine"] for r in diagnostic])),
            "median_relation_to_membership_norm": float(
                np.median(
                    [
                        r["relation_norm"] / max(r["membership_norm"], 1e-20)
                        for r in diagnostic
                    ]
                )
            ),
            "sampling": "first16_UID_sorted_train_per_charged_and_mixed_fixed_before_diagnostic",
            "not_causal_evidence": True,
        }
        private["arms"][arm] = {
            "terminal": binding(root / "terminal.json"),
            "summary": terminal["summary"],
            "stages": stages,
            "positive_cases": proven,
            "gradient_rows": diagnostic,
        }
        public["arms"][arm] = pub
        print(arm, pub["gradient_diagnostic"], flush=True)
    assert all(len(x) == 1 for x in orders.values())
    private["matched_orders_verified"] = True
    write(output / "terminal-audit-private.json", private)
    write(output / "phase74-review-public.json", public)
    print("TERMINAL_AUDIT_PASS", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    review(a.parent, a.output)
