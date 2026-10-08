#!/usr/bin/env python3
"""Add shared tag/channel accounting to already fitted experimental flat groups.

No fitting or truth-dependent candidate decisions. Neutral groups use a fixed
B0 token (flavour is not predicted); exact hierarchy/PID quality is not claimed.
This is a separate flat-group ontology, never the production level decoder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from diagnose_search_survival import checked, sha, write  # noqa: E402


def run(pilot_root, output):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("CPU Slurm allocation required")
    import torch
    from hypertagging.evaluation.trained_context import load_trained_evaluation_context
    from hypertagging.evaluation.tag_efficiency import evaluate_tag_efficiency_event, summarize_tag_efficiency_events
    from hypertagging.models.direct_membership import DirectMembershipHead
    from hypertagging.reconstruction.hierarchical_inference import project_schema_v4_fsps
    from hypertagging.reconstruction.beam_search import _truth_free_model_view
    from hypertagging.reconstruction.level_rollout import (
        CompositeProposal, _constrained_rollout_model_batch, _with_predicted_leaf_p4, append_composite_proposals,
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = json.loads((pilot_root / "plan.json").read_text())
    receipt = json.loads((pilot_root / "run/receipt.json").read_text())
    if sha(pilot_root / "run/features.pt") != receipt["features_sha256"]:
        raise ValueError("Pilot features changed")
    paths = {k: checked(v) for k, v in plan["inputs"].items()}
    uids = json.loads(paths["cohort"].read_text())["event_uids"]
    context = load_trained_evaluation_context(checkpoint=paths["reconstruction-checkpoint"], data=paths["selection"],
        dataset_index=paths["index"], split="validation", max_events=len(uids), device="cpu",
        diagnostic_allow_external_independent_sample=True, event_selection="explicit_uids", explicit_event_uids=uids)
    cache = torch.load(pilot_root / "run/features.pt", map_location="cpu", weights_only=False)
    features = {r["uid"]: r for r in cache["development"]}
    models, bindings, rows = {}, {}, {}
    for name in ("tiny_memorization", "pilot_384"):
        path = pilot_root / "run" / f"{name}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["contract_sha256"] != sha(pilot_root / "run/contract.json"):
            raise ValueError("Head contract mismatch")
        model = DirectMembershipHead(saved["input_dim"])
        model.load_state_dict(saved["state_dict"], strict=True)
        models[name] = model.eval()
        bindings[name] = sha(path)
        rows[name] = []
    for index, event in enumerate(context.events):
        truth = context.collated_event_batch(index)
        projection = project_schema_v4_fsps(truth)
        input_batch = _constrained_rollout_model_batch(projection.batch, target_level=1, policy=context.constraint_policy)
        with torch.inference_mode():
            encoded = context.model(_truth_free_model_view(input_batch), target_level=1,
                pid_kinematics_mode_override="soft_expectation", pid_temperature_override=.5)
            state = _with_predicted_leaf_p4(projection.batch, encoded.leaf_pid_logits, mode="hard", temperature=.5)
        row = features[event.event_uid]
        torch.testing.assert_close(row["features"], encoded.node_embeddings[0], rtol=0, atol=0)
        for name, model in models.items():
            with torch.inference_mode():
                logits, objects = model(row["features"][None], torch.ones(1, len(row["targets"]), dtype=torch.bool))
                probabilities = logits[0].softmax(-1)
                assignment = probabilities.argmax(-1)
            proposals, used = [], set()
            for slot in sorted(range(2), key=lambda s: -float(objects[0, s])):
                presence = float(objects[0, slot].sigmoid())
                if presence < .5:
                    continue
                positions = (assignment == slot + 1).nonzero().flatten().tolist()
                daughters, local = [], set()
                for position in sorted(positions, key=lambda p: -float(probabilities[p, slot + 1])):
                    sources = set(row["sources"][position].nonzero().flatten().tolist())
                    if sources and not sources & (used | local):
                        daughters.append(position)
                        local.update(sources)
                charge = float(row["charge"][daughters].sum()) if daughters else 0.
                if len(daughters) < 2 or min(abs(charge - q) for q in (-1, 0, 1)) > 1e-5:
                    continue
                # Fixed charge-to-token mapping; no true B type/flavour is read.
                token = 22 if charge > .5 else 39 if charge < -.5 else 21
                proposals.append(CompositeProposal(slot, token, tuple(daughters), presence, presence))
                used.update(local)
            predicted, _ = append_composite_proposals(state, proposals, target_level=1) if proposals else (dict(state), [])
            predicted["evaluation_leaf_source_keys"] = projection.evaluation_leaf_source_keys.clone()
            rows[name].append(evaluate_tag_efficiency_event(truth, [predicted],
                source_category=event.source_category, event=event))
    result = {"version": "flat-membership-shared-tag-evaluation-v1", "events_per_head": 60,
        "truth_used_for_inference": False, "head_sha256": bindings,
        "feature_cache_sha256": receipt["features_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tag_efficiency": {name: summarize_tag_efficiency_events(records) for name, records in rows.items()},
        "ontology": "experimental_flat_optional_B_groups_neutral_flavour_fixed_B0",
        "limitations": ["no_internal_hierarchy_prediction", "neutral_B_flavour_not_predicted", "development_reuse",
                        "not_production_level_ontology", "no_physical_FEI_claim"]}
    output.parent.mkdir(exist_ok=True, parents=True)
    write(output, result)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.pilot_root.resolve(), args.output.resolve())
