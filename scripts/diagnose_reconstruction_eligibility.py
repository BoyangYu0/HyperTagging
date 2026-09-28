"""Read-only development census of legal scheduled-sampling target support."""

from pathlib import Path
import argparse, hashlib, json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from hypertagging.evaluation.trained_context import load_trained_evaluation_context
from hypertagging.reconstruction.level_rollout import (
    RolloutConfig,
    evaluation_reference_rollout,
    cached_context_for_level,
)
from hypertagging.training.scheduled_sampling import aligned_level_targets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data", "dataset-index", "cohort", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--events", type=int, default=16)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cohort = json.loads(args.cohort.read_text())
    uids = cohort["checkpoint_selection_event_uids"][: args.events]
    assert (
        args.events > 0
        and len(uids) == args.events
        and not set(uids) & set(cohort["event_uids"])
    )
    torch.set_num_threads(2)
    context = load_trained_evaluation_context(
        checkpoint=args.checkpoint,
        data=args.data,
        dataset_index=args.dataset_index,
        split="validation",
        max_events=args.events,
        device="cpu",
        event_selection="explicit_uids",
        explicit_event_uids=uids,
    )
    cfg = context.checkpoint["config"]
    results = []
    with torch.inference_mode():
        for index, event in enumerate(context.events):
            truth = context.collated_event_batch(index)
            rollout = evaluation_reference_rollout(
                context.model,
                truth,
                config=RolloutConfig(
                    max_level=6,
                        constraint_policy=context.constraint_policy,
                    rollout_pid_kinematics_mode=context.rollout_pid_kinematics_mode,
                    rollout_pid_temperature=cfg.get("rollout_pid_temperature", 0.5),
                    object_threshold=cfg.get("rollout_object_threshold", 0.6),
                    pointer_threshold=cfg.get("rollout_pointer_threshold", 0.35),
                    use_learned_confidence=bool(
                        context.checkpoint.get("confidence_head_trained")
                    ),
                    confidence_trained=bool(
                        context.checkpoint.get("confidence_head_trained")
                    ),
                    continue_through_empty_levels=True,
                ),
            )
            levels = []
            for level in range(1, 7):
                state = cached_context_for_level(rollout, level)
                old = aligned_level_targets(truth, state, target_level=level)
                legal = context.constraint_policy.forest_pointer_validity_mask(
                    state, level
                )
                new = aligned_level_targets(
                    truth, state, target_level=level, pointer_eligibility=legal
                )
                levels.append(
                    {
                        "level": level,
                        "truth_targets": old.truth_target_count,
                        "historical_representable": old.representable_count,
                        "legal_representable": new.representable_count,
                    }
                )
            results.append(
                {
                    "event_uid_sha256": hashlib.sha256(
                        event.event_uid.encode()
                    ).hexdigest(),
                    "levels": levels,
                }
            )
            print(
                json.dumps({"completed": index + 1, "events": args.events}), flush=True
            )
    totals = {
        k: sum(row[k] for event in results for row in event["levels"])
        for k in ("truth_targets", "historical_representable", "legal_representable")
    }
    output = {
        "version": "development-eligibility-census-v1",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "event_count": len(results),
        "strict_overlap": 0,
        "sealed_test_accessed": False,
        "totals": totals,
        "events": results,
        "interpretation": "Post-generation truth audit only. Legal support is not proposal survival or a causal estimate of training benefit.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
