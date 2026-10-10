"""Conditional TRAIN-role pair-message connection contrast; no automatic successor."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.prepare_phase74_development_data import binding, read, write  # noqa:E402
from scripts.phase76_development import guarded, initial, settings  # noqa:E402
from scripts.phase81_pair_supervision import initial_state_digest  # noqa:E402
from scripts.phase84_exposure import (  # noqa:E402
    load_cache,
    nested,
    exposure,
    REPLICATION_INITIAL_DIGEST,
)
from scripts.run_phase74_development import fit, save, verify_bindings  # noqa:E402

ARMS = ("connection_off", "connection_on")
ADDITION_SEED = 20261010085
MILESTONES = (1500, 6000)
RESOURCES = dict(cpus=2, memory_gib=32, hours=8, gpus=0, requeue=False)
ORIGINAL_REVIEW_SHA256 = (
    "10396dad02625441645f417b584c12b47712486975f74252b44e36a90d57163e"
)
REPLICATION_CONTRACT_SHA256 = (
    "53c4daaed933f53b34d78e9843ba2ae8a5d0b5a15665b3998c78dc1249b39a51"
)
REPLICATION_SOURCE_SHA = "72de398ab5b5cc4477d73f692b175a830ea7af61"


def study_settings():
    return {**settings(), "downstream_updates": 6000}


def initialize(pretrain, cache, arm):
    import torch
    from hypertagging.models.pair_conditioned_membership import (
        PairConditionedMembershipDecoder,
    )

    if arm not in ARMS:
        raise ValueError("Unknown connection arm")
    model, native, checkpoint = initial(pretrain, cache)
    base_digest = initial_state_digest(model, native)
    if base_digest != REPLICATION_INITIAL_DIGEST:
        raise ValueError("Historical base initializer changed")
    # The base initializer and global RNG are exactly the historical ones.
    # Additional parameters are matched, not fitted probe transfer.
    with torch.random.fork_rng(devices=[]):
        decoder = PairConditionedMembershipDecoder(
            connection_enabled=arm == "connection_on", addition_seed=ADDITION_SEED
        )
    missing = decoder.load_state_dict(native.state_dict(), strict=False)
    if missing.unexpected_keys or any(
        not k.startswith(("semantic_edges.", "message.")) for k in missing.missing_keys
    ):
        raise ValueError("Native decoder transfer mismatch")
    if any(
        not torch.equal(value, decoder.state_dict()[key])
        for key, value in native.state_dict().items()
    ):
        raise ValueError("Native decoder initialization changed")
    return model, decoder, checkpoint, base_digest


def validate(c):
    if (
        c.get("kind") != "phase85_pair_connection"
        or c.get("arms") != list(ARMS)
        or c.get("pool_size") != 384
        or c.get("settings") != study_settings()
        or c.get("milestones") != list(MILESTONES)
        or c.get("addition_seed") != ADDITION_SEED
        or c.get("resources") != RESOURCES
        or c.get("semantic_pair_coefficient") != 1.0
        or c.get("heldout_events") != 0
        or c.get("sealed_test_access") is not False
        or c.get("automatic_successor") is not False
        or c.get("maximum_scientific_jobs") != 2
    ):
        raise ValueError("Scientific contrast/controls changed")
    if (
        len(c["exposure_prerequisites"]) != 2
        or c["exposure_prerequisites"][0]["sha256"] != ORIGINAL_REVIEW_SHA256
    ):
        raise ValueError("Exact original384 review required first")
    for index, item in enumerate(c["exposure_prerequisites"]):
        receipt = read(item)
        if (
            receipt.get("status") != "PASS"
            or receipt.get("pool") != 384
            or receipt.get("review_scheduler") != "COMPLETED|0:0"
            or any(
                receipt.get("gate", {}).get(key) is not True
                for key in ("absolute_trainable", "exposure_gain", "fixed6000_eligible")
            )
        ):
            raise ValueError("Replicated TRAIN learnability prerequisite absent")
        if index == 1 and (
            receipt.get("source_sha") != REPLICATION_SOURCE_SHA
            or REPLICATION_CONTRACT_SHA256
            not in {x["sha256"] for x in receipt.get("bindings", [])}
        ):
            raise ValueError("Exact reserved sampling-seed replication required")
    if (
        len(c["exposure_prerequisites"]) != 2
        or len({x["sha256"] for x in c["exposure_prerequisites"]}) != 2
    ):
        raise ValueError("Original and replication reviews required")


def parameter_gradients(module):
    return sum(
        float(p.grad.detach().abs().sum())
        for p in module.parameters()
        if p.grad is not None
    )


def smoke(c, out):
    """Discarded four-step fit per arm, real gradients and worst-row capacity."""
    import torch
    from hypertagging.training.capacity_development import (
        detector_features,
        membership_loss,
    )
    from hypertagging.models.pair_conditioned_membership import semantic_pair_loss
    from scripts.diagnose_phase80_optimizer_steps import audit

    cache, admission = load_cache(Path(c["parent"]))
    rows = nested(cache["train"], 384)
    worst = max(cache["train"], key=lambda r: len(r["targets"]))
    probe = [worst] + [r for r in rows if r["uid"] != worst["uid"]][:7]
    gradient_row = next(
        r
        for r in rows
        if all(bool((r["targets"] == label).any()) for label in (0, 1, 2))
    )
    results = {}
    for arm in ARMS:
        model, decoder, checkpoint, base_digest = initialize(
            Path(c["pretrain"]), cache, arm
        )
        initial_digest = initial_state_digest(model, decoder)
        initial_rng = hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()
        ).hexdigest()
        folder = out / arm
        folder.mkdir()
        compute = Counter()
        started = time.monotonic()
        history = fit(
            model,
            decoder,
            probe,
            stage="downstream",
            objective="existing",
            settings={**study_settings(), "downstream_updates": 4},
            output=folder,
            compute=compute,
            cache_binding=admission["cache"],
            source_sha=c["source_sha"],
            semantic_pair_supervision=True,
        )
        fit_seconds = time.monotonic() - started
        sample = exposure(probe, 4, study_settings()["seed"])
        if (
            sample["per_uid"][worst["uid"]] == 0
            or sample["sequence_sha256"] != history["data_order_sha256"]
        ):
            raise ValueError("Worst-cardinality smoke row was not exercised")
        # The zero output projection initially blocks membership-to-edge gradients.
        # Test functional use after the finite discarded updates, separately from CE.
        model.zero_grad(set_to_none=True)
        decoder.zero_grad(set_to_none=True)
        h, _, _ = detector_features(model, gradient_row["detector"])
        result = decoder(h, gradient_row["sources"])
        membership_loss(result, gradient_row["targets"]).backward()
        member_grad = {
            "edge": parameter_gradients(decoder.semantic_edges),
            "message": parameter_gradients(decoder.message),
            "encoder": parameter_gradients(model.encoder),
        }
        if not all(math.isfinite(x) for x in member_grad.values()):
            raise FloatingPointError("Nonfinite membership gradient")
        if (
            member_grad["encoder"] <= 0
            or (
                arm == "connection_on"
                and min(member_grad["edge"], member_grad["message"]) <= 0
            )
            or (
                arm == "connection_off"
                and (member_grad["edge"] != 0 or member_grad["message"] != 0)
            )
        ):
            raise ValueError(
                "Connection gradient branch does not match declared factor"
            )
        model.zero_grad(set_to_none=True)
        decoder.zero_grad(set_to_none=True)
        h, _, _ = detector_features(model, gradient_row["detector"])
        result = decoder(h, gradient_row["sources"])
        semantic, support = semantic_pair_loss(
            result["edge_logits"], result["edge_pairs"], gradient_row["targets"]
        )
        if not all(support[key] > 0 for key in ("same_b", "cross_b", "background")):
            raise ValueError("Real semantic class gradient support absent")
        semantic.backward()
        auxiliary_gradient = parameter_gradients(decoder.semantic_edges)
        if not math.isfinite(auxiliary_gradient) or auxiliary_gradient <= 0:
            raise ValueError("Semantic edge supervision gradient absent")
        started = time.monotonic()
        audit(model, decoder, probe, folder / "model-only.jsonl.gz")
        evaluation_seconds_per_event = (time.monotonic() - started) / len(probe)
        forecast = (
            2 * fit_seconds / 4 * 6000
            + 2 * evaluation_seconds_per_event * 384 * 2
            + 1200
        )
        if not math.isfinite(forecast) or forecast > 7 * 3600:
            raise ValueError("Runtime forecast exceeds unchanged allocation")
        results[arm] = dict(
            initial_digest=initial_digest,
            initial_torch_rng_sha256=initial_rng,
            base_digest=base_digest,
            initial_checkpoint=checkpoint,
            history=history,
            compute=dict(compute),
            discarded_presentations=32,
            actual_exposure=sample,
            gradient_row_uid=gradient_row["uid"],
            support=support,
            membership_only_gradients=member_grad,
            auxiliary_only_edge_gradient=auxiliary_gradient,
            fit_seconds=fit_seconds,
            evaluation_seconds_per_event=evaluation_seconds_per_event,
            forecast_seconds=forecast,
        )
    if len({x["initial_digest"] for x in results.values()}) != 1:
        raise ValueError("Arm initialization mismatch")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 24 * 1024**2:
        raise ValueError("Memory admission exceeded")
    write(
        out / "admission.json",
        dict(
            status="PASS",
            source_sha=c["source_sha"],
            contract=binding(Path(c["contract_path"])),
            job_id=os.environ["SLURM_JOB_ID"],
            results=results,
            peak_rss_kib=rss,
            probe_uids=[r["uid"] for r in probe],
            isolation=admission["current_isolation"],
            selection=[r["uid"] for r in rows],
            heldout_events=0,
            all_fitting_discarded=True,
        ),
    )


def run(c, arm, out):
    import torch
    from scripts.diagnose_phase80_optimizer_steps import audit

    admitted = read(c["admission"])
    if admitted.get("status") != "PASS" or admitted["source_sha"] != c["source_sha"]:
        raise ValueError("Stale or absent runtime admission")
    cache, a = load_cache(Path(c["parent"]))
    rows = nested(cache["train"], 384)
    if [r["uid"] for r in rows] != admitted["selection"]:
        raise ValueError("Training selection changed")
    model, decoder, cp, base_digest = initialize(Path(c["pretrain"]), cache, arm)
    digest = initial_state_digest(model, decoder)
    if digest != admitted["results"][arm]["initial_digest"]:
        raise ValueError("Initialization changed after admission")
    if (
        hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        != admitted["results"][arm]["initial_torch_rng_sha256"]
    ):
        raise ValueError("Initial Torch RNG changed after admission")
    write(
        out / "startup.json",
        dict(
            job_id=os.environ["SLURM_JOB_ID"],
            arm=arm,
            source_sha=c["source_sha"],
            contract=binding(Path(c["contract_path"])),
            initial_checkpoint=cp,
            initial_digest=digest,
            base_digest=base_digest,
            uid_order_sha256=hashlib.sha256(
                "\n".join(r["uid"] for r in rows).encode()
            ).hexdigest(),
            initial_torch_rng_sha256=hashlib.sha256(
                torch.get_rng_state().numpy().tobytes()
            ).hexdigest(),
            resources=RESOURCES,
            connection_enabled=decoder.connection_enabled,
        ),
    )
    write(out / "isolation.json", a["current_isolation"])
    start, cpu = time.monotonic(), time.process_time()
    compute, snapshots = Counter(), []
    stop_reason = "fixed6000"

    def callback(**kw):
        nonlocal stop_reason
        step = kw["step"]
        if (
            time.monotonic() - start > 7.25 * 3600
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            stop_reason = "runtime_or_memory_guard"
            return True
        if step not in MILESTONES:
            return False
        rngstate, modes = torch.get_rng_state(), (model.training, decoder.training)
        save(
            out / f"step-{step}.pt",
            dict(
                model_state_dict=model.state_dict(),
                decoder_state_dict=decoder.state_dict(),
                optimizer_state_dict=kw["optimizer"].state_dict(),
                scheduler_state_dict=None,
                scaler_state_dict=None,
                torch_rng_state=rngstate,
                sampling_rng_state=kw["rng"].getstate(),
                step=step,
                source_sha=c["source_sha"],
                contract=binding(Path(c["contract_path"])),
                data_order_sha256=kw["sequence_sha256"],
                normalizer=model.runtime_feature_normalizer,
                settings=study_settings(),
                architecture=dict(context=128, hyperbolic=32, head=256, depth=4),
                connection_enabled=decoder.connection_enabled,
                semantic_pair_supervision=True,
                resume_authorized=False,
            ),
        )
        result = audit(model, decoder, rows, out / f"step-{step}-model-only.jsonl.gz")
        if not all(math.isfinite(v) for v in result["mean_risk"].values()):
            raise FloatingPointError("Nonfinite native audit risk")
        write(out / f"step-{step}-evaluation.json", result)
        sample = exposure(rows, step, study_settings()["seed"])
        if sample["sequence_sha256"] != kw["sequence_sha256"]:
            raise ValueError("Sampling order mismatch")
        write(out / f"step-{step}-exposure-private.json", sample)
        snapshots.append(
            dict(
                step=step,
                counts=result["native"]["counts"],
                checkpoint=binding(out / f"step-{step}.pt"),
                evaluation=binding(out / f"step-{step}-evaluation.json"),
            )
        )
        compute["evaluation_event_views"] += len(rows)
        compute["evaluation_encoder_passes"] += 2 * len(rows)
        compute["evaluation_node_squared_proxy"] += sum(
            len(r["targets"]) ** 2 for r in rows
        )
        torch.set_rng_state(rngstate)
        model.train(modes[0])
        decoder.train(modes[1])
        if (
            time.monotonic() - start > 7.25 * 3600
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 26 * 1024**2
        ):
            stop_reason = "runtime_or_memory_guard"
            return True
        return False

    history = fit(
        model,
        decoder,
        rows,
        stage="downstream",
        objective="existing",
        settings=study_settings(),
        output=out,
        compute=compute,
        cache_binding=a["cache"],
        source_sha=c["source_sha"],
        semantic_pair_supervision=True,
        step_callback=callback,
    )
    write(
        out / "summary.json",
        dict(
            status="COMPLETED",
            arm=arm,
            stop_reason=stop_reason,
            fixed_final_eligible=stop_reason == "fixed6000"
            and history["updates"] == 6000,
            source_sha=c["source_sha"],
            history=history,
            snapshots=snapshots,
            pool_size=len(rows),
            compute={
                **compute,
                "wall_seconds": time.monotonic() - start,
                "cpu_seconds": time.process_time() - cpu,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "parameters": sum(p.numel() for p in model.encoder.parameters())
                + sum(p.numel() for p in decoder.parameters()),
            },
            heldout_events=0,
            physical_tree_beam_p4_metrics="UNAVAILABLE_FLAT_OPTIONAL_B",
            audit_risk_scope="Historical native membership+within-B risk; semantic training loss separately logged, not included in native risk.",
            final_candidate_rule="Fixed6000 only; no best checkpoint/seed or automatic successor.",
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("smoke", "run"))
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    guarded()
    c = json.loads(args.contract.read_text())
    validate(c)
    verify_bindings(c, ROOT)
    if (
        c["contract_path"] != str(args.contract.resolve())
        or c["source_root"] != str(ROOT)
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        != c["source_sha"]
    ):
        raise ValueError("Source/contract location changed")
    if args.action == "run" and (
        args.arm not in ARMS or str(args.output.resolve()) != c["outputs"][args.arm]
    ):
        raise ValueError("Unbound arm/output")
    if args.action == "smoke" and str(args.output.resolve()) != c["smoke_output"]:
        raise ValueError("Unbound smoke output")
    args.output.mkdir(exist_ok=False)
    if args.action == "smoke":
        smoke(c, args.output)
    else:
        run(c, args.arm, args.output)
    write(
        args.output / "terminal.json",
        dict(
            status="COMPLETED",
            bindings=[binding(x) for x in sorted(args.output.iterdir()) if x.is_file()],
        ),
    )


if __name__ == "__main__":
    main()
