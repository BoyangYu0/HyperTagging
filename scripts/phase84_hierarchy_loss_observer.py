"""Diagnostic-only observation of native hierarchy losses; no optimizer changes.

Wrappers return original objects unchanged and are restored even on failure.
The caller supplies the checkpoint's actual trainable parameter groups. Reports
are local sensitivities under the supplied context/sampling policy, not replay
of historical training exposure. Tensor captures are post-hoc diagnostic data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from threading import RLock
import time

import torch

_LOCK = RLock()


def detached(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value):
        return {
            field.name: detached(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(detached(v) for v in value)
    return value


@contextmanager
def capture_native_losses(trainer, *, capture_tensors=True):
    """Temporarily observe the two exact native module functions (single process)."""
    with _LOCK:
        original_loss = trainer.level_reconstruction_loss
        original_mean = trainer._normalized_weighted_mean
        calls, normalizations = [], []

        def loss_wrapper(*args, **kwargs):
            output = original_loss(*args, **kwargs)
            record = {
                "target_level": kwargs["target_level"],
                "target_override": kwargs.get("target_override") is not None,
                "grad_enabled": torch.is_grad_enabled(),
                "output": output,
                # Original graph references for cheap matched-query logit
                # sensitivities; never serialized as inference input.
                "pointer_output": args[0],
                "loss_batch": args[1],
                "target_override_values": kwargs.get("target_override"),
                "controls": {
                    k: detached(v)
                    for k, v in kwargs.items()
                    if k not in ("target_override", "constraint_policy")
                },
            }
            if capture_tensors:
                record["posthoc"] = {
                    "pointer_output": detached(args[0]),
                    "input_batch": detached(args[1]),
                    "target_override": detached(kwargs.get("target_override")),
                    "matches": detached(output.matches),
                }
            calls.append(record)
            return output

        def mean_wrapper(losses, weights):
            output = original_mean(losses, weights)
            normalizations.append(
                {"losses": tuple(losses), "weights": tuple(weights), "output": output}
            )
            return output

        trainer.level_reconstruction_loss = loss_wrapper
        trainer._normalized_weighted_mean = mean_wrapper
        try:
            yield calls, normalizations
        finally:
            trainer.level_reconstruction_loss = original_loss
            trainer._normalized_weighted_mean = original_mean


def scalar_derivative(output, value):
    if not output.requires_grad or not value.requires_grad:
        return None
    result = torch.autograd.grad(output, value, allow_unused=True, retain_graph=True)[0]
    if result is None:
        return None
    if result.numel() != 1 or not torch.isfinite(result):
        raise ValueError("Observer requires finite scalar loss coefficients")
    return float(result.detach())


def gradients(value, parameters):
    if not value.requires_grad:
        return tuple(None for _ in parameters)
    result = torch.autograd.grad(
        value, parameters, retain_graph=True, allow_unused=True
    )
    if any(g is not None and not torch.isfinite(g).all() for g in result):
        raise FloatingPointError("Nonfinite observed native gradient")
    return result


def gradient_summary(values, reference):
    norm2 = sum(
        float(g.detach().double().square().sum()) for g in values if g is not None
    )
    reference2 = sum(
        float(g.detach().double().square().sum()) for g in reference if g is not None
    )
    dot = sum(
        float((g.detach().double() * r.detach().double()).sum())
        for g, r in zip(values, reference)
        if g is not None and r is not None
    )
    return {
        "l2_norm": norm2**0.5,
        "dot_with_native_gradient": dot,
        "cosine_with_native_gradient": dot / (norm2 * reference2) ** 0.5
        if norm2 and reference2
        else None,
        "parameters_with_gradient": sum(g is not None for g in values),
        "interpretation": "Norms are not additive; zero/unused is not proof of adequate supervision",
    }


def analyze_native_capture(
    native_result, calls, normalizations, parameter_groups, *, rtol=1e-4, atol=1e-6
):
    """Differentiate the original graph; never populate .grad or step parameters."""
    reconstruction, leaf_pid = native_result[:2]
    total = reconstruction + leaf_pid
    if total.numel() != 1 or not torch.isfinite(total):
        raise FloatingPointError("Nonfinite native total")
    names, parameters, slices = [], [], {}
    seen = set()
    for group, named_parameters in parameter_groups.items():
        start = len(parameters)
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                raise ValueError("Observer group includes frozen parameter: " + name)
            if id(parameter) in seen:
                raise ValueError("Overlapping observer parameter groups")
            seen.add(id(parameter))
            names.append(name)
            parameters.append(parameter)
        slices[group] = slice(start, len(parameters))
    if not parameters:
        raise ValueError("No configured trainable parameters")
    reference = gradients(total, parameters)

    def describe(value):
        grad = gradients(value, parameters)
        return {
            group: gradient_summary(grad[positions], reference[positions])
            for group, positions in slices.items()
        }

    native_weighted_sum = total * 0
    loss_contributions = []
    reports = []
    for index, call in enumerate(calls):
        loss = call["output"]
        coefficient = scalar_derivative(total, loss.total)
        normalization_membership = []
        for norm_index, norm in enumerate(normalizations):
            sensitivity = scalar_derivative(norm["output"], loss.total)
            if sensitivity is not None:
                normalization_membership.append(
                    {
                        "index": norm_index,
                        "coefficient": sensitivity,
                        "direct_scalar_in_list": any(
                            x is loss.total for x in norm["losses"]
                        ),
                    }
                )
        if not call["grad_enabled"] or not loss.total.requires_grad:
            role = "detached_readout"
        elif coefficient is None:
            role = "unused_differentiable_call"
        elif normalization_membership and all(
            x["direct_scalar_in_list"] for x in normalization_membership
        ):
            role = "auxiliary_direct_normalized_loss"
        elif normalization_membership and all(
            not x["direct_scalar_in_list"] for x in normalization_membership
        ):
            role = "primary_derived_normalized_loss"
        else:
            role = "unclassifiable_graph_membership"
        report = {
            "call_index": index,
            "height": call["target_level"],
            "target_override": call["target_override"],
            "context_label": "generated_aligned_targets"
            if call["target_override"]
            else "teacher_or_fallback_or_readout",
            "role": role,
            "effective_native_coefficient": coefficient,
            "normalization_membership": normalization_membership,
            "matched_mothers": sum(len(x) for x in loss.matches),
            "unweighted_total": float(loss.total.detach()),
            "components": [],
        }
        # Group aliased scalar components (usually unsupported zero losses) once.
        unique = {}
        for name, component in loss.components.items():
            unique.setdefault(id(component), {"names": [], "value": component})[
                "names"
            ].append(name)
        rebuilt = loss.total * 0
        for item in unique.values():
            value = item["value"]
            component_weight = scalar_derivative(loss.total, value)
            component_report = {
                "names": item["names"],
                "unweighted_value": float(value.detach()),
                "native_component_coefficient": component_weight,
                "shared_scalar_alias": len(item["names"]) > 1,
            }
            if coefficient is not None and component_weight is not None:
                contribution = value * component_weight * coefficient
                rebuilt = rebuilt + value * component_weight
                native_weighted_sum = native_weighted_sum + contribution
                loss_contributions.append(contribution)
                component_report["effective_weighted_value"] = float(
                    contribution.detach()
                )
                component_report["gradients"] = describe(contribution)
            report["components"].append(component_report)
        if coefficient is not None:
            torch.testing.assert_close(rebuilt, loss.total, rtol=rtol, atol=atol)
        reports.append(report)
    # Native recovery and any other graph paths not exposed as component losses
    # remain explicit residuals, never falsely assigned to a component/height.
    residual = reconstruction - native_weighted_sum
    reconstructed = sum(loss_contributions, total * 0) + residual + leaf_pid
    torch.testing.assert_close(reconstructed, total, rtol=rtol, atol=atol)
    replay_gradients = gradients(reconstructed, parameters)
    max_error = 0.0
    for actual, expected in zip(replay_gradients, reference):
        if actual is None or expected is None:
            require_same = actual is None and expected is None
            if not require_same:
                nonnull = actual if actual is not None else expected
                torch.testing.assert_close(
                    nonnull, torch.zeros_like(nonnull), rtol=rtol, atol=atol
                )
            continue
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        max_error = max(max_error, float((actual - expected).abs().max()))
    normalized_reports, normalized_total = [], total * 0
    for index, item in enumerate(normalizations):
        coefficient = scalar_derivative(total, item["output"])
        if coefficient is not None:
            normalized_total = normalized_total + coefficient * item["output"]
        normalized_reports.append(
            {
                "index": index,
                "weights": list(item["weights"]),
                "denominator": sum(item["weights"]),
                "scalar_count": len(item["losses"]),
                "effective_native_coefficient": coefficient,
            }
        )
    torch.testing.assert_close(normalized_total, reconstruction, rtol=rtol, atol=atol)
    normalized_gradients = gradients(normalized_total + leaf_pid, parameters)
    for actual, expected in zip(normalized_gradients, reference):
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                value = actual if actual is not None else expected
                torch.testing.assert_close(
                    value, torch.zeros_like(value), rtol=rtol, atol=atol
                )
        else:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {
        "native_total": float(total.detach()),
        "reconstruction_loss": float(reconstruction.detach()),
        "leaf_pid_loss": float(leaf_pid.detach()),
        "calls": reports,
        "normalizations": normalized_reports,
        "native_gradient": describe(total),
        "leaf_pid_gradient": describe(leaf_pid),
        "unallocated_recovery_or_other_residual": {
            "value": float(residual.detach()),
            "gradients": describe(residual),
        },
        "replay": {
            "scalar_sum": "PASS",
            "normalized_native_aggregation": "PASS",
            "normalized_native_gradient": "PASS",
            "gradient_sum": "PASS",
            "maximum_absolute_gradient_error": max_error,
            "rtol": rtol,
            "atol": atol,
        },
        "trainable_parameter_names": names,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "limitations": "Residual accounting is explicit, not proof every recovery path is attributed. Supplied sampling/context policy is diagnostic unless independently authenticated as historical replay. No optimizer fitting occurred.",
    }


def matched_query_logit_diagnostics(call, effective_coefficient, policy=None):
    """Post-hoc PID/size cells; local logit sensitivities, not parameter gradients.

    Uses original matching and target contract without rematching queries. Policy
    is optional provenance only: native captured controls determine target masks.
    Missing/invalid target metadata stays unavailable, never an inferred label.
    """
    from hypertagging.losses.level_reconstruction import (
        targets_for_level,
        focal_binary_cross_entropy_with_logits,
    )
    from hypertagging.preprocessing.pid_filter import PDG_TOKENS
    import torch.nn.functional as F

    output, loss, batch = call["pointer_output"], call["output"], call["loss_batch"]
    controls = call["controls"]
    level = call["target_level"]
    override = call.get("target_override_values")
    if override is None:
        target_types, target_masks, _, _ = targets_for_level(
            batch,
            level,
            min_daughters=controls.get("min_daughters", 2),
            target_policy=controls.get("target_policy", "complete_only"),
        )
    else:
        target_types, target_masks, _, _ = override
    count = sum(len(matches) for matches in loss.matches)
    contributions = {}
    for name, logits in (
        ("type", output.type_logits),
        ("pointer", output.pointer_logits),
    ):
        component = loss.components.get(name)
        weight = (
            scalar_derivative(loss.total, component) if component is not None else None
        )
        available = bool(
            call["grad_enabled"]
            and component is not None
            and component.requires_grad
            and logits.requires_grad
            and effective_coefficient is not None
            and weight is not None
        )
        gradient = None
        if available:
            gradient = torch.autograd.grad(
                component, logits, allow_unused=True, retain_graph=True
            )[0]
            if gradient is not None:
                gradient = gradient.detach() * effective_coefficient * weight
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite matched-query logit gradient")
        contributions[name] = (gradient, weight)
    rows = []
    for bi, matches in enumerate(loss.matches):
        context = batch["node_mask"][bi] & (batch["level_ids"][bi] < level)
        positions = context.nonzero(as_tuple=False).flatten()
        for query, target in matches:
            token = int(target_types[bi][target])
            mask = target_masks[bi][target].bool()
            if mask.numel() != positions.numel():
                raise ValueError("Matched target/context axis mismatch")
            selected = positions[mask]
            sources = batch.get("recursive_leaf_source_mask")
            source_count = (
                int(sources[bi, selected].any(0).sum()) if sources is not None else None
            )
            valid_token = 0 < token < len(PDG_TOKENS) and PDG_TOKENS[token] != 0
            record = {
                "batch_index": bi,
                "query_index": query,
                "matched_target_index": target,
                "height": level,
                "pid_token": token if valid_token else None,
                "signed_pdg": PDG_TOKENS[token] if valid_token else None,
                "pid_status": "AVAILABLE"
                if valid_token
                else "UNAVAILABLE_INVALID_TARGET_TOKEN",
                "daughter_count": len(selected),
                "recursive_source_count": source_count,
                "source_status": "AVAILABLE"
                if source_count is not None
                else "UNAVAILABLE_SOURCE_MASK",
                "matched_component_denominator": count,
                "gradients": {},
            }
            with torch.no_grad():
                record["local_type_ce"] = (
                    float(
                        F.cross_entropy(
                            output.type_logits[bi, query][None],
                            target_types[bi][target][None],
                        )
                    )
                    if valid_token
                    else None
                )
                record["local_pointer_focal_loss"] = float(
                    focal_binary_cross_entropy_with_logits(
                        output.pointer_logits[bi, query, context],
                        mask.float(),
                        positive_weight=controls.get("pointer_positive_weight", 4.0),
                        gamma=controls.get("pointer_focal_gamma", 1.0),
                    )
                )
            for name, (gradient, weight) in contributions.items():
                grad = gradient[bi, query] if gradient is not None else None
                record["gradients"][name] = {
                    "status": "AVAILABLE_LOCAL_LOGIT_GRADIENT"
                    if grad is not None
                    else "UNAVAILABLE_DETACHED_OR_UNUSED_GRAPH",
                    "effective_component_coefficient": effective_coefficient * weight
                    if effective_coefficient is not None and weight is not None
                    else None,
                    "l1": float(grad.double().abs().sum())
                    if grad is not None
                    else None,
                    "l2": float(grad.double().square().sum().sqrt())
                    if grad is not None
                    else None,
                }
            rows.append(record)
    return {
        "scope": "Matched-query local type/pointer logit sensitivities; not encoder/parameter gradients or correct-group-conditioned PID accuracy",
        "matched_queries": count,
        "rows": rows,
        "unmatched_queries": int(
            output.type_logits.shape[0] * output.type_logits.shape[1] - count
        ),
        "constraint_policy_supplied": policy is not None,
        "limitations": "Native Hungarian matches may have wrong daughter groups. Conditional correct-group PID requires a separate post-generation exact-source join. Source count is provenance-column union, not daughter count.",
    }


def observe_optimization_loss(
    model, batch, *, native_kwargs, parameter_groups, capture_tensors=True, trainer=None
):
    """Return (native_result, serializable_report, graph_capture) for one batch."""
    if trainer is None:
        from hypertagging.training import reconstruction_trainer as trainer
    started = time.monotonic()
    with capture_native_losses(trainer, capture_tensors=capture_tensors) as (
        calls,
        normalizations,
    ):
        result = trainer._optimization_loss(model, batch, **native_kwargs)
    forward_elapsed = time.monotonic() - started
    analysis_started = time.monotonic()
    report = analyze_native_capture(result, calls, normalizations, parameter_groups)
    report["timing"] = {
        "forward_wall_seconds": forward_elapsed,
        "parameter_attribution_wall_seconds": time.monotonic() - analysis_started,
    }
    return result, report, {"calls": calls, "normalizations": normalizations}
