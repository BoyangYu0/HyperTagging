"""Training-role information probes; utilities only, no admission or CLI runner.

Detector input is strictly projected reconstructed FSP state. Its 128 coordinates
are 41 measurements, 41 availability flags, and 46 categorical one-hot flags.
Only the constant projected daughter count is omitted (checked zero). Learned
embeddings already contain learned contextual/PID transformations; equal probe
capacity does NOT equalize those upstream information-processing advantages.
"""

from __future__ import annotations

import hashlib
import time
import resource
import torch
from torch.nn import functional as F
from hypertagging.preprocessing.schema_v2 import NODE_KIND_TO_ID
from hypertagging.preprocessing.schema_v3 import (
    V3_COMMON_FEATURE_NAMES,
    V3_TRACK_FEATURE_NAMES,
    V3_CLUSTER_FEATURE_NAMES,
)
from hypertagging.preprocessing.schema_v4 import KLM_FEATURE_NAMES, LEAF_MODE_TO_ID
from hypertagging.preprocessing.pid_filter import PDG_TOKENS

COMMON = (0, 1, 2, 3, 4, 5, 11)
KINDS = tuple(NODE_KIND_TO_ID[k] for k in ("track", "ecl_cluster", "klm_cluster"))
MODES = tuple(
    LEAF_MODE_TO_ID[k]
    for k in ("raw_track_predicted_pid", "fixed_hypothesis_candidate")
)
BLOCKS = (
    ("common", len(V3_COMMON_FEATURE_NAMES)),
    ("track", len(V3_TRACK_FEATURE_NAMES)),
    ("cluster", len(V3_CLUSTER_FEATURE_NAMES)),
    ("klm", len(KLM_FEATURE_NAMES)),
)
READ_KEYS = frozenset(
    ["node_mask", "node_kind_ids", "leaf_kinematics_mode_ids", "pid_labels"]
    + [name + suffix for name, _ in BLOCKS for suffix in ("_features", "_availability")]
)


def detector_nodes(detector):
    """Return node values/observations/continuous mask, never target fields.

    Caller authenticates the native strict projection and input provenance.
    Unknown extra keys are ignored: only READ_KEYS are accessed, never inferred.
    """
    d = {key: detector[key] for key in READ_KEYS}
    mask = d["node_mask"]
    if (
        mask.dtype != torch.bool
        or mask.ndim != 2
        or mask.shape[0] != 1
        or not mask.any()
    ):
        raise ValueError("One nonempty projected event required")
    n = mask.shape[1]
    kind, mode, pid = [
        d[k][0, mask[0]]
        for k in ("node_kind_ids", "leaf_kinematics_mode_ids", "pid_labels")
    ]
    for v, allowed in ((kind, KINDS), (mode, MODES), (pid, range(len(PDG_TOKENS)))):
        if (
            v.dtype != torch.long
            or not torch.isin(v, torch.tensor(list(allowed))).all()
        ):
            raise ValueError("Unsupported reconstructed category")
    if ((mode == MODES[0]) & (pid != 0)).any():
        raise ValueError("Raw track PID must be unknown before inference")
    values, observed = [], []
    for block, width in BLOCKS:
        x, a = d[block + "_features"], d[block + "_availability"]
        if x.shape != (1, n, width) or a.shape != x.shape or a.dtype != torch.bool:
            raise ValueError("Feature/mask schema changed")
        x, a = x[0, mask[0]].float(), a[0, mask[0]]
        if block == "common":
            if (x[:, 10] != 0).any() or a[:, [6, 7, 8, 9]].any():
                raise ValueError("Not a projected detector-only common block")
            x, a = x[:, COMMON], a[:, COMMON]
        else:
            expected = KINDS[("track", "cluster", "klm").index(block)]
            if (a & (kind != expected)[:, None]).any():
                raise ValueError("Availability outside detector kind")
        if not torch.isfinite(x[a]).all():
            raise ValueError("Nonfinite observed measurement")
        values.append(torch.where(a, x, torch.zeros_like(x)))
        observed.append(a)
    v, a = torch.cat(values, -1), torch.cat(observed, -1)
    flags = torch.cat(
        (
            (kind[:, None] == torch.tensor(KINDS)).float(),
            (mode[:, None] == torch.tensor(MODES)).float(),
            F.one_hot(pid, len(PDG_TOKENS)).float(),
        ),
        -1,
    )
    x = torch.cat((v, a.float(), flags), -1)
    obs = torch.cat(
        (a, torch.ones_like(a), torch.ones_like(flags, dtype=torch.bool)), -1
    )
    continuous = torch.arange(128) < v.shape[1]
    if x.shape[1] != 128:
        raise ValueError("Lossless fixed input width changed")
    return x, obs, continuous


def validate_roles(records):
    ids = [r["uid"] for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty/duplicate event identities")
    if any(
        r["source_role"] != "train" or r["partition"] not in ("fit", "assessment")
        for r in records
    ):
        raise ValueError("Only event-disjoint TRAIN probe roles allowed")
    if {r["partition"] for r in records} != {"fit", "assessment"}:
        raise ValueError("Both probe roles required")


def fit_normalizer(records, feature):
    """Masked fit-only statistics; flags remain unchanged; absent slots identity."""
    validate_roles(records)
    fit = [r for r in records if r["partition"] == "fit"]
    x = torch.cat([r[feature] for r in fit]).double()
    a = torch.cat([r[feature + "_observed"] for r in fit])
    continuous = fit[0][feature + "_continuous"]
    if x.shape[1] != 128 or a.shape != x.shape or a.dtype != torch.bool:
        raise ValueError("Normalizer feature contract")
    if (
        continuous.shape != (128,)
        or continuous.dtype != torch.bool
        or not torch.isfinite(x[a]).all()
    ):
        raise ValueError("Invalid observed normalizer values or flags")
    if any(not torch.equal(r[feature + "_continuous"], continuous) for r in records):
        raise ValueError("Inconsistent categorical contract")
    count = a.sum(0)
    clean = torch.where(a, x, torch.zeros_like(x))
    mean = clean.sum(0) / count.clamp_min(1)
    var = torch.where(a, (x - mean).square(), torch.zeros_like(x)).sum(
        0
    ) / count.clamp_min(1)
    mean = torch.where(continuous & (count > 0), mean, 0.0)
    scale = torch.where(continuous & (count > 1) & (var > 1e-12), var.sqrt(), 1.0)
    return {
        "mean": mean.float(),
        "scale": scale.float(),
        "observations": count,
        "fit_uids": sorted(r["uid"] for r in fit),
    }


def normalize(x, observed, stats):
    return torch.where(
        observed, (x - stats["mean"]) / stats["scale"], torch.zeros_like(x)
    )


def pair_inputs(nodes, ij):
    """Symmetric pair and identical mean-event context; no labels accepted."""
    if nodes.ndim != 2 or nodes.shape[1] != 128 or not torch.isfinite(nodes).all():
        raise ValueError("Finite 128-wide nodes required")
    a, b = nodes[ij[0]], nodes[ij[1]]
    return torch.cat(((a - b).abs(), a * b, nodes.mean(0).expand(len(a), -1)), -1)


def pair_labels(target, ij):
    if target.dtype != torch.long or not ((target >= -1) & (target <= 2)).all():
        raise ValueError("Membership target contract")
    a, b = target[ij[0]], target[ij[1]]
    known = (a >= 0) & (b >= 0)
    y = torch.where((a == 0) | (b == 0), 2, torch.where(a == b, 0, 1))
    return torch.where(known, y, -1)


def shuffled_targets(target, uid, seed):
    """Within-event null preserves unknown positions and exact class counts."""
    positions = torch.where(target >= 0)[0]
    g = torch.Generator().manual_seed(
        int(hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()[:12], 16)
    )
    output = target.clone()
    output[positions] = target[positions[torch.randperm(len(positions), generator=g)]]
    return output


def new_probe(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return torch.nn.Sequential(
            torch.nn.Linear(384, 64), torch.nn.GELU(), torch.nn.Linear(64, 3)
        )


def fit_probe(events, *, seed, updates, batch_size, lr=0.001):
    """Finite fit; events contain fit-only uid,x(P,384),y(P) incl unknown=-1.

    Uniform event then available-class sampling keeps continuum events, does
    not invent absent classes. Feature arms use identical target/order schedule.
    No assessment data, validation selection or automatic budget extension.
    """
    if not 1 <= updates <= 4096 or not 1 <= batch_size <= 256 or lr != 0.001:
        raise ValueError("Finite diagnostic budget exceeded")
    if not events or len({r["uid"] for r in events}) != len(events):
        raise ValueError("Empty/duplicate fitting events")
    classes = []
    for r in events:
        if r["source_role"] != "train" or r["partition"] != "fit":
            raise ValueError("Non-fitting event supplied")
        if r["x"].shape != (len(r["y"]), 384) or not torch.isfinite(r["x"]).all():
            raise ValueError("Pair feature contract")
        if r["y"].dtype != torch.long or not ((r["y"] >= -1) & (r["y"] <= 2)).all():
            raise ValueError("Pair label contract")
        c = [torch.where(r["y"] == k)[0] for k in range(3)]
        c = [v for v in c if len(v)]
        if not c:
            raise ValueError("No supported pairs: account exclusions before fitting")
        classes.append(c)
    model = new_probe(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    order, used, sampled, curve = hashlib.sha256(), set(), set(), []
    start = time.monotonic()
    for step in range(updates):
        pairs = []
        for _ in range(batch_size):
            e = int(torch.randint(len(events), (), generator=generator))
            c = classes[e][int(torch.randint(len(classes[e]), (), generator=generator))]
            p = int(c[int(torch.randint(len(c), (), generator=generator))])
            pairs.append((e, p))
            used.add(events[e]["uid"])
            sampled.add((events[e]["uid"], p))
        order.update(str(pairs).encode())
        x = torch.stack([events[e]["x"][p] for e, p in pairs]).detach()
        y = torch.stack([events[e]["y"][p] for e, p in pairs])
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite diagnostic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        opt.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite diagnostic parameters")
        if step == 31 and (time.monotonic() - start) / 32 * updates * 3 > 600:
            raise ValueError("Probe runtime forecast exceeds per-head600s guard")
        if (
            step % 128 == 0
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 12 * 1024**2
        ):
            raise ValueError("Probe memory guard")
        if step == 0 or (step + 1) % 128 == 0 or step + 1 == updates:
            curve.append({"update": step + 1, "minibatch_loss": float(loss.detach())})
    return model.eval(), {
        "updates": updates,
        "pair_presentations": updates * batch_size,
        "unique_sampled_events": len(used),
        "unique_sampled_pairs": len(sampled),
        "order_sha256": order.hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "curve": curve,
    }


def score_pairs(model, x):
    """Detached model-only output for serialization before evaluation joins."""
    if x.ndim != 2 or x.shape[1] != 384 or not torch.isfinite(x).all():
        raise ValueError("Pair scoring contract")
    if not len(x):
        return torch.empty((0, 3), dtype=x.dtype)
    with torch.inference_mode():
        return torch.cat([model(chunk).softmax(-1) for chunk in x.split(4096)]).detach()


def source_pairs(sources):
    """Only unordered detector-source-disjoint pairs; aliases counted explicitly."""
    if sources.ndim != 2 or sources.dtype != torch.bool or not sources.any(-1).all():
        raise ValueError("Nonempty boolean recursive detector source support required")
    ij = torch.triu_indices(len(sources), len(sources), offset=1)
    shared = (sources[ij[0]] & sources[ij[1]]).any(-1)
    return ij[:, ~shared], {
        "possible_pairs": ij.shape[1],
        "source_alias_excluded": int(shared.sum()),
        "eligible_pairs": int((~shared).sum()),
    }
