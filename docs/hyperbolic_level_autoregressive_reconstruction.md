# Hyperbolic Level-Autoregressive Reconstruction

HyperTagging reconstruction is a hierarchy of unordered sets, not a language
sequence.  The factorization used here is:

```text
p(tree | S0) = product_t p(S_{t+1} | S_{<=t})
```

`S0` contains reconstructed final-state particles.  Each later level contains
symbolic mother nodes predicted from lower levels.

## Truth Topology And Reco Kinematics

MC truth and MC matching may supervise topology labels, same-mother labels,
same-branch labels, LCA depth, B-side labels, and daughter links.  Model inputs
use reconstructed quantities.  A reconstructed mother four-vector is always
computed from selected reconstructed daughter four-vectors:

```text
px = sum(px_daughter)
py = sum(py_daughter)
pz = sum(pz_daughter)
E  = sum(E_daughter)
```

MCParticle mother p4 may only appear in diagnostic `mc_*` fields.

## Hyperbolic Pretraining

The encoder produces Euclidean node embeddings and Poincare-ball embeddings.
Dense pairwise losses provide O(N^2) supervision: same-mother BCE,
same-branch BCE, parent-child margin, and radius-depth regularization.

## Level Reconstruction

For target level `t+1`, learned query slots attend to nodes in `S_{<=t}` and
predict object/no-object logits, mother type logits, daughter pointer logits,
cardinality logits, and confidence logits.  Next-level truth nodes are matched
to query slots with Hungarian matching when SciPy is available and a
deterministic fallback for tiny tests.

## Teacher Forcing, Rollout, And Scheduled Sampling

Teacher forcing uses truth-guided previous levels as context.  Rollout appends
hard-decoded predicted mother nodes and constructs their p4 from daughters.
Scheduled sampling gradually mixes predicted previous levels into the context;
the schedule is configured by start, end, and warmup steps.

## What Runs Locally

CPU unit tests, tiny synthetic fixtures, and CPU dry-runs are local-safe.
Full data, full training, and normal CUDA training are SLURM-only.  A local CUDA
smoke test is allowed only for tiny runs after explicit `squeue` and
`nvidia-smi` safety checks.
