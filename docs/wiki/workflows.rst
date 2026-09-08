Training, evaluation and development
====================================

CPU development
---------------

In the installed project environment, with bounded CPU threads:

.. code-block:: bash

   export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
   python -m pytest -q
   python scripts/train_hyperbolic_pretrain.py --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
   python scripts/train_level_reconstruction.py --dry-run --tiny --device cpu --max-steps 2 --batch-size 2
   python scripts/execute_notebook_smoke_tests.py --list

Notebook execution is separate from a docs build. Use the notebook runner's
``--keep-output`` in a new directory when running the registered CPU fixture
notebooks. GPU and basf2 checks require their documented external environments.

Preparing a scientific run
--------------------------

Real runs bind immutable source-role selections to an authenticated dataset
index. Fit statistics on train only and preserve a fixed validation UID cohort;
sealed test is not a source for tuning or normalization. Validate query and
cardinality capacities for the selected target policy before training.
``complete_only``, ``reconstructable_partial`` and diagnostic populations have
different denominators. Complete-only targets do not promise complete physical
event reconstruction.

YAML resolution is defaults, then YAML, then explicitly supplied CLI flags.
Architecture, feature/PID contracts, normalizers, constraints, data-order identity,
sampling and rollout PID mode are persisted. Checkpoints also carry optimizer,
scheduler, scaler, RNG and curriculum state. Exact mid-epoch replay is supported
with zero loader workers; prefetched multiworker replay is not claimed.

Use the tracked guarded Slurm or HTCondor renderers for production. Rendering
does not submit a job. Inspect :doc:`_generated/status/index` and the original
readiness/provenance documents before interpreting an old contract as launch
authorization. Local CUDA diagnostics require the explicit admission/watchdog
or tiny-test guards; a docs build never launches them.

Strict offline evaluation
-------------------------

Inspect required arguments without starting inference:

.. code-block:: bash

   python scripts/validate_reconstruction_checkpoint_pair.py --help
   python scripts/evaluate_full_decay.py --help

Validate the trusted encoder/reconstruction checkpoint lineage first, then
evaluate the immutable validation/test selection with ``evaluate_full_decay.py``.
Output must not alias any input shard, index, manifest or checkpoint. The strict
path is CPU-only and uses evaluation plus inference mode.

Full scope contributes one eligible root unit per event. B-half contributes two
units and an event-level both-halves result. Continuum uses explicit top-level
retained components. Direct-target incompatibilities remain failed primary
trials; contraction is diagnostic. Match by reconstructed source/topology before
PID or p4 scoring, aggregate counts before division, and preserve unavailable
versus failed results. Keep greedy, beam top-1 and oracle@K namespaces separate.

Development evidence
--------------------

The existing CPU and notebook workflows remain independent of the new Pages
workflow. Docs checks validate coverage, reproducibility, source provenance,
local links/assets, and workflow permissions. They do not update scientific
readiness or promote historical test counts to the current checkout.

For exact options and operator commands, use :doc:`_generated/catalog/index`.
Consult the checkout's ``docs/training.md``, ``docs/condor.md``,
``docs/full_decay_reconstruction_evaluation.md``, ``notebooks/README.md`` and
``environment/gpu/README.md``. These source documents are not included in the
public artifact.
