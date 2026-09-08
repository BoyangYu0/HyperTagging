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

Production workflows use the tracked Slurm or HTCondor renderers; see the
:doc:`script catalogue <_generated/catalog/index>` for entry points.

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

Further reading
---------------

See :doc:`dataset`, :doc:`training`, :doc:`reconstruction` and :doc:`evaluation`
for the scientific definitions, and :doc:`maintaining` for documentation builds.
