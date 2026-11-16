# HyperTagging working rules

Read this file, check Git status, then consult [the study loop](docs/study_loop.md)
and the current controller/protocol for study work. The latest user instruction
wins; historical plans and receipts are evidence, not new work orders.

- **Be practical, not overly defensive.** Carry authorized work through to a
  result. Resolve routine choices yourself; ask only for missing decisions,
  destructive scope or a real blocker. Run checks relevant to the change and
  reuse valid evidence. Avoid repeated approvals, generic checks and blind retries.
- **Put files in the correct paths.** Keep code, configs, tests and concise
  guides in Git. Put datasets, checkpoints, logs, executed notebooks, builds and
  private study records on the data volume. Use separate study/operation outputs;
  retain immutable inputs and recovery manifests. See [storage](docs/repository_storage.md).
- **Use computing resources well.** Run independent training and full-reconstruction
  studies in parallel when allocations and dependencies permit. Evaluate completed
  checkpoints on CPU while GPU training continues. Register each operation with
  the existing controller; isolate source versions, outputs and resource ownership.
- **Respect shared GPUs.** Use the site's scheduler or its supported local
  diagnostic admission path. Free memory alone is not permission. Use only
  assigned devices, account for CPU/RAM/I/O too, and never disturb another job.
  Follow [GPU rules](environment/gpu/README.md#shared-gpu-rules).
- **Track every study.** Record hypothesis, control, source/config/data/checkpoint
  identities, jobs, actual results, failures, cost and next decision. The controller
  owns the campaign ledger; workers provide operation-local evidence.
- **Evaluate reconstruction, not just loss.** Compare fixed cohorts and report
  continuous quality, exact topology/PID, coverage, uncertainty and compute.
  Keep truth out of inference, fit normalization on train only, preserve source
  uniqueness and recursive daughter sums, and retain failed/unavailable trials.
  Follow [evaluation](docs/full_decay_reconstruction_evaluation.md#evaluation-workflow).
- **Publish meaningful findings.** Record publish/defer at review. Update the
  dashboard for a supported result, useful accumulated diagnosis or material
  correction; summarize older studies and retain downloads. Follow the
  [publication procedure](docs/wiki/maintaining.rst) through the existing owner.
- **Preserve ongoing work.** Keep user edits, live worker sources, checkpoint
  lineage and explicit pauses. Use one controller; reconcile terminal handoffs
  before reusing ownership. Follow the authorized Git/publication scope.

Keep one definition per topic: workflow in the study loop, technical contracts
in their reference guides, results in evidence records. Link rather than copy.
Other references: [training](docs/training.md), [preprocessing](docs/preprocessing_design.md),
[current audit](docs/audits/current_status.md), [remote setup](docs/codex_remote_setup.md),
[deployment](docs/basf2_onnx_full_decay.md).
