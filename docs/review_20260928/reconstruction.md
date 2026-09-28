# Reconstruction, inference and deployment review — 2026-09-28

Reviewed source: `/home/b/Boyang.Yu/HyperTagging_uni/HyperTagging_phase61_review_20260927`, commit `db1ff41`. Source was not edited. This review reads actual implementation and tracked Phase58–62 evidence; it does not claim independent regeneration of millions of recorded metrics or inspect sealed test data. References below are repository-relative file:line at that commit.

## Assessment

The direction is scientifically cautious and mostly justified: hold the 70,000-event training set, retain independent validation, finish the already defined stability replication, and do not promote a model on shallow recovery or merely passing implementation gates. Neither larger training data nor longer pretraining has a controlled demonstrated benefit; that does **not** establish that either is ineffective. Objective/representation quality is an unproved hypothesis too.

However, a verified scheduled-sampling eligibility defect should precede another objective-dose campaign. Training can label a target representable and reward a daughter pointer that deployment must reject because the daughter already has a parent. This is distinct from the disclosed low physics performance and cannot be resolved by more pretraining. Deployment also has an acknowledged export blocker for the current models and intentionally different search semantics.

## Verified correctness defect: predicted-context supervision permits illegal re-parenting

`training/scheduled_sampling.py:212` selects every active predicted node, then at lines 227–242 matches a truth daughter to the first node with the same recursive source set. It does not require a parentless root, exclude a detector alias reserved by an existing composite, or consume the same pointer-eligibility mask as inference.

`training/reconstruction_trainer.py:1512` collates these predicted contexts. `_with_allowed_types` at line 1956 uses only `constraint_policy.pointer_validity_mask`, whose implementation (`reconstruction/constraints.py:149`) checks node availability, level and kind, not parenting. By contrast `reconstruction/level_rollout.py:231` explicitly intersects `parent_ids < 0` and excludes committed-source aliases at lines 234–235. The model merely passes the provided mask to its decoder (`models/level_autoregressive.py:341`). Appended daughter nodes remain active for contextual information, so active is not equivalent to eligible.

A six-node CPU reproducer establishes the discrepancy. A level-two truth mother needs leaf 0 plus correctly formed composite 4. An earlier incorrect composite 5 has consumed leaf 0. Current alignment nevertheless reports the target as representable and selects `[0,4]`:

```json
{
  "truth_targets": 1,
  "reported_representable": 1,
  "training_target_mask": [[true,false,false,false,true,false]],
  "training_pointer_valid": [[true,true,true,true,true,true]],
  "inference_pointer_valid": [[false,false,false,false,true,true]]
}
```

The reproduction is saved as `reproduce_parented_alignment.py`, with observed output in `reproduce_parented_alignment.json`, next to this review. It ran with the installed CPU environment, current repository `PYTHONPATH=src`, CUDA disabled and one BLAS thread. This is a demonstrated contract defect; its incidence and effect on real trained checkpoints remain unmeasured. It does not by itself invalidate every historical model or prove that it caused shallow reconstruction.

Recommended repair design:

1. Define a shared truth-free forest eligibility helper: lower-level, active, kind-permitted, parentless, and not an alias of a source already committed to another surviving composite. Keep consumed nodes available to the encoder as context; only mask their pointer eligibility.
2. For predicted training contexts, use that exact helper both for alignment candidates and for decoder pointer masks. Pass eligibility explicitly into `aligned_level_targets`, and count a target as unrepresentable if any required daughter lacks a legal exact-source candidate. Avoid silently choosing an earlier ineligible alias when a later eligible candidate exists.
3. Handle teacher contexts separately. Stored full truth `parent_ids` include future mothers; applying `parent_ids < 0` directly would incorrectly reject legitimate teacher daughters. Construct the visible generation prefix (parents formed below the target level only) or keep the current teacher path until its prefix is represented explicitly. Never derive predicted eligibility from future truth parents.
4. Add meaningful regressions for consumed daughters, source aliases, an eligible alternative with the same source set, retained consumed-node context, and teacher daughters whose stored parent is at the target/future level. Re-run matching, scheduled-sampling, source-exclusivity, strict inference and training smoke checks. Version the training contract; do not overwrite historical checkpoints or rerun old studies under the old label.
5. Before retraining, quantify affected event-level targets on a frozen development cohort. Compare source-match representability with legal-forest representability, by generation and by first wrong merge. No new strict cohort is needed to measure this software discrepancy.

## Reconstruction objective review

The actual current loss is an event/level-weighted mean of set loss plus recovery, with optional teacher auxiliary loss and leaf-PID CE. The set-loss definitions are `losses/level_reconstruction.py:99`; the total assembly is `training/reconstruction_trainer.py:1527`, `:1635` and `:1015`. Phase62 arm overrides use pointer positive weight **32**, despite common_config retaining 16 (`configs/reconstruction/ht_reconstruction_phase62_20260927.json:13`, `:34`, `:9772`); object positive weight is 12, recovery weight 2, query repulsion 0.01. Interpret resolved arm configuration, not one shared field in isolation.

| Term | Current role | Recommendation |
|---|---|---|
| Object focal BCE, coefficient 1 | Hungarian-matched queries positive; unmatched queries negative, except a limited mask for unrepresentable targets. Positive weight 12. | Keep sparsity control, but measure precision/recall by generation and event size. Weighted/focal outputs are not calibrated probabilities; calibrate before interpreting them as likelihoods. |
| Mother type CE, coefficient 1 | Applied to matched queries under ontology/type policy. | Keep ontology and charge constraints; diagnose confusion conditional on exact daughter sets separately from joint reconstruction. |
| Pointer focal BCE, coefficient 1 | Applied to matched queries over lower-level context. Arm positive weight 32. | Fix legal candidate masks first. Inspect precision/recall and gradients by eligible-root multiplicity; do not infer calibrated link probability from weighted focal BCE. |
| Cardinality CE, coefficient 0.2 | Matched truth daughter count. | Diagnose joint correctness of count and ranked daughter subset; independent count accuracy alone does not show recoverable mothers. |
| Confidence BCE, coefficient 0.2 | Raw pointer-IoU at threshold 0.5 × hard type correctness × source validity, with zero for unmatched queries. | Train/calibrate on the actual decoded candidate, including threshold, cardinality, charge, source eligibility and alternative type. Report calibration conditional on generation and candidates actually considered. |
| Physics consistency, coefficient 0.1 | Soft selected-daughter p4 and charge versus the current true daughter subset's sums. | Keep exact persistent daughter sums. Report p4 and charge subterms separately and fit any scales on train only. Default p4 scales are all 1; the raw squared residual grows with multiplicity/energy. This is not physical momentum resolution. |
| Recursive-source conflict, coefficient 0.1 | Soft penalty for conflicting selected source pairs, gated by object probability/mask. | Keep hard inference exclusivity regardless of loss; report its gradient and violation reduction separately. |
| Mother charge, coefficient 1 × policy soft weight 0.1 | Soft daughter charge versus the target mother's nominal charge. | Keep hard rollout compatibility; examine redundancy or opposing gradients with charge-sum consistency before changing weights. |
| Query repulsion, coefficient 0.01 | Cosine overlap of matched proposals whose targets are disjoint. | Useful narrow regularizer; unmatched duplicate high-object proposals are not covered. Quantify duplicate candidate pressure before broadening it. |
| Recovery, coefficient 2 | `softplus(-top_object_logits)` when a target is unrepresentable. | This only encourages object presence; it does not recover missing subtrees. Replace its label/interpretation, and test a real first-error correction objective after the eligibility repair. |
| Leaf PID CE, coefficient 1 | First selected target level per event. | Phase61/62 freeze the PID head, but when the encoder becomes trainable, this CE can still backpropagate through the fixed head to encoder representations. Audit shared gradients accordingly. |
| Optional auxiliary teacher loss | Default zero; adds teacher supervision on predicted-context samples when enabled. | A possible controlled rescue ablation for unrepresentable states, keeping exposure and effective optimization budget explicit. |

Two additional objective concerns are demonstrated by code, but their scientific effect remains hypothetical:

- Confidence supervision uses a hard pointer threshold 0.5 (`level_reconstruction.py:193`) while scientific inference uses 0.35 and cardinality/exclusivity constraints. Beam candidates with alternative types and daughter sets reuse the same query confidence (`beam_search.py:462`) even though training's target is tied to the query's argmax type and raw thresholded set. The confidence score is therefore not trained to estimate correctness of every candidate it ranks.
- Missing-target queries are masked out of object BCE (`level_reconstruction.py:264–276`), but confidence BCE still includes their zero targets (`:288–291`). Recovery then increases the highest object logits over **all** queries (`reconstruction_trainer.py:1543–1546`), potentially an already matched query rather than a missing-target slot. No pointer/type target is added and the earlier rollout is under `torch.no_grad()` (`:1412`). The recovery term can encourage a high-object/low-confidence query without teaching a legal repair. This helps explain why further recovery-dose tuning is weakly motivated; it does not establish that changing the term will improve physics.

Hungarian assignment itself is appropriate for unordered mother sets. The matching surrogate combines type NLL, soft Jaccard, object softplus and count NLL (`losses/set_matching.py:33–58`); this differs from the weighted focal training objective. Log assignment stability and component scale before tuning the surrogate. Production SciPy and fail-closed capacity checks should remain.

## Inference and beam search

The strict path is architecturally sound in the inspected areas: it physically projects detector FSPs, supplies reconstructed state, attaches evaluation keys only after generation, and uses CPU eval/inference mode (`reconstruction/hierarchical_inference.py:551–611`, `:677–688`). Mother state is constructed from daughters; no free mother p4 regressor should be added. Current links and source bookkeeping are legal state; truth hierarchy, channel labels and completeness are not inference inputs.

Full-depth beam really retains coherent states across generations; it is not merely local top-k followed by greedy collapse. It has deterministic canonical state deduplication, source exclusivity, proposal/state caps, a separate completed-root lane and explicit no-object choices. Live deduplication includes confidence-dependent future features (`beam_search.py:143–209`). These protections deserve preservation.

Its score is nevertheless a heuristic, not a normalized tree likelihood: accepted candidate factors include object × type × **mean selected pointer probability** × cardinality × optional confidence; omitted queries contribute log(1-object), and cumulative totals are normalized by scored-query count (`beam_search.py:461–492`, `:533`, `:568`). Mean selected pointers omit unselected-pointer factors; focal-weighted probabilities are uncalibrated; terminal-root priority precedes numerical score (`:572–574`). Root construction is therefore neither exactness nor a calibrated probability of a correct complete event.

Increasing beam width cannot recover a needed daughter eliminated by the pointer threshold (`beam_search.py:347–352`) or undo an earlier committed merge absent another surviving state. Next diagnose the complete bottleneck ladder: legal target exists → target inside candidate proposal support → survives per-query/per-level caps → survives forest beam → chosen top-1. Inspect this after generation, never with truth-guided search/pruning.

On Phase61's common 20-event beam cohort, control greedy, model rankings and oracle each achieve 1/22 exact full components. Candidate greedy achieves 1/22, normalized-joint 2/22, average-link 3/22 and oracle 3/22; all have 0/20 coherent forests (`docs/wiki/phase61.rst:285–357`). This is weak evidence of some shallow ranking/support headroom, not deep reconstruction. Ranking selected after viewing this cohort is exploratory; freeze a chosen rule on development data before a fresh strict comparison. Per-component oracle maxima may be mutually incompatible; report coherent event oracle separately.

## ONNX and basf2: two independent blockers

1. **Current model shapes do not export as a full v1 bundle.** Scientific per-level queries are 16/8/6/4/3/2, and cardinalities also vary. `_uniform_decoder_capacity` rejects heterogeneous capacities (`deployment/export_onnx.py:663–679`). This is an intentional checked limitation, documented at `docs/basf2_onnx_full_decay.md:46–51`, not evidence that the current best checkpoint has been deployed. Extend the manifest to per-level output shapes and validate each graph; do not reshape/retrain a checkpoint merely to fit the exporter.
2. **Host search is not the offline policy.** basf2 chooses one count, one type and one daughter set per query (`basf2_integration/runtime.py:619–642`), excludes the empty proposal set whenever proposals exist (`:762`), and sums confidence over accepted mothers (`:790`, `:851`). Offline search enumerates alternatives and normalizes cumulative log scores. The difference is explicitly documented at `docs/basf2_onnx_full_decay.md:29–36`. ONNX logit/feature parity and synthetic beam tests do not imply event-candidate or ranking parity.

Required deployment sequence: per-level manifest support; same-checkpoint PyTorch versus ONNX feature/logit tolerance and hard decisions on identical fixed states; common host-search implementation or an explicitly separate validated deployment policy; event-level comparison of proposal support, top-1, source exclusivity, PID and p4 across realistic multiplicities; then a bounded current-revision real-mDST basf2 pilot. Record truncation/cap failures explicitly. No current-revision real pilot is established by this review.

## Accumulated scientific evidence and next studies

Phase58 lost its control to objective-gradient dominance. Phase59 completed both settings but had only 25 strict events and needed count-bound evaluator recovery. Phase60 failed identically before the late-PID treatment. Phase61 completed with mixed shallow quality and undocumented-low-level runtime variation: 123/141 tensors differ at the common-prefix step 1094, losses first diverge at step 944 (`docs/wiki/phase61.rst:39–43`). Equal seed labels do not establish common-random-number identity. Earlier matched adaptation/PID/recovery-dose studies likewise did not establish reproducible deep benefit; the all-study synthesis at `docs/wiki/phase61.rst:504–528` is appropriately restrained.

Phase61 primary full forests remain 0/100; 14/142 versus 15/142 exact nontrivial components are all depth one. Only 96/142 nontrivial full components are classified representable, despite the leaf-dominated 3185/3323 overall representability (`docs/wiki/phase61.rst:208–218`, `:268–281`). Strong representation diagnostics and high source retention do not certify recursive assembly. Physical resolution remains unmeasured.

Recommended order, with falsifiable outcomes:

1. **Eligibility repair and incidence audit.** Above fixture must become zero legal representable targets while unchanged teacher targets remain available. Measure real affected targets and repeated source use. Run one matched corrected-versus-current reconstruction study only after a frozen corrected contract exists, preserving the historical model/evidence.
2. **Assembly ceiling audit, no training required.** For a frozen development checkpoint, measure teacher-forced versus free rollout by generation, then the proposal-support/survival/top-1 ladder, jointly with topology depth, PID and policy compatibility. If oracle candidate support is poor, prioritize representation/proposals; if support is strong but top-1 poor, prioritize calibration/ranking. Keep all-truth diagnostic inputs entirely outside deployable inference.
3. **Reproducible treatment branching.** For future late-only interventions, produce one authenticated common-prefix checkpoint including optimizer, scheduler, RNG and memory, then branch both treatments from it. Use a bounded deterministic diagnostic to locate the current pre-treatment divergence; do not change kernels mid-campaign or claim seed pairing removes it. Retain independent seed replications to estimate training variation.
4. **Decoded-confidence ablation.** Freeze candidate generator/checkpoint and compare existing confidence with development-fitted actual-decoder calibration/ranking, then evaluate one frozen model-only top-1 rule on fresh strict events. Success requires improvement in nontrivial exact/deep components without precision/validity loss; an oracle-only gain is failure of deployment ranking.
5. **Actual missing-subtree learning.** Replace object-only recovery with a clearly specified teacher corrective action at the first representational divergence, or a dynamic-oracle objective only for legal feasible actions. Test against masking-only and existing recovery at matched exposure. Do not teach impossible daughters or differentiate through truth-driven inference. Demand better reachable target coverage and deep exactness, not only more mothers.
6. **Target-policy and data scaling studies.** First report compatibility by nontrivial topology, multiplicity, single-source composites, missing daughters and category. Any intentional target contraction must define a separate estimand. Later compare 35k/70k (and then larger) at matched optimizer presentations/architecture/cohort, plus a compute-controlled axis, rather than conflating data size and steps again.

Checkpoint selection should add a preregistered nontrivial topology/coherent-forest criterion or a clearly defined lexicographic rule after development diagnosis. Existing micro source-set-plus-mother-PID recovery, depth fraction, structural validity and p4 closure are valuable gates but weak proxies for exact hierarchy. Retain both full and B-half denominators, all failed/incompatible trials, count-based aggregation, event-cluster uncertainty and source-domain validation; do not open the sealed test for this exploratory cycle.

No production jobs were launched, no source was changed, and no new physics result is claimed. The one executed reproducer verifies the specific eligibility mismatch above.
