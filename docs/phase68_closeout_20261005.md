# Phase68 completed hybrid encoder review — 2026-10-05 (Asia/Shanghai)

**Incremental encoder refinement has no established recursive-quality benefit. Hold 70,000 training events for one independent-seed confirmation; prioritize representation and assembly usefulness over simply longer pretraining.** Both arms fail original gates. Immediate training growth is not demonstrated necessary, but neither scaling nor better pretraining has established comparative superiority. No model is promoted.

## Native validity and intervention

Both native jobs completed with exit 0 from frozen source `ee4c7255024aa75295c35e32e438c26cde1ed468`. Receipt/config/input/report hashes, all 34 data shards, fourteen registered full/half reports, exact repeats, five registered checkpoint tracks per arm and all 26 saved checkpoint files verify. No evaluation recovery was needed. Every native JSON/JSONL scalar and available checkpoint scalar metadata is exported. Strict evaluations of unregistered teacher-forced, edge-F1, periodic and signal tracks remain unavailable, not zero.

This is a HYBRID PARAMETER ABLATION: refined versus train-rescaled pre-refinement encoder parameters, with the same refined PID, train normalization and other pretraining-model state. Independent tensor inspection verifies 119 changed entries among 121 encoder entries, 20 identical nonencoder entries including two PID entries, identical normalization, and exact candidate agreement with the authenticated pre-refinement encoder. The hybrid has step zero and no resumable optimizer/scheduler/RNG state. Both encoders have historical pretraining; this is not pretraining versus no pretraining, an untrained baseline, or a hyperbolic-versus-Euclidean test.

Both arms execute 4,376 reconstruction updates and 280,064 replay slots, seed 20261004, mixed-context probability rising to 0.5, late encoder adaptation after 2,188 updates, frozen PID and zero extra pretraining. Architecture, inference, thresholds, policy, selection and original gates match. Historical refinement compute differs between encoders; equal downstream updates/slots are not an equal-FLOP claim. Identical PID-head weights do not guarantee identical PID predictions because encoder features differ.

## Cohorts and populations

Fresh selection 1,000 and strict 100 exclude 57,700 prior reservations; beam is the fixed first 20 strict events. Strict-selection overlap is zero and 41,200 validation events remain unreserved after Phase68. Full retained scope has 2,972 isolated leaves, 84 single-source composites and 137 nontrivial units; half/component scope has 2,384 leaves, 84 single-source composites and 159 nontrivial units. Nontrivial representability is 92/137 full and 101/159 half. Only 24/100 events have no flagged incompatibility. These necessary eligibility diagnostics are not attainable efficiencies. Policy-incompatible direct targets remain failed primary trials. Explicit B halves and continuum component fallback keep separate semantics.

## Primary results

| Endpoint | Refined encoder | Pre-refinement encoder |
| --- | ---: | ---: |
| Selection source-set plus mother PID | 287/3666 | 296/3666 |
| Original exact mother coverage | 7/206 | 8/206 |
| Original full lcag | 7/3542 | 8/3542 |
| Original full root completion | 43/100 | 44/100 |
| Original full source precision | 95/118 | 85/114 |
| Original full source recall | 95/397 | 85/397 |
| Original half lcag | 31/2268 | 34/2268 |
| Original half perfect lcag | 2/147 | 1/147 |
| Original half root pid accuracy | 8/147 | 9/147 |
| Original half source precision | 165/329 | 160/331 |
| Original half source recall | 165/687 | 160/687 |
| Retained full lcag pair accuracy | 28/3944 | 27/3944 |
| Retained full perfect lcag | 3/137 | 2/137 |
| Retained full coherent retained forest | 0/100 | 0/100 |
| Retained full source precision | 3127/3382 | 3096/3381 |
| Retained full source recall | 3127/3768 | 3096/3768 |
| Retained full mother pid coverage | 28/540 | 27/540 |
| Retained full mother pid accuracy | 28/28 | 25/27 |
| Retained full leaf pid accuracy | 1159/2348 | 1169/2348 |
| Retained full both halves perfect lcag | UNAVAILABLE (0/0) | UNAVAILABLE (0/0) |
| Retained half lcag pair accuracy | 32/2282 | 34/2282 |
| Retained half perfect lcag | 2/159 | 2/159 |
| Retained half coherent retained forest | 0/100 | 0/100 |
| Retained half source precision | 2570/2816 | 2555/2811 |
| Retained half source recall | 2570/3180 | 2555/3180 |
| Retained half mother pid coverage | 32/518 | 34/518 |
| Retained half mother pid accuracy | 32/32 | 32/34 |
| Retained half leaf pid accuracy | 1091/2048 | 1100/2048 |
| Retained half both halves perfect lcag | 0/22 | 0/22 |

Both primary tracks select step 3,000. Both fail half-perfect-LCAG; pre-refinement additionally fails full-source precision. Structural validity, source exclusivity, daughter-sum closure and exact repeat checks pass. Selection source-set plus mother PID and configured root construction are not exact recursive topology.

All exact components in every registered view and retained candidate are two-leaf, one-mother, depth-one components. There is no depth-at-least-two exact success. Both primary models have zero coherent forests. The pre-refinement auxiliary depth/validity tracks recover the same mother-free event; their step-1,000 model tensors are identical. A different shallow event with one mother is recovered by a proposal candidate in both arms and scopes. These are two distinct exploratory events, not independent successes for every track, scope or rank, and neither establishes a treatment advantage.

Teacher-forced scores remain separate from strict detector-only rollout. Source/topology matching precedes PID and p4 scoring. Schema-v4 p4 references reconstructed daughter sums; physical reconstructed-minus-MC momentum resolution is unavailable. Algebraic closure is an implementation invariant. Missing strict tracks and physical-resolution values are not zero.

## Paired uncertainty

Intervals use 10,000 paired event-cluster bootstrap draws, seed 20261004, ratios of summed counts, and pre-refinement minus refined differences. They are exploratory, conditional on the two fitted models, unadjusted for multiple comparisons, and omit training-seed and source-domain uncertainty. A zero-width empirical forest interval from all-zero outcomes does not establish zero population success probability. Under an independent identical Bernoulli-event model, zero of 100 implies a one-sided 95% upper bound of about 2.95%; heterogeneous events limit that model.

| Endpoint | Difference (percentage points) | Paired 95% interval |
| --- | ---: | ---: |
| full lcag pair accuracy | -0.025 | -0.161 to 0.116 |
| full perfect lcag | -0.730 | -3.185 to 1.600 |
| full coherent retained forest | 0.000 | 0.000 to 0.000 |
| Nontrivial full source precision | -0.755 | -6.708 to 5.196 |
| Nontrivial full source recall | -2.949 | -5.541 to -0.403 |
| half lcag pair accuracy | 0.088 | -0.121 to 0.328 |
| half perfect lcag | 0.000 | -1.829 to 1.875 |
| half coherent retained forest | 0.000 | 0.000 to 0.000 |
| Nontrivial half source precision | -1.087 | -5.812 to 3.703 |
| Nontrivial half source recall | -0.562 | -2.536 to 1.443 |

Nontrivial full-source precision is 228/315 refined versus 207/289 pre-refinement; recall is 228/712 versus 207/712. Half precision is 243/368 versus 239/368; recall 243/712 versus 239/712. The full recall interval conditionally favors refined encoding, while LCAG, exact-component and precision intervals include zero. This isolated exploratory recall signal fails the joint recursive-usefulness requirement. No cross-phase cohort pooling or causal size/pretraining conclusion follows.

## Execution and representation

Both arms encounter 442,789 repeated target exposures. Refined preserves 331,551 and loses 111,238 as unrepresentable; pre-refinement preserves 331,182 and loses 111,607. Both execute 175,091 teacher and 104,973 predicted slots, with zero recovery/fallback/skipped counts. Model-forward counts are 395,272 versus 395,329; batching prevents interpreting these as FLOPs. All 4,376 steps per arm pass finite loss/model/optimizer checks.

All 121 encoder entries and both PID-head entries transfer without mismatch. All saved checkpoint PID weights and train normalization remain fixed; encoder entries remain identical through the freeze boundary and change afterward. Fixed-input geometry uses the same 128 train and 32 previously used development samples, four curriculum views, and initial/selected/final encoder stages. No strict or test samples are used for this diagnostic; it is not a census of generated rollout states.

Across 48 measured stage/sample/view combinations, saturation ranges from 0.000000 to 0.000000; mean radial derivatives range from 0.624 to 0.802. Usable geometry on these samples does not imply useful recursive transfer.

## Search and confidence

All 134 returned proposal candidates (65 refined, 69 pre-refinement) are checked in both scopes. This registered proposal-set ranking diagnostic is distinct from full-depth beam search, which was not run. Candidate ranks, score components, confidence/search counters, reduced rank denominators and every retained candidate remain in the exports. Oracle uses truth only after generation and is never deployable.

| Retained beam scope / ranking | Refined LCAG | Pre-refinement LCAG | Refined exact | Pre-refinement exact |
| --- | ---: | ---: | ---: | ---: |
| full/greedy | 9/596 | 8/596 | 0/31 | 0/31 |
| full/proposal beam/average link probability | 6/596 | 8/596 | 2/31 | 2/31 |
| full/proposal beam/learned confidence mean | 8/596 | 8/596 | 2/31 | 0/31 |
| full/proposal beam/learned confidence sum | 9/596 | 8/596 | 0/31 | 0/31 |
| full/proposal beam/normalized joint log probability | 8/596 | 8/596 | 2/31 | 0/31 |
| full/proposal beam/oracle diagnostic | 11/596 | 10/596 | 5/31 | 3/31 |
| half/greedy | 9/349 | 8/349 | 0/35 | 0/35 |
| half/proposal beam/average link probability | 6/349 | 8/349 | 2/35 | 2/35 |
| half/proposal beam/learned confidence mean | 8/349 | 8/349 | 2/35 | 0/35 |
| half/proposal beam/learned confidence sum | 9/349 | 8/349 | 0/35 | 0/35 |
| half/proposal beam/normalized joint log probability | 8/349 | 8/349 | 2/35 | 0/35 |
| half/proposal beam/oracle diagnostic | 11/349 | 11/349 | 5/35 | 3/35 |

Greedy has no exact component or coherent forest on the 20-event beam subset. Average-link ranking recovers the same one-mother shallow forest in both arms; refined learned-confidence mean and normalized-joint ranking also recover it. Learned-confidence sum does not. Oracle recovers one coherent forest in each arm, with full exact-component counts 5/31 versus 3/31. This proposal oracle chooses one candidate per event and scope by its registered truth-only ranking. Its component counts aggregate across events and are not counts of coherent forests. Separate oracle-at-K summaries may recover different truth components from different candidates; their coherent-forest conjunction still requires one candidate. Small candidate/ranking improvements do not produce deep topology or justify a general search winner.

## Complete exports

The private export contains 12,555,087 native JSON/JSONL scalar records plus 18,174 checkpoint metadata scalars: **12,573,261 total**. The closeout projection has 165,795 rows. Public downloads preserve **178,022 aggregate values** across seven report views, training-history summaries and all saved-checkpoint metric tracks. Null, false, zero, numerator and denominator remain distinct. Integrity manifests authenticate every exported file. Private event identities, native paths, logs and checkpoint contents remain outside public pages; historical downloads are preserved.

## Cumulative evidence and allocation

The dated [all-study ledger](review_20260928/studies.md) covers migration Phases1–13, named early production/pretraining/transfer families, Stage A and reconstruction Phase34 onward. No standalone reconstruction Phase14–33 closeouts were found; no such results are invented. Its dated Phase62 pending state is superseded by the completed [Phase62](phase62_review_phase63_20260928.md), [Phase63](phase63_closeout_20260930.md), [Phase64](phase64_closeout_20261001.md) and [Phase65](phase65_closeout_20261003.md) reviews. Different cohorts, seeds, source corrections and metric definitions prevent pooling these studies as one learning curve.

| Evidence family | Decision-relevant finding |
| --- | --- |
| Migration, early pilots, Stage A, Phases34–39 | Establish contracts and small local recovery signals, not modern recursive physics quality. Phase34 confirmations did not establish a superior earlier pretraining checkpoint. |
| Phase40/40r1 | Doubling training data also doubled updates and changed validation cohort. This does not isolate a training-size effect. |
| Phases41–47 | Pointer and adaptation changes were sparse/mixed. Phase42 longer historical pretraining did not improve its primary endpoint. Phase44 has seed deviations; corrected source boundaries limit pooling. |
| Phases48–51 | Phase48 did not execute PID adaptation; repaired treatments did not yield a replicated joint topology benefit. |
| Phases52–54 | Recovery doses did not establish a useful recursive gain. Phase53 has a shared depth-two component, not a treatment advantage or coherent forest. |
| Phases55–58 | Parent/PID effects were mixed; missing controls in56/58 and strict-selection contamination in57 prevent controlled quality conclusions. |
| Phases59–60 | Phase59 is a25-event feasibility pilot; Phase60 failed before the late treatment and has no reconstruction endpoints. |
| Phases61–62 | Stable completion under early PID0.5, but mixed shallow quality and no late-PID winner. Phase62 control passes original gates without establishing deep quality. |
| Phase63 | Corrected recovery versus masking ties at3/155 exact full components, with no deep success. Masking is a simpler baseline, not a demonstrated quality winner. |
| Phase64 | Train-only rescaling durably repairs measured saturation, but neither arm passes all gates or recovers a coherent forest. A single auxiliary depth-two component is exploratory. |
| Phase65 | Full freezing preserves repaired geometry but yields no exact nontrivial component in any view; half LCAG declines conditionally, and the extra forest has no mothers. Stop freeze-versus-late repetition as the immediate priority. |
| Phase66 | Removing generated contexts restores target support but exchanges strict nontrivial precision for recall; no exact depth-two success. Neither arm passes all gates. |
| Phase67 | Target preservation and precision/recall tradeoff repeat; primary exact components are zero. A shared auxiliary depth-two event does not select a regime. End this context contrast. |
| Phase68 | Hybrid encoder-only refinement contrast has nearly tied LCAG, shallow exact components and zero primary forests. A conditional full-source recall signal is exploratory; all pretraining benefit remains unidentified. |

**Training size:** hold 70,000 for the next bounded diagnostic. Phase40 changed size, updates and cohort together; no controlled size-by-compute curve establishes an immediate data bottleneck or rules out scaling. A future 35k/70k/140k study should separate equal-update from equal-exposure comparisons, fix source mixture and evaluate independent seeds. That future campaign is not submitted here.

**Independent validation:** keep it separate from training growth. Phase60 source-disjoint expansion was necessary after selection contamination and pool exhaustion. Phase69 excludes all 58,800 prior reservations and reserves fresh 1,000 selection/100 strict/20 nested beam events, leaving 40,100 unreserved. No further validation expansion is needed for this pair; repeated draws from these sources do not establish new-domain generalization.

**Representation and objectives:** prioritize falsifiable task-aligned representation/assembly hypotheses over simply longer pretraining, without claiming they already outperform data growth. Phase42 did not support longer historical duration, Phase64 repaired geometry without recursive utility, Phase65 freezing did not help, and Phase66–67 target support traded precision for recall without a regime winner. Phase68 now supplies the missing incremental encoder contrast, but shared refined PID/normalization and prior pretrained weights prevent any conclusion about all pretraining. Ordered within-branch negatives, actual decoder-state negatives, target representability, proposal support and irreversible merge errors remain distinct future hypotheses, not proven remedies.

## One bounded next campaign: Phase69

Confirm the same hybrid encoder contrast once with independent seed 20261005 and fresh cohorts. Hypothesis A is that incremental refinement reproduces its nontrivial source-recall signal and supports joint strict topology improvement; hypothesis B is that the apparent advantage is model/seed/cohort variation and objective/geometry gains do not transfer. Repeating the existing intervention once avoids selecting a new objective from a single noisy comparison. This does not resume context tuning or establish an untrained baseline.

Keep the same authenticated encoder artifacts, shared refined PID, train normalization, 70,000 training events, mixed probability 0.5, late adaptation after 2,188, 4,376 updates, 280,064 slots, zero new pretraining, architecture, target policy, inference/search, checkpoint selection and original gates. Record full retained LCAG, exact nontrivial components, nontrivial precision/recall, coherent forests and depth-at-least-two outcomes with paired event uncertainty. Require every original gate and joint nontrivial topology/source improvement plus nontrivial coherent forests for a useful-transfer claim. Isolated leaves, auxiliary selection or recall alone do not qualify. No bitwise common-prefix claim is made.

Exactly two guarded tasks request one H100 NVL, eight CPUs, 64 GiB and 36 hours each, no requeue: at most 72 allocated GPU-hours. Source, config, checkpoint and fresh-cohort hashes are frozen before admission and submission. Preserve native failures, truth-free inference, source exclusivity and train-only fitting. Acceptance and actual scheduling/startup observations are separate from scientific success. No campaign beyond this confirmation, sealed test, model promotion or automatic chain is authorized.

## Verification and publication

Focused scientific and frozen-source admission checks pass. Broad CPU, full documentation layouts, audit, privacy, export/link checks, desktop/narrow browser verification, exact branch/master CI and actual live downloads are recorded in the final manifest. The publication retains existing hard byte/file/work/runtime limits and all historical studies. The prior capacity plan is remeasured for this archive; unchanged checks remain mandatory.


## Phase69 accepted scheduling snapshot

The two bounded Phase69 tasks were accepted and released from source 54cc4c89af8c0792e09d25f5c64a664660d74b96. The 2026-10-05 08:45 Asia/Shanghai snapshot verifies both PENDING (Priority), with exact contract/resource/no-requeue bindings. GPU startup is not yet verified. Acceptance is not scientific success.
