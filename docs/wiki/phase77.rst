Phase77 training-only proposal diagnosis
========================================

The completed :doc:`Phase76 <phase76>` context ablation did not improve exact
membership. This separately authorized pass follows its recommendation to
investigate true-B constituents assigned unassigned, with explicit background
controls. It selects **no new training**: the diagnosis does not isolate a
supported single training intervention. No fresh validation is consumed or
reserved, and no primary evaluation, model promotion or automatic successor runs.

Reproducible scope
-------------------

The fixed context-on final main and independent tiny checkpoints are replayed
on all1536 training-pool identities and24 tiny-fit views drawn from that pool.
There are256 training collisions in each of six categories; tiny has eight each
of charged, mixed and ccbar. The original70000-event corpus, widths128/32,
depth four,256-wide decoder, train-fitted normalization, PID and source/charge
rules remain fixed. Every detached proposal/refinement probability reproduces
the previous immutable trace exactly. Labels enter only after generation.
The shared authenticated cache contains prior development rows, but those rows
are discarded and never iterated, evaluated or used for this hypothesis choice.

Before inference, the diagnostic registers one aggregate-foreground decision,
not a threshold grid, and96 gradient probes chosen by UID hash,16 per category.
No optimizer step is taken. Existing native train/tiny acceptance, relation,
latent-source and channel metrics are reused after exact output verification;
all previous held-out results retain their original historical scope.

Measured failure
-----------------

Of9437 true-B source assignments in512 collisions,3437 are correctly assigned,
4172 are unassigned and1828 go to the other B at the proposal stage. Refinement
has3205 correct,4480 unassigned and1752 other-B assignments. Across all six
categories,2981 of52590 background sources are assigned a proposal B slot;
refinement reduces this to2491. These are source assignments, not accepted
fake-B events. Within B collisions, proposal background extras are1536; adding
1828 other-B sources reproduces the earlier3364 extra-source count. Refinement
has1209 background extras plus1752 other-B sources, reproducing2961.
Both stages recover0 of1024 raw exact B sets; native final
accepted recovery is also0. Tiny raw is32 of32 and accepted27 of32, with five
truth targets incompatible with the unchanged charge guard.

Only709 of the4172 unassigned B sources have combined B probability above0.5.
The prespecified aggregate-B decision recovers392 correct-slot assignments and
317 additional other-B assignments, while adding882 background-to-B assignments.
Raw exact recovery stays0 of1024. Foreground precision decreases3.12 percentage
points, with a descriptive paired collision interval of-3.65 to-2.61 points.
Resampling preserves whole collisions within categories; correlated nodes and
B trials are not independent samples. The zero-exact bootstrap interval is
degenerate and does not establish population equivalence.

Foreground rank AUC is0.906 for proposals and0.911 after refinement, pooled
across training nodes. This is not a held-out calibration claim. Unassigned B
sources have median winning-B minus background logit-1.377, so most omissions
are not simply a near-tie caused by splitting B probability across slots.
Under the native hard-error event permutation, conditional B-slot classification
is correct for5676 of9437 B nodes. Even a
truth-assisted perfect foreground mask followed by the model's B-slot choice
recovers0 exact sets. Correcting only the B-slot swaps with truth, while keeping
native omissions and extras, yields37 of1024. These are labelled oracle
diagnostics, not deployable efficiencies, a correct model-produced pool, or
proof of legal physical decoder reachability.

All size bins fail exact recovery. Their B-trial denominators are61,413,518
and32 for sizes1–4,5–8,9–16 and17 or more. Complete counts and margins by size
and category, including all background categories, remain in the download.

Supervision and optimization audit
------------------------------------

All62027 training detector nodes have known membership supervision here;
unknown-label masking is nevertheless covered by negative tests. Native loss
chooses a permutation using CE, soft-Jaccard and presence. The hard-error
matching used in diagnostics optimizes a different criterion and disagrees
in120 of512 B collisions at proposal stage,88 after refinement. Loss and
permutation-gradient regression tests pass; disagreement alone is not a bug.
Presence is applied later for acceptance and cannot explain zero raw recovery.

True-B nodes contribute321.58 of506.81 summed event-normalized proposal CE,
versus185.24 from background, despite background being more numerous. Summed
squared CE logit-gradient norms are4.254 versus2.021. Overlap gradients also
reach true-B nodes. Actual proposal-parameter gradients are nonzero on all96
training probes, and the refinement path reaches them too. These quantities
do not measure cancellation of the full dataset parameter gradient. They do
not support declaring background domination or repeating gradient projection.

The last100 main updates have mean loss1.2746 versus1.1885 in the preceding
window. Convergence is unproven, not disproven; noisy final windows and tiny
memorization do not establish that longer training, data growth or new
pretraining would help. The fixed main history has1500 updates and12000
presentations, covering1535 distinct identities from the1536 pool. Independent
tiny fitting has1000 updates and8000 presentations. All curves remain preserved.

Decision, resources and limitations
-------------------------------------

No target/interface implementation defect was demonstrated. Foreground and
B-slot failures coexist; the measured decision change worsens background
assignment and does not repair exact membership. A new loss weight or threshold
would be speculative. Keep corpus, widths and guards fixed. Before another
scientific training, establish a specific mechanism for weak within-event
B-slot partition learning using training-only conditional assignment and
pair-separability diagnostics with background controls. This pass does not
launch that additional investigation or another campaign.

The diagnostic completed on two CPUs with16GiB and a one-hour limit, no GPU
and no requeue. Its model section takes148.90 wall seconds and147.86 process
CPU seconds, peak RSS1499336KiB. Scheduler elapsed time is171 seconds. It runs
1560 forward views and96 additional gradient views, with1322658 encoder and
1385991 decoder parameters. Node-squared proxies are reported separately from
actual CPU/memory accounting and are not FLOPs. There are zero new fit updates.

This is training-only evidence at one final checkpoint and seed, not fresh
confirmation or a complete post-study evaluation. Primary evaluation still
requires at least2000 distinct processed collisions in each of six categories
and the development gates; primary scale-up remains closed. Exact physical
trees, physical beam pools, physical deep survival and p4 closure remain
unavailable for flat optional-B groups. Retained-source and latent diagnostics
are not physical FEI efficiency.

See the :doc:`dashboard <_generated/status/index>` and its lossless
:doc:`downloads <_generated/status/downloads>` for all supported metrics,
uncertainty, source bindings, native channel coverage and historical evidence.
