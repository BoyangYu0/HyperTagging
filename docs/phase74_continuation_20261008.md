# Phase74 fresh-cohort development continuation

The user explicitly authorized designation of fresh non-primary development
identities after the original admission failed. That failure remains immutable:
all60 old candidates overlap Phase71 primary. It is superseded, not repaired by
relabeling those identities. No primary reservation or sealed test is used.

Metadata admission authenticated the original70000-event train corpus and160000
validation identities. The transitive historical exclusion union contains111000
validation identities. A deterministic SHA256 ordering,seed202610081,designates
600 fresh held-out development events,100 in each required category,without
reading outcome labels or restricting to easy/complete events. The shared
supplementary reservation registry records stage=development. Fixed final
checkpoints and threshold0.5 avoid adaptive held-out checkpoint/threshold tuning.
The remaining taupair validation category is not part of this six-category assay.

| Category | Validation identities | Previously excluded | Available before designation | New development |
| --- | ---: | ---: | ---: | ---: |
| charged | 20000 | 14079 | 5921 | 100 |
| mixed | 20000 | 14099 | 5901 | 100 |
| ccbar | 30000 | 20175 | 9825 | 100 |
| uubar | 30000 | 20222 | 9778 | 100 |
| ddbar | 20000 | 14087 | 5913 | 100 |
| ssbar | 20000 | 14167 | 5833 | 100 |

## Scientific comparison

Exactly four arms compare128-existing,128-assembly,256-existing,256-assembly.
Fresh initial states share the same seed and controlled architecture; objective
arms at each width load the same immutable initialization. Hyperbolic width32,
depth4,curvature1,radius convention, tangent cap4 and tangent variance target0.05
stay fixed. FFN width is four times contextual width; dropout is zero.

The existing arm calls the native topology/parent/distance/radius/channel,
variance/covariance,PID and corruption objectives with their native coefficients
and progressive200/250/300/250-step curriculum. Its teacher/corrupted training
views remain explicitly identified as such. They are not called generated states.
The assembly arm adds unordered optional two-group membership with unassigned,
and within-B categorical discrimination between exact direct-parent unions,
incomplete sibling unions,and other same-B relations. Truth supplies targets
only after model-generated proposals exist; legitimate relatives are not globally
repelled. Detector-only and genuinely generated latent partial states are both
supervised.

Generated partial states use two rounds,at most four disjoint model-ranked merges
per round. Their groups contain detector-source-exclusive FSP indices; they are
latent membership states,not persisted physical mothers. Their pooled encoder
representations and predicted soft global memberships explicitly condition the
common downstream decoder. The128-wide features are zero-padded to256,without
losing components. Every downstream decoder has1385991 parameters and is reset
to the same initialization after pretraining,so pretrained auxiliary-head history
cannot confound the encoder contrast. All four encoders adapt downstream;
PID-head weights remain frozen in downstream training. No truth teacher enters
this inference interface. Physical particle/PID construction semantics are not
replaced by the latent grouping diagnostic.

## Bounded budgets and evaluation

The shared training subset contains1536 original training events,256/category.
It is not full70000-event training. Normalization is the authenticated unchanged
train-fitted contract; historical model parameters are never transferred.
Each arm uses1000 pretraining updates,batch8 (8000 presentations),then1500 main
downstream updates,batch8 (12000 presentations). An independent copy of the
post-pretraining encoder runs1000 tiny updates,batch8 on24 training events
(8000 presentations). Tiny training cannot alter the main model or its history.
All sampling orders and fixed-final checkpoint rules match across arms.

The extra detector/generated-state views are presented in both pretraining arms;
only assembly adds their gradients. Updates/presentations are therefore controlled,
but FLOPs are not equal. Record parameters, view/encoder-pass counts, node-pair
workload proxies, wall/CPU time, peak memory and losses. No FLOP measurement is
claimed. CPU allocations are bounded to four jobs,2CPU/32GiB/6h each,no GPU and
no requeue. Real-data gradient and worst-size timing admission precedes submission.

Count raw and accepted tiny/held-out memberships,unknown targets,within-B relation
classes,continuum flat-group acceptance,charge/cardinality rejections and detector
source conflicts. Report latent exact-source-set proposal/retention for known
retained nodes of generation height at least two. This is not proof of legal
physical deep-proposal reachability. Inclusive source-membership top1 refers to
the single predicted partition. Exact-B hierarchy top1/pool,inclusive beam pool
and physical daughter-sum closure are unavailable because this decoder constructs
neither physical mothers nor a beam pool; unavailable results are not zero successes.

The600 held-out events contain400 nominal B trials and400 continuum collisions.
This is development,not policy-sized primary evaluation. The comparison script
reports width/objective effects and their interaction with collision-paired
bootstrap intervals and sparse zero-success event bounds. One seed and sparse
successes limit inference. At least95% tiny raw exactness plus count-backed
held-out gain with background/validity controls is required before proposing
primary scale-up. There is no automatic follow-on,hierarchy campaign,512-wide
model,coefficient grid or promotion.

The [machine-readable plan](../configs/reconstruction/phase74_capacity_development_plan.json)
and guarded runtime bind source,data,normalization,initializations and resources.
Submission requires all four arms,authenticated cohort isolation,real gradients,
immutable hashes and an atomic duplicate-submission lock. Original Phase73 and
failed Phase74 evidence remain unchanged.
