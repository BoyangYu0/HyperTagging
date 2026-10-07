Phase71 auxiliary teacher objective and efficiency review
=========================================================

The same original reserved cohort supplies **12,000 processed collisions per arm**, **2,000 in each required category**, in full and half scope. Checkpoint selection uses a separate 1,000 collisions. All failures and unavailable truth retain their original denominators; no difficult event is replaced. The 70,000-event training set and fitted normalization are unchanged.

The contrast is auxiliary teacher objective weight 0.5 versus 1.0, with matched mixed contexts, enabled decoder bias, source-exclusive inference and bounded updates. These retained-source metrics are not physical FEI or original quark reconstruction efficiencies.

See the :doc:`dashboard <_generated/status/index>` and :doc:`complete download catalogue <_generated/status/downloads>`. Historical studies and all native diagnostic gates remain preserved.

Neither Phase71 arm produces an exact retained B tag or inclusive FSP group: 0/8,000 nominal B trials in both full and half scope, with zero any-B, both-B and coherent-pair successes. Thirteen unavailable exact-truth trials remain nominal; B membership is available for all 8,000. Weight 1.0 reduces exact nontrivial components from 308/18,536 to 148/18,536, full retained LCAG from 3,147/726,498 to 2,939/726,498, and nontrivial source recall from 32,124/112,959 to 31,260/112,959. Its slight nontrivial precision increase and fake-B decrease (293/8,000 to 281/8,000 continuum events) do not establish a joint efficiency winner.

Primary tagging and coherent pairs
----------------------------------

.. list-table:: Strict greedy; nominal denominators include unavailable truth
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Teacher weight 0.5
     - Teacher weight 1.0
   * - full exact per_b_correct
     - 0/8000 (0%)
     - 0/8000 (0%)
   * - full exact event_any_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full exact event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full exact coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full inclusive per_b_correct
     - 0/8000 (0%)
     - 0/8000 (0%)
   * - full inclusive event_any_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full inclusive event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - full inclusive coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half exact per_b_correct
     - 0/8000 (0%)
     - 0/8000 (0%)
   * - half exact event_any_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half exact event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half exact coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half inclusive per_b_correct
     - 0/8000 (0%)
     - 0/8000 (0%)
   * - half inclusive event_any_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half inclusive event_both_correct
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - half inclusive coherent pair
     - 0/4000 (0%)
     - 0/4000 (0%)

.. list-table:: Full-scope B truth and channel accounting; half scope is separately exported
   :header-rows: 1
   :widths: auto

   * - Arm
     - Definition
     - Unknown nominal B trials
     - Channel rows
     - Recovered channels
     - Nominal channel trials
   * - aux_teacher_050
     - exact
     - 13
     - 7855
     - 0
     - 8000
   * - aux_teacher_050
     - inclusive
     - 0
     - 7855
     - 0
     - 8000
   * - aux_teacher_100
     - exact
     - 13
     - 7855
     - 0
     - 8000
   * - aux_teacher_100
     - inclusive
     - 0
     - 7855
     - 0
     - 8000

Each B-pair collision contributes two nominal B trials. Distinct truth B successes are deduplicated across every accepted candidate. Inclusive success requires an actual valid reconstructed composite with precisely the retained FSP membership; no disconnected partition is invented. Exact topology/PID availability and inclusive membership availability remain separate. Missing truth makes observed success rates proven-success lower bounds. Channel identifiers describe the retained signed representation; full generator decay descriptions and original quark ancestry are unavailable.

Continuum recovery and fake-B background
----------------------------------------

.. list-table:: Explicit retained components; fake slots use two nominal slots per collision
   :header-rows: 1
   :widths: auto

   * - Population and endpoint
     - Teacher weight 0.5
     - Teacher weight 1.0
   * - full ccbar exact
     - 22/6829 (0.32216%)
     - 19/6829 (0.27823%)
   * - full ccbar inclusive
     - 240/6829 (3.5144%)
     - 240/6829 (3.5144%)
   * - full ccbar fake-B event
     - 55/2000 (2.75%)
     - 78/2000 (3.9%)
   * - full ccbar fake-B slot
     - 58/4000 (1.45%)
     - 88/4000 (2.2%)
   * - full uubar exact
     - 56/5717 (0.97953%)
     - 39/5717 (0.68218%)
   * - full uubar inclusive
     - 455/5717 (7.9587%)
     - 442/5717 (7.7313%)
   * - full uubar fake-B event
     - 86/2000 (4.3%)
     - 62/2000 (3.1%)
   * - full uubar fake-B slot
     - 88/4000 (2.2%)
     - 62/4000 (1.55%)
   * - full ddbar exact
     - 69/5752 (1.1996%)
     - 52/5752 (0.90403%)
   * - full ddbar inclusive
     - 534/5752 (9.2837%)
     - 493/5752 (8.5709%)
   * - full ddbar fake-B event
     - 73/2000 (3.65%)
     - 56/2000 (2.8%)
   * - full ddbar fake-B slot
     - 77/4000 (1.925%)
     - 57/4000 (1.425%)
   * - full ssbar exact
     - 121/5863 (2.0638%)
     - 33/5863 (0.56285%)
   * - full ssbar inclusive
     - 752/5863 (12.826%)
     - 742/5863 (12.656%)
   * - full ssbar fake-B event
     - 79/2000 (3.95%)
     - 85/2000 (4.25%)
   * - full ssbar fake-B slot
     - 85/4000 (2.125%)
     - 88/4000 (2.2%)
   * - half ccbar exact
     - 15/6829 (0.21965%)
     - 11/6829 (0.16108%)
   * - half ccbar inclusive
     - 240/6829 (3.5144%)
     - 240/6829 (3.5144%)
   * - half ccbar fake-B event
     - 55/2000 (2.75%)
     - 78/2000 (3.9%)
   * - half ccbar fake-B slot
     - 58/4000 (1.45%)
     - 88/4000 (2.2%)
   * - half uubar exact
     - 47/5717 (0.82211%)
     - 35/5717 (0.61221%)
   * - half uubar inclusive
     - 455/5717 (7.9587%)
     - 442/5717 (7.7313%)
   * - half uubar fake-B event
     - 87/2000 (4.35%)
     - 62/2000 (3.1%)
   * - half uubar fake-B slot
     - 89/4000 (2.225%)
     - 62/4000 (1.55%)
   * - half ddbar exact
     - 63/5752 (1.0953%)
     - 44/5752 (0.76495%)
   * - half ddbar inclusive
     - 534/5752 (9.2837%)
     - 493/5752 (8.5709%)
   * - half ddbar fake-B event
     - 73/2000 (3.65%)
     - 56/2000 (2.8%)
   * - half ddbar fake-B slot
     - 77/4000 (1.925%)
     - 57/4000 (1.425%)
   * - half ssbar exact
     - 108/5863 (1.8421%)
     - 28/5863 (0.47757%)
   * - half ssbar inclusive
     - 752/5863 (12.826%)
     - 742/5863 (12.656%)
   * - half ssbar fake-B event
     - 79/2000 (3.95%)
     - 85/2000 (4.25%)
   * - half ssbar fake-B slot
     - 85/4000 (2.125%)
     - 88/4000 (2.2%)

Component recovery is not q/qbar recovery: reduced data lack parton-to-FSP ancestry. Background acceptance and reconstruction correctness use different denominators. Unavailable component truth is retained in the complete aggregate tables. All accepted B candidates are inspected for tag recovery; slot-based fake-B accounting remains distinct from the total accepted-candidate count.

Retained topology and paired uncertainty
----------------------------------------

.. list-table:: All retained versus nontrivial populations
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Teacher weight 0.5
     - Teacher weight 1.0
   * - full lcag_pair_accuracy
     - 3147/726498 (0.43317%)
     - 2939/726498 (0.40454%)
   * - full source_precision
     - 379267/416516 (91.057%)
     - 379393/415937 (91.214%)
   * - full source_recall
     - 379267/481553 (78.759%)
     - 379393/481553 (78.785%)
   * - full coherent_retained_forest
     - 41/12000 (0.34167%)
     - 50/12000 (0.41667%)
   * - full nontrivial perfectLCAG
     - 308/18536 (1.6616%)
     - 148/18536 (0.79845%)
   * - full nontrivial source_precision
     - 32124/42002 (76.482%)
     - 31260/40587 (77.02%)
   * - full nontrivial source_recall
     - 32124/112959 (28.439%)
     - 31260/112959 (27.674%)
   * - half lcag_pair_accuracy
     - 3518/394119 (0.89262%)
     - 3447/394119 (0.87461%)
   * - half source_precision
     - 274974/309177 (88.937%)
     - 275142/308688 (89.133%)
   * - half source_recall
     - 274974/369890 (74.339%)
     - 275142/369890 (74.385%)
   * - half coherent_retained_forest
     - 41/12000 (0.34167%)
     - 50/12000 (0.41667%)
   * - half nontrivial perfectLCAG
     - 265/22534 (1.176%)
     - 122/22534 (0.5414%)
   * - half nontrivial source_precision
     - 35976/51781 (69.477%)
     - 35740/51675 (69.163%)
   * - half nontrivial source_recall
     - 35976/112957 (31.849%)
     - 35740/112957 (31.64%)

.. list-table:: Exact shapes are not interchangeable with deep B reconstruction
   :header-rows: 1
   :widths: auto

   * - Arm
     - Scope
     - Depth-one exact components
     - Depth-two-or-deeper exact components
     - Forests with mothers
     - Mother-free forests
   * - aux_teacher_050
     - full
     - 307
     - 1
     - 21
     - 20
   * - aux_teacher_050
     - half
     - 264
     - 1
     - 21
     - 20
   * - aux_teacher_100
     - full
     - 148
     - 0
     - 28
     - 22
   * - aux_teacher_100
     - half
     - 122
     - 0
     - 28
     - 22

.. list-table:: Weight1.0 minus0.5; paired stratified collision bootstrap, 10,000 resamples
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Difference
     - 95% percentile interval
   * - full exact_tag per_b_correct
     - 0.0
     - [0.0, 0.0]
   * - full inclusive_group per_b_correct
     - 0.0
     - [0.0, 0.0]
   * - full retained lcag_pair_accuracy
     - -0.00028630498638674857
     - [-0.00041028059226495134, -0.00016416543168465075]
   * - full nontrivial_retained perfectLCAG
     - -0.008631851532153647
     - [-0.010738886375355266, -0.006571495063519596]
   * - full nontrivial_retained source_precision
     - 0.005376631009785249
     - [0.00011343758181140433, 0.010705573598109296]
   * - full nontrivial_retained source_recall
     - -0.007648792924866532
     - [-0.010629065926935648, -0.004734483987440866]
   * - full retained coherent_forest
     - 0.0007499999999999998
     - [-0.00041666666666666675, 0.0019166666666666663]
   * - full pooled_continuum exact_component_recovery
     - -0.005173626919415587
     - [-0.006680315107885581, -0.003691552415897226]
   * - full pooled_continuum inclusive_component_recovery
     - -0.0026488969827407893
     - [-0.005165155906974453, -0.00012370893484947552]
   * - full pooled_continuum top1_fake_b_event_acceptance
     - -0.0014999999999999944
     - [-0.006874999999999999, 0.003999999999999997]

The downloads contain both-arm and difference intervals for every registered paired endpoint, including each continuum category, coherent pairs and fake-B slots. Both B trials stay within their collision cluster. Sparse category-specific event-any binomial bounds accompany zero/single successes. Degenerate zero-success bootstrap intervals do not prove zero population efficiency.

Coverage, unavailable truth and immutable lineage
-------------------------------------------------

.. list-table:: Identical authenticated populations in both arms and scopes
   :header-rows: 1
   :widths: auto

   * - Category and scope
     - Requested
     - Attempted
     - Processed
     - Failed
     - Tree unavailable
     - Exact-tag unavailable events
     - Membership unavailable events
   * - charged full
     - 2000
     - 2000
     - 2000
     - 0
     - 0
     - 4
     - 0
   * - charged half
     - 2000
     - 2000
     - 2000
     - 0
     - 19
     - 4
     - 0
   * - mixed full
     - 2000
     - 2000
     - 2000
     - 0
     - 0
     - 9
     - 0
   * - mixed half
     - 2000
     - 2000
     - 2000
     - 0
     - 20
     - 9
     - 0
   * - ccbar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 6
     - 6
   * - ccbar half
     - 2000
     - 2000
     - 2000
     - 0
     - 13
     - 6
     - 6
   * - uubar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 9
     - 9
   * - uubar half
     - 2000
     - 2000
     - 2000
     - 0
     - 590
     - 9
     - 9
   * - ddbar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 2
     - 2
   * - ddbar half
     - 2000
     - 2000
     - 2000
     - 0
     - 571
     - 2
     - 2
   * - ssbar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 2
     - 2
   * - ssbar half
     - 2000
     - 2000
     - 2000
     - 0
     - 422
     - 2
     - 2

The primary full-root tree metric is unavailable for continuum collisions because it does not invent a B-pair root. Explicit retained-component metrics remain separately measurable. Half-scope direct-target compatibility and exact/inclusive tag availability have their own unavailable counts; none of these events is removed.

All 85,000 prior reservations and the separate 1,000 selection events are excluded from this original 12,000-event primary cohort. The supplementary registry authenticates the combined 13,000-event reservation. Input hashes cover 70,000 training and 160,000 validation identities, with no sealed-test payload access. Original training payloads and train-only normalization authenticate against the Phase70 lineage. Expanded independent validation is not training growth.

The original selection-only checkpoints are step 3,000 for weight 0.5 and step 4,000 for weight 1.0. No primary result is used to select another checkpoint. Shared cardinality limits 17 globally, 12 at level 1 and 17 at level 2 admit the four previously exposed targets in both arms. Their recorded repair prevents causal Phase70-to-Phase71 attribution.

The immutable native cohort also retains descriptive pre-repair index metadata and an inherited selection path. Actual native job contracts and the primary evaluator bind the repaired index and expanded selection independently by hash. The discrepancy is recorded, not silently edited; original memberships and training payloads are preserved. Phase72 writes current descriptive bindings.

Native source: ``1545d8ee80f26bc291498ae8718cb6e4a984f0bb``. Primary evaluator: ``9b0d97f663bd382e7a048c90e2670147217dcd97``. Cohort digest: ``c1da7c69dfbf8d414f4b8b683aac8d8a0404b9c842ad50df41e3e08f4d1bc3bf``. Every input, report, checkpoint and chunk is hash-bound. Full and half scopes, repeated reports and candidates do not multiply collision coverage.

Native diagnostic tracks, geometry and execution
------------------------------------------------

.. list-table:: Original gates are retained, not replaced by primary efficiency
   :header-rows: 1
   :widths: auto

   * - Arm
     - Updates
     - Selected checkpoint
     - All original gates pass
     - Failed gates
     - Exact repeat
   * - aux_teacher_050
     - 4376
     - 3000
     - True
     - none
     - True
   * - aux_teacher_100
     - 4376
     - 4000
     - False
     - minimum_half_perfect_lcag
     - True

.. list-table:: Registered native full/half views: actual diagnostic category counts
   :header-rows: 1
   :widths: auto

   * - Arm
     - View
     - Collisions
     - Category counts
   * - aux_teacher_050
     - independent_complete_target_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_050
     - independent_depth_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_050
     - independent_tree_validity_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_050
     - primary_complete_target_beam_direct
     - 20
     - ccbar:1, charged:1, ddbar:5, mixed:5, ssbar:2, uubar:6
   * - aux_teacher_050
     - primary_complete_target_contracted_diagnostic
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_050
     - primary_complete_target_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_050
     - primary_complete_target_repeat2_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_100
     - independent_complete_target_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_100
     - independent_depth_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_100
     - independent_tree_validity_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_100
     - primary_complete_target_beam_direct
     - 20
     - ccbar:1, charged:1, ddbar:5, mixed:5, ssbar:2, uubar:6
   * - aux_teacher_100
     - primary_complete_target_contracted_diagnostic
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_100
     - primary_complete_target_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29
   * - aux_teacher_100
     - primary_complete_target_repeat2_direct
     - 100
     - ccbar:8, charged:12, ddbar:24, mixed:13, ssbar:14, uubar:29

.. list-table:: Actual training exposure; matched updates are not equal FLOPs
   :header-rows: 1
   :widths: auto

   * - Arm
     - Teacher slots
     - Predicted slots
     - Target exposures
     - Representable
     - Unrepresentable
     - Model forwards
   * - aux_teacher_050
     - 175106.0
     - 104958.0
     - 442666.0
     - 331045.0
     - 111621.0
     - 392565.0
   * - aux_teacher_100
     - 175106.0
     - 104958.0
     - 442666.0
     - 331061.0
     - 111605.0
     - 395369.0

Both arms execute 4,376 updates and 280,064 replay slots, seed 20261007, with zero additional pretraining. All 26 saved checkpoints retain exhaustive scalar exports, finite-state and frozen-PID lineage checks. Missing strict evaluations of unregistered tracks remain unavailable. All registered auxiliary full/half reports and every returned candidate are preserved.

Independent current-CPU replays preserve all original scientific counts and prediction fingerprints. The original native repeat gates pass exactly. Against the archived runtime, 9,446 continuous kinematic or score values differ, with maximum absolute difference 1.36e-5; these bounded differences and original values are preserved separately. No bitwise archived tensor equality is claimed.

Geometry diagnostics use 128 fixed training and 32 previously used development collisions across four curriculum views and initial/selected/final encoders. They are diagnostic, not independent primary coverage or a census of generated states. Their exact sample counts, per-level variance, radial derivatives, saturation and gradient/transfer diagnostics remain in the complete bundle. Persistent p4 is an exact daughter sum; closure is not generator momentum resolution.

.. list-table:: Geometry input samples: actual category counts
   :header-rows: 1
   :widths: auto

   * - Population
     - Collisions
     - Category counts
   * - development
     - 32
     - {'ccbar': 7, 'charged': 3, 'ddbar': 7, 'mixed': 2, 'ssbar': 2, 'taupair': 6, 'uubar': 5}
   * - train
     - 128
     - {'ccbar': 13, 'charged': 22, 'ddbar': 17, 'mixed': 20, 'ssbar': 20, 'taupair': 21, 'uubar': 15}

Diagnostic full-depth beam and candidate accounting
---------------------------------------------------

.. list-table:: Width-two beam: 60 collisions, 10/category; not policy-complete
   :header-rows: 1
   :widths: auto

   * - Arm
     - Scope
     - Collisions
     - Category counts
     - Returned candidates
   * - aux_teacher_050
     - full
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120
   * - aux_teacher_050
     - half
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120
   * - aux_teacher_100
     - full
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120
   * - aux_teacher_100
     - half
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120

.. list-table:: Beam subset efficiencies:40 nominal B trials,20 B-pair collisions
   :header-rows: 1
   :widths: auto

   * - Arm
     - Scope and definition
     - Selection
     - Per B
     - Any B
     - Both B pooled
     - Coherent pair
   * - aux_teacher_050
     - full exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - full exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - full exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - full inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - full inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - full inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - half exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - half exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - half exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - half inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - half inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_050
     - half inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - full exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - full exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - full exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - full inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - full inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - full inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - half exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - half exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - half exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - half inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - half inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - aux_teacher_100
     - half inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)

Greedy, model-ranked top1 and retained pool at each K remain separate. Pool recovery deduplicates each true B and inspects every accepted candidate; incompatible successes are not a coherent pair. The complete download includes all beam/oracle, rank/score, candidate and uncertainty records, plus native 20-event proposal-beam diagnostics. Neither small subset inherits primary coverage.

Synthesis of all studies
------------------------

.. list-table:: Study families and negative evidence
   :header-rows: 1

   * - Family
     - Evidence and consequence
   * - Historical implementations and migration phases 1–13
     - Preserve software equivalence and distinct legacy denominators. They are not modern trained-physics comparisons. No standalone reconstruction closeouts numbered 14–33 were established by the historical inventory.
   * - Early production, pretraining and resource/transfer pilots
     - Data, runtime and transfer contracts were established with recorded failures and finite checkpoints. Throughput and objective stability are not recursive quality.
   * - Stage A and phases 34–39
     - Local relation/pointer and replay gains did not establish deep reconstruction or satisfy a promotion-grade joint endpoint.
   * - Phase40 and its recovery
     - Size, compute and evaluation population changed together. This is not a controlled training-data learning curve.
   * - Phases41–47
     - Pointer, pretraining duration and encoder variations produced mixed sparse results; scientific repairs limit comparability. Longer historical pretraining is not all pretraining versus none.
   * - Phases48–51
     - Phase48 did not execute its PID treatment. Later repaired treatments execute, but do not establish joint topology improvement.
   * - Phases52–54
     - Recovery-weight changes alter local counts while exact components and forests stagnate or regress. More dose tuning is not supported.
   * - Phases55–58
     - Evaluator repairs, missing controls and Phase57 selection contamination limit causal conclusions. Preserve failures and unavailable endpoints.
   * - Phases59–62
     - Phase59 is a 25-event feasibility pilot; Phase60 fails before treatment. Completed later schedules retain mixed shallow results and pre-treatment variation. Equal seeds are not identical realized prefixes.
   * - Phase63
     - Corrected forest-root target eligibility repairs supervision. This is necessary correctness work, not demonstrated end-to-end efficiency.
   * - Phase64
     - Radial reconditioning repairs measured saturation. A geometry repair does not demonstrate useful recursive assembly.
   * - Phase65
     - Freeze versus late adaptation has mixed sparse results and no primary exact nontrivial component. Geometry remains usable without establishing tags.
   * - Phases66–67
     - Teacher-only preserves supervision that generated contexts lose. The precision/recall tradeoff does not establish a joint recursive winner.
   * - Phases68–69
     - Hybrid refined versus pre-refinement encoder contrasts share refined PID and normalization. The confirmation does not establish a joint advantage or identify the total benefit of pretraining.
   * - Phase69 category supplement
     - Exact retained tagging is zero of 8,000 nominal B trials in each arm; inclusive grouping is zero versus one. This supersedes neither its historical 16-collision diagnostic nor the limitations of a single fitted pair.
   * - Phase70
     - The module intervention executes; its category-sized exact tags are0/8000 in both arms and inclusive tags0 versus1/8000. Better shallow recovery trades against recall and background. No joint efficiency winner.

The historical review and all dated studies remain in :doc:`studies` and the
complete download catalogue. Comparisons across different seeds, cohorts,
source mixtures, evaluator revisions or target policies are descriptive.
Phase69 and Phase70 cannot be treated as a paired cross-study comparison.
Neither full/half scopes, checkpoint tracks, repeated inference nor nested
beam pools multiply the collision sample size.


Phase71 adds a category-sized auxiliary teacher objective contrast to that evidence. Its two arms can be compared within their shared cohort; historical cohorts, seeds, source domains, repaired capacities and evaluator versions must not be pooled as causal effects. Negative replications, failed controls and unavailable outcomes remain evidence.

Decision, uncertainty and one bounded successor
-----------------------------------------------

Hold the training set at 70,000 for one bounded objective-alignment experiment. Prioritize improved representation/task supervision before training-data growth now; the immediate experiment changes reconstruction supervision, not pretraining duration. Do not simply add pretraining steps. This allocation decision does not establish that improved pretraining is superior to scaling, that scaling cannot help, or that current validation precision is sufficient for rare tags. Enlarged independent evaluation improved measurement coverage without growing training data.

End auxiliary-teacher dose tuning. Phase71 has no observed exact or inclusive tagging gain, while stronger anchoring worsens nontrivial exact recovery and recall. Phase34/42 do not establish benefit from longer pretraining; repeated geometry, exposure, PID, hybrid-encoder and decoder studies have not demonstrated joint tagging improvement. Repaired unsaturated geometry and historical 70,000-event holds do not rule out scaling. Test a distinct local membership objective: matched-daughter soft-Jaccard off versus on, alongside unchanged focal pointer loss, with the existing weight-0.5 auxiliary reference in both arms. This directly tests joint omitted/foreign-daughter penalties while preserving truth-free inference and source exclusivity. It is a falsifiable local surrogate, not a demonstrated cure for recursive B reconstruction or a physical FEI claim. Do not pool historical cohorts as causal effects; keep single-seed and sparse-efficiency uncertainty explicit.

The paired intervals resample whole collisions within source category and condition on two fitted models. They do not estimate training-seed variation or provide multiplicity-adjusted discovery. Zero-width empirical bootstrap intervals do not show zero population efficiency: the downloads also report collision-level exact-binomial sparse-success bounds. Missing truth means proven-success lower bounds. Equal-category pooled results are not physical mixture estimates.

Phase72 tests the **presence of matched-daughter soft-Jaccard supervision**, coefficient 0 versus 1, alongside unchanged focal pointer supervision. The surrogate penalizes foreign and omitted daughters jointly on existing legal, matched target sets. It does not alter truth-free inference, matching assignments, source exclusivity or target eligibility. It is local daughter membership, not an end-to-end exact B objective.

Both arms use 70,000 training events, the same reconditioned encoder and frozen PID, mixed context 0.5, auxiliary teacher weight 0.5, enabled bias, late adaptation after 2,188 updates and 4,376 total updates/280,064 slots; seed 20261008. No extra pretraining is purchased. Resources are bounded to one H100 NVL, 8 CPUs, 64 GiB and 36 hours per arm, with no requeue.

Primary endpoints are exact/inclusive per-B and any/both/coherent-pair efficiencies with channel coverage, continuum recovery, fake-B acceptance, source precision and recursive structural guards. The useful-effect target is at least 4 additional successes per 8,000 nominal B trials with positive paired lower bounds and no background/precision/structure deterioration; this is not a power guarantee. Native diagnostic gates remain unchanged.

A fresh 12,000-event cohort, 2,000/category, and separate 1,000 selection events exclude the 98,000 prior reservations. Reservation alone is not future processed coverage. Admission, frozen-source hashes and accepted scheduling are recorded separately. There is no automatic successor, sealed-test access or scientific model promotion.

At the 2026-10-07 09:38 CEST admission snapshot, both Phase72 jobs are accepted: set_overlap_off: PENDING (Resources), set_overlap_on: PENDING (Priority). Frozen source ``38124a9e05ca63cd5ac5469e0687952eec2b14dc`` passes 63 CPU admission tests and both native preflights. Startup and full training completion are not yet verified; scheduling is not scientific validation. Immutable private receipts bind both jobs, resources, configurations, checkpoint and fresh cohorts.

Complete aggregate downloads and integrity
------------------------------------------

:download:`Complete policy and scientific bundle <_generated/status/phase71-policy-reevaluation-v1.json>`; :download:`integrity <_generated/status/phase71-policy-integrity.json>`; :download:`standalone decoder <_generated/status/phase71-policy-decoder.txt>`.

:download:`All native aggregate scalars <_generated/status/phase71-native-aggregates.json>`; :download:`native integrity <_generated/status/phase71-native-integrity.json>`; :download:`native decoder <_generated/status/phase71-native-decoder.txt>`.

:download:`phase71-policy-part-0.json <_generated/status/phase71-policy-part-0.json>`.

:download:`phase71-policy-part-1.json <_generated/status/phase71-policy-part-1.json>`.

:download:`phase71-policy-part-2.json <_generated/status/phase71-policy-part-2.json>`.

The complete policy/scientific aggregate retains 1,756,039 numeric records; the native download retains 173,335 aggregate scalar records. Private scalar censuses retain 13,236,906 native values, 59,686 saved-checkpoint metadata values and 12,536,947 additive values, plus every policy report. These are exported records, not independent observations.

The lossless compact formats retain numeric counts, channel/type labels and null availability. Public archives omit collision identifiers, native machine paths, scheduler records, logs and weights. Private evidence retains full reports, logs, scalar censuses, native failures, hash inventories and reproducible receipts. Historical downloads and their immutable bytes remain preserved.
