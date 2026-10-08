Phase72 set-overlap objective and efficiency review
===================================================

The same original reserved cohort supplies **12,000 processed collisions per arm**, **2,000 in each required category**, in full and half scope. Checkpoint selection uses a separate 1,000 collisions. All failures and unavailable truth retain their original denominators; no difficult event is replaced. The 70,000-event training set and fitted normalization are unchanged.

The contrast adds matched-daughter soft-Jaccard weight 0 versus 1 alongside the same focal pointer objective, auxiliary teacher weight 0.5, mixed contexts and enabled decoder bias. The shared seed is 20261008, with 4,376 updates and no extra pretraining. Inference and target eligibility are unchanged. These retained-source metrics are not physical FEI or original quark reconstruction efficiencies.

See the :doc:`dashboard <_generated/status/index>` and :doc:`complete download catalogue <_generated/status/downloads>`. Historical studies and all native diagnostic gates remain preserved.

Set-overlap off/on produces 0/8,000 exact B tags and 2/8,000 versus 1/8,000 inclusive memberships in both scopes; no both-B or coherent-pair success occurs. Seventeen unavailable exact-truth trials remain nominal, while B membership is available for all8,000 trials. The on-minus-off inclusive difference is -0.000125, with paired95% interval [-0.000375,0]. Set-overlap improves nontrivial exact components260/18,409 to342/18,409 and reduces continuum fake-B events243/8,000 to160/8,000 (difference -0.010375; paired95% interval [-0.014625,-0.006]). These shallow and background gains do not establish a B-efficiency winner.

Primary tagging and coherent pairs
----------------------------------

.. list-table:: Strict greedy; nominal denominators include unavailable truth
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Set overlap off
     - Set overlap on
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
     - 2/8000 (0.025%)
     - 1/8000 (0.0125%)
   * - full inclusive event_any_correct
     - 2/4000 (0.05%)
     - 1/4000 (0.025%)
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
     - 2/8000 (0.025%)
     - 1/8000 (0.0125%)
   * - half inclusive event_any_correct
     - 2/4000 (0.05%)
     - 1/4000 (0.025%)
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
   * - set_overlap_off
     - exact
     - 17
     - 7852
     - 0
     - 8000
   * - set_overlap_off
     - inclusive
     - 0
     - 7852
     - 2
     - 8000
   * - set_overlap_on
     - exact
     - 17
     - 7852
     - 0
     - 8000
   * - set_overlap_on
     - inclusive
     - 0
     - 7852
     - 1
     - 8000

Each B-pair collision contributes two nominal B trials. Distinct truth B successes are deduplicated across every accepted candidate. Inclusive success requires an actual valid reconstructed composite with precisely the retained FSP membership; no disconnected partition is invented. Exact topology/PID availability and inclusive membership availability remain separate. Missing truth makes observed success rates proven-success lower bounds. Channel identifiers describe the retained signed representation; full generator decay descriptions and original quark ancestry are unavailable.

.. list-table:: Per-category nominal B tagging
   :header-rows: 1
   :widths: auto

   * - Population
     - Set overlap off
     - Set overlap on
   * - charged full exact
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - charged full inclusive
     - 1/4000 (0.025%)
     - 1/4000 (0.025%)
   * - charged half exact
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - charged half inclusive
     - 1/4000 (0.025%)
     - 1/4000 (0.025%)
   * - mixed full exact
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - mixed full inclusive
     - 1/4000 (0.025%)
     - 0/4000 (0%)
   * - mixed half exact
     - 0/4000 (0%)
     - 0/4000 (0%)
   * - mixed half inclusive
     - 1/4000 (0.025%)
     - 0/4000 (0%)

All three positive inclusive memberships reproduce exactly in separate tree-retaining replays. Inspection identifies actual depth-one composites containing two retained FSPs, predicted as D0 or anti-D0, rather than exact B trees. Source sets match the corresponding retained B memberships; no partition or disconnected union is invented. The charged success is shared between arms; only the off arm adds a mixed-category success. These very small retained groups reinforce the limitation of treating inclusive success as physical B-tagging efficiency.

Continuum recovery and fake-B background
----------------------------------------

.. list-table:: Explicit retained components; fake slots use two nominal slots per collision
   :header-rows: 1
   :widths: auto

   * - Population and endpoint
     - Set overlap off
     - Set overlap on
   * - full ccbar exact
     - 22/6868 (0.32033%)
     - 42/6868 (0.61153%)
   * - full ccbar inclusive
     - 283/6868 (4.1206%)
     - 284/6868 (4.1351%)
   * - full ccbar fake-B event
     - 73/2000 (3.65%)
     - 49/2000 (2.45%)
   * - full ccbar fake-B slot
     - 88/4000 (2.2%)
     - 61/4000 (1.525%)
   * - full uubar exact
     - 69/5592 (1.2339%)
     - 85/5592 (1.52%)
   * - full uubar inclusive
     - 479/5592 (8.5658%)
     - 480/5592 (8.5837%)
   * - full uubar fake-B event
     - 70/2000 (3.5%)
     - 37/2000 (1.85%)
   * - full uubar fake-B slot
     - 76/4000 (1.9%)
     - 42/4000 (1.05%)
   * - full ddbar exact
     - 86/5689 (1.5117%)
     - 97/5689 (1.705%)
   * - full ddbar inclusive
     - 540/5689 (9.492%)
     - 532/5689 (9.3514%)
   * - full ddbar fake-B event
     - 45/2000 (2.25%)
     - 31/2000 (1.55%)
   * - full ddbar fake-B slot
     - 52/4000 (1.3%)
     - 33/4000 (0.825%)
   * - full ssbar exact
     - 66/5747 (1.1484%)
     - 95/5747 (1.653%)
   * - full ssbar inclusive
     - 780/5747 (13.572%)
     - 778/5747 (13.537%)
   * - full ssbar fake-B event
     - 55/2000 (2.75%)
     - 43/2000 (2.15%)
   * - full ssbar fake-B slot
     - 63/4000 (1.575%)
     - 52/4000 (1.3%)
   * - half ccbar exact
     - 16/6868 (0.23296%)
     - 28/6868 (0.40769%)
   * - half ccbar inclusive
     - 283/6868 (4.1206%)
     - 284/6868 (4.1351%)
   * - half ccbar fake-B event
     - 73/2000 (3.65%)
     - 49/2000 (2.45%)
   * - half ccbar fake-B slot
     - 88/4000 (2.2%)
     - 61/4000 (1.525%)
   * - half uubar exact
     - 68/5592 (1.216%)
     - 79/5592 (1.4127%)
   * - half uubar inclusive
     - 479/5592 (8.5658%)
     - 480/5592 (8.5837%)
   * - half uubar fake-B event
     - 70/2000 (3.5%)
     - 37/2000 (1.85%)
   * - half uubar fake-B slot
     - 76/4000 (1.9%)
     - 42/4000 (1.05%)
   * - half ddbar exact
     - 79/5689 (1.3886%)
     - 89/5689 (1.5644%)
   * - half ddbar inclusive
     - 540/5689 (9.492%)
     - 532/5689 (9.3514%)
   * - half ddbar fake-B event
     - 45/2000 (2.25%)
     - 31/2000 (1.55%)
   * - half ddbar fake-B slot
     - 52/4000 (1.3%)
     - 33/4000 (0.825%)
   * - half ssbar exact
     - 52/5747 (0.90482%)
     - 84/5747 (1.4616%)
   * - half ssbar inclusive
     - 780/5747 (13.572%)
     - 778/5747 (13.537%)
   * - half ssbar fake-B event
     - 55/2000 (2.75%)
     - 43/2000 (2.15%)
   * - half ssbar fake-B slot
     - 63/4000 (1.575%)
     - 52/4000 (1.3%)

Component recovery is not q/qbar recovery: reduced data lack parton-to-FSP ancestry. Background acceptance and reconstruction correctness use different denominators. Unavailable component truth is retained in the complete aggregate tables. All accepted B candidates are inspected for tag recovery; slot-based fake-B accounting remains distinct from the total accepted-candidate count.

Retained topology and paired uncertainty
----------------------------------------

.. list-table:: All retained versus nontrivial populations
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Set overlap off
     - Set overlap on
   * - full lcag_pair_accuracy
     - 3285/731311 (0.44919%)
     - 3432/731311 (0.46929%)
   * - full source_precision
     - 382858/415626 (92.116%)
     - 384292/415622 (92.462%)
   * - full source_recall
     - 382858/482097 (79.415%)
     - 384292/482097 (79.713%)
   * - full coherent_retained_forest
     - 32/12000 (0.26667%)
     - 40/12000 (0.33333%)
   * - full nontrivial perfectLCAG
     - 260/18409 (1.4124%)
     - 342/18409 (1.8578%)
   * - full nontrivial source_precision
     - 31552/40036 (78.809%)
     - 32282/41057 (78.627%)
   * - full nontrivial source_recall
     - 31552/113051 (27.91%)
     - 32282/113051 (28.555%)
   * - half lcag_pair_accuracy
     - 3709/395671 (0.93739%)
     - 3775/395671 (0.95408%)
   * - half source_precision
     - 277858/307841 (90.26%)
     - 279220/308109 (90.624%)
   * - half source_recall
     - 277858/370406 (75.014%)
     - 279220/370406 (75.382%)
   * - half coherent_retained_forest
     - 32/12000 (0.26667%)
     - 40/12000 (0.33333%)
   * - half nontrivial perfectLCAG
     - 225/22409 (1.0041%)
     - 297/22409 (1.3254%)
   * - half nontrivial source_precision
     - 35664/50234 (70.996%)
     - 35844/50340 (71.204%)
   * - half nontrivial source_recall
     - 35664/113051 (31.547%)
     - 35844/113051 (31.706%)

.. list-table:: Exact shapes are not interchangeable with deep B reconstruction
   :header-rows: 1
   :widths: auto

   * - Arm
     - Scope
     - Depth-one exact components
     - Depth-two-or-deeper exact components
     - Forests with mothers
     - Mother-free forests
   * - set_overlap_off
     - full
     - 259
     - 1
     - 32
     - 0
   * - set_overlap_off
     - half
     - 224
     - 1
     - 32
     - 0
   * - set_overlap_on
     - full
     - 341
     - 1
     - 40
     - 0
   * - set_overlap_on
     - half
     - 296
     - 1
     - 40
     - 0

.. list-table:: On minus off; paired stratified collision bootstrap, 10,000 resamples
   :header-rows: 1
   :widths: auto

   * - Endpoint
     - Difference
     - 95% percentile interval
   * - full exact_tag per_b_correct
     - 0.0
     - [0.0, 0.0]
   * - full inclusive_group per_b_correct
     - -0.000125
     - [-0.000375, 0.0]
   * - full retained lcag_pair_accuracy
     - 0.00020100887310597025
     - [0.00011375314324134003, 0.0002867138436894924]
   * - full nontrivial_retained perfectLCAG
     - 0.0044543429844098
     - [0.002885111338760444, 0.006079464646708556]
   * - full nontrivial_retained source_precision
     - -0.001817975581238418
     - [-0.005644964928317975, 0.0020670848930904586]
   * - full nontrivial_retained source_recall
     - 0.006457262651369755
     - [0.0039487551721255594, 0.008991307976814621]
   * - full retained coherent_forest
     - 0.000666666666666667
     - [0.0002499999999999998, 0.0011666666666666665]
   * - full pooled_continuum exact_component_recovery
     - 0.0031804486106461333
     - [0.002052169073712625, 0.004338394793926248]
   * - full pooled_continuum inclusive_component_recovery
     - -0.00033478406427854546
     - [-0.0014281663875280637, 0.0007575287452897557]
   * - full pooled_continuum top1_fake_b_event_acceptance
     - -0.010374999999999999
     - [-0.014625000000000003, -0.005999999999999998]

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
     - 12
     - 0
   * - charged half
     - 2000
     - 2000
     - 2000
     - 0
     - 17
     - 12
     - 0
   * - mixed full
     - 2000
     - 2000
     - 2000
     - 0
     - 0
     - 5
     - 0
   * - mixed half
     - 2000
     - 2000
     - 2000
     - 0
     - 28
     - 5
     - 0
   * - ccbar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 8
     - 8
   * - ccbar half
     - 2000
     - 2000
     - 2000
     - 0
     - 8
     - 8
     - 8
   * - uubar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 6
     - 6
   * - uubar half
     - 2000
     - 2000
     - 2000
     - 0
     - 600
     - 6
     - 6
   * - ddbar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 4
     - 4
   * - ddbar half
     - 2000
     - 2000
     - 2000
     - 0
     - 627
     - 4
     - 4
   * - ssbar full
     - 2000
     - 2000
     - 2000
     - 0
     - 2000
     - 3
     - 3
   * - ssbar half
     - 2000
     - 2000
     - 2000
     - 0
     - 468
     - 3
     - 3

Full-root tree metrics do not invent a B-pair root for continuum collisions. Explicit retained components remain separately measurable. Direct-target incompatibility, exact-tag availability and membership availability have distinct denominators; missing truth never causes replacement.

The original primary cohort excludes all 98,000 prior reservations and 1,000 checkpoint-selection collisions. It is disjoint from the 70,000-event training set and train-fitted normalization. The source-safe 160,000-event validation expansion is not training-data growth. Both original selection-only checkpoints are step 4,000; primary outcomes never choose another checkpoint.

Native source: ``38124a9e05ca63cd5ac5469e0687952eec2b14dc``. Evaluator: ``7f6b28bd62c28656fd2f6600492cbfbb2225d23b``. Cohort digest: ``d837c7aae079b600ec8135b96671a278f74bea74220113c66cabb0ff66cd3115``. Selection/index digests and every checkpoint, chunk and receipt are retained. Frozen contracts and original failed gates are unchanged.

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
   * - set_overlap_off
     - 4376
     - 4000
     - False
     - minimum_half_perfect_lcag
     - True
   * - set_overlap_on
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
   * - set_overlap_off
     - independent_complete_target_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_off
     - independent_depth_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_off
     - independent_tree_validity_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_off
     - primary_complete_target_beam_direct
     - 20
     - ccbar:2, charged:3, ddbar:5, mixed:3, ssbar:3, uubar:4
   * - set_overlap_off
     - primary_complete_target_contracted_diagnostic
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_off
     - primary_complete_target_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_off
     - primary_complete_target_repeat2_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_on
     - independent_complete_target_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_on
     - independent_depth_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_on
     - independent_tree_validity_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_on
     - primary_complete_target_beam_direct
     - 20
     - ccbar:2, charged:3, ddbar:5, mixed:3, ssbar:3, uubar:4
   * - set_overlap_on
     - primary_complete_target_contracted_diagnostic
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_on
     - primary_complete_target_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20
   * - set_overlap_on
     - primary_complete_target_repeat2_direct
     - 100
     - ccbar:19, charged:11, ddbar:15, mixed:18, ssbar:17, uubar:20

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
   * - set_overlap_off
     - 175167.0
     - 104897.0
     - 442427.0
     - 331374.0
     - 111053.0
     - 396197.0
   * - set_overlap_on
     - 175167.0
     - 104897.0
     - 442427.0
     - 331401.0
     - 111026.0
     - 396081.0

Both arms execute 4,376 updates and 280,064 replay slots, seed 20261008, with zero additional pretraining. All 26 saved checkpoints have scalar censuses, finite-state and frozen-PID transfer checks. Strict results of unregistered checkpoint tracks remain unavailable.

Independent current-CPU replays retain all scientific counts and prediction fingerprints. Native repeat gates pass exactly. Against the archived runtime, 6,452 continuous values differ, with maximum absolute difference 1.63913e-06; original values and differences remain archived. No unsupported bitwise archived equality is claimed.

Geometry uses 128 fixed training and 32 previously used development collisions across four curriculum views and initial/selected/final encoders. These are diagnostic inputs, not primary coverage or a generated-state census. The complete bundle retains variance, radial derivatives, saturation, transfer and objective execution diagnostics. Daughter-sum p4 closure is an implementation invariant, not physical resolution.

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
   * - set_overlap_off
     - full
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120
   * - set_overlap_off
     - half
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120
   * - set_overlap_on
     - full
     - 60
     - {'ccbar': 10, 'charged': 10, 'ddbar': 10, 'mixed': 10, 'ssbar': 10, 'uubar': 10}
     - 120
   * - set_overlap_on
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
   * - set_overlap_off
     - full exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - full exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - full exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - full inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - full inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - full inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - half exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - half exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - half exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - half inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - half inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_off
     - half inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - full exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - full exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - full exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - full inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - full inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - full inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - half exact
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - half exact
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - half exact
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - half inclusive
     - greedy
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - half inclusive
     - model top1
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
   * - set_overlap_on
     - half inclusive
     - retained pool@2
     - 0/40 (0%)
     - 0/20 (0%)
     - 0/20 (0%)
     - 0/20 (0%)

Greedy, model-ranked top1 and retained pool at each K remain separate. Pool recovery deduplicates each true B and inspects every accepted candidate; incompatible successes are not a coherent pair. The complete download includes all beam/oracle, rank/score, candidate and uncertainty records, plus native 20-event proposal-beam diagnostics. Neither small subset inherits primary coverage.

Synthesis of all studies and allocation recommendation
------------------------------------------------------

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
     - Category-sized exact tags are 0/8000 in both arms and inclusive tags 0 versus 1/8000. Better shallow recovery trades against recall and higher fake-B acceptance; no joint winner.

The historical review and all dated studies remain in :doc:`studies` and the
complete download catalogue. Comparisons across different seeds, cohorts,
source mixtures, evaluator revisions or target policies are descriptive.
Phase69 and Phase70 cannot be treated as a paired cross-study comparison.
Neither full/half scopes, checkpoint tracks, repeated inference nor nested
beam pools multiply the collision sample size.

Training growth and pretraining recommendation
----------------------------------------------

Do not expand training automatically now. Keep 70,000 training events for the
next bounded development study. This is a decision under uncertainty, not
evidence that 70,000 is optimal or that scaling cannot help. A later controlled
learning curve must distinguish more unique examples from additional updates,
exposures and compute. Historical Phase40 cannot answer that question.

The additional validation materialization improves independent evaluation
precision and source coverage; it does not grow the training set. The former
Phase69 16-collision/32-B-trial diagnostic could not resolve rare tagging.
Even the policy minimum can leave sparse channels and zero-success uncertainty.

Task-aligned representation and objective improvements are plausible priorities;
merely running more pretraining is not supported as a remedy. Geometry repair,
repeated 70,000-event holds and hybrid encoder ablations do not prove improved
pretraining is more beneficial than data growth. The total benefit of
pretraining and a controlled scaling advantage both remain unidentified.


Phase71 auxiliary weights 0.5/1.0 yield 0/8000 exact and inclusive tags in both arms. Nontrivial exact components fall 308/18536 to 148/18536, while fake-B events change 293/8000 to 281/8000 with a paired interval spanning zero. This ends auxiliary dose tuning. Phase72 tests a different local set objective; it does not estimate the benefit of training growth, encoder width or more pretraining.

Set-overlap off/on produces 0/8,000 exact B tags and 2/8,000 versus 1/8,000 inclusive memberships in both scopes; no both-B or coherent-pair success occurs. Seventeen unavailable exact-truth trials remain nominal, while B membership is available for all8,000 trials. The on-minus-off inclusive difference is -0.000125, with paired95% interval [-0.000375,0]. Set-overlap improves nontrivial exact components260/18,409 to342/18,409 and reduces continuum fake-B events243/8,000 to160/8,000 (difference -0.010375; paired95% interval [-0.014625,-0.006]). These shallow and background gains do not establish a B-efficiency winner.

End this set-overlap coefficient contrast without a coefficient sweep or model promotion. Hold the authenticated70,000-event corpus and prioritize one bounded direct-membership development contrast: the same encoder frozen versus adapted, identical128-wide heads and shared original384-event training/60-event reused development cohorts. The existing evidence does not identify whether training-data growth or better pretraining is superior.

The new objective has measurable shallow structural and background effects, but no exact B tags or count-backed held-out B-membership improvement. Prior five-condition search has no correct complete B pool, and frozen-head tiny memorization does not generalize. This supports testing global assembly-aligned membership gradients before more local coefficient tuning, reranking, or primary scale-up. A later capacity/pretraining factorial must compare matched histories and head interfaces; simply extending unchanged pretraining is not an assembly-aligned intervention.

Latest structural policy and one bounded development campaign
-------------------------------------------------------------

The integrated policy is structural-reconstruction-study-policy-v2. The uncommitted source snapshot and its evidence were authenticated before integration; later remote/live-source checks preserve lineage. Historical study contracts remain immutable.

Five search conditions on the same reused 60 development collisions (40 B trials) all recover 0 exact and inclusive B groups at top1 and in the pool. Broader search generates more shallow proposals, but no correct complete B pool. Reranking-only work lacks the required pool/top1 gap. Optimistic clean-root coverage is a necessary bound, not legal decoder reachability.

The frozen head memorizes 32/32 raw tiny memberships, of which 28 pass candidate guards; both tiny and 384-event fits remain 0/40 held out. This is learnability evidence, not generalization or proof that encoder capacity is sufficient. The pilot has a separate 64-wide head bottleneck.

Phase73 therefore tests the policy’s direct-membership joint-learning family in development: the same historical encoder frozen versus adapted, a common 128-wide head, frozen PID parameters, identical initialization, identities and presentation budgets. The original 384 training identities and 60 reused development collisions are disjoint from Phase72 primary and selection. Fixed final checkpoints and presence threshold 0.5 avoid tuning on development. Each arm has a 1,000-update tiny-head assay and 1,000-update main fit; seed 20261008. Two CPU-only jobs each have 2 CPUs, 32 GiB, 24 hours and no requeue.

The proposed 128/256 contextual-width by existing/assembly-targeted pretraining factorial remains an unexecuted representation plan. It must hold hyperbolic width 32, depth and geometry fixed, use comparable pretraining histories, a controlled downstream interface and measured compute. Historical pretrained128 versus fresh256 cannot measure a width effect. Phase73 is an adaptation contrast and makes no capacity or pretraining claim.

Before any primary scale-up or hierarchy, require at least95% tiny raw membership plus a count-backed held-out membership gain over control, background comparison and source validity. Lower loss, memorization, extra candidates and shallow recovery alone cannot pass. No new primary cohort is reserved; the sealed test stays closed. There is no automatic second campaign or model promotion. Policy planning validation is separate from runtime/data/resource admission.

Prioritize assembly-aligned objectives and representations before purchasing uncontrolled training growth or simply longer unchanged pretraining. This allocation recommendation does not establish that better pretraining beats data growth. A later growth study must separate more unique examples from optimizer presentations and compute; repeated70,000-event holds do not prove scaling useless.

At 2026-10-08 08:31 CEST, the two development jobs are accepted: frozen: RUNNING, adapted: RUNNING. Frozen source ``1eba6c7643d8c2674fb068575aea22dce1df9dd8``. Runtime admission authenticates the checkpoint, data, source, cohort exclusions and fixed resource budget. Job completion and scientific effectiveness are not required or inferred from this scheduling snapshot.

Uncertainty and interpretation limits
-------------------------------------

All paired intervals resample whole collisions within category, keeping both B trials together. They condition on two fitted models; single-seed uncertainty and multiplicity are not covered. Exact binomial collision bounds accompany sparse or zero successes. A zero-width empirical bootstrap is not zero population uncertainty. Missing truth remains nominal, and equal-category pooled rates are not physical mixture estimates.

.. list-table:: Sparse bounds for collision event-any proven success
   :header-rows: 1
   :widths: auto

   * - Arm
     - Population
     - Successes
     - Collisions
     - One-sided95% upper
   * - set_overlap_off
     - full greedy charged exact
     - 0.0
     - 2000.0
     - 0.0014967448951882837
   * - set_overlap_off
     - full greedy charged inclusive
     - 1.0
     - 2000.0
     - 0.002369713367656843
   * - set_overlap_off
     - full greedy mixed exact
     - 0.0
     - 2000.0
     - 0.0014967448951882837
   * - set_overlap_off
     - full greedy mixed inclusive
     - 1.0
     - 2000.0
     - 0.002369713367656843
   * - set_overlap_on
     - full greedy charged exact
     - 0.0
     - 2000.0
     - 0.0014967448951882837
   * - set_overlap_on
     - full greedy charged inclusive
     - 1.0
     - 2000.0
     - 0.002369713367656843
   * - set_overlap_on
     - full greedy mixed exact
     - 0.0
     - 2000.0
     - 0.0014967448951882837
   * - set_overlap_on
     - full greedy mixed inclusive
     - 0.0
     - 2000.0
     - 0.0014967448951882837

Complete metric downloads and integrity
---------------------------------------

:doc:`Download all latest and historical metric files <_generated/status/downloads>`. Policy manifests, all parts, standard-library decoders and integrity/cardinality records are preserved together. The policy bundle includes all full/half primary aggregates, channel/type coverage, exact/inclusive tagging, beam top1/pool/coherent accounting, paired uncertainty and native diagnostics.

New payload files use bounded gzip/base32 JSON envelopes of authenticated public aggregates. The supplied standalone decoders verify compressed and decoded hashes, expansion limits and cardinalities. This lossless transport preserves every historical download and retains the fixed site/file/privacy limits.

Every per-step training and checkpoint metadata scalar is also downloadable in the four ``phase72-set-overlap-*-scalars.json`` files, with ``phase72-scalar-integrity.json`` and the bounded ``phase72-scalar-decoder.txt``. Native aggregate records: ``phase72-native-aggregates.json``; integrity and decoder: ``phase72-native-integrity.json`` and ``phase72-native-decoder.txt``. Policy entrypoint: ``phase72-policy-reevaluation-v1.json``; integrity and decoder: ``phase72-policy-integrity.json`` and ``phase72-policy-decoder.txt``. Download every referenced part before decoding.

The complete public scientific bundle contains 1,761,151 numeric records; the native bundle contains 184,348 aggregate scalar records. Full private censuses preserve 12,788,071 native scalars and 59,528 checkpoint metadata scalars, plus every primary report, candidate, per-step training scalar and repeat. Counts measure exported records, not independent observations.

Public aggregates retain null availability and all reviewed count/label fields while excluding event identifiers, private paths, logs, scheduler IDs and weights. Full private evidence and hash manifests remain in the task artifact bundle. No historical download or original failed gate has been removed.
