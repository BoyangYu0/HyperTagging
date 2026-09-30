Phase64 radial reconditioning study
===================================

The completed Phase63 encoders remain radially saturated on a development sample.
A scale fitted only on 128 training events removes saturation in the measured
views, restoring mean cap derivatives to roughly 0.67--0.72. This supports a
controlled experiment; it does not establish reconstruction improvement.

Phase64 compares unchanged parameter initialization with a one-time projection
weight and bias multiplier of approximately 0.01694. Both arms receive the same
2188-step pretraining refinement and 4376-step masking-only reconstruction.
The 70000 training events, loss weights, architecture, radius cap, normalization
policy, inference thresholds and search budgets remain matched.

A fresh reservation of 1000 selection and 100 strict validation events excludes
all 53300 historical reservations; beam evaluation uses a fixed 20-event subset.
There are 45600 validation events left unreserved. Review radial usability after
refinement together with nontrivial topology, exact component depth, source
precision/recall and coherent forests. Isolated leaves and generated depth must
not stand in for recursive accuracy.

Both study arms were accepted and released from the verified immutable source.
The first scheduler observation is pending priority; GPU startup is unverified.
The full CPU suite passed 2011 tests with 34 skips, and the frozen source passed
all 34 focused tests.
There is no automatic restart, successor, promotion or sealed-test access.
The detailed repository specification is ``docs/phase64_plan_20260930.md``.
This page records the study contract and does not poll the scheduler.
