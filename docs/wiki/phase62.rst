Phase62 closeout and Phase63 corrected supervision
==================================================

Both Phase62 jobs completed under their original frozen contract. The control
passes original gates; the lower late-PID candidate fails full LCAG and exact
mother coverage. Neither establishes a reproducible recursive quality winner.

.. list-table:: Native retained-tree results
   :header-rows: 1

   * - Endpoint
     - Control
     - Lower late PID
   * - Full LCAG
     - 23 / 3274
     - 21 / 3274
   * - Exact nontrivial full components
     - 9 / 154
     - 11 / 154
   * - Coherent retained forests
     - 1 / 100
     - 2 / 100

Every exact component and coherent-forest mother remains depth one. Original
full LCAG is 2/2789 versus 0/2789. Component beam oracle is only 3/559 LCAG pairs
and 1/28 exact nontrivial components in both arms. Larger beams or another
PID-weight sweep are not supported by these results.

The repairs share legal forest eligibility, correct missing-slot confidence and
recovery, exclude repeated-event retrieval peers, publish supported parent
metrics, diagnose radial saturation and support per-level ONNX capacities.
Frozen development checks find two falsely representable targets among 48 truth
targets on 16 events, and radial saturation throughout a 32-event sample.

Phase63 compares corrected unmatched-slot recovery with masking-only supervision
from one shared Phase62 control encoder. No additional pretraining, promotion,
sealed-test access or automatic successor is authorized. It reserves fresh 1000
selection and 100 strict events; 46700 validation UIDs remain untouched.

The detailed review is available in the repository as
``docs/phase62_review_phase63_20260928.md``. Source-bound compact results are in
the repository evidence index; large scalar and tree exports remain on the
project data volume. This page records reviewed evidence;
it does not poll the scheduler.

Both Phase63 arms were accepted and released from immutable source
``aeefabdaab6f05bce817081f43109c0a8641827d``. The submission snapshot records
both waiting for scheduler priority; it does not establish GPU startup or
training completion. Verification passed 1,958 CPU tests with 34 skips, both
two-step training dry runs, and source/contract integrity checks.
