# Phase-3/4 reconstruction terminal decision

Status: **terminal comparison complete; no production promotion**.

The preregistered primary metric selects the step-54,064 exploratory baseline: its best rollout edge F1 was 0.046 at reconstruction step 500, versus 0.042 for step 81,096 and 0.030 for step 108,128. Terminal scores preserve the same ordering (0.034, 0.028, 0.026). Later checkpoints improve teacher-forced validation loss but do not win the declared rollout endpoint.

Step 54,064 did not pass the pretraining success gate. Steps 81,096 and 108,128 did, but neither wins the downstream primary. Therefore there is no checkpoint that satisfies both conditions. The decision is `NO_PROMOTION`; the sealed test remains closed, and step 81,096 must not be selected post hoc.

Job 16251299 failed before training because its legacy selection entry lacked builder metadata. Metadata-only repromotion repaired that input, and jobs 16251359, 16251363, and 16251364 completed normally with unchanged source checkpoints, finite outputs, identical cohorts and configuration, and all structural gates passing.

The next authorized scientific action is validation-only paired confirmation of the saved step-500 checkpoints from the 54,064 and 81,096 arms on two new disjoint validation cohorts. It cannot promote a checkpoint automatically and cannot access sealed-test data.
