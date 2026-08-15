# HyperTagging pretraining run 15749624 report

This package is the verified forensic report for the H100-NVL pretraining job
`15749624` at repository commit
`1ae7d74f9d82822b55c4ff8923f1c07981b953e4`.

The Slurm job ended `FAILED` after 3,342 of 17,500 planned optimizer steps when
the next clipped gradient norm became non-finite. All 14 saved checkpoints are
CPU-loadable and contain finite model and optimizer tensors, but the run never
left curriculum phase 1 and no checkpoint is promoted as a completed pretrained
model. `best.pt` at step 500 is the safest available diagnostic snapshot.

Files:

- `report.html`: self-contained interactive technical report.
- `artifact.json`: validated canonical report artifact.
- `evidence.json`: deterministic source evidence, including full real-sample
  embedding rows for 64 fixed-cohort validation events.
- `collect_pretraining_evidence.py`: remote evidence extractor; explicitly uses
  the validation role and never requests the sealed test.
- `build_pretraining_report.py`: report dataset and artifact builder.
- `checkpoint_audit.py`: compact checkpoint tensor/hash audit.
- `render_qa.cjs`: independent Microsoft Edge render and console-error QA.

The report artifact passed the portable report schema/package checks. Independent
Edge rendering produced 43 SVG elements, 25 rendered tables, and no browser
console errors.
