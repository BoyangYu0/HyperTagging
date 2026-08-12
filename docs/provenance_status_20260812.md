# Training provenance status — 2026-08-12

The integration source is local branch `training-integration-20260812` at the
clean no-ff merge `83776a2c4f9e7edd776915edf30816730881f9eb`. The two inputs are
preserved without rewriting history:

- `pre-training-master-537d7cc` preserves master tip `537d7cc`.
- `pre-training-focused-8e9d0b8` preserves focused tip `8e9d0b8`.

The 1M-event reduced campaign declares source commit
`f4e54df23b5c60115e475c5d68df4651899d678e` and tree
`b6e3a4118b960e3a4676a61af9601438d56cef96`. The commit object is unavailable
in the inspected repositories, and GitHub returned `not our ref`. Therefore the
tree cannot yet be independently derived from the commit object.

This is a fail-closed scientific provenance blocker, not a CPU-development
blocker. CPU implementation, manifest generation, and focused tests may
continue. Scientific Slurm submission remains forbidden until the object is
recovered from an archive/campaign worktree or equivalent immutable bundle,
its tree is verified, the UID/index gate passes, and later trainer/Slurm gates
are complete.

The authoritative machine-readable status is
`configs/training_selection/production_1m_20260812/provenance_status.json`.
Validate it with:

```bash
python scripts/validate_training_provenance.py
python scripts/validate_training_provenance.py --require-scientific-slurm-ready
```

The first command validates the document and reports blockers. The second must
exit nonzero while any scientific blocker remains.
