# V13 verifier v4 implementation report

Verdict: **BLOCKED — critical F01/capability/path behavior is corrected, but exact per-oracle T002–T194 semantic fixtures are not complete.** All authorization and action flags remain false.

Implementation commit `4d167110ea46a814c4e9ed5fcce5622fe26c234e`, tag `ht-joint-terminal-evidence-verifier-v13-implementation-20260827-v4`, was applied on sealed v3 report head `51e81d8f0bf717c0ba1b949126f4e86a84a7fecb`.

V4 validates terminal and pair, checks the immutable compiled F01 tuple, raises exact `E_FEASIBILITY_SHAPE_AUTHORITY`, does not open locator or downstream nodes, and cannot issue a capability. The capability class, constructor seal, and weak live-identity registry exist only inside a closure and are not importable or mutable module globals. Trusted-root containment now checks the root and every descendant with `lstat`, owner/mode policy, `O_NOFOLLOW`, and stable open/fstat identity; nested JSON refs are recursively reopened and routed to specialized validators.

Targeted v13 suites: 206 passed. V4 collection: 196 substantive production calls, covering named T001–T194 plus capability/API hardening, with no skip or xfail. However, T002–T194 currently use five concrete fail-closed mutation families and therefore do not yet satisfy the stronger requirement for every oracle's exact domain mutation and exact expected code. The repository has no complete non-null synthetic terminal/pair/native evidence corpus for those calculations; the large 118,500-record/1,500-batch and ABBA/statistical fixtures remain to be built.

Full repository collection truthfully fails on two missing `nbformat` imports. Without those two modules: 762 passed, 8 skipped, and 5 unrelated existing tests failed. No package was installed.

Runtime used only uv 0.5.20 with `--frozen --no-sync`; no payload, GPU, Slurm, scheduler, or scientific execution occurred.
